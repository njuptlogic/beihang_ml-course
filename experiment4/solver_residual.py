"""Endpoint correction from a frozen few-step velocity history, following the spatial operator idea of SpanLift."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from course.common import dump, table
from experiment4.learned_solver import (
    OUT,
    SPLITS,
    combine,
    initialize,
    noise,
    parameters,
    velocity,
)
from experiment4.research_evaluate import atomic_array
from experiment4.research_train import file_hash

DIRECTORY = OUT / "residual"
SEEDS = (17, 73)
VARIANTS = ("point", "spatial")
BUDGET = 4
PARENT = "bands_4"


def protocol():
    DIRECTORY.mkdir(exist_ok=True)
    cfg = dict(
        parent=PARENT,
        parent_sha256=file_hash(OUT / f"{PARENT}.json"),
        target="24 midpoint steps (48 calls) of frozen time-only epoch 32, guidance 3",
        fit=1200,
        validation=400,
        epochs=24,
        batch=32,
        seeds=list(SEEDS),
        primary_budget=4,
        variants=list(VARIANTS),
        checkpoint="lowest validation clipped pixel MSE for each seed",
        selection="lowest validation MSE of the two-seed prediction mean including unmodified base",
        origin="Learning to Solve Generative ODEs Beyond the Linear Span (2026-06-07), endpoint-only adaptation",
    )
    p = DIRECTORY / "protocol.json"
    if p.exists() and json.loads(p.read_text()) != cfg:
        raise ValueError("Different residual training protocol at destination")
    dump(p, cfg)


@torch.no_grad()
def buffer(model, z, y):
    k, family, rows = parameters(PARENT)
    history = [z]
    x = z
    for i in range(k):
        history.append(velocity(model, x, y, i / k))
        x = combine(history, rows[i], family)
    return torch.cat([x, *history], 1)


class Corrector(nn.Module):
    def __init__(self, variant):
        super().__init__()
        kernel = 3 if variant == "spatial" else 1
        width = 48
        self.body = nn.Sequential(
            nn.Conv2d(3 * (BUDGET + 2), width, 1),
            nn.SiLU(),
            nn.Conv2d(
                width,
                width,
                kernel,
                padding=kernel // 2,
                groups=width,
                padding_mode="replicate",
            ),
            nn.Conv2d(width, width, 1),
            nn.SiLU(),
            nn.Conv2d(
                width,
                width,
                kernel,
                padding=kernel // 2,
                groups=width,
                padding_mode="replicate",
            ),
            nn.Conv2d(width, 3, 1),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, inputs):
        return inputs[:, :3] + self.body(inputs)


def cache(model, device):
    for name, (n, seed) in SPLITS.items():
        path = DIRECTORY / f"{name}_buffer.npy"
        if path.is_file():
            if np.load(path, mmap_mode="r").shape != (n, 3 * (BUDGET + 2), 32, 32):
                raise ValueError(path)
            continue
        z, y = noise(n, seed)
        x = torch.cat(
            [
                buffer(model, zz.to(device), yy.to(device)).cpu().half()
                for zz, yy in zip(z.split(32), y.split(32))
            ]
        )
        atomic_array(path, x.numpy())


def data(split):
    x = torch.from_numpy(np.load(DIRECTORY / f"{split}_buffer.npy"))
    y = torch.from_numpy(
        np.load(OUT / f"teacher_{split}.npy", mmap_mode="r")[:, -1].copy()
    )
    return x, y


@torch.no_grad()
def predict_set(model, x, device):
    return torch.cat(
        [model(xx.to(device, dtype=torch.float32)).cpu() for xx in x.split(32)]
    )


def load_variant(variant, device):
    models = []
    for seed in SEEDS:
        net = Corrector(variant).to(device)
        ck = torch.load(
            DIRECTORY / f"{variant}_{seed}.pt", map_location=device, weights_only=True
        )
        net.load_state_dict(ck["weights"])
        models.append(net.eval())
    return models


def train(device):
    x, y = data("fit")
    vx, vy = data("validation")
    vy = vy.float().clamp(-1, 1)
    summaries = []
    for variant in VARIANTS:
        for seed in SEEDS:
            ck = DIRECTORY / f"{variant}_{seed}.pt"
            record = DIRECTORY / f"{variant}_{seed}.csv"
            if record.exists() and ck.exists():
                continue
            torch.manual_seed(seed)
            rng = torch.Generator().manual_seed(seed)
            model = Corrector(variant).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.01)
            schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, 24, eta_min=0.0002
            )
            best = float("inf")
            trace = []
            for epoch in range(1, 25):
                model.train()
                total = 0.0
                for ids in torch.randperm(len(x), generator=rng).split(32):
                    xx = x[ids].to(device, dtype=torch.float32)
                    yy = y[ids].to(device, dtype=torch.float32)
                    output = model(xx)
                    loss = F.mse_loss(
                        output.clamp(-1, 1), yy.clamp(-1, 1)
                    ) + 0.05 * F.mse_loss(output, yy)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    total += loss.item() * len(ids)
                schedule.step()
                model.eval()
                err = F.mse_loss(predict_set(model, vx, device).clamp(-1, 1), vy).item()
                trace.append(
                    dict(epoch=epoch, training_loss=total / len(x), validation_mse=err)
                )
                if err < best:
                    temp = ck.with_suffix(".tmp")
                    torch.save(
                        dict(
                            weights=model.state_dict(), epoch=epoch, validation_mse=err
                        ),
                        temp,
                    )
                    temp.replace(ck)
                    best = err
            table(record, trace)
            print(variant, seed, best, flush=True)
        nets = load_variant(variant, device)
        predictions = torch.stack([predict_set(net, vx, device) for net in nets]).mean(
            0
        )
        summaries.append(
            dict(
                method=variant,
                mse=float(F.mse_loss(predictions.clamp(-1, 1), vy)),
                trainable_per_seed=sum(p.numel() for p in nets[0].parameters()),
            )
        )
    summaries.append(
        dict(
            method="base",
            mse=float(F.mse_loss(vx[:, :3].float().clamp(-1, 1), vy)),
            trainable_per_seed=0,
        )
    )
    table(DIRECTORY / "validation.csv", summaries)
    dump(DIRECTORY / "selection.json", min(summaries, key=lambda v: v["mse"]))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["cache", "train"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    d = torch.device(a.device)
    model = initialize(d)
    protocol()
    cache(model, d)
    if a.action == "train":
        del model
        train(d)
