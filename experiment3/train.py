"""Spatial-density regression with factorial tests of feature fusion and a gaze prior."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from course.common import ROOT, configure, table, dump, batches

BASE = ROOT / "experiment3"
VARIANTS = ("deep", "multiscale", "deep_prior", "multiscale_prior")
SEEDS = (17, 42, 73)
GRID = (24, 40)


def density(x):
    x = x.flatten(1).clamp_min(1e-12)
    return x / x.sum(1, keepdim=True)


def compare(truth, pred):
    p = density(truth)
    q = density(pred)
    cc = F.cosine_similarity(
        p - p.mean(1, keepdim=True), q - q.mean(1, keepdim=True), dim=1, eps=1e-12
    )
    kl = (p * (p.log() - q.log())).sum(1)
    sim = torch.minimum(p, q).sum(1)
    return torch.stack((cc, kl, sim), 1)


def resize(x, size):
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class DensityHead(nn.Module):
    def __init__(self, variant, log_prior):
        super().__init__()
        self.multiscale = "multiscale" in variant
        self.prior = "prior" in variant
        channels = [128, 256, 512] if self.multiscale else [512]
        self.paths = nn.ModuleList(
            [
                nn.Sequential(nn.Conv2d(c, 16, 1), nn.GroupNorm(4, 16), nn.SiLU())
                for c in channels
            ]
        )
        self.readout = nn.Sequential(
            nn.Conv2d(16 * len(channels), 32, 3, padding=1),
            nn.GroupNorm(4, 32),
            nn.SiLU(),
            nn.Conv2d(32, 1, 1),
        )
        if self.prior:
            self.gain = nn.Parameter(torch.ones(()))
            self.register_buffer("log_prior", log_prior)

    def forward(self, features):
        chosen = features if self.multiscale else features[-1:]
        h = torch.cat(
            [resize(layer(x), GRID) for x, layer in zip(chosen, self.paths)], 1
        )
        logits = self.readout(h)
        if self.prior:
            logits = logits + self.gain * self.log_prior
        return logits.flatten(1).softmax(1).reshape(-1, 1, *GRID)


def main(out, device):
    configure(42)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(BASE / "data/images.csv")
    splits = df.split.to_numpy()
    fit = np.flatnonzero(splits == "fit")
    val = np.flatnonzero(splits == "validation")
    # Memory mapping keeps the large feature bank on host. Only current mini-batches use GPU memory.
    feats = [np.load(BASE / f"data/stage{s}.npy", mmap_mode="r") for s in (2, 3, 4)]
    gt = (
        torch.from_numpy(np.load(BASE / "data/targets.npy")).to(device)[:, None].float()
    )
    low = F.interpolate(gt, size=GRID, mode="area")
    prior = density(low[fit]).mean(0).reshape(1, 1, *GRID)
    np.save(out / "prior.npy", prior.cpu().numpy())
    rows = []

    def evaluate(name, seed, pred, epoch, params):
        parts = []
        for ids in batches(np.arange(len(df)), 48):
            pred_batch = torch.as_tensor(pred[ids], device=device).float()
            parts.append(
                compare(gt[ids], resize(pred_batch, gt.shape[-2:])).cpu().numpy()
            )
        result = df[["index", "member", "category", "split"]].copy()
        result[["cc", "kl", "sim"]] = np.concatenate(parts)
        run = out / (name if seed == -1 else f"{name}_{seed}")
        run.mkdir(exist_ok=True)
        result.to_csv(run / "per_image.csv", index=False)
        for part in ("fit", "validation", "train_full", "test"):
            data = (
                result[result.split != "test"]
                if part == "train_full"
                else result[result.split == part]
            )
            rows.append(
                dict(
                    variant=name,
                    seed=seed,
                    split=part,
                    epoch=epoch,
                    parameters=params,
                    **data[["cc", "kl", "sim"]].mean().to_dict(),
                )
            )

    evaluate("spatial_prior", -1, prior.expand(len(df), -1, -1, -1), 0, 0)
    for variant in VARIANTS:
        for seed in SEEDS:
            configure(seed)
            run = out / f"{variant}_{seed}"
            run.mkdir(exist_ok=True)
            model = DensityHead(variant, prior.log()).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=0.0015, weight_decay=0.01)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 14, eta_min=0.00015)

            def forward(ids):
                chosen = feats if model.multiscale else feats[-1:]
                return model(
                    [
                        torch.from_numpy(a[ids]).to(device, dtype=torch.float32)
                        for a in chosen
                    ]
                )

            best = -float("inf")
            history = []
            for epoch in range(1, 15):
                model.train()
                total = 0
                for ids in batches(fit, 32, True):
                    pred = forward(ids)
                    cc, kl, _ = compare(low[ids], pred).mean(0)
                    loss = kl + 0.3 * (1 - cc)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    total += loss.detach() * len(ids)
                model.eval()
                with torch.no_grad():
                    metrics = torch.cat(
                        [
                            compare(gt[ids], resize(forward(ids), gt.shape[-2:]))
                            for ids in batches(val, 32)
                        ]
                    ).mean(0)
                cc, kl, sim = metrics.tolist()
                history.append(
                    dict(
                        epoch=epoch,
                        training_loss=float(total) / len(fit),
                        val_cc=cc,
                        val_kl=kl,
                        val_sim=sim,
                    )
                )
                if cc > best:
                    best = cc
                    torch.save(
                        dict(
                            model=model.state_dict(),
                            variant=variant,
                            seed=seed,
                            epoch=epoch,
                        ),
                        run / "best.pt",
                    )
                sched.step()
            ck = torch.load(run / "best.pt", weights_only=True, map_location=device)
            model.load_state_dict(ck["model"])
            with torch.inference_mode():
                pred = torch.cat(
                    [forward(ids).cpu() for ids in batches(np.arange(len(df)), 32)]
                ).numpy()
            np.save(run / "density.npy", pred.astype(np.float32))
            evaluate(
                variant,
                seed,
                pred,
                ck["epoch"],
                sum(p.numel() for p in model.parameters()),
            )
            table(run / "history.csv", history)
            table(out / "metrics.csv", rows)
            print(variant, seed, ck["epoch"], best, flush=True)
    frame = pd.DataFrame(rows)
    selected = (
        frame.query("split=='validation' and seed!=-1")
        .groupby("variant")
        .cc.mean()
        .idxmax()
    )
    ensemble = np.mean(
        [np.load(out / f"{selected}_{seed}/density.npy") for seed in SEEDS], axis=0
    )
    evaluate("ensemble", -1, ensemble, 0, 0)
    np.save(out / "ensemble/density.npy", ensemble)
    table(out / "metrics.csv", rows)
    dump(out / "selection.json", dict(variant=selected, seeds=SEEDS))
    images = pd.read_csv(out / "ensemble/per_image.csv")
    images.query("split=='test'").groupby("category")[["cc", "kl", "sim"]].agg(
        ["mean", "std", "count"]
    ).to_csv(out / "categories.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=BASE / "outputs")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    main(a.output, a.device)
