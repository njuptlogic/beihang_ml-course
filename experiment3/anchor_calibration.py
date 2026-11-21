"""Adapt an independent frozen fixation predictor, then choose its contribution on validation."""
import argparse, json
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.nn import functional as F
from course.common import batches, configure, dump, table
from experiment3.multiscale_train import BASE, MID, OUT, Dataset, OUTPUT_GRID, SEEDS
from experiment3.train import compare, resize


class Calibrator(nn.Module):
    """Temperature, shared fixation prior and smooth low-frequency spatial residual."""

    def __init__(self, adaptive):
        super().__init__()
        self.adaptive = adaptive
        h, w = OUTPUT_GRID
        yy = torch.pi * (torch.arange(h) + 0.5) / h
        xx = torch.pi * (torch.arange(w) + 0.5) / w
        harmonics = [(i, j) for i in range(5) for j in range(5) if 0 < i + j <= 4]
        basis = torch.stack(
            [(i * yy[:, None]).cos() * (j * xx[None, :]).cos() for i, j in harmonics]
        )
        self.register_buffer("basis", basis)
        n = len(basis) + 2
        self.control = nn.Parameter(torch.zeros(n))
        if adaptive:
            self.ctx = nn.Sequential(
                nn.Linear(48 + 2, 32), nn.SiLU(), nn.Dropout(0.25), nn.Linear(32, n)
            )
            nn.init.zeros_(self.ctx[-1].weight)
            nn.init.zeros_(self.ctx[-1].bias)

    def forward(self, ext, global_context, prior):
        logu = ext.clamp_min(1e-12).log()
        logu = logu - logu.mean((-2, -1), keepdim=True)
        logpi = prior.clamp_min(1e-12).log()
        logpi = logpi - logpi.mean((-2, -1), keepdim=True)
        controls = self.control[None].expand(len(ext), -1)
        if self.adaptive:
            summary = torch.stack(
                [
                    logu.flatten(1).std(1),
                    torch.log(ext.flatten(1).amax(1) * ext[0].numel()),
                ],
                1,
            )
            controls = controls + self.ctx(torch.cat([global_context, summary], 1))
        score = (
            logu * controls[:, 0, None, None, None].clamp(-1.5, 1.5).exp()
            + controls[:, 1, None, None, None] * logpi
            + torch.einsum("bc,chw->bhw", controls[:, 2:], self.basis)[:, None]
        )
        return score.flatten(1).softmax(1).reshape_as(ext)


def setup(device):
    ds = Dataset(device)
    u = torch.from_numpy(np.load(OUT / "unisal.npy"))
    ctx = torch.from_numpy(ds.ctx)
    return ds, u, ctx


def train(device):
    ds, u, ctx = setup(device)
    for arm in ["global", "adaptive"]:
        banks = []
        for seed in [17] if arm == "global" else SEEDS:
            configure(seed)
            m = Calibrator(arm == "adaptive").to(device)
            opt = torch.optim.AdamW(m.parameters(), lr=0.003, weight_decay=0.025)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 18, eta_min=0.0002)
            best = -np.inf
            run = OUT / f"cal_{arm}_{seed}"
            run.mkdir(exist_ok=True)
            history = []

            @torch.inference_mode()
            def predict(ids):
                m.eval()
                return torch.cat(
                    [
                        m(u[b].to(device), ctx[b].to(device), ds.prior).cpu()
                        for b in batches(ids, 48)
                    ]
                )

            for e in range(1, 19):
                m.train()
                total = 0.0
                for ids in batches(ds.fit, 40, shuffle=True):
                    out = m(u[ids].to(device), ctx[ids].to(device), ds.prior)
                    cc, kl, sim = compare(ds.targets[ids].to(device), out).mean(0)
                    loss = 0.8 * kl + 0.8 * (1 - cc) + 0.25 * (1 - sim)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    total += len(ids) * loss.item()
                cc, kl, sim = ds.score(predict(ds.val), ds.val)
                history.append(
                    dict(epoch=e, loss=total / len(ds.fit), cc=cc, kl=kl, sim=sim)
                )
                sched.step()
                if cc > best:
                    best = cc
                    torch.save(
                        dict(model=m.state_dict(), epoch=e, arm=arm, seed=seed),
                        run / "best.pt",
                    )
            table(run / "history.csv", history)
            m.load_state_dict(
                torch.load(run / "best.pt", map_location=device, weights_only=True)[
                    "model"
                ]
            )
            banks.append(predict(ds.allids))
            print(arm, seed, best, flush=True)
        np.save(OUT / f"cal_{arm}.npy", torch.stack(banks).mean(0).numpy())


def combine(ds, weights, collection):
    return sum(w * collection[n] for n, w in weights.items())


def select(device):
    ds, u, ctx = setup(device)
    sources = {
        "dino_selected": torch.from_numpy(np.load(OUT / "dino_selected.npy")),
        "unisal": u,
        **{
            n: torch.from_numpy(np.load(OUT / f"{n}.npy"))
            for n in ("cal_global", "cal_adaptive")
        },
    }
    val = {n: x[ds.val] for n, x in sources.items()}
    candidates = []
    for anchor in ("unisal", "cal_global", "cal_adaptive"):
        for w in (0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.65, 0.80, 1.0):
            weights = {"dino_selected": 1 - w, anchor: w}
            cc, kl, sim = ds.score(combine(ds, weights, val), ds.val)
            candidates.append(
                dict(anchor=anchor, anchor_weight=w, cc=cc, kl=kl, sim=sim)
            )
    table(OUT / "fusion_validation.csv", candidates)
    r = max(candidates, key=lambda t: t["cc"])
    weights = {
        n: v
        for n, v in {
            "dino_selected": 1 - r["anchor_weight"],
            r["anchor"]: r["anchor_weight"],
        }.items()
        if v > 0
    }
    dump(
        OUT / "selection.json",
        dict(
            **r,
            weights=weights,
            selection_split="validation",
            metric="CC",
            pretrained_external_checkpoint="original UNISAL weights_best, SALICON branch",
        ),
    )
    np.save(OUT / "selected.npy", combine(ds, weights, sources).numpy())
    print("Final validation choice", r, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["train", "select"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    configure(42)
    (train if a.action == "train" else select)(a.device)
