"""Factorial study of local features and image-routed gaze priors. Lock choices on validation."""
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from torch import nn
from torch.nn import functional as F
from course.common import batches, configure, dump, table
from experiment3.train import BASE, SEEDS, GRID, compare, density, resize
from experiment3.semantic_features import CACHE, OUT

VARIANTS = ("tokens", "tokens_local", "tokens_routed", "tokens_local_routed")
EPOCHS = 16


def make_priors(meta, fit, low):
    tokens = np.load(CACHE / "cls.npy").astype("float64")
    mu = tokens[fit].mean(0)
    sd = tokens[fit].std(0).clip(1e-5)
    x = (tokens - mu) / sd
    cls = LogisticRegression(C=0.025, max_iter=500)
    with threadpool_limits(2):
        cls.fit(x[fit], meta.category.iloc[fit])
        p = cls.predict_proba(x)
    shared = density(low[fit]).mean(0).reshape(1, 1, *GRID)
    probs = torch.tensor(p, dtype=torch.float32)
    classes = cls.classes_
    by_class = torch.stack(
        [density(low[fit[meta.category.to_numpy()[fit] == c]]).mean(0) for c in classes]
    )
    bank = 0.85 * by_class + 0.15 * shared.flatten()
    adapted = (probs @ bank).reshape(-1, 1, *GRID)
    np.savez(
        OUT / "router.npz",
        mean=mu,
        std=sd,
        coef=cls.coef_,
        intercept=cls.intercept_,
        classes=classes.astype("U"),
        bank=bank.numpy(),
        shared=shared.numpy(),
    )
    results = meta[["index", "category", "split"]].copy()
    results["predicted"] = classes[np.argmax(p, axis=1)]
    results["max_probability"] = p.max(1)
    for k, c in enumerate(classes):
        results["p_" + c] = p[:, k]
    results.to_csv(OUT / "routes.csv", index=False)
    return shared, adapted


class SemanticHead(nn.Module):
    def __init__(self, variant):
        super().__init__()
        self.local = "local" in variant
        self.tokens = nn.Sequential(
            nn.Conv2d(128, 48, 1), nn.GroupNorm(6, 48), nn.SiLU()
        )
        if self.local:
            self.texture = nn.Sequential(
                nn.Conv2d(128, 16, 1), nn.GroupNorm(4, 16), nn.SiLU()
            )
        self.read = nn.Sequential(
            nn.Conv2d(64 if self.local else 48, 40, 3, padding=1),
            nn.GroupNorm(5, 40),
            nn.SiLU(),
            nn.Dropout2d(0.12),
            nn.Conv2d(40, 1, 1),
        )
        self.gain = nn.Parameter(torch.ones(()))
        nn.init.zeros_(self.read[-1].weight)
        nn.init.zeros_(self.read[-1].bias)

    def forward(self, mid, deep, prior, local=None):
        x = resize(self.tokens(torch.cat((mid, deep), dim=1)), GRID)
        if self.local:
            x = torch.cat((x, self.texture(local)), dim=1)
        score = self.read(x) + self.gain * prior.clamp_min(1e-10).log()
        return score.flatten(1).softmax(1).reshape(-1, 1, *GRID)


def train(device):
    configure(42)
    OUT.mkdir(exist_ok=True, parents=True)
    meta = pd.read_csv(BASE / "data/images.csv")
    fit = meta.index[meta.split == "fit"].to_numpy(copy=True)
    val = meta.index[meta.split == "validation"].to_numpy(copy=True)
    gt = torch.from_numpy(np.load(BASE / "data/targets.npy"))[:, None].float()
    low = F.interpolate(gt, size=GRID, mode="area")
    shared, routed = make_priors(meta, fit, low)
    features = [
        np.load(p, mmap_mode="r")
        for p in (CACHE / "block6.npy", CACHE / "block12.npy", BASE / "data/stage2.npy")
    ]

    def forward(model, variant, ids, flip=False):
        tensors = [
            torch.from_numpy(x[ids]).to(device, dtype=torch.float32)
            for x in features[: 3 if model.local else 2]
        ]
        prior = (
            routed[ids] if "routed" in variant else shared.expand(len(ids), -1, -1, -1)
        )
        prior = prior.to(device)
        if flip:
            tensors = [t.flip(-1) for t in tensors]
            prior = prior.flip(-1)
        return model(tensors[0], tensors[1], prior, tensors[2] if model.local else None)

    @torch.inference_mode()
    def predict(model, variant, ids):
        model.eval()
        return torch.cat(
            [forward(model, variant, idx).cpu() for idx in batches(ids, 32)]
        )

    def measures(pred, idx):
        parts = []
        for start in range(0, len(idx), 48):
            sl = slice(start, start + 48)
            parts.append(compare(gt[idx[sl]], resize(pred[sl], gt.shape[-2:])))
        return torch.cat(parts).numpy()

    val_preds = {}
    seed_rows = []
    dump(
        OUT / "protocol.json",
        dict(
            variants=VARIANTS,
            seeds=SEEDS,
            epochs=EPOCHS,
            grid=GRID,
            learning_rate=0.0012,
            min_lr=0.00008,
            batch=32,
            dropout=0.12,
            category_prior_weight=0.85,
            blend_grid=[0, 0.25, 0.5, 0.75, 1.0],
            selection="Highest validation CC over equal-weight three-seed variant ensembles and old/new probability mixtures; ties keep source order.",
        ),
    )
    for variant in VARIANTS:
        predictions = []
        for seed in SEEDS:
            configure(seed)
            run = OUT / f"{variant}_{seed}"
            run.mkdir(exist_ok=True)
            model = SemanticHead(variant).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=0.0012, weight_decay=0.01)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, EPOCHS, eta_min=0.00008
            )
            best = -np.inf
            history = []
            for ep in range(1, EPOCHS + 1):
                model.train()
                total = 0.0
                for ids in batches(fit, 32, shuffle=True):
                    mirror = np.random.rand() < 0.5
                    target = low[ids].to(device)
                    if mirror:
                        target = target.flip(-1)
                    prediction = forward(model, variant, ids, mirror)
                    cc, kl, _ = compare(target, prediction).mean(0)
                    loss = kl + 0.3 * (1 - cc)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    total += float(loss.detach()) * len(ids)
                pred = predict(model, variant, val)
                cc, kl, sim = measures(pred, val).mean(0).tolist()
                history.append(
                    dict(epoch=ep, loss=total / len(fit), cc=cc, kl=kl, sim=sim)
                )
                if cc > best:
                    best = cc
                    torch.save(
                        dict(
                            model=model.state_dict(),
                            epoch=ep,
                            variant=variant,
                            seed=seed,
                        ),
                        run / "best.pt",
                    )
                sched.step()
            table(run / "history.csv", history)
            ck = torch.load(run / "best.pt", map_location=device, weights_only=True)
            model.load_state_dict(ck["model"])
            pred = predict(model, variant, val)
            predictions.append(pred)
            cc, kl, sim = measures(pred, val).mean(0).tolist()
            seed_rows.append(
                dict(
                    variant=variant,
                    seed=seed,
                    epoch=ck["epoch"],
                    split="validation",
                    cc=cc,
                    kl=kl,
                    sim=sim,
                    parameters=sum(p.numel() for p in model.parameters()),
                    gain=model.gain.item(),
                )
            )
            print(
                variant,
                seed,
                "epoch",
                ck["epoch"],
                "validation CC",
                round(cc, 5),
                flush=True,
            )
        val_preds[variant] = torch.stack(predictions).mean(0)
    table(OUT / "validation_by_seed.csv", seed_rows)
    old = torch.from_numpy(np.load(BASE / "outputs/ensemble/density.npy"))
    candidates = []
    for variant, pred in val_preds.items():
        for w in (0.0, 0.25, 0.5, 0.75, 1.0):
            cc, kl, sim = measures((1 - w) * old[val] + w * pred, val).mean(0).tolist()
            candidates.append(
                dict(variant=variant, semantic_weight=w, cc=cc, kl=kl, sim=sim)
            )
    table(OUT / "validation_search.csv", candidates)
    selected = max(candidates, key=lambda r: r["cc"])
    dump(
        OUT / "selection.json",
        dict(
            **selected, seeds=SEEDS, selection_split="validation", selection_metric="CC"
        ),
    )
    print("Validation choice locked", selected, flush=True)
    rows = []
    allids = np.arange(len(meta))
    for variant in VARIANTS:
        accum = torch.zeros(len(meta), 1, *GRID)
        for seed in SEEDS:
            run = OUT / f"{variant}_{seed}"
            ck = torch.load(run / "best.pt", map_location=device, weights_only=True)
            model = SemanticHead(variant).to(device)
            model.load_state_dict(ck["model"])
            pred = predict(model, variant, allids)
            accum += pred
            mm = measures(pred, allids)
            for split in ("fit", "validation", "test"):
                cc, kl, sim = mm[meta.split == split].mean(0).tolist()
                rows.append(
                    dict(
                        variant=variant,
                        seed=seed,
                        epoch=ck["epoch"],
                        split=split,
                        cc=cc,
                        kl=kl,
                        sim=sim,
                    )
                )
        np.save(OUT / f"{variant}.npy", (accum / len(SEEDS)).numpy())
    table(OUT / "metrics_by_seed.csv", rows)
    bestpred = np.load(OUT / f"{selected['variant']}.npy")
    w = selected["semantic_weight"]
    np.save(OUT / "selected.npy", w * bestpred + (1 - w) * old.numpy())
    np.save(OUT / "routed_prior.npy", routed.numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    train(parser.parse_args().device)
