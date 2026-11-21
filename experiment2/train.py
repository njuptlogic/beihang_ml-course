"""A controlled 2x2 ablation of multi-scale evidence and spatial attention."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    balanced_accuracy_score,
    precision_score,
    brier_score_loss,
    roc_curve,
)
from course.common import ROOT, configure, batches, table, dump

BASE = ROOT / "experiment2"
VARIANTS = ("global", "pyramid", "attention", "pyramid_attention")
SEEDS = (17, 42, 73)


class RetinaHead(nn.Module):
    def __init__(self, variant):
        super().__init__()
        self.pyramid = "pyramid" in variant
        self.attention = "attention" in variant
        self.proj = nn.Sequential(
            nn.Conv2d(768 if self.pyramid else 512, 64, 1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
        )
        if self.attention:
            self.score = nn.Sequential(
                nn.Conv2d(64, 32, 1), nn.Tanh(), nn.Conv2d(32, 1, 1)
            )
        self.readout = nn.Sequential(nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, middle, deep):
        h = self.proj(torch.cat((middle, deep), 1) if self.pyramid else deep).flatten(2)
        pooled = h.mean(-1)
        if self.attention:
            weight = self.score(h.unflatten(-1, deep.shape[-2:])).flatten(2).softmax(-1)
            pooled = (pooled + (h * weight).sum(-1)) / 2
        return self.readout(pooled).flatten()


def scores(y, p, threshold):
    pred = p >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(
        accuracy=float((pred == y).mean()),
        auc=roc_auc_score(y, p),
        ap=average_precision_score(y, p),
        sensitivity=tp / (tp + fn),
        specificity=tn / (tn + fp),
        precision=precision_score(y, pred, zero_division=0),
        f1=f1_score(y, pred),
        balanced_accuracy=balanced_accuracy_score(y, pred),
        brier=brier_score_loss(y, p),
        tp=int(tp),
        fp=int(fp),
        tn=int(tn),
        fn=int(fn),
    )


def choose_threshold(y, p):
    fpr, tpr, thr = roc_curve(y, p, drop_intermediate=False)
    good = np.isfinite(thr)
    return float(thr[good][np.argmax((tpr - fpr)[good])])


def main(out, device):
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(BASE / "data/images.csv")
    split = df.split.to_numpy()
    fit = np.where(split == "fit")[0]
    val = np.where(split == "validation")[0]
    feats = [
        torch.from_numpy(np.load(BASE / f"data/stage{s}.npy")).to(device)
        for s in (3, 4)
    ]
    y = (df.category == "disease").astype(int).to_numpy()
    labels = torch.tensor(y, device=device, dtype=torch.float32)
    summary = []
    for variant in VARIANTS:
        for seed in SEEDS:
            configure(seed)
            run = out / f"{variant}_{seed}"
            run.mkdir(exist_ok=True)
            model = RetinaHead(variant).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 24, eta_min=0.00008)
            history = []
            best = -1

            def forward(ids):
                return model(*(f[ids].float() for f in feats))

            @torch.no_grad()
            def predict(ids):
                model.eval()
                return torch.cat(
                    [forward(idx).sigmoid().cpu() for idx in batches(ids, 96)]
                ).numpy()

            for epoch in range(1, 25):
                model.train()
                total = 0
                for ids in batches(fit, 64, True):
                    logits = forward(ids)
                    loss = F.binary_cross_entropy_with_logits(logits, labels[ids])
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    total += loss.detach() * len(ids)
                pv = predict(val)
                auc = roc_auc_score(y[val], pv)
                history.append(
                    dict(
                        epoch=epoch,
                        training_bce=float(total) / len(fit),
                        val_auc=auc,
                        val_brier=brier_score_loss(y[val], pv),
                        lr=opt.param_groups[0]["lr"],
                    )
                )
                if auc > best:
                    best = auc
                    torch.save(
                        dict(
                            model=model.state_dict(),
                            epoch=epoch,
                            threshold=choose_threshold(y[val], pv),
                            variant=variant,
                            seed=seed,
                        ),
                        run / "best.pt",
                    )
                sched.step()
            ck = torch.load(run / "best.pt", map_location=device, weights_only=True)
            model.load_state_dict(ck["model"])
            prob = predict(np.arange(len(df)))
            pred = df[["index", "member", "category", "split"]].copy()
            pred["label"] = y
            pred["probability"] = prob
            pred["threshold"] = ck["threshold"]
            pred.to_csv(run / "predictions.csv", index=False)
            table(run / "history.csv", history)
            for name, idx in [
                ("fit", fit),
                ("validation", val),
                ("train_full", np.where(split != "test")[0]),
                ("test", np.where(split == "test")[0]),
            ]:
                summary.append(
                    dict(
                        variant=variant,
                        seed=seed,
                        split=name,
                        epoch=ck["epoch"],
                        threshold=ck["threshold"],
                        parameters=sum(p.numel() for p in model.parameters()),
                        **scores(y[idx], prob[idx], ck["threshold"]),
                    )
                )
            table(out / "metrics.csv", summary)
            print(variant, seed, ck["epoch"], best, flush=True)
    # Freeze a model family by mean validation AUC, then evaluate a three-seed probability ensemble.
    metrics = pd.DataFrame(summary)
    selected = (
        metrics.query("split=='validation'").groupby("variant").auc.mean().idxmax()
    )
    runs = [pd.read_csv(out / f"{selected}_{s}/predictions.csv") for s in SEEDS]
    prob = np.mean([r.probability for r in runs], axis=0)
    thr = choose_threshold(y[val], prob[val])
    pred = runs[0].copy()
    pred["probability"] = prob
    pred["threshold"] = thr
    pred.to_csv(out / "ensemble_predictions.csv", index=False)
    dump(out / "selection.json", dict(variant=selected, seeds=SEEDS, threshold=thr))
    for part, idx in [
        ("fit", fit),
        ("validation", val),
        ("train_full", np.where(split != "test")[0]),
        ("test", np.where(split == "test")[0]),
    ]:
        summary.append(
            dict(
                variant="ensemble",
                seed=-1,
                split=part,
                threshold=thr,
                **scores(y[idx], prob[idx], thr),
            )
        )
    table(out / "metrics.csv", summary)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=BASE / "outputs")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    main(a.output, a.device)
