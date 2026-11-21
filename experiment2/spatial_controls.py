"""Matched interventions on the selected feature family after locking the primary model."""
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from sklearn.metrics import roc_auc_score
from course.common import table
from experiment2.foundation import splits
from experiment2.spatial_features import OUT, CACHE
from experiment2.train import BASE, scores
from experiment2.spatial import (
    features,
    train_solutions,
    calibrate,
    oof_threshold,
    width_auc,
    nuisance,
)


def main():
    lock = json.loads((OUT / "selection.json").read_text())
    meta = pd.read_csv(BASE / "data/images.csv")
    y, g, fit, folds = splits(meta)
    te = meta.index[meta.split.eq("test")].to_numpy()
    bank = np.load(CACHE / "natural.npz")["embeddings"]
    geom = np.load(CACHE / "natural_gram.npz")["gram"]
    reg = lock["ridge"]
    mode = lock["weighting"]
    configs = [(lock["design"], i) for i in (0, 4, 12, 18, 24)]
    for design in (
        "field",
        "local",
        "balanced",
        "annulus",
        "global_local",
        "all_views",
        "all_regions_shape",
    ):
        if (design, lock["rank"]) not in configs:
            configs.append((design, lock["rank"]))
    rows = []
    preds = meta.iloc[te][["index", "member", "category"]].copy()
    expected = pd.read_csv(OUT / "predictions.csv").selected.to_numpy()[te]
    for design, rank in configs:
        x = features(bank, design, geom)
        oof = np.zeros(len(meta))
        fold_scores = []
        effective = []
        for a, b in folds:
            _, _, _, oof[b] = next(
                train_solutions(x[a], y[a], g[a], x[b], (rank,), (reg,), (mode,))
            )
            effective.append(min(rank, len(nuisance(x[a], y[a], g[a]))))
            fold_scores.append(roc_auc_score(y[b], oof[b]))
        cal = calibrate(oof[fit], y[fit])
        thr = oof_threshold(y[fit], g[fit], cal.predict_proba(oof[fit, None])[:, 1])
        _, _, _, raw = next(
            train_solutions(x[fit], y[fit], g[fit], x[te], (rank,), (reg,), (mode,))
        )
        p = cal.predict_proba(raw[:, None])[:, 1]
        if design == lock["design"] and rank == lock["rank"]:
            np.testing.assert_allclose(p, expected, atol=1e-7)
        rows.append(
            dict(
                design=design,
                max_rank=rank,
                fold_ranks="/".join(map(str, effective)),
                refit_rank=min(rank, len(nuisance(x[fit], y[fit], g[fit]))),
                ridge=reg,
                weighting=mode,
                threshold=thr,
                fold_auc=np.mean(fold_scores),
                within_width_auc=width_auc(y[fit], g[fit], oof[fit]),
                **scores(y[te], p, thr),
            )
        )
        preds[f"{design}_{rank}"] = p
    table(OUT / "matched_controls.csv", rows)
    preds.to_csv(OUT / "matched_predictions.csv", index=False)
    print(
        pd.DataFrame(rows)[
            ["design", "max_rank", "fold_ranks", "refit_rank", "auc", "accuracy"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    with threadpool_limits(2):
        main()
