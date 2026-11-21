"""Choose operating points from source out-of-fold scores before official evaluation."""
import json
import hashlib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_curve
from course.common import dump, table
from experiment2.spatial import OUT, fit_weights
from experiment2.train import scores


def lock_operating_points():
    cfg = json.loads((OUT / "selection.json").read_text())
    df = pd.read_csv(OUT / "oof_decisions.csv")
    y = df.category.eq("disease").astype(int).to_numpy()
    g = df.width.to_numpy() // 32
    p = expit(
        df[cfg["design"]].to_numpy() * cfg["platt_scale"] + cfg["platt_intercept"]
    )
    _, _, thresholds = roc_curve(y, p, drop_intermediate=False)
    thresholds = thresholds[np.isfinite(thresholds)]
    correct = (p[:, None] >= thresholds[None, :]) == y[:, None]
    # Source image accuracy and mix of equally weighted size-held-out folds, with
    # each fold retaining its observed prevalence, quantify the same 0/1 cost.
    accuracy = np.mean(
        [correct[df.fold == k].mean(axis=0) for k in range(1, 5)], axis=0
    )
    metric = (accuracy + correct.mean(axis=0)) / 2
    ids = np.flatnonzero(np.isclose(metric, metric.max()))
    acc = float(thresholds[ids[len(ids) // 2]])
    values = {
        "balanced_primary": cfg["threshold"],
        "equal_cost": acc,
        "sensitivity_90": float(np.quantile(p[y == 1], 0.10, method="lower")),
        "specificity_90": float(np.quantile(p[y == 0], 0.90, method="higher")),
    }
    locked = dict(
        design=cfg["design"],
        selection_sha256=hashlib.sha256(
            (OUT / "selection.json").read_bytes()
        ).hexdigest(),
        policy="primary uses weighted Youden; equal cost maximizes average of pooled accuracy and equal-fold mean accuracy; other two use the named empirical OOF class quantile",
        thresholds=values,
    )
    dump(OUT / "operating_points.json", locked)
    table(
        OUT / "source_operating.csv",
        [dict(rule=n, threshold=t, **scores(y, p, t)) for n, t in values.items()],
    )
    print(locked)


if __name__ == "__main__":
    lock_operating_points()
