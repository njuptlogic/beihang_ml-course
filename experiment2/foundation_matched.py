"""Hold the chosen readout, regularisation and source weights fixed across all views."""
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from course.common import table
from experiment2.train import BASE, choose_threshold, scores
from experiment2.foundation import OUT, load_features, splits, fit_model


def main():
    meta = pd.read_csv(BASE / "data/images.csv")
    data = load_features(meta)
    name = json.loads((OUT / "selection.json").read_text())["feature"]
    cfg = pd.read_csv(OUT / "selected_by_feature.csv").set_index("feature").loc[name]
    y, g, fit, folds = splits(meta)
    t = meta.index[meta.split == "test"].to_numpy()
    rows = []
    preds = meta[["index", "member", "split", "category"]].copy()
    with threadpool_limits(2):
        for view, x in data.items():
            oof = np.zeros(len(meta))
            aucs = []
            for a, b in folds:
                m = fit_model(x[a], y[a], g[a], cfg["head"], cfg.C, cfg.weighting)
                oof[b] = m.decision_function(x[b])
                aucs.append(roc_auc_score(y[b], oof[b]))
            pl = LogisticRegression(C=1000, solver="lbfgs", max_iter=1000).fit(
                oof[fit, None], y[fit]
            )
            threshold = choose_threshold(y[fit], pl.predict_proba(oof[fit, None])[:, 1])
            m = fit_model(x[fit], y[fit], g[fit], cfg["head"], cfg.C, cfg.weighting)
            p = pl.predict_proba(m.decision_function(x)[:, None])[:, 1]
            preds[view] = p
            rows.append(
                dict(
                    feature=view,
                    cv_auc=np.mean(aucs),
                    head=cfg["head"],
                    C=cfg.C,
                    weighting=cfg.weighting,
                    threshold=threshold,
                    **scores(y[t], p[t], threshold)
                )
            )
    table(OUT / "matched_readout.csv", rows)
    preds.to_csv(OUT / "matched_predictions.csv", index=False)
    original = pd.read_csv(OUT / "predictions.csv")
    assert np.allclose(preds[name], original[name], atol=1e-6)


if __name__ == "__main__":
    main()
