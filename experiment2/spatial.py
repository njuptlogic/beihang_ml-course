"""Source-only ridge readouts with optional removal of acquisition-associated subspaces."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.linalg import solve
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from threadpoolctl import threadpool_limits

from course.common import dump, table
from experiment2.foundation import splits
from experiment2.spatial_features import OUT, CACHE, encode, get_views
from experiment2.retinal_features import backbone
from experiment2.train import BASE, scores

REGIONS = {
    "field": [(0, 0), (0, 1)],
    "local": [(1, 0), (1, 1)],
    "balanced": [(2, 0), (2, 1)],
    "annulus": [(1, 2), (1, 3), (2, 2), (2, 3)],
}
DESIGNS = {
    "field": ["field"],
    "local": ["local"],
    "balanced": ["balanced"],
    "annulus": ["annulus"],
    "local_balanced": ["local", "balanced"],
    "global_local": ["field", "local"],
    "all_views": ["field", "local", "balanced"],
    "all_regions": ["field", "local", "balanced", "annulus"],
}
DESIGNS.update(
    {
        "shape": ["shape"],
        "annulus_shape": ["annulus", "shape"],
        "all_regions_shape": ["field", "local", "balanced", "annulus", "shape"],
    }
)
RANKS = (0, 4, 12, 18, 24)
RIDGE = (0.05, 0.5, 5.0)
PROTOCOL = dict(
    encoder="DINOv2-S/14-reg4",
    roi="42% of the short edge, centred on the local bright-region contrast maximum",
    fit="original fit split only; four fixed width-blocked folds",
    ranks=RANKS,
    ridge=RIDGE,
    selection="equal average of mean held-fold AUROC and harmonic-support-weighted within-width AUROC (both labels >=2)",
    threshold="maximum weighted Youden on grouped out-of-fold decisions, 50% ordinary class balance + 50% within-width overlap",
    designs=DESIGNS,
    weights=["balanced_class", "width_overlap"],
    comparator="second retinal encoder, same design, projection, ridge and training weights as the source-selected natural encoder",
)


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-10)


def features(bank, design, gram_bank=None):
    parts = []
    for name in DESIGNS[design]:
        if name == "shape":
            # Centre each Gram vector on its own spatial mean; offset and amplitude
            # vary with the contrast of the photograph. Both views receive equal weight.
            blocks = [
                unit(v - v.mean(1, keepdims=True))
                for v in (gram_bank[:, 0], gram_bank[:, 1])
            ]
        else:
            blocks = [unit(bank[:, v, r, :]) for v, r in REGIONS[name]]
        parts.append(np.concatenate(blocks, 1) / np.sqrt(len(blocks)))
    return np.ascontiguousarray(
        np.concatenate(parts, 1) / np.sqrt(len(parts)), dtype=np.float64
    )


def balanced(y):
    return len(y) / (2 * np.bincount(y)[y])


def overlap(y, g):
    levels = np.unique(g)
    weights = np.zeros(len(y))
    for level in levels:
        choice = g == level
        n = np.bincount(y[choice], minlength=2)
        if n.min() < 2:
            continue
        weights[choice] = n.min() / n[y[choice]]
    return weights * (len(y) / weights.sum())


def fit_weights(y, g, mode):
    return (
        balanced(y) if mode == "balanced_class" else (balanced(y) + overlap(y, g)) / 2
    )


def width_auc(y, g, p):
    aucs = []
    weights = []
    for level in np.unique(g):
        k = g == level
        cnt = np.bincount(y[k], minlength=2)
        if cnt.min() >= 2:
            aucs.append(roc_auc_score(y[k], p[k]))
            weights.append(2 * cnt.prod() / cnt.sum())
    return np.average(aucs, weights=weights)


def nuisance(x, y, g):
    means = []
    for label in (0, 1):
        mu = x[y == label].mean(0)
        for level in np.unique(g[y == label]):
            subset = (y == label) & (g == level)
            n = subset.sum()
            if n >= 3:
                means.append((x[subset].mean(0) - mu) * np.sqrt(n / (n + 15)))
    matrix = np.stack(means)
    _, s, v = np.linalg.svd(matrix, full_matrices=False)
    return v[s > s[0] * 1e-8]


def train_solutions(
    x,
    y,
    g,
    xval,
    ranks=RANKS,
    penalties=RIDGE,
    modes=("balanced_class", "width_overlap"),
):
    basis = nuisance(x, y, g)
    for rank in ranks:
        v = basis[:rank]
        z = x - (x @ v.T) @ v if rank else x
        for mode in modes:
            w = fit_weights(y, g, mode)
            w = w / w.mean()
            mu = np.average(z, axis=0, weights=w)
            centered = z - mu
            # Solve the weighted centred ridge regression in the dual space.
            weight = np.sqrt(w)
            zw = centered * weight[:, None]
            target = (2 * y - 1) * weight
            gram = zw @ zw.T
            for reg in penalties:
                h = gram.copy()
                h.flat[:: len(h) + 1] += reg
                alpha = solve(h, target, assume_a="pos", check_finite=False)
                coef = zw.T @ alpha
                if rank:
                    coef -= v.T @ (v @ coef)
                bias = -mu @ coef
                yield (rank, reg, mode), coef, bias, xval @ coef + bias


def oof_threshold(y, g, p):
    fp, tp, thresholds = roc_curve(
        y, p, sample_weight=fit_weights(y, g, "width_overlap"), drop_intermediate=False
    )
    ix = np.flatnonzero(np.isclose(tp - fp, np.max(tp - fp)))
    k = ix[len(ix) // 2]
    return float(thresholds[k])


def save_model(file, config, coef, bias, cal, threshold):
    np.savez(
        file,
        weight=coef,
        bias=bias,
        scale=cal.coef_[0, 0],
        intercept=cal.intercept_[0],
        threshold=threshold,
    )
    assert cal.coef_[0, 0] > 0
    return {
        **config,
        "threshold": threshold,
        "platt_scale": float(cal.coef_[0, 0]),
        "platt_intercept": float(cal.intercept_[0]),
    }


def calibrate(p, y):
    return LogisticRegression(C=100, solver="lbfgs").fit(p[:, None], y)


def choose(meta):
    y, g, fit, folds = splits(meta)
    dump(OUT / "protocol.json", PROTOCOL)
    natural = np.load(CACHE / "natural.npz")
    assert np.array_equal(natural["index"], meta["index"])
    nat = natural["embeddings"]
    grams = np.load(CACHE / "natural_gram.npz")["gram"]
    records = []
    solutions = []
    oof_by_design = {}
    for design in DESIGNS:
        x = features(nat, design, grams)
        oof = {}
        held = {}
        for j, (a, b) in enumerate(folds):
            for cfg, _, _, values in train_solutions(x[a], y[a], g[a], x[b]):
                if cfg not in oof:
                    oof[cfg] = np.full(len(y), np.nan)
                    held[cfg] = []
                oof[cfg][b] = values
                held[cfg].append(roc_auc_score(y[b], values))
        grid = []
        for (rank, reg, mode), p in oof.items():
            key = (rank, reg, mode)
            fold_mean = np.mean(held[key])
            matched = width_auc(y[fit], g[fit], p[fit])
            row = dict(
                design=design,
                rank=rank,
                ridge=reg,
                weighting=mode,
                fold_auc=fold_mean,
                within_width_auc=matched,
                criterion=(fold_mean + matched) / 2,
                **{f"fold{i+1}": v for i, v in enumerate(held[key])},
            )
            grid.append(row)
            records.append(row)
        best = max(grid, key=lambda v: v["criterion"])
        key = (best["rank"], best["ridge"], best["weighting"])
        source = oof[key][fit]
        oof_by_design[design] = source
        cal = calibrate(source, y[fit])
        threshold = oof_threshold(
            y[fit], g[fit], cal.predict_proba(source[:, None])[:, 1]
        )
        _, coef, bias, _ = next(
            train_solutions(
                x[fit], y[fit], g[fit], x[:0], (key[0],), (key[1],), (key[2],)
            )
        )
        solutions.append(
            save_model(OUT / f"{design}.npz", best, coef, bias, cal, threshold)
        )
        print("Source CV", best, flush=True)
    selected = max(solutions, key=lambda v: v["criterion"])
    table(OUT / "cv_grid.csv", records)
    table(OUT / "ablation_selection.csv", solutions)
    frame = meta.iloc[fit][["index", "category", "width", "member"]].copy()
    frame["fold"] = -1
    for j, (_, b) in enumerate(folds):
        frame.loc[b, "fold"] = j + 1
    for name, p in oof_by_design.items():
        frame[name] = p
    frame.to_csv(OUT / "oof_decisions.csv", index=False)
    compare = features(
        np.load(CACHE / "retina.npz")["embeddings"],
        selected["design"],
        np.load(CACHE / "retina_gram.npz")["gram"],
    )
    cfg = (selected["rank"], selected["ridge"], selected["weighting"])
    raw = np.zeros(len(meta))
    for a, b in folds:
        _, _, _, raw[b] = next(
            train_solutions(
                compare[a], y[a], g[a], compare[b], (cfg[0],), (cfg[1],), (cfg[2],)
            )
        )
    cal = calibrate(raw[fit], y[fit])
    thr = oof_threshold(y[fit], g[fit], cal.predict_proba(raw[fit, None])[:, 1])
    _, coef, bias, _ = next(
        train_solutions(
            compare[fit], y[fit], g[fit], compare[:0], (cfg[0],), (cfg[1],), (cfg[2],)
        )
    )
    comp = save_model(
        OUT / "retina_comparator.npz",
        {"design": selected["design"]},
        coef,
        bias,
        cal,
        thr,
    )
    dump(
        OUT / "selection.json",
        dict(
            **selected,
            encoder="natural",
            comparator=comp,
            protocol_sha256=hashlib.sha256(
                (OUT / "protocol.json").read_bytes()
            ).hexdigest(),
        ),
    )
    from experiment2.spatial_operating import lock_operating_points

    lock_operating_points()
    print("LOCKED", selected, flush=True)


def probability(x, path):
    with np.load(path) as m:
        score = x @ m["weight"] + m["bias"]
        return expit(score * m["scale"] + m["intercept"]), float(m["threshold"])


def test(meta):
    lock = json.loads((OUT / "selection.json").read_text())
    y = meta.category.eq("disease").astype(int).to_numpy()
    nat = np.load(CACHE / "natural.npz")["embeddings"]
    results = {}
    rows = []
    grams = np.load(CACHE / "natural_gram.npz")["gram"]
    for d in DESIGNS:
        results[d] = probability(features(nat, d, grams), OUT / f"{d}.npz")
    results["retina_comparator"] = probability(
        features(
            np.load(CACHE / "retina.npz")["embeddings"],
            lock["design"],
            np.load(CACHE / "retina_gram.npz")["gram"],
        ),
        OUT / "retina_comparator.npz",
    )
    results["selected"] = results[lock["design"]]
    points = json.loads((OUT / "operating_points.json").read_text())
    if (
        points["selection_sha256"]
        != hashlib.sha256((OUT / "selection.json").read_bytes()).hexdigest()
    ):
        raise ValueError("Operating points belong to a different locked model")
    for rule, point in points["thresholds"].items():
        if rule != "balanced_primary":
            results[rule] = (results["selected"][0], point)
    prev = BASE / "outputs/foundation"
    old = pd.read_csv(prev / "predictions.csv")
    prevmets = (
        pd.read_csv(prev / "metrics.csv").query("split=='test'").set_index("feature")
    )
    assert old.member.equals(meta.member)
    for oldname in ("previous_resnet", "dino_global", "retina_triple"):
        results[oldname] = old[oldname].to_numpy(), float(
            prevmets.loc[oldname, "threshold"]
        )
    preds = meta[["index", "member", "category", "split", "width"]].copy()
    for name, (p, t) in results.items():
        preds[name] = p
        for split in ("fit", "validation", "train_full", "test"):
            mask = (
                meta.split.ne("test") if split == "train_full" else meta.split.eq(split)
            )
            rows.append(
                dict(
                    model=name,
                    split=split,
                    n=int(mask.sum()),
                    threshold=t,
                    **scores(y[mask], p[mask], t),
                )
            )
    table(OUT / "metrics.csv", rows)
    preds.to_csv(OUT / "predictions.csv", index=False)
    test = meta.split.eq("test").to_numpy()
    yy = y[test]
    rng = np.random.default_rng(873)
    groups = [np.where(yy == c)[0] for c in (0, 1)]
    boot = {k: [] for k in ("dino_global", "retina_triple", "previous_resnet")}
    s, t = results["selected"]
    s = s[test]
    for _ in range(2000):
        i = np.concatenate(
            [rng.choice(group, len(group), replace=True) for group in groups]
        )
        auc = roc_auc_score(yy[i], s[i])
        acc = np.mean((s[i] >= t) == yy[i])
        for model, draws in boot.items():
            old, thr = results[model]
            q = old[test][i]
            draws.append(
                (auc - roc_auc_score(yy[i], q), acc - np.mean((q >= thr) == yy[i]))
            )
    table(
        OUT / "paired.csv",
        [
            dict(
                reference=k,
                delta_auc_l=np.quantile(v, 0.025, axis=0)[0],
                delta_auc_h=np.quantile(v, 0.975, axis=0)[0],
                delta_acc_l=np.quantile(v, 0.025, axis=0)[1],
                delta_acc_h=np.quantile(v, 0.975, axis=0)[1],
            )
            for k, v in boot.items()
        ],
    )
    print(
        pd.DataFrame(rows)
        .query("split=='test'")[
            ["model", "auc", "accuracy", "sensitivity", "specificity"]
        ]
        .to_string(index=False)
    )


def raw_probability(blob, model, device, lock):
    import torch

    imgs, center = get_views(blob)
    t = torch.from_numpy(imgs).permute(0, 3, 1, 2)
    bank = np.stack([encode(model, t[i : i + 1], device) for i in range(3)], axis=1)
    from experiment2.spatial_shapes import gram

    geom = (
        np.stack([gram(model, t[i : i + 1], device) for i in (1, 2)], axis=1)
        if "shape" in DESIGNS[lock["design"]]
        else None
    )
    p, thr = probability(
        features(bank, lock["design"], geom), OUT / (lock["design"] + ".npz")
    )
    return dict(
        probability=float(p[0]),
        threshold=thr,
        prediction="disease" if p[0] >= thr else "normal",
        center=center.tolist(),
    )


def main():
    import torch

    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["select", "evaluate", "verify", "predict"])
    p.add_argument("--image", type=Path)
    p.add_argument(
        "--rule",
        choices=("balanced_primary", "equal_cost", "sensitivity_90", "specificity_90"),
        default="balanced_primary",
        help="Operating point for predict",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    meta = pd.read_csv(BASE / "data/images.csv")
    with threadpool_limits(3):
        if a.action == "select":
            return choose(meta)
        if a.action == "evaluate":
            return test(meta)
        from course.common import configure

        configure(42)
        d = torch.device(a.device)
        model, _ = backbone("natural", d)
        lock = json.loads((OUT / "selection.json").read_text())
        if a.action == "predict":
            if not a.image:
                p.error("--image is required")
            res = raw_probability(a.image.read_bytes(), model, d, lock)
            res["threshold"] = json.loads((OUT / "operating_points.json").read_text())[
                "thresholds"
            ][a.rule]
            res["prediction"] = (
                "disease" if res["probability"] >= res["threshold"] else "normal"
            )
            print(json.dumps({**res, "rule": a.rule}, indent=2))
            return
        data = pd.read_csv(OUT / "predictions.csv")
        check = []
        for i in [50, 457, 1260, 1650, 1790, 1870]:
            row = meta.iloc[i]
            with ZipFile(BASE / "data/raw" / row.archive) as z:
                res = raw_probability(z.read(row.member), model, d, lock)
            error = abs(res["probability"] - data.selected[i])
            assert error < 0.004 and (res["prediction"] == "disease") == (
                data.selected[i] >= res["threshold"]
            )
            check.append(dict(index=i, member=row.member, error=error, **res))
        dump(OUT / "verification.json", check)
        print("Verified", len(check), "independent raw images")


if __name__ == "__main__":
    main()
