"""Source-only model selection with resolution-blocked validation, followed by a locked test."""
import argparse
import json
from pathlib import Path
from zipfile import ZipFile
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from course.common import table, dump
from experiment2.train import BASE, scores, choose_threshold
from experiment2.retinal_features import CACHE, VIEWS, backbone, encode, views

OUT = BASE / "outputs/foundation"
# Each candidate sees the same 1,393 images as the original image model.
# Feature-set, classifier and weighting comparisons are enumerated before test evaluation.
FEATURES = {
    "resnet_global": ["resnet_global"],
    "dino_global": ["natural_field"],
    "retina_global": ["retina_field"],
    "retina_crop": ["retina_crop"],
    "retina_balanced": ["retina_balanced_crop"],
    "retina_dual": ["retina_field", "retina_crop"],
    "retina_triple": ["retina_field", "retina_crop", "retina_balanced_crop"],
}
PENALTIES = (0.1, 1.0, 10.0)


def load_features(meta):
    banks = {}
    for name in ("natural", "retina"):
        with np.load(CACHE / f"{name}.npz") as data:
            assert np.array_equal(data["index"], meta["index"].to_numpy())
            for k in ("field",) if name == "natural" else VIEWS:
                banks[name + "_" + k] = data[k]
    res = np.load(BASE / "data/stage4.npy", mmap_mode="r")
    banks["resnet_global"] = res.mean((2, 3), dtype=np.float32)
    return {
        k: np.concatenate([banks[s] for s in spec], 1) for k, spec in FEATURES.items()
    }


def classifier(head, C):
    # Fold-local z scoring followed by unit-length vectors yields a defined RBF scale across dimensions.
    est = (
        LogisticRegression(C=C, solver="lbfgs", max_iter=1000, tol=1e-7)
        if head == "linear"
        else SVC(C=C, kernel="rbf", gamma=1.0, cache_size=180)
    )
    return make_pipeline(StandardScaler(), Normalizer(), est)


def fit_model(x, y, groups, head, C, weighting):
    model = classifier(head, C)
    kwargs = {}
    if weighting == "balanced_size":
        _, inv, count = np.unique(
            np.stack([y, groups], 1), axis=0, return_inverse=True, return_counts=True
        )
        weights = 1 / np.sqrt(count[inv])
        for c in (0, 1):
            weights[y == c] *= (y == c).sum() / weights[y == c].sum()
        kwargs[model.steps[-1][0] + "__sample_weight"] = weights
    return model.fit(x, y, **kwargs)


def splits(meta):
    labels = meta.category.eq("disease").astype(int).to_numpy()
    groups = meta.width.to_numpy() // 32
    fit = meta.index[meta.split.eq("fit")].to_numpy()
    folds = []
    for a, b in StratifiedGroupKFold(n_splits=4).split(fit, labels[fit], groups[fit]):
        train, valid = fit[a], fit[b]
        assert not set(groups[train]) & set(groups[valid])
        assert len(np.unique(labels[valid])) == 2
        folds.append((train, valid))
    return labels, groups, fit, folds


def protocol(meta, features):
    y, group, fit, folds = splits(meta)
    records = []
    winners = []
    fold_outputs = []
    splitmeta = meta[["index", "member", "split", "width", "height", "category"]].copy()
    splitmeta["width_band"] = group
    splitmeta["fold"] = -1
    for k, (_, v) in enumerate(folds):
        splitmeta.loc[v, "fold"] = k + 1
    splitmeta.to_csv(OUT / "folds.csv", index=False)
    for name, x in features.items():
        candidates = []
        for head in ("linear", "rbf"):
            for weighting in ("ordinary", "balanced_size"):
                for C in PENALTIES:
                    oof = np.full(len(meta), np.nan)
                    aucs = []
                    for a, b in folds:
                        model = fit_model(x[a], y[a], group[a], head, C, weighting)
                        oof[b] = model.decision_function(x[b])
                        aucs.append(roc_auc_score(y[b], oof[b]))
                    cfg = dict(
                        feature=name,
                        head=head,
                        C=C,
                        weighting=weighting,
                        cv_auc=float(np.mean(aucs)),
                        cv_worst_auc=float(np.min(aucs)),
                        pooled_oof_auc=roc_auc_score(y[fit], oof[fit]),
                    )
                    records.append(
                        {**cfg, **{f"fold_{i+1}_auc": v for i, v in enumerate(aucs)}}
                    )
                    candidates.append((cfg, oof))
        # A deterministic tie rule follows the declared order of head, weighting and increasing C.
        cfg, oof = max(candidates, key=lambda a: a[0]["cv_auc"])
        best = fit_model(
            x[fit], y[fit], group[fit], cfg["head"], cfg["C"], cfg["weighting"]
        )
        # Map source out-of-fold scores to probabilities, preserving their rank.
        platt = LogisticRegression(C=1000, solver="lbfgs", max_iter=1000).fit(
            oof[fit, None], y[fit]
        )
        scale = float(platt.coef_[0, 0])
        bias = float(platt.intercept_[0])
        assert scale > 0
        prob = expit(oof[fit] * scale + bias)
        threshold = choose_threshold(y[fit], prob)
        locked = {
            **cfg,
            "platt_scale": scale,
            "platt_bias": bias,
            "threshold": threshold,
        }
        joblib.dump((best, locked), OUT / f"{name}.joblib")
        winners.append(locked)
        for i, p in zip(fit, prob):
            fold_outputs.append(dict(index=i, feature=name, label=y[i], probability=p))
        print(name, cfg["cv_auc"], cfg["head"], cfg["C"], cfg["weighting"], flush=True)
    frame = pd.DataFrame(winners)
    selected = frame.loc[frame.cv_auc.idxmax(), "feature"]
    table(OUT / "cv_grid.csv", records)
    table(OUT / "selected_by_feature.csv", winners)
    table(OUT / "oof_predictions.csv", fold_outputs)
    dump(
        OUT / "selection.json",
        dict(
            feature=selected,
            selection="largest equal-fold mean AUROC",
            threshold_source="Youden on grouped fit-only out-of-fold scores",
            width_band_pixels=32,
            fit_images=len(fit),
            folds=4,
            positive="disease",
            views=FEATURES[selected],
            classifiers=["linear", "rbf"],
            penalties=PENALTIES,
        ),
    )


def probability(model, cfg, x):
    return expit(model.decision_function(x) * cfg["platt_scale"] + cfg["platt_bias"])


def evaluate(meta, features):
    select = json.loads((OUT / "selection.json").read_text())
    y = meta.category.eq("disease").astype(int).to_numpy()
    ids = {
        s: meta.index[meta.split.eq(s)].to_numpy()
        for s in ("fit", "validation", "test")
    }
    ids["train_full"] = np.sort(np.r_[ids["fit"], ids["validation"]])
    rows = []
    allpred = meta[["index", "member", "split", "width", "height", "category"]].copy()
    run = BASE / "outputs/followup"
    old = np.load(run / "ensemble.npy")
    oldthr = json.loads((run / "selection.json").read_text())["threshold"]
    allpred["previous_resnet"] = old
    operating = {"previous_resnet": oldthr}
    for name, x in features.items():
        model, cfg = joblib.load(OUT / f"{name}.joblib")
        allpred[name] = probability(model, cfg, x)
        operating[name] = cfg["threshold"]
    for name, thr in operating.items():
        for split, i in ids.items():
            rows.append(
                dict(
                    feature=name,
                    split=split,
                    n=len(i),
                    threshold=thr,
                    **scores(y[i], allpred[name].to_numpy()[i], thr),
                )
            )
    table(OUT / "metrics.csv", rows)
    allpred.to_csv(OUT / "predictions.csv", index=False)
    selected = select["feature"]
    p = allpred[selected].to_numpy()
    test = ids["test"]
    thr = operating[selected]
    o = y[test]
    a = old[test]
    b = p[test]
    # Reuse the earlier ResNet threshold rule on the same 246-image holdout.
    # Neither threshold uses the official test labels or changes the chosen head.
    val = ids["validation"]
    val_threshold = choose_threshold(y[val], p[val])
    threshold_rows = []
    for source, value in (
        ("grouped_oof", thr),
        ("random_validation", val_threshold),
    ):
        for target, ix in (("validation", val), ("test", test)):
            threshold_rows.append(
                dict(
                    feature=selected,
                    source=source,
                    split=target,
                    threshold=value,
                    n=len(ix),
                    **scores(y[ix], p[ix], value),
                )
            )
    table(OUT / "threshold_comparison.csv", threshold_rows)
    old_pred = a >= oldthr
    new_pred = b >= thr
    new_right = int(np.sum((old_pred != o) & (new_pred == o)))
    old_right = int(np.sum((old_pred == o) & (new_pred != o)))
    rng = np.random.default_rng(531)
    by_y = [np.flatnonzero(o == k) for k in (0, 1)]
    delta = []
    for _ in range(2000):
        ind = np.concatenate([rng.choice(g, len(g), replace=True) for g in by_y])
        new_auc = roc_auc_score(o[ind], b[ind])
        old_auc = roc_auc_score(o[ind], a[ind])
        acc = (new_pred[ind] == o[ind]).mean()
        prev = (old_pred[ind] == o[ind]).mean()
        delta.append((new_auc, new_auc - old_auc, acc, acc - prev))
    intervals = np.quantile(np.array(delta), [0.025, 0.975], axis=0).T
    comparison = dict(
        previous_corrected=new_right,
        previous_correct_now_wrong=old_right,
        exact_mcnemar_p=float(binomtest(new_right, new_right + old_right, 0.5).pvalue)
        if new_right + old_right
        else 1.0,
        n_resamples=2000,
        **{
            k: val.tolist()
            for k, val in zip(
                ("auc_ci", "delta_auc_ci", "acc_ci", "delta_acc_ci"), intervals
            )
        },
    )
    dump(OUT / "paired.json", comparison)
    known = set(zip(meta.loc[ids["fit"], "width"], meta.loc[ids["fit"], "height"]))
    known_flags = np.array([(w, h) in known for w, h in zip(meta.width, meta.height)])
    subgroup = []
    for flag in (True, False):
        ix = test[known_flags[test] == flag]
        for name in ("previous_resnet", selected):
            subgroup.append(
                dict(
                    feature=name,
                    resolution="seen" if flag else "unseen",
                    n=len(ix),
                    disease=y[ix].sum(),
                    **scores(y[ix], allpred[name].to_numpy()[ix], operating[name]),
                )
            )
    table(OUT / "subgroups.csv", subgroup)
    print(
        pd.DataFrame(rows)
        .query("split=='test'")[
            ["feature", "accuracy", "auc", "sensitivity", "specificity"]
        ]
        .to_string(index=False)
    )


def classify_views(raw, selected, device):
    import torch

    specs = FEATURES[selected]
    inputs, box = views(raw)
    if selected == "resnet_global":
        from torchvision.models import resnet18, ResNet18_Weights
        from course.prepare import resize
        from course.predict import normalized

        enc = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval().to(device)
        rgb, _ = resize(raw, "RGB", (224, 224))
        x = normalized(rgb, device)
        with torch.inference_mode():
            h = enc.layer4(
                enc.layer3(
                    enc.layer2(enc.layer1(enc.maxpool(enc.relu(enc.bn1(enc.conv1(x))))))
                )
            )
            return h.half().float().mean((2, 3)).cpu().numpy(), box
    name = specs[0].split("_", 1)[0]
    enc, _ = backbone(name, device)
    vals = []
    inputs = torch.from_numpy(inputs).permute(0, 3, 1, 2)
    for key in specs:
        view = key.split("_", 1)[1]
        vals.append(
            encode(enc, inputs[VIEWS.index(view) : VIEWS.index(view) + 1], name, device)
        )
    return np.concatenate(vals, axis=1), box


def predict(raw, device, selected=None):
    selected = selected or json.loads((OUT / "selection.json").read_text())["feature"]
    x, box = classify_views(raw, selected, device)
    model, cfg = joblib.load(OUT / f"{selected}.joblib")
    p = float(probability(model, cfg, x)[0])
    return dict(
        feature=selected,
        probability=p,
        threshold=cfg["threshold"],
        prediction="disease" if p >= cfg["threshold"] else "normal",
        view_box=box.tolist(),
    )


def verify(meta, device):
    old = pd.read_csv(OUT / "predictions.csv")
    select = json.loads((OUT / "selection.json").read_text())["feature"]
    rows = []
    for i in (0, 821, 1639, 1888):
        r = meta.iloc[i]
        with ZipFile(BASE / "data/raw" / r.archive) as z:
            pred = predict(z.read(r.member), device)
        d = abs(pred["probability"] - old.loc[i, select])
        assert d < 0.01 and (pred["probability"] >= pred["threshold"]) == (
            old.loc[i, select] >= pred["threshold"]
        ), d
        rows.append(
            dict(index=i, member=r.member, abs_error=d, prediction=pred["prediction"])
        )
    dump(OUT / "verification.json", rows)
    print("Verified raw-image probability and operating point for", len(rows), "images")


def main():
    import torch

    p = argparse.ArgumentParser()
    p.add_argument("task", choices=["select", "test", "verify", "predict"])
    p.add_argument(
        "--view-set",
        choices=FEATURES,
        help="Single-image prediction only; defaults to the CV-selected representation.",
    )
    p.add_argument("--image", type=Path)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    OUT.mkdir(exist_ok=True, parents=True)
    meta = pd.read_csv(BASE / "data/images.csv")
    device = torch.device(a.device)
    if a.task == "predict":
        if not a.image:
            p.error("--image is required")
        print(json.dumps(predict(a.image.read_bytes(), device, a.view_set), indent=2))
        return
    if a.task == "verify":
        verify(meta, device)
        return
    x = load_features(meta)
    with threadpool_limits(2):
        if a.task == "select":
            protocol(meta, x)
        else:
            evaluate(meta, x)


if __name__ == "__main__":
    main()
