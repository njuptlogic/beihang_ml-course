"""Plot matched ablations, threshold trade-offs and original-image failure examples."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
)
from course.figures import (
    TEAL,
    BLUE,
    AMBER,
    INK,
    GRAY,
    COLORS,
    STYLE,
    setup,
    architecture,
    page,
    save,
    axes_title,
    picture,
)
from course.common import dump, table
from experiment2.train import BASE, VARIANTS, SEEDS

OUT = BASE / "outputs"
NAMES = ["Global", "Multi-scale", "Attention", "Both"]


def main():
    setup()
    architecture(2)
    meta = pd.read_csv(BASE / "data/images.csv")
    images = np.load(BASE / "data/images.npy", mmap_mode="r")
    p = pd.read_csv(OUT / "ensemble_predictions.csv")
    follow = np.load(OUT / "followup/ensemble.npy")
    p["follow"] = follow
    p["size_score"] = np.load(OUT / "followup/size_predictions.npy")
    test = p[p.split == "test"]
    val = p[p.split == "validation"]
    thresholds = {
        "probability": float(p.threshold.iloc[0]),
        "follow": json.loads((OUT / "followup/selection.json").read_text())[
            "threshold"
        ],
    }
    fig, ax = page(2, 6, size=(11.5, 4.2))
    for col in range(6):
        for row, part in enumerate(["fit", "test"]):
            category = "disease" if col < 3 else "normal"
            ids = meta.index[(meta.split == part) & (meta.category == category)]
            i = ids[[len(ids) // 5, len(ids) // 2, 4 * len(ids) // 5][col % 3]]
            picture(
                ax[row, col],
                images[i],
                f"{part.title()} · {category}"
                if row == 0 or col in (0, 3)
                else part.title(),
            )
            ax[row, col].set_xlabel(f"{meta.width[i]} × {meta.height[i]}", fontsize=8)
    save(fig, 2, "cohort")
    fig, ax = page(1, 2, size=(10, 4.0))
    met = pd.read_csv(OUT / "metrics.csv").query("seed!=-1")
    for axi, (column, title) in zip(
        ax,
        [
            ("training_bce", "A   Fitting loss"),
            ("val_auc", "B   Validation discrimination"),
        ],
    ):
        for j, v in enumerate(VARIANTS):
            vals = np.stack(
                [pd.read_csv(OUT / f"{v}_{s}/history.csv")[column] for s in SEEDS]
            )
            ep = np.arange(1, vals.shape[1] + 1)
            m = vals.mean(0)
            sd = vals.std(0, ddof=1)
            axi.plot(ep, m, label=NAMES[j], color=COLORS[j], ls=STYLE[j])
            axi.fill_between(ep, m - sd, m + sd, color=COLORS[j], alpha=0.09)
        axes_title(
            axi,
            title,
            "Epoch",
            "Binary cross-entropy" if column == "training_bce" else "AUROC",
        )
    ax[0].legend()
    save(fig, 2, "training")
    fig, ax = page(1, 2, size=(10, 4.1))
    for axi, (metric, title) in zip(
        ax, [("auc", "A   Ranking"), ("balanced_accuracy", "B   Thresholded decisions")]
    ):
        for j, part in enumerate(["validation", "test"]):
            d = (
                met[met.split == part]
                .groupby("variant")[metric]
                .agg(["mean", "std"])
                .loc[list(VARIANTS)]
            )
            axi.errorbar(
                np.arange(4) + (-0.07 if j == 0 else 0.07),
                d["mean"],
                yerr=d["std"],
                color=[BLUE, AMBER][j],
                fmt=["o", "s"][j],
                capsize=4,
                label=part.title(),
                ms=7,
            )
        axi.set_xticks(range(4), NAMES)
        axes_title(axi, title, y="AUROC" if metric == "auc" else "Balanced accuracy")
        axi.legend()
        axi.set_ylim(0.45, 1.01)
    save(fig, 2, "ablation")
    fig, ax = page(1, 2, size=(10.7, 4.3))
    for inp, col, style, caption in [
        (val, "probability", (GRAY, "--"), "Frozen · val"),
        (test, "probability", (BLUE, "-"), "Frozen · test"),
        (test, "follow", (TEAL, "-"), "Fine-tuned · test"),
        (test, "size_score", (AMBER, ":"), "Size only · test"),
    ]:
        y = inp.label
        v = inp[col]
        fpr, tpr, _ = roc_curve(y, v)
        prec, rec, _ = precision_recall_curve(y, v)
        ax[0].plot(
            fpr,
            tpr,
            c=style[0],
            ls=style[1],
            label=f"{caption}  {roc_auc_score(y,v):.3f}",
        )
        ax[1].plot(
            rec,
            prec,
            c=style[0],
            ls=style[1],
            label=f"{caption}  {average_precision_score(y,v):.3f}",
        )
    ax[0].plot([0, 1], [0, 1], c="#ACB8BD", ls=":", lw=1)
    ax[1].axhline(test.label.mean(), c="#ACB8BD", ls=":", lw=1)
    for a, t, x, y in [
        (ax[0], "A   ROC curves", "False-positive rate", "Sensitivity"),
        (ax[1], "B   Precision–recall curves", "Sensitivity", "Precision"),
    ]:
        axes_title(a, t, x, y)
        a.legend(loc="lower right" if "ROC" in t else "lower left")
    save(fig, 2, "discrimination")
    fig, ax = page(1, 3, size=(11.5, 3.65), width_ratios=[0.85, 1.2, 1.1])
    thr = thresholds["follow"]
    pred = test.follow >= thr
    cm = confusion_matrix(test.label, pred)
    ax[0].imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
    for (i, j), v in np.ndenumerate(cm):
        ax[0].text(
            j,
            i,
            str(v),
            ha="center",
            va="center",
            color="white" if v > cm.max() * 0.55 else INK,
            weight="bold",
            size=17,
        )
    ax[0].set_xticks([0, 1], ["Normal", "Disease"])
    ax[0].set_yticks([0, 1], ["Normal", "Disease"])
    ax[0].set(xlabel="Prediction", ylabel="Ground truth", title="A   Test counts")
    for k, (name, c) in enumerate([("Normal", BLUE), ("Disease", AMBER)]):
        a = test.loc[test.label == k, "follow"]
        ax[1].hist(a, bins=np.linspace(0, 1, 16), color=c, alpha=0.55, label=name)
    ax[1].axvline(thr, color=INK, ls="--", label=f"Cutoff {thr:.2f}")
    ax[1].legend()
    axes_title(ax[1], "B   Score distribution", "Disease probability", "Images")
    for d, label, c in [(val, "Validation", BLUE), (test, "Test", AMBER)]:
        # Equal-count bins keep the sparse middle of this distribution visible.
        bins = np.array_split(d.sort_values("follow").index, 8)
        xs = [d.loc[b, "follow"].mean() for b in bins]
        ys = [d.loc[b, "label"].mean() for b in bins]
        ax[2].plot(xs, ys, marker="o", ms=4, c=c, label=label)
    ax[2].plot([0, 1], [0, 1], ls=":", color=GRAY)
    axes_title(
        ax[2],
        "C   Reliability",
        "Mean predicted probability",
        "Observed disease fraction",
    )
    ax[2].legend()
    save(fig, 2, "decisions")
    fig, ax = page(1, 2, size=(10.2, 3.8))
    for a, (part, title) in zip(
        ax, [("fit", "A   Fitting images"), ("test", "B   Official test images")]
    ):
        for k, c in [("normal", BLUE), ("disease", AMBER)]:
            a.hist(
                meta.loc[(meta.split == part) & (meta.category == k), "width"],
                bins=np.arange(480, 2650, 64),
                alpha=0.58,
                color=c,
                label=k.title(),
            )
        axes_title(a, title, "Original width (pixels)", "Images")
        a.legend()
    save(fig, 2, "resolution")
    fig, ax = page(2, 4, size=(9.6, 5.1))
    cases = []
    for row, (mask, asc, title) in enumerate(
        [
            (test.label.eq(0) & pred, False, "False positive"),
            (test.label.eq(1) & ~pred, True, "False negative"),
        ]
    ):
        ds = test[mask].sort_values("follow", ascending=asc).head(4)
        for a in ax[row]:
            a.axis("off")
        for col, r in enumerate(ds.itertuples()):
            i = r.index
            name = Path(r.member).stem
            a = ax[row, col]
            a.axis("on")
            picture(a, images[i], f"{title} · {r.follow:.4f}")
            a.set_xlabel(name, fontsize=9)
            cases.append(
                dict(
                    member=r.member,
                    kind=title,
                    score=r.follow,
                    width=int(meta.width[i]),
                    height=int(meta.height[i]),
                )
            )
    save(fig, 2, "errors")
    table(OUT / "error_gallery.csv", cases)
    # Test-image resampling holds the trained predictors and operating points fixed.
    rng = np.random.default_rng(601)
    y = test.label.to_numpy()
    v = test.follow.to_numpy()
    old = test.probability.to_numpy()
    ix = [np.flatnonzero(y == k) for k in (0, 1)]
    vals = []
    for _ in range(1500):
        i = np.concatenate([rng.choice(g, len(g), replace=True) for g in ix])
        au = roc_auc_score(y[i], v[i])
        vals.append(
            [au, (v[i] >= thr).__eq__(y[i]).mean(), au - roc_auc_score(y[i], old[i])]
        )
    dump(
        OUT / "bootstrap.json",
        {
            k: np.quantile(np.array(vals)[:, j], [0.025, 0.975]).tolist()
            for j, k in enumerate(["auc", "accuracy", "paired_auc_delta"])
        },
    )


if __name__ == "__main__":
    main()
