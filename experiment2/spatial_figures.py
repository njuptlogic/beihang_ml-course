"""Summarise anatomical region readouts, source selection and locked decisions."""
import json
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch
from sklearn.metrics import roc_curve

from course.common import table
from course.figures import (
    setup,
    page,
    picture,
    axes_title,
    save,
    BLUE,
    TEAL,
    AMBER,
    PLUM,
    GRAY,
)
from experiment2.spatial import OUT, DESIGNS
from experiment2.spatial_features import CACHE, get_views
from experiment2.train import BASE

NAMES = {
    "field": "Full field",
    "local": "Raw crop",
    "balanced": "Balanced crop",
    "annulus": "Core + ring",
    "local_balanced": "Paired crops",
    "global_local": "Full + raw",
    "all_views": "Full + both crops",
    "all_regions": "All regions",
    "shape": "Patch relations",
    "annulus_shape": "Core + ring + relations",
    "all_regions_shape": "All regions + relations",
}


def raw(meta, idx):
    row = meta.iloc[idx]
    with ZipFile(BASE / "data/raw" / row.archive) as z:
        return z.read(row.member)


def architecture():
    fig, ax = plt.subplots(figsize=(12.4, 4.5), layout="constrained")
    ax.set_xlim(0, 12.8)
    ax.set_ylim(0, 4.4)
    ax.axis("off")

    def box(x, y, w, h, head, sub, fill, edge):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.025,rounding_size=0.09",
                linewidth=1.3,
                facecolor=fill,
                edgecolor=edge,
                zorder=3,
            )
        )
        ax.text(
            x + w / 2,
            y + h * 0.65,
            head,
            ha="center",
            va="center",
            size=10,
            weight="bold",
            zorder=4,
        )
        ax.text(
            x + w / 2,
            y + h * 0.29,
            sub,
            ha="center",
            va="center",
            size=8.4,
            color="#40515D",
            zorder=4,
        )

    def arrow(a, b, diagnostic=False):
        ax.add_patch(
            FancyArrowPatch(
                a,
                b,
                arrowstyle="-|>",
                mutation_scale=14,
                lw=1.5,
                ls="--" if diagnostic else "-",
                color=AMBER if diagnostic else "#456370",
                zorder=2,
            )
        )

    scan = "#E8F2F8"
    read = "#E4F3EE"
    train = "#FBEDDA"
    out = "#F2ECF6"
    ax.text(0.1, 3.98, "A   Single photograph", fontsize=13, weight="bold")
    ax.text(5.35, 1.12, "B   Fitting the readout", fontsize=12, weight="bold")
    box(0.13, 2.05, 1.65, 1.05, "Fundus RGB", "Canonical orientation", scan, BLUE)
    for i, (name, sub) in enumerate(
        [
            ("Full field", "CLS + patch mean"),
            ("Local RGB", "CLS + mean + regions"),
            ("Local balanced", "CLS + mean + regions"),
        ]
    ):
        y = 3.13 - i * 0.88
        box(2.37, y, 2.15, 0.70, name, sub, scan, BLUE)
        arrow((1.78, 2.57), (2.36, y + 0.35))
        arrow((4.52, y + 0.35), (5.13, 2.57))
    box(5.14, 2.05, 2.14, 1.05, "Shared DINOv2", "Frozen ViT-S/14 · 392 px", read, TEAL)
    box(
        7.81,
        2.05,
        2.04,
        1.05,
        "Region readout",
        "10 vectors · 3,840 values",
        read,
        TEAL,
    )
    box(
        10.42, 2.05, 2.19, 1.05, "Disease score", "Linear head + calibration", out, PLUM
    )
    arrow((7.28, 2.57), (7.80, 2.57))
    arrow((9.85, 2.57), (10.41, 2.57))
    box(5.30, 0.12, 2.09, 0.70, "Fit split only", "Class and width bins", train, AMBER)
    box(
        7.84,
        0.12,
        2.02,
        0.70,
        "Acquisition axes",
        "Within-class centroids",
        train,
        AMBER,
    )
    box(
        10.44,
        0.12,
        2.16,
        0.70,
        "Project, then fit",
        "Grouped OOF threshold",
        train,
        AMBER,
    )
    arrow((7.39, 0.47), (7.83, 0.47), True)
    arrow((9.86, 0.47), (10.43, 0.47), True)
    arrow((11.52, 0.83), (11.52, 2.02), True)
    save(fig, 2, "spatial_architecture")


def diagram(meta):
    fig, ax = page(2, 4, size=(11.9, 6.4))
    data = np.load(CACHE / "natural_gram.npz")["gram"]
    t = np.triu_indices(49, 1)
    # First fitting image of either class with the bright candidate in the outer horizontal quintiles.
    center = np.load(CACHE / "natural.npz")["center"]
    ids = []
    for label in ("disease", "normal"):
        ix = np.flatnonzero(
            meta.category.eq(label)
            & meta.split.eq("fit")
            & ((center[:, 0] < 0.2) | (center[:, 0] > 0.8))
        )[0]
        ids.append(ix)
    for i, idx in enumerate(ids):
        views, c = get_views(raw(meta, idx))
        q = 1 - c[0] if c[0] < 0.5 else c[0]
        picture(ax[i, 0], views[0])
        ax[i, 0].plot(
            q * (views.shape[2] - 1),
            c[1] * (views.shape[1] - 1),
            "+",
            color="#B3FBFD",
            ms=16,
            mew=1.8,
        )
        picture(ax[i, 1], views[1])
        picture(ax[i, 2], views[2])
        w = views.shape[1]
        for rr, ls in ((0.23, "-"), (0.42, "--")):
            ax[i, 2].add_patch(
                Circle(
                    ((w - 1) / 2, (w - 1) / 2),
                    w * rr,
                    fill=False,
                    edgecolor="white",
                    lw=1.1,
                    ls=ls,
                )
            )
        mat = np.eye(49)
        mat[t] = data[idx, 0]
        mat[(t[1], t[0])] = data[idx, 0]
        im = ax[i, 3].imshow(
            mat, cmap="viridis", vmin=-0.25, vmax=1, interpolation="nearest"
        )
        ax[i, 3].set_xticks([0, 24, 48], [1, 25, 49])
        ax[i, 3].set_yticks([0, 24, 48], [1, 25, 49])
        ax[i, 0].set_ylabel(
            meta.iloc[idx].category.capitalize() + " · fit", weight="bold"
        )
    for a, title in zip(
        ax[0],
        [
            "A  Canonical field",
            "B  42% crop",
            "C  Core and ring",
            "D  Patch Gram matrix",
        ],
    ):
        a.set_title(title, size=11)
    fig.colorbar(im, ax=ax[:, 3], shrink=0.76, label="Cosine of regional embeddings")
    table(
        OUT / "localization_examples.csv",
        [
            {
                "index": i,
                "member": meta.iloc[i].member,
                "x": center[i, 0],
                "y": center[i, 1],
            }
            for i in ids
        ],
    )
    save(fig, 2, "spatial_regions")


def selection():
    sel = json.loads((OUT / "selection.json").read_text())
    grid = pd.read_csv(OUT / "cv_grid.csv")
    best = pd.read_csv(OUT / "ablation_selection.csv").set_index("design")
    mets = pd.read_csv(OUT / "metrics.csv").query("split=='test'").set_index("model")
    fig, ax = page(1, 2, size=(12.2, 6), width_ratios=[1, 1.1])
    ds = list(DESIGNS)
    y = np.arange(len(ds))
    for i, (kind, color, mark, label) in enumerate(
        [
            ("fold_auc", BLUE, "o", "Held-fold AUROC"),
            ("within_width_auc", AMBER, "s", "Within-width AUROC"),
            ("test", TEAL, "D", "Official test AUROC"),
        ]
    ):
        vals = mets.loc[ds, "auc"] if kind == "test" else best.loc[ds, kind]
        ax[0].scatter(vals, y + (i - 1) * 0.15, c=color, s=32, marker=mark, label=label)
    ax[0].set_yticks(y, [NAMES[n] for n in ds])
    ax[0].invert_yaxis()
    axes_title(ax[0], "A  Where region features transfer", "AUROC")
    ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8)
    sub = grid[(grid.design == sel["design"]) & (grid.weighting == sel["weighting"])]
    xs = sorted(sub["rank"].unique())
    for reg, st, col in zip(
        sorted(sub.ridge.unique()), ("o-", "s--", "D-."), (BLUE, AMBER, TEAL)
    ):
        d = sub[sub.ridge == reg].sort_values("rank")
        ax[1].plot(d["rank"], d.criterion, st, color=col, label=f"Ridge {reg:g}")
    ax[1].scatter(
        [sel["rank"]],
        [sel["criterion"]],
        s=140,
        edgecolors="black",
        facecolors="none",
        zorder=4,
        label="Source selection",
    )
    ax[1].set_xticks(xs)
    axes_title(
        ax[1],
        "B  Removing width-associated directions",
        "Maximum projection rank",
        "Source selection criterion",
    )
    ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8)
    save(fig, 2, "spatial_selection")


def outcomes(pred, mets):
    fig, ax = page(1, 3, size=(13, 4.2), width_ratios=[1.05, 1.12, 0.93])
    y = pred.category.eq("disease").to_numpy()
    for name, color, ls, label in [
        ("previous_resnet", GRAY, ":", "ResNet"),
        ("retina_triple", AMBER, "-.", "RETFound, previous"),
        ("dino_global", BLUE, "--", "DINO, previous"),
        ("selected", TEAL, "-", "DINO, selected"),
    ]:
        r = mets.loc[name]
        f, t, _ = roc_curve(y, pred[name])
        ax[0].plot(f, t, c=color, ls=ls, label=f"{label} · {r.auc:.3f}")
        ax[0].scatter(1 - r.specificity, r.sensitivity, c=color, marker="o", s=25)
    ax[0].plot([0, 1], [0, 1], ":", c="0.65", lw=1)
    ax[0].set(xlim=(-0.015, 1.015), ylim=(-0.015, 1.015))
    ax[0].legend(fontsize=7, loc="lower right")
    axes_title(
        ax[0], "A  Locked ROC curves", "False-positive rate", "True-positive rate"
    )
    d = pred.selected.to_numpy()
    for a, color, ls, label in [
        (False, BLUE, "--", "Normal"),
        (True, AMBER, "-", "Disease"),
    ]:
        vals = np.sort(d[y == a])
        ax[1].step(
            vals,
            np.arange(1, len(vals) + 1) / len(vals),
            where="post",
            ls=ls,
            color=color,
            label=label,
        )
    for name, col, ls, label in [
        ("selected", TEAL, "-", "Weighted Youden"),
        ("equal_cost", PLUM, "-.", "Equal-cost rule"),
    ]:
        ax[1].axvline(mets.loc[name].threshold, color=col, ls=ls, lw=1.3, label=label)
    axes_title(
        ax[1],
        "B  Ranked disease scores",
        "Calibrated probability",
        "Cumulative fraction",
    )
    ax[1].legend(fontsize=7, loc="lower right")
    ax[1].set_xlim(0, 1)
    c = mets.loc["selected"]
    mat = np.array([[c.tn, c.fp], [c.fn, c.tp]], dtype=int)
    ax[2].imshow(mat, cmap="Blues", vmin=0, vmax=max(150, mat.max()))
    for (row, col), val in np.ndenumerate(mat):
        ax[2].text(
            col,
            row,
            str(val),
            ha="center",
            va="center",
            fontsize=22,
            color="white" if val > 75 else "#19394D",
        )
    ax[2].set_xticks([0, 1], ["Normal", "Disease"])
    ax[2].set_yticks([0, 1], ["Normal", "Disease"])
    ax[2].set(
        xlabel="Predicted class", ylabel="Official class", title="C  Test decisions"
    )
    save(fig, 2, "spatial_outcomes")


def cases(meta, pred, mets):
    p = pred.selected
    old = pred.dino_global
    t = mets.loc["selected"].threshold
    ot = mets.loc["dino_global"].threshold
    y = pred.category.eq("disease")
    new_correct = p.ge(t).eq(y)
    prev_correct = old.ge(ot).eq(y)
    rule = [
        ("Corrected", ~prev_correct & new_correct),
        ("New error", prev_correct & ~new_correct),
        ("Both correct", prev_correct & new_correct),
        ("Both wrong", ~prev_correct & ~new_correct),
    ]
    fig, ax = page(2, 4, size=(10, 5.1))
    picked = []
    for j, (label, mask) in enumerate(rule):
        opts = pred[mask]
        if len(opts) == 0:
            for a in ax[:, j]:
                a.axis("off")
            ax[0, j].set_title(label + " · none")
            continue
        # Nearest to the median absolute score change, with row-order tie-breaking.
        delta = (opts.selected - opts.dino_global).abs()
        ix = (delta - delta.median()).abs().idxmin()
        r = pred.loc[ix]
        views, _ = get_views(raw(meta, r["index"]))
        picture(ax[0, j], views[0], label)
        picture(ax[1, j], views[1])
        ax[1, j].set_xlabel(
            f"{r.category}  #{int(r['index'])}\n{r.dino_global:.3f} → {r.selected:.3f}"
        )
        picked.append(
            dict(
                group=label,
                index=r["index"],
                member=r.member,
                previous=r.dino_global,
                selected=r.selected,
                n=len(opts),
            )
        )
    ax[0, 0].set_ylabel("Full field")
    ax[1, 0].set_ylabel("Local view")
    table(OUT / "cases.csv", picked)
    save(fig, 2, "spatial_cases")


def main():
    setup()
    meta = pd.read_csv(BASE / "data/images.csv")
    pred = pd.read_csv(OUT / "predictions.csv").query("split=='test'")
    mets = pd.read_csv(OUT / "metrics.csv").query("split=='test'").set_index("model")
    architecture()
    diagram(meta)
    selection()
    outcomes(pred, mets)
    cases(meta, pred, mets)


if __name__ == "__main__":
    main()
