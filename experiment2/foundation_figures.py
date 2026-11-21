"""Figures and case data for the width-blocked retinal foundation-model experiment."""
import json
from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve

from course.figures import (
    BLUE,
    TEAL,
    AMBER,
    PLUM,
    INK,
    GRAY,
    axes_title,
    setup,
    page,
    save,
    picture,
)
from course.common import table
from experiment2.foundation import OUT
from experiment2.retinal_features import views
from experiment2.train import BASE

LABELS = {
    "resnet_global": "ResNet · field",
    "dino_global": "DINOv2 · field",
    "retina_global": "Green · field",
    "retina_crop": "Green · crop",
    "retina_balanced": "Green · balanced crop",
    "retina_dual": "Green · field + crop",
    "retina_triple": "Green · all three",
}


def arch(meta):
    # A deterministic source image is used to show exactly the three input transformations.
    index = meta.index[(meta.split == "fit") & (meta.category == "disease")][0]
    r = meta.loc[index]
    with ZipFile(BASE / "data/raw" / r.archive) as z:
        images, box = views(z.read(r.member))
    fig = plt.figure(figsize=(12, 4.7))
    gs = fig.add_gridspec(
        3,
        3,
        width_ratios=[1, 1.1, 3.55],
        left=0.035,
        right=0.985,
        bottom=0.07,
        top=0.82,
        hspace=0.12,
        wspace=0.05,
    )
    fig.text(
        0.035,
        0.91,
        "Retinal transfer across acquisition scales",
        fontsize=19,
        weight="bold",
    )
    fig.text(
        0.035,
        0.846,
        "Source image widths define validation folds and training weights. The classifier takes pixel-derived embeddings.",
        fontsize=10,
        color=GRAY,
    )
    a = fig.add_subplot(gs[:, 0])
    picture(a, images[0], "Original field")
    xmin, ymin, xmax, ymax = box * 392
    a.add_patch(
        Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color="#FECD68", lw=2
        )
    )
    a.set_xlabel(Path(r.member).stem, fontsize=8)
    for j, (im, lab) in enumerate(
        zip(images, ["Field", "Local crop", "Color-balanced crop"])
    ):
        a = fig.add_subplot(gs[j, 1])
        picture(a, im)
        a.set_ylabel(lab, size=8)
    a = fig.add_subplot(gs[:, 2])
    a.set(xlim=(0, 10), ylim=(0, 5))
    a.axis("off")

    def boxn(x, y, w, h, title, desc, fill):
        a.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=.045,rounding_size=.1",
                fc=fill,
                ec="#4B6B77",
                lw=1.1,
            )
        )
        a.text(
            x + w / 2,
            y + h * 0.64,
            title,
            weight="bold",
            fontsize=10.5,
            ha="center",
            va="center",
        )
        a.text(x + w / 2, y + h * 0.31, desc, fontsize=8.5, ha="center", va="center")

    def arrow(x1, y1, x2, y2, dash=False):
        a.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=13,
                color=GRAY,
                linestyle="--" if dash else "-",
                lw=1.2,
            )
        )

    boxn(0.34, 1.8, 2.45, 1.25, "RETFound-Green", "shared frozen ViT-S/14", "#E5F3F0")
    boxn(
        3.44, 1.8, 2.68, 1.25, "Average patch tokens", "384 values per view", "#E7EDF6"
    )
    boxn(
        6.83,
        1.8,
        2.84,
        1.25,
        "Classifier + calibration",
        "fit on source images",
        "#F0EBF4",
    )
    arrow(0, 2.43, 0.32, 2.43)
    arrow(2.82, 2.43, 3.39, 2.43)
    arrow(6.17, 2.43, 6.78, 2.43)
    a.text(
        4.8,
        0.72,
        "Selection: four width-disjoint folds",
        ha="center",
        fontsize=10,
        color=AMBER,
        weight="bold",
    )
    a.text(
        4.8,
        0.27,
        "Views, probe family, size weighting and regularization",
        ha="center",
        fontsize=8.3,
        color=GRAY,
    )
    arrow(6.9, 0.78, 8.26, 1.75, True)
    a.text(
        4.95,
        4.17,
        "28 × 28 patch grid  ·  4 registers  ·  6 attention heads",
        ha="center",
        fontsize=9,
        color=GRAY,
    )
    save(fig, 2, "foundation_architecture")


def main():
    setup()
    meta = pd.read_csv(BASE / "data/images.csv")
    arch(meta)
    folds = pd.read_csv(OUT / "folds.csv").query("fold>0")
    fig, axes = page(2, 2, size=(11.3, 5.6), sharex=True, sharey=True)
    for j, ax in enumerate(axes.ravel(), 1):
        block = folds.query("fold==@j")
        gr = block.groupby(["width_band", "category"]).size().unstack(fill_value=0)
        ax.bar(gr.index * 32 + 16, gr.normal, 30, color=BLUE, label="Normal")
        ax.bar(
            gr.index * 32 + 16,
            gr.disease,
            30,
            bottom=gr.normal,
            color=AMBER,
            label="Disease",
        )
        ax.set_title(
            f"Fold {j}   ·   {len(block)} images   ·   {int(block.category.eq('disease').sum())} positive",
            size=11,
        )
        if j > 2:
            ax.set_xlabel("Source image width (pixels)")
        if j in (1, 3):
            ax.set_ylabel("Images")
        ax.grid(axis="y")
    axes[0, 0].legend()
    save(fig, 2, "foundation_folds")
    selected = json.loads((OUT / "selection.json").read_text())["feature"]
    full = pd.read_csv(OUT / "cv_grid.csv")
    cfg = pd.read_csv(OUT / "selected_by_feature.csv").set_index("feature")
    met = pd.read_csv(OUT / "metrics.csv")
    test = met[met.split == "test"].set_index("feature")
    valid = met[met.split == "validation"].set_index("feature")
    fig, axes = page(1, 2, size=(11, 4.5), sharey=True)
    for j, (name, lab) in enumerate(LABELS.items()):
        this = cfg.loc[name]
        entry = full[
            (full.feature == name)
            & (full.C == this.C)
            & (full["head"] == this["head"])
            & (full.weighting == this.weighting)
        ].iloc[0]
        vals = entry[[f"fold_{k}_auc" for k in range(1, 5)]].to_numpy(float)
        axes[0].plot([vals.min(), vals.max()], [j, j], c="#A4B9BC", lw=1.5)
        axes[0].scatter(vals, j + np.linspace(-0.1, 0.1, 4), s=21, c=GRAY, alpha=0.7)
        axes[0].scatter(this.cv_auc, j, marker="D", s=45, c=TEAL, zorder=4)
        v, t = valid.loc[name, "auc"], test.loc[name, "auc"]
        axes[1].plot([t, v], [j, j], c="#B1C1C5", lw=1.3)
        axes[1].scatter(v, j, marker="o", s=37, facecolor="white", edgecolor=BLUE)
        axes[1].scatter(t, j, marker="s", s=39, color=AMBER)
    axes[0].set_yticks(
        range(len(LABELS)),
        [x + (" *" if k == selected else "") for k, x in LABELS.items()],
    )
    axes[0].invert_yaxis()
    axes_title(axes[0], "A   Held-out width groups", "AUROC")
    axes_title(axes[1], "B   Fixed splits", "AUROC")
    axes[0].plot([], [], "D", color=TEAL, label="Mean across four folds")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), fontsize=8)
    axes[1].plot([], [], "o", mfc="white", mec=BLUE, label="Random validation")
    axes[1].plot([], [], "s", color=AMBER, label="Official test")
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), fontsize=8, ncol=2)
    save(fig, 2, "foundation_ablation")
    preds = pd.read_csv(OUT / "predictions.csv")
    limits = (
        pd.read_csv(OUT / "threshold_comparison.csv")
        .query("split == 'test'")
        .set_index("source")
    )
    fig, axes = page(1, 2, size=(9.4, 3.6), sharex=True, sharey=True)
    for ax, split, title in zip(
        axes, ("validation", "test"), ("A   Random validation", "B   Official test")
    ):
        subset = preds[preds.split == split]
        for category, color in (("normal", BLUE), ("disease", AMBER)):
            values = np.sort(subset.loc[subset.category == category, selected])
            ax.step(
                np.r_[0, values, 1],
                np.r_[0, np.arange(1, len(values) + 1) / len(values), 1],
                where="post",
                color=color,
                label=f"{category.title()}   n = {len(values)}",
            )
        for source, style, color in (
            ("grouped_oof", "--", TEAL),
            ("random_validation", ":", PLUM),
        ):
            ax.axvline(
                limits.at[source, "threshold"], linestyle=style, color=color, lw=1.5
            )
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.02, 1.02)
        axes_title(ax, title, "Fixed model's estimated probability of disease")
        ax.legend(loc="lower right", fontsize=8)
    axes[0].set_ylabel("Fraction at or below this score")
    fig.suptitle(
        f"Threshold from width folds = {limits.at['grouped_oof','threshold']:.3f} (dashed)   ·   "
        f"from random validation = {limits.at['random_validation','threshold']:.3f} (dotted)",
        fontsize=10,
    )
    save(fig, 2, "foundation_thresholds")
    data = preds[preds.split == "test"]
    y = data.category.eq("disease").to_numpy()
    fig, axes = page(1, 3, size=(12.2, 3.75), width_ratios=[1, 1, 0.92])
    for name, label, c, ls in (
        ("previous_resnet", "Earlier ensemble", BLUE, "--"),
        (selected, "Selected probe", TEAL, "-"),
    ):
        s = data[name].to_numpy()
        f, t, _ = roc_curve(y, s)
        prec, rec, _ = precision_recall_curve(y, s)
        axes[0].plot(f, t, c=c, ls=ls, label=f"{label}   {test.at[name,'auc']:.3f}")
        axes[1].plot(rec, prec, c=c, ls=ls, label=f"{label}   {test.at[name,'ap']:.3f}")
    axes[0].plot([0, 1], [0, 1], c=GRAY, ls=":", lw=1)
    axes[1].axhline(y.mean(), c=GRAY, ls=":", lw=1)
    axes_title(axes[0], "A   ROC", "False-positive rate", "Sensitivity")
    axes_title(axes[1], "B   Precision–recall", "Sensitivity", "Precision")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), fontsize=8)
    axes[1].legend(loc="lower left", fontsize=8)
    a = data.previous_resnet.to_numpy() >= test.at["previous_resnet", "threshold"]
    b = data[selected].to_numpy() >= test.at[selected, "threshold"]
    agreement = np.array(
        [
            [int(np.sum(((a == y) == i) & ((b == y) == j))) for j in (True, False)]
            for i in (True, False)
        ]
    )
    axes[2].imshow(agreement, cmap="Blues", vmin=0, vmax=agreement.max())
    axes[2].set_title("C   Paired decisions")
    axes[2].set(
        xticks=[0, 1],
        xticklabels=["Correct", "Wrong"],
        yticks=[0, 1],
        yticklabels=["Correct", "Wrong"],
        xlabel="Selected probe",
        ylabel="Earlier ensemble",
    )
    for (i, j), v in np.ndenumerate(agreement):
        axes[2].text(
            j,
            i,
            str(v),
            weight="bold",
            ha="center",
            va="center",
            color="white" if v > 0.6 * agreement.max() else INK,
            fontsize=16,
        )
    save(fig, 2, "foundation_pairing")
    paths = np.load(BASE / "data/images.npy", mmap_mode="r")
    fig, axes = page(1, 4, size=(10.4, 3.05))
    cases = []
    for ax, title, flg in zip(
        axes,
        ["Corrected FP", "Corrected FN", "New FP", "New FN"],
        [
            (a != y) & (b == y) & ~y,
            (a != y) & (b == y) & y,
            (a == y) & (b != y) & ~y,
            (a == y) & (b != y) & y,
        ],
    ):
        subset = data[flg]
        if len(subset):
            # Largest absolute change in distance to decision; ties use file order.
            contrast = (subset[selected] - test.at[selected, "threshold"]) - (
                subset.previous_resnet - test.at["previous_resnet", "threshold"]
            )
            j = contrast.abs().idxmax()
            r = subset.loc[j]
            picture(ax, paths[j], title)
            ax.set_xlabel(
                f"{Path(r.member).stem}\nOld {r.previous_resnet:.3f}   New {r[selected]:.3f}",
                fontsize=8,
            )
            cases.append(
                dict(
                    kind=title,
                    n=len(subset),
                    index=int(j),
                    member=r.member,
                    old=r.previous_resnet,
                    new=r[selected],
                )
            )
        else:
            ax.axis("off")
            ax.text(0.5, 0.55, "No such case", ha="center", va="center")
            ax.set_title(title)
    table(OUT / "paired_cases.csv", cases)
    save(fig, 2, "foundation_cases")


if __name__ == "__main__":
    main()
