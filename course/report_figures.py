"""Create the three main replication figures from per-example measurements."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve
from course.common import ROOT
from course.figures import (
    setup,
    save,
    picture,
    axes_title,
    TEAL,
    BLUE,
    AMBER,
    GRAY,
)


def retina(out):
    p = pd.read_csv(out / "predictions.csv").query("split=='test'")
    y = p.category.eq("disease")
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.4), layout="constrained")
    for key, color in [("rank_0", AMBER), ("rank_18", TEAL)]:
        fpr, tpr, _ = roc_curve(y, p[key])
        prec, rec, _ = precision_recall_curve(y, p[key])
        tag = key.replace("rank_", "Rank ")
        axes[0].plot(fpr, tpr, label=tag, color=color)
        axes[1].plot(rec, prec, color=color, label=tag)
    axes[0].plot([0, 1], [0, 1], ":", color=GRAY)
    axes[1].axhline(y.mean(), ls=":", color=GRAY)
    axes_title(
        axes[0], "A   Discrimination", "False positive rate", "True positive rate"
    )
    axes_title(axes[1], "B   Precision–recall", "Recall", "Precision")
    axes[0].legend(loc="lower right")
    bins = np.linspace(0, 1, 15)
    ax = axes[2]
    for flag, c, tag in [(False, BLUE, "Normal"), (True, AMBER, "Disease")]:
        ax.hist(
            p.loc[y == flag, "rank_18"],
            bins,
            histtype="step",
            linewidth=2,
            color=c,
            label=tag,
        )
    cutoff = (
        pd.read_csv(out / "metrics.csv")
        .query("split=='test' and rank==18")
        .threshold.item()
    )
    ax.axvline(cutoff, ls="--", color=GRAY)
    axes_title(
        ax, "C   Fixed decision threshold", "Predicted probability", "Images per bin"
    )
    ax.legend(loc="upper center")
    save(fig, 2, "report_replay")


def saliency(out):
    p = pd.read_csv(out / "predictions.csv")
    order = p.groupby("category").cc.median().sort_values().index
    fig, axes = plt.subplots(
        1, 2, figsize=(10.6, 6.6), layout="constrained", sharey=True
    )
    for ax, key, label, color in zip(
        axes,
        ["cc", "kl"],
        ["A   Correlation (higher is better)", "B   KL divergence (lower is better)"],
        [TEAL, BLUE],
    ):
        samples = [p.loc[p.category == c, key] for c in order]
        b = ax.boxplot(
            samples,
            orientation="horizontal",
            tick_labels=order,
            widths=0.62,
            showfliers=True,
            patch_artist=True,
            flierprops=dict(marker=".", markersize=2, markeredgecolor=GRAY),
            medianprops=dict(color="#142F3C", lw=1.3),
        )
        for patch in b["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
        ax.axvline(p[key].mean(), ls="--", color=AMBER, lw=1.4)
        ax.set_title(label, loc="left")
        ax.grid(axis="x", alpha=0.22)
        ax.set_xlabel("Per-image " + key.upper())
    save(fig, 3, "report_replay")


def color(out):
    p = pd.read_csv(out / "predictions.csv")
    names = [
        "Plane",
        "Car",
        "Bird",
        "Cat",
        "Deer",
        "Dog",
        "Frog",
        "Horse",
        "Ship",
        "Truck",
    ]
    fig = plt.figure(figsize=(12, 4.7), layout="constrained")
    grids = fig.add_gridspec(2, 6, width_ratios=[4, 1, 1, 1, 1, 1])
    ax = fig.add_subplot(grids[:, 0])
    rates = {
        key: p[key].eq(p.condition).groupby(p.condition).mean().reindex(range(10)) * 100
        for key in ("uncorrected", "selected")
    }
    ax.hlines(np.arange(10), rates["uncorrected"], rates["selected"], color=GRAY, lw=1.3)
    for key, c in [("uncorrected", AMBER), ("selected", TEAL)]:
        ax.scatter(
            rates[key],
            np.arange(10),
            marker="o" if key == "uncorrected" else "|",
            s=30 if key == "uncorrected" else 70,
            label="Original" if key == "uncorrected" else "Corrected",
            color=c,
        )
        images = np.load(out / f"{key}_images.npy", mmap_mode="r")
        for j, i in enumerate([0, 2, 3, 5, 8]):
            a = fig.add_subplot(grids[int(key == "selected"), j + 1])
            picture(a, images[i].transpose(1, 2, 0), interpolation="nearest")
            if key == "uncorrected":
                a.set_title(names[i], fontsize=10)
            if j == 0:
                a.set_ylabel("Original" if key == "uncorrected" else "Corrected")
    ax.set_yticks(np.arange(10), names)
    ax.invert_yaxis()
    axes_title(ax, "A   Probe by class", "Condition agreement (%)", None)
    ax.legend(loc="lower left", fontsize=8)
    save(fig, 4, "report_replay")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("experiment", type=int, choices=[2, 3, 4])
    p.add_argument("--input-dir", type=Path, help="Directory created by course.report_replay")
    args = p.parse_args()
    n = args.experiment
    setup()
    {2: retina, 3: saliency, 4: color}[n](args.input_dir or ROOT / f"experiment{n}/outputs/report_replay")
