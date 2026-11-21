"""Compare gaze density predictions across scenes, categories and prior difficulty."""
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from course.figures import (
    TEAL,
    BLUE,
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
from experiment3.train import BASE, VARIANTS, SEEDS

OUT = BASE / "outputs"
NAMES = ["Deep", "Multi-scale", "Deep + prior", "Multi-scale + prior"]


def up(a):
    return np.asarray(
        Image.fromarray(a.squeeze().astype("float32")).resize(
            (160, 90), Image.Resampling.BILINEAR
        )
    )


def relative(a):
    return a / a.mean()


def main():
    setup()
    architecture(3)
    meta = pd.read_csv(BASE / "data/images.csv")
    photos = np.load(BASE / "data/images.npy", mmap_mode="r")
    gt = np.load(BASE / "data/targets.npy", mmap_mode="r")
    den = np.load(OUT / "ensemble/density.npy", mmap_mode="r")
    prior = relative(up(np.load(OUT / "prior.npy")))
    raw = pd.read_csv(OUT / "native_per_image.csv").query("split=='test'")
    test = raw[raw.model == "ensemble"].set_index("index").copy()
    comp = raw[raw.model == "spatial_prior"].set_index("index")
    test["delta_cc"] = test.cc - comp.cc
    test["kl_reduction"] = comp.kl - test.kl
    fig, axes = page(4, 5, size=(11.3, 6.8))
    for ax, (name, ds) in zip(axes.flat, test.groupby("category")):
        i = ds.index[0]
        picture(ax, photos[i], name)
    save(fig, 3, "atlas")
    fig, axes = page(1, 3, size=(12.1, 3.8))
    for ax, col, title, yl in zip(
        axes,
        ["training_loss", "val_cc", "val_kl"],
        ["A   Fitting", "B   Spatial correlation", "C   Density divergence"],
        ["KL + 0.3 (1 − CC)", "Validation CC", "Validation KL"],
    ):
        for j, v in enumerate(VARIANTS):
            vals = np.array(
                [pd.read_csv(OUT / f"{v}_{s}/history.csv")[col] for s in SEEDS]
            )
            m = vals.mean(0)
            sd = vals.std(0, ddof=1)
            ep = np.arange(1, len(m) + 1)
            ax.plot(ep, m, c=COLORS[j], ls=STYLE[j], label=NAMES[j])
            ax.fill_between(ep, m - sd, m + sd, color=COLORS[j], alpha=0.10)
        axes_title(ax, title, "Epoch", yl)
    axes[0].legend(fontsize=8)
    save(fig, 3, "training")
    metrics = pd.read_csv(OUT / "metrics.csv").query("split=='test'")
    fig, axes = page(1, 3, size=(11, 3.7), sharey=True)
    for ax, metric, title in zip(
        axes, ["cc", "kl", "sim"], ["A   CC ↑", "B   KL ↓", "C   SIM ↑"]
    ):
        for j, v in enumerate(VARIANTS):
            vals = metrics.query("variant==@v")[metric]
            ax.scatter(
                vals,
                np.full(len(vals), j) + np.array([-0.09, 0, 0.09]),
                s=20,
                color=COLORS[j],
                alpha=0.65,
            )
            ax.plot(vals.mean(), j, "D", ms=7, color=COLORS[j], mec="white")
        ax.axvline(
            metrics.query("variant=='spatial_prior'")[metric].iloc[0],
            ls="--",
            c=GRAY,
            label="Prior only",
        )
        ax.set_yticks(range(4), NAMES)
        axes_title(ax, title, metric.upper())
        ax.legend(loc="lower right")
    save(fig, 3, "ablation")
    cats = pd.read_csv(OUT / "native_categories.csv")
    full = cats[cats.model == "ensemble"].set_index("category").sort_values("cc")
    null = cats[cats.model == "spatial_prior"].set_index("category").loc[full.index]
    fig, axes = page(1, 2, size=(10.4, 7.0), sharey=True)
    for ax, col, title in zip(
        axes, ["cc", "kl"], ["A   Correlation ↑", "B   Divergence ↓"]
    ):
        yy = np.arange(len(full))
        ax.hlines(
            yy,
            np.minimum(full[col], null[col]),
            np.maximum(full[col], null[col]),
            color="#ACBABA",
            lw=1.5,
        )
        ax.scatter(
            null[col],
            yy,
            marker="o",
            facecolor="white",
            edgecolor=GRAY,
            s=33,
            label="Prior only",
            zorder=4,
        )
        ax.scatter(
            full[col],
            yy,
            marker="D",
            c=TEAL,
            s=31,
            label="Multi-scale + prior",
            zorder=5,
        )
        ax.set_yticks(yy, full.index)
        axes_title(ax, title, col.upper())
        ax.legend(loc="lower right", bbox_to_anchor=(1, 1.01), fontsize=8)
    save(fig, 3, "categories")
    fig, axes = page(2, 4, size=(11, 3.7))
    for j, name in enumerate(["Social", "Pattern", "Action", "Satelite"]):
        ids = test.index[test.category == name].to_numpy()
        a = np.mean([relative(gt[i].astype(float)) for i in ids], 0)
        b = np.mean([relative(up(den[i])) for i in ids], 0)
        for k, arr in enumerate([a, b]):
            picture(
                axes[k, j],
                arr,
                title=name if k == 0 else None,
                cmap="magma",
                vmin=0,
                vmax=4,
            )
            if j == 0:
                axes[k, j].set_ylabel(["Observed", "Predicted"][k])
    save(fig, 3, "category_maps")
    selections = [
        ("Largest gain", test.delta_cc.idxmax()),
        ("Median CC", (test.cc - test.cc.median()).abs().idxmin()),
        ("Lowest CC", test.cc.idxmin()),
        ("Largest loss", test.delta_cc.idxmin()),
    ]
    fig, axes = page(4, 5, size=(12.0, 6.55))
    selected = []
    for row, (label, i) in enumerate(selections):
        r = test.loc[i]
        g = relative(gt[i].astype(float))
        p = relative(up(den[i]))
        selected.append(
            dict(
                rank=label,
                index=int(i),
                member=r.member,
                cc=r.cc,
                delta_cc=r.delta_cc,
                kl=r.kl,
            )
        )
        for j, (arr, title) in enumerate(
            zip(
                [photos[i], g, prior, p, p - g],
                [
                    "Stimulus",
                    "Observed",
                    "Fit prior",
                    "Prediction",
                    "Prediction − observed",
                ],
            )
        ):
            a = axes[row, j]
            params = (
                {}
                if j == 0
                else (
                    dict(cmap="RdBu_r", vmin=-4, vmax=4)
                    if j == 4
                    else dict(cmap="magma", vmin=0, vmax=4)
                )
            )
            picture(a, arr, title if row == 0 else None, **params)
            if j == 0:
                a.set_ylabel(label, fontsize=9)
            if j == 1:
                a.set_xlabel(f"{r.category} / {Path(r.member).stem}", fontsize=8)
            if j == 3:
                a.set_xlabel(f"CC {r.cc:.3f}   Δ {r.delta_cc:+.3f}", fontsize=8)
    save(fig, 3, "cases")
    table(OUT / "case_gallery.csv", selected)
    fig, axes = page(1, 2, size=(10.5, 4))
    axes[0].scatter(comp.cc, test.cc, c=TEAL, s=13, alpha=0.49)
    lim = [min(comp.cc.min(), test.cc.min()) - 0.025, 1]
    axes[0].plot(lim, lim, ls="--", c=GRAY)
    axes[0].set_xlim(lim)
    axes[0].set_ylim(lim)
    axes_title(axes[0], "A   One point per test scene", "Prior CC", "Model CC")
    axes[1].hist(test.delta_cc, bins=26, color=BLUE, alpha=0.84)
    axes[1].axvline(0, c=GRAY, ls="--")
    axes_title(
        axes[1], "B   Gain over the spatial prior", "Model CC − prior CC", "Scenes"
    )
    save(fig, 3, "gain")
    rng = np.random.default_rng(401)
    groups = [data.index.to_numpy() for _, data in test.groupby("category")]
    records = []
    for _ in range(2000):
        ids = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        records.append(
            test.loc[ids, ["cc", "delta_cc", "kl_reduction"]].mean().to_numpy()
        )
    dump(
        OUT / "bootstrap.json",
        {
            k: np.quantile(np.array(records)[:, j], [0.025, 0.975]).tolist()
            for j, k in enumerate(["cc", "delta_cc", "kl_reduction"])
        },
    )


if __name__ == "__main__":
    main()
