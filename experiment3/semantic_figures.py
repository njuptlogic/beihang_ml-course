"""Render study figures from native paired measurements and saved validation histories."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.ticker import MaxNLocator
from PIL import Image
from sklearn.metrics import confusion_matrix
from course.common import table
from course.figures import (
    setup,
    page,
    save,
    axes_title,
    picture,
    INK,
    GRAY,
    TEAL,
    BLUE,
    AMBER,
    PLUM,
)
from experiment3.train import BASE, SEEDS
from experiment3.semantic_features import OUT
from experiment3.semantic_train import VARIANTS
from experiment3.semantic_coverage import main as coverage

LABELS = ("Tokens", "Tokens + local", "Tokens + routing", "All three")
COLORS = (BLUE, AMBER, PLUM, TEAL)


def architecture():
    fig, ax = plt.subplots(figsize=(13, 4.5))
    fig.subplots_adjust(left=0.01, right=0.985, top=0.96, bottom=0.04)
    ax.set(xlim=(0, 14), ylim=(0, 4.15))
    ax.axis("off")
    nodes = {}

    def box(key, x, y, w, h, title, note, color):
        nodes[key] = (x, y, w, h)
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=.02,rounding_size=.08",
                facecolor=color,
                edgecolor="#536A78",
                lw=1.05,
                zorder=2,
            )
        )
        ax.text(
            x + w / 2,
            y + h * 0.66,
            title,
            ha="center",
            va="center",
            weight="bold",
            size=10.5,
            zorder=3,
        )
        ax.text(
            x + w / 2, y + h * 0.29, note, ha="center", va="center", size=8.8, zorder=3
        )

    def connect(a, b):
        x, y, w, h = nodes[a]
        u, v, ww, hh = nodes[b]
        ax.add_patch(
            FancyArrowPatch(
                (x + w, y + h / 2),
                (u, v + hh / 2),
                arrowstyle="-|>",
                mutation_scale=12,
                shrinkA=3,
                shrinkB=3,
                color=GRAY,
                lw=1.2,
                zorder=1,
            )
        )

    ax.text(
        0.15,
        3.8,
        "Reading gaze from scene semantics and local structure",
        weight="bold",
        size=18,
    )
    box("image", 0.12, 1.62, 1.35, 0.76, "RGB image", "full field", "#E7F0F3")
    box(
        "dino", 2.02, 2.37, 1.75, 0.84, "DINOv2-S/14", "frozen · 4 registers", "#DFEFEB"
    )
    box("cnn", 2.02, 0.57, 1.75, 0.84, "ResNet-18", "frozen · stage 2", "#DFEFEB")
    box(
        "tok", 4.42, 1.64, 1.97, 0.78, "Patch readout", "blocks 6 + 12 · PCA", "#EAE9F4"
    )
    box("route", 4.42, 2.73, 1.97, 0.72, "Scene router", "CLS → 20 weights", "#F9EFDF")
    box("local", 4.42, 0.57, 1.97, 0.73, "Local texture", "1 × 1 projection", "#EAE9F4")
    box(
        "prior",
        7.20,
        2.70,
        2.02,
        0.8,
        "Spatial prior",
        "mixture of fit maps",
        "#F9EFDF",
    )
    box(
        "fuse",
        7.20,
        1.03,
        2.02,
        0.83,
        "Spatial fusion",
        "24 × 40 score grid",
        "#EAE9F4",
    )
    box("soft", 10, 1.65, 1.76, 0.8, "Density head", "score + α log π", "#DFEFEB")
    box("out", 12.25, 1.65, 1.5, 0.8, "Softmax", "unit mass", "#E7F0F3")
    for a, b in [
        ("image", "dino"),
        ("image", "cnn"),
        ("dino", "tok"),
        ("dino", "route"),
        ("cnn", "local"),
        ("tok", "fuse"),
        ("local", "fuse"),
        ("route", "prior"),
        ("prior", "soft"),
        ("fuse", "soft"),
        ("soft", "out"),
    ]:
        connect(a, b)
    ax.text(
        0.15,
        0.13,
        "Ablations replace the image-routed prior with one global fit map or remove the local branch.",
        fontsize=10,
        color=GRAY,
    )
    save(fig, 3, "semantic_architecture")


def ablations():
    seeds = pd.read_csv(OUT / "metrics_by_seed.csv")
    fig, axes = page(1, 3, size=(12.6, 3.75))
    for j, v in enumerate(VARIANTS):
        allh = np.stack([pd.read_csv(OUT / f"{v}_{s}/history.csv").cc for s in SEEDS])
        ep = np.arange(allh.shape[1]) + 1
        m = allh.mean(0)
        sd = allh.std(0, ddof=1)
        axes[0].plot(
            ep, m, color=COLORS[j], ls=["-", "--", "-.", ":"][j], label=LABELS[j]
        )
        axes[0].fill_between(ep, m - sd, m + sd, color=COLORS[j], alpha=0.09)
        for ax, metric in zip(axes[1:], ["cc", "kl"]):
            obs = seeds.query("variant==@v and split=='test'")[metric].to_numpy()
            ax.scatter(
                obs,
                np.full(3, j) + np.array([-0.115, 0, 0.115]),
                s=18,
                color=COLORS[j],
                alpha=0.58,
            )
            ax.plot(obs.mean(), j, "D", c=COLORS[j], ms=7, mec="white", mew=0.6)
    axes_title(axes[0], "A   Validation", "Epoch", "CC")
    axes[0].legend(fontsize=8)
    for ax, title, metric in zip(
        axes[1:], ["B   Single-model CC ↑", "C   Single-model KL ↓"], ["CC", "KL"]
    ):
        ax.set_yticks(np.arange(4), LABELS)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        axes_title(ax, title, metric)
    save(fig, 3, "semantic_ablation")


def routing():
    route = pd.read_csv(OUT / "routes.csv").query("split=='test'")
    bank = np.load(OUT / "router.npz")
    names = np.asarray(bank["classes"])
    mat = confusion_matrix(
        route.category, route.predicted, labels=names, normalize="true"
    )
    fig = plt.figure(figsize=(12.4, 6.3), layout="constrained")
    gs = fig.add_gridspec(
        3, 4, width_ratios=[0.8, 0.8, 0.13, 2.05], height_ratios=[0.12, 1, 1]
    )
    txt = fig.add_subplot(gs[0, :2])
    txt.axis("off")
    txt.text(0, 0.05, "A   Learned scene priors", fontsize=13, weight="bold")
    for j, n in enumerate(["Action", "Social", "Pattern", "Satelite"]):
        ax = fig.add_subplot(gs[1 + j // 2, j % 2])
        im = bank["bank"][np.where(names == n)[0][0]].reshape(24, 40) * 960
        picture(ax, im, n, cmap="magma", vmin=0, vmax=4)
    ax = fig.add_subplot(gs[:, 3])
    m = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(range(20), names, rotation=75, ha="right", fontsize=7)
    ax.set_yticks(range(20), names, fontsize=7)
    ax.set_xlabel("Image-predicted scene")
    ax.set_ylabel("Folder category")
    ax.set_title("B   Test routing accuracy by scene", pad=8)
    fig.colorbar(m, ax=ax, shrink=0.65, label="Fraction of true category")
    save(fig, 3, "semantic_routing")


def gains():
    d = pd.read_csv(OUT / "paired_images.csv").query("split=='test'")
    cats = pd.read_csv(OUT / "paired_categories.csv").sort_values("delta_cc")
    rng = np.random.default_rng(555)
    ci = []
    yy = np.arange(len(cats))
    for _, r in cats.iterrows():
        x = d.loc[d.category == r.category, "delta_cc"].to_numpy()
        ci.append(
            np.quantile(
                rng.choice(x, (2000, len(x)), replace=True).mean(1), [0.025, 0.975]
            )
        )
    fig, axes = page(1, 2, size=(11.4, 6.2), width_ratios=[1.18, 1])
    lo, hi = np.array(ci).T
    axes[0].hlines(yy, lo, hi, color="#829CA8", lw=1.2)
    axes[0].scatter(cats.delta_cc, yy, color=TEAL, marker="D", s=28, zorder=4)
    axes[0].set_yticks(yy, cats.category, fontsize=8.4)
    axes[0].axvline(0, color=GRAY, ls="--", lw=1)
    axes_title(
        axes[0], "A   Change by scene category", "Paired ΔCC vs previous ensemble"
    )
    axes[1].scatter(
        d.centroid_radius, d.delta_cc, alpha=0.46, s=14, color=BLUE, edgecolors="none"
    )
    axes[1].axhline(0, color=GRAY, ls="--", lw=1)
    groups = pd.qcut(d.centroid_radius, q=4)
    grouped = d.groupby(groups, observed=True)[["centroid_radius", "delta_cc"]].mean()
    axes[1].plot(
        grouped.centroid_radius,
        grouped.delta_cc,
        "D-",
        color=AMBER,
        label="Quartile means",
        ms=7,
    )
    axes_title(
        axes[1],
        "B   Distance of the observed hotspot",
        "Normalized centroid distance from center",
        "Image-level ΔCC",
    )
    axes[1].legend()
    save(fig, 3, "semantic_gains")
    table(
        OUT / "category_intervals.csv",
        dict(category=cats.category, mean=cats.delta_cc, lower=lo, upper=hi),
    )


def mean_unit(x):
    y = np.asarray(
        Image.fromarray(np.squeeze(x).astype("float32")).resize(
            (160, 90), Image.Resampling.BILINEAR
        )
    )
    return y / y.mean()


def cases():
    d = pd.read_csv(OUT / "paired_images.csv").query("split=='test'").set_index("index")
    imgs = np.load(BASE / "data/images.npy", mmap_mode="r")
    gt = np.load(BASE / "data/targets.npy", mmap_mode="r")
    new = np.load(OUT / "selected.npy", mmap_mode="r")
    old = np.load(BASE / "outputs/ensemble/density.npy", mmap_mode="r")
    choices = [
        ("Largest gain", d.delta_cc.idxmax()),
        ("Median gain", (d.delta_cc - d.delta_cc.median()).abs().idxmin()),
        ("Largest drop", d.delta_cc.idxmin()),
    ]
    fig, axes = page(3, 5, size=(12, 5.3))
    rows = []
    for row, (label, idx) in enumerate(choices):
        r = d.loc[idx]
        g = mean_unit(gt[idx])
        a = mean_unit(old[idx])
        b = mean_unit(new[idx])
        change = np.abs(a - g) - np.abs(b - g)
        for col, (im, t) in enumerate(
            zip(
                [imgs[idx], g, a, b, change],
                [
                    "Stimulus",
                    "Observed density",
                    "Previous model",
                    "Selected model",
                    "Error reduction",
                ],
            )
        ):
            params = (
                dict()
                if col == 0
                else dict(cmap="magma", vmin=0, vmax=4)
                if col < 4
                else dict(cmap="BrBG", vmin=-2, vmax=2)
            )
            picture(axes[row, col], im, t if row == 0 else None, **params)
        axes[row, 0].set_ylabel(label, fontsize=9)
        axes[row, 0].set_xlabel(
            r.category + " / " + r.member.rsplit("/", 1)[-1], fontsize=8.5
        )
        axes[row, 2].set_xlabel(f"CC {r.old_cc:.3f}")
        axes[row, 3].set_xlabel(f"CC {r.new_cc:.3f}")
        rows.append(
            dict(
                selection=label,
                index=int(idx),
                member=r.member,
                old_cc=r.old_cc,
                new_cc=r.new_cc,
                delta_cc=r.delta_cc,
                delta_kl=r.delta_kl,
            )
        )
    save(fig, 3, "semantic_cases")
    table(OUT / "gallery.csv", rows)


def swaps():
    data = pd.read_csv(OUT / "native_images.csv").query("split=='test'")
    piv = data.pivot(index="index", columns="model", values="cc")
    rng = np.random.default_rng(696)
    meta = data[data.model == "tokens_local_routed"].set_index("index").loc[piv.index]
    groupids = [
        np.flatnonzero(meta.category.to_numpy() == name)
        for name in sorted(meta.category.unique())
    ]
    draws = np.concatenate(
        [rng.choice(ids, (3000, len(ids)), replace=True) for ids in groupids], axis=1
    )
    labels = ("Token maps", "Local grid", "Spatial prior")
    items = ("swap_tokens", "swap_local", "swap_route")
    fig, axes = page(1, 2, size=(10.5, 4.0), width_ratios=[1.35, 1])
    rows = []
    for j, name in enumerate(items):
        delta = (piv[name] - piv.tokens_local_routed).to_numpy()
        pos = np.arange(len(delta))
        order = np.sort(delta)
        axes[0].plot(pos, order, c=COLORS[j], label=labels[j], ls=["-", "--", "-."][j])
        v = delta.mean()
        lo, hi = np.quantile(delta[draws].mean(1), [0.025, 0.975])
        axes[1].errorbar(
            v,
            j,
            xerr=[[v - lo], [hi - v]],
            c=COLORS[j],
            fmt=["o", "s", "D"][j],
            capsize=4,
        )
        rows.append(
            dict(
                swap=name,
                delta_cc=v,
                lower=lo,
                upper=hi,
                negative=int((delta < 0).sum()),
            )
        )
    axes[0].axhline(0, c=GRAY, lw=1)
    axes[1].axvline(0, c=GRAY, lw=1)
    axes_title(
        axes[0],
        "A   Effect of a same-class donor",
        "Rank within each perturbation (400 images)",
        "CC after exchange − intact CC",
    )
    axes[0].legend()
    axes[1].set_yticks(range(3), labels)
    axes_title(axes[1], "B   Average paired change", "ΔCC with 95% bootstrap interval")
    save(fig, 3, "semantic_exchange")
    table(OUT / "counterfactual_effects.csv", rows)


def main():
    setup()
    architecture()
    ablations()
    routing()
    gains()
    cases()
    swaps()
    coverage()


if __name__ == "__main__":
    main()
