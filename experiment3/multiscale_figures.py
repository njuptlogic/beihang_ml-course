"""Evidence-linked plots for the external saliency anchor and multiscale ablations."""
import json
from io import BytesIO
from zipfile import ZipFile
import numpy as np
import pandas as pd
import torch
from PIL import Image
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from course.figures import (
    setup,
    page,
    save,
    picture,
    axes_title,
    TEAL,
    BLUE,
    PLUM,
    AMBER,
    GRAY,
    INK,
)
from course.common import table
from experiment3.train import BASE, resize
from experiment3.multiscale_train import OUT, CACHE, PREV, ARMS, SEEDS

NAMES = {
    "middle": "DINO · middle scale",
    "pyramid": "DINO · three scales",
    "guided": "DINO + local weights",
    "scatter": "DINO + fixation spread",
    "full": "DINO + both modules",
    "residual": "DINO + old-model residual",
    "pe_core": "PE-Core · one scale",
    "pe_spatial": "PE-Spatial · one scale",
    "dual": "DINO + PE-Spatial",
    "unisal_native": "UNISAL · original",
    "cal_global": "UNISAL · global calibration",
    "cal_adaptive": "UNISAL · scene calibration",
    "dino_selected": "Feature ensemble",
    "selected": "Final mixture",
    "previous": "Previous course model",
    "unisal": "UNISAL · coarse grid",
}


def architecture():
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    ax.set(xlim=(0, 12), ylim=(0, 6))
    ax.axis("off")

    def box(x, y, w, h, title, note, c):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=.03,rounding_size=.11",
                linewidth=1.4,
                edgecolor=c,
                facecolor=c + "13",
            )
        )
        ax.text(
            x + w / 2,
            y + h * 0.65,
            title,
            ha="center",
            va="center",
            fontsize=10,
            weight="semibold",
            color=INK,
        )
        ax.text(
            x + w / 2,
            y + h * 0.28,
            note,
            ha="center",
            va="center",
            fontsize=8.1,
            color="#42545e",
        )

    def arrow(x1, y1, x2, y2, c=GRAY):
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=12,
                color=c,
                lw=1.3,
                shrinkA=2,
                shrinkB=2,
            )
        )

    box(0.18, 2.65, 1.4, 0.9, "Scene RGB", "one image", BLUE)
    box(2.12, 4.55, 2.6, 1.0, "DINO spatial tokens", "224, 448, 672 px wide", TEAL)
    box(2.12, 3.05, 2.6, 1.0, "Perception Encoder", "PE-Core or PE-Spatial", PLUM)
    box(2.12, 1.55, 2.6, 1.0, "ResNet texture", "stage 2 · local guidance", BLUE)
    box(2.12, 0.12, 2.6, 1.0, "UNISAL · frozen", "public SALICON branch", AMBER)
    for yy in (5.05, 3.55, 2.05, 0.62):
        arrow(1.58, 3.1, 2.1, yy)
    box(5.3, 3.45, 2.65, 1.05, "Spatial readout", "patch fusion / 3 × 3 kernels", TEAL)
    box(
        5.3,
        1.85,
        2.65,
        1.05,
        "Fixation distribution",
        "center prior / learned spread",
        TEAL,
    )
    box(5.3, 0.12, 2.65, 1.0, "Source calibration", "temperature + spatial bias", AMBER)
    for yy in (5.05, 3.55, 2.05):
        arrow(4.75, yy, 5.27, 3.92)
    arrow(6.63, 3.44, 6.63, 2.95)
    arrow(4.73, 0.62, 5.26, 0.62)
    box(
        8.55,
        2.08,
        3.05,
        1.15,
        "Validation-selected mixture",
        "45 × 80 density → native grid",
        PLUM,
    )
    arrow(7.98, 2.37, 8.52, 2.58)
    arrow(7.98, 0.62, 9.55, 2.05)
    ax.text(
        0.2,
        5.87,
        "Frozen encoders and competing density readouts",
        va="center",
        fontsize=14,
        weight="semibold",
        color=INK,
    )
    ax.text(8.65, 4.22, "Fit: 1,360  /  Validate: 240", fontsize=9, color=GRAY)
    ax.text(8.65, 3.76, "Held-out course test: 400", fontsize=9, color=GRAY)
    save(fig, 3, "multiscale_architecture")


def training():
    fig, axes = page(1, 2, size=(11.5, 5), width_ratios=[1, 1.3])
    order = list(ARMS)
    styles = ["-", "--", "-.", ":"]
    colors = [BLUE, TEAL, AMBER, PLUM, GRAY]
    metrics = (
        pd.read_csv(OUT / "native_metrics.csv")
        .query("split=='test'")
        .set_index("model")
    )
    rows = []
    for i, arm in enumerate(order):
        hs = [pd.read_csv(OUT / f"{arm}_{s}/history.csv") for s in SEEDS]
        vals = np.stack([h.cc for h in hs])
        best = vals.max(1)
        rows.extend(dict(arm=arm, seed=s, validation_cc=v) for s, v in zip(SEEDS, best))
        col = colors[i % len(colors)]
        if arm in ("middle", "pyramid", "scatter", "pe_spatial", "dual"):
            j = ["middle", "pyramid", "scatter", "pe_spatial", "dual"].index(arm)
            axes[0].plot(
                hs[0].epoch,
                vals.mean(0),
                c=colors[j],
                ls=styles[j % 4],
                label=NAMES[arm],
                lw=1.8,
            )
        axes[1].scatter(best, [i] * 3, c=col, alpha=0.7, marker="o", s=28)
        axes[1].scatter(metrics.loc[arm, "cc"], i, c=INK, marker="D", s=38)
    axes[1].set_yticks(range(len(order)), [NAMES[n] for n in order])
    axes[1].invert_yaxis()
    axes[1].scatter([], [], marker="o", c=GRAY, label="Validation seeds")
    axes[1].scatter([], [], marker="D", c=INK, label="Native test ensemble")
    axes[1].legend(loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=2)
    axes[0].legend(loc="lower right", fontsize=8)
    axes_title(axes[0], "A   Validation over training", "Epoch", "Image-wise mean CC")
    axes_title(axes[1], "B   Seeds and held-out predictions", "Mean CC")
    table(OUT / "ablation_plot.csv", rows)
    save(fig, 3, "multiscale_training")


def gains():
    f = pd.read_csv(OUT / "paired_images.csv")
    cats = f.groupby("category").delta_cc.mean().sort_values()
    groups = {n: g.delta_cc.values for n, g in f.groupby("category")}
    rng = np.random.default_rng(843)
    fig, (a, b) = page(1, 2, size=(11.5, 6.2), width_ratios=[1, 1.1])
    intervals = []
    for i, (c, v) in enumerate(cats.items()):
        d = groups[c]
        lo, hi = np.quantile(rng.choice(d, (3000, len(d))).mean(1), [0.025, 0.975])
        a.plot([lo, hi], [i, i], c=TEAL, lw=1.6)
        a.scatter(v, i, c=TEAL, s=24)
        intervals.append(dict(category=c, mean=v, lo=lo, hi=hi))
    a.set_yticks(range(len(cats)), cats.index, fontsize=7.8)
    a.axvline(0, c=GRAY, lw=1, ls="--")
    axes_title(a, "A   Paired change by scene", "Final minus previous CC")
    s = b.scatter(
        f.previous_cc,
        f.selected_cc,
        c=f.delta_cc,
        cmap="BrBG",
        vmin=-0.06,
        vmax=0.06,
        s=17,
        alpha=0.77,
        edgecolors="none",
    )
    lim = [min(f.selected_cc.min(), f.previous_cc.min()) - 0.01, 1]
    b.plot(lim, lim, "--", color=GRAY, lw=1)
    b.set(xlim=lim, ylim=lim, aspect="equal")
    axes_title(b, "B   The same 400 test images", "Previous CC", "Final CC")
    b.text(
        0.04,
        0.96,
        f"{(f.delta_cc>0).sum()} improved / 400",
        transform=b.transAxes,
        va="top",
        fontsize=9,
    )
    fig.colorbar(s, ax=b, shrink=0.72, label="Paired CC change")
    table(OUT / "category_intervals.csv", intervals)
    save(fig, 3, "multiscale_gains")


def anchors():
    rows = pd.read_csv(OUT / "native_images.csv").query("split == 'test'")
    final = rows[rows.model == "selected"].sort_values("index").reset_index(drop=True)
    alone = (
        rows[rows.model == "dino_selected"].sort_values("index").reset_index(drop=True)
    )
    assert final.member.equals(alone.member)
    rng = np.random.default_rng(569)
    draws = np.concatenate(
        [
            rng.choice(g.index, (4000, len(g)), replace=True)
            for _, g in final.groupby("category")
        ],
        axis=1,
    )
    contrasts = []
    for metric in ("cc", "kl", "sim"):
        delta = (final[metric] - alone[metric]).to_numpy()
        lo, hi = np.quantile(delta[draws].mean(1), [0.025, 0.975])
        contrasts.append(
            dict(
                metric=metric,
                point=delta.mean(),
                lower=lo,
                upper=hi,
                improved=int((delta * (-1 if metric == "kl" else 1) > 0).sum()),
            )
        )
    table(OUT / "anchor_paired.csv", contrasts)
    f = pd.read_csv(OUT / "fusion_validation.csv")
    m = (
        pd.read_csv(OUT / "native_metrics.csv")
        .query("split=='test'")
        .set_index("model")
    )
    fig, (a, b) = page(1, 2, size=(11, 4.5), width_ratios=[1, 1.13])
    for (n, g), c, ma in zip(
        f.groupby("anchor", sort=False), (AMBER, BLUE, TEAL), ("o", "D", "s")
    ):
        a.plot(g.anchor_weight, g.cc, c=c, marker=ma, label=NAMES[n], ms=4)
    a.legend(fontsize=8)
    axes_title(
        a,
        "A   Choosing the external weight",
        "Weight on the UNISAL branch",
        "Validation CC",
    )
    sel = json.loads((OUT / "selection.json").read_text())
    a.scatter(
        sel["anchor_weight"],
        sel["cc"],
        facecolors="none",
        edgecolors=INK,
        s=104,
        lw=1.4,
        zorder=5,
    )
    names = [
        "unisal_native",
        "cal_global",
        "cal_adaptive",
        "previous",
        "dino_selected",
        "selected",
    ]
    y = np.arange(len(names))
    b.scatter(m.loc[names, "cc"], y, c=[AMBER, AMBER, AMBER, GRAY, BLUE, TEAL], s=62)
    b.set_yticks(y, [NAMES[n] for n in names])
    b.invert_yaxis()
    b.set_xlim(m.loc[names, "cc"].min() - 0.012, m.loc[names, "cc"].max() + 0.025)
    for v, i in zip(m.loc[names, "cc"], y):
        b.text(v + 0.0018, i, f"{v:.4f}", va="center", fontsize=8)
    axes_title(b, "B   Frozen, adapted and fused models", "Native test CC")
    save(fig, 3, "multiscale_anchor")


def cases():
    paired = pd.read_csv(OUT / "paired_images.csv")
    order = paired.sort_values(["delta_cc", "index"])
    inds = [order.iloc[-1], order.iloc[(len(order) - 1) // 2], order.iloc[0]]
    real = np.load(BASE / "data/images.npy", mmap_mode="r")
    truth = np.load(BASE / "data/targets.npy", mmap_mode="r")
    old = np.load(PREV / "selected.npy", mmap_mode="r")
    new = np.load(OUT / "selected.npy", mmap_mode="r")
    unisal = np.load(OUT / "unisal.npy", mmap_mode="r")
    fig, axes = page(3, 6, size=(13.4, 5.35))
    rows = []
    for i, r in enumerate(inds):
        idx = int(r["index"])
        gt = truth[idx].astype(float)
        gt /= gt.mean()
        vals = []
        for arr in (old, unisal, new):
            u = resize(torch.tensor(np.array(arr[idx : idx + 1])), gt.shape)[
                0, 0
            ].numpy()
            vals.append(u / u.mean())
        improvement = np.abs(vals[0] - gt) - np.abs(vals[-1] - gt)
        picture(axes[i, 0], real[idx])
        axes[i, 0].set_ylabel(f"{r.category}\nΔCC {r.delta_cc:+.3f}", fontsize=8)
        for j, data in enumerate((gt, *vals)):
            picture(axes[i, j + 1], data, cmap="magma", vmin=0, vmax=4)
        picture(axes[i, 5], improvement, cmap="BrBG", vmin=-2, vmax=2)
        rows.append(
            dict(
                index=idx,
                member=r.member,
                case=["largest gain", "median change", "largest loss"][i],
                delta_cc=r.delta_cc,
            )
        )
    for a, t in zip(
        axes[0],
        [
            "Image",
            "Ground truth",
            "Previous",
            "Frozen UNISAL",
            "Final mixture",
            "Error reduction",
        ],
    ):
        a.set_title(t, size=9)
    table(OUT / "gallery.csv", rows)
    save(fig, 3, "multiscale_cases")


def representations():
    meta = pd.read_csv(BASE / "data/images.csv")
    ids = [
        int(meta[meta.member.str.endswith(n)].index[0])
        for n in ["OutdoorNatural/199.jpg", "Jumbled/179.jpg", "Social/039.jpg"]
        if any(meta.member.str.endswith(n))
    ]
    if len(ids) < 3:
        ids += (
            meta.query("split=='test' and category=='Social'")
            .index[: 3 - len(ids)]
            .tolist()
        )
    source = [
        ("DINO coarse", CACHE / "coarse.npy"),
        ("DINO fine", CACHE / "fine.npy"),
        ("PE-Core", CACHE / "pe_core.npy"),
        ("PE-Spatial", CACHE / "pe_spatial.npy"),
    ]
    features = [np.load(p, mmap_mode="r") for _, p in source]
    raw = np.load(BASE / "data/images.npy", mmap_mode="r")
    fig, axes = page(len(ids), 5, size=(11.6, 5))
    fit = meta.index[meta.split == "fit"][::24].to_numpy()
    limits = []
    for j, x in enumerate(features):
        channels = [64, 65, 66] if j < 2 else [127, 126, 125]
        sample = np.stack([x[i, channels] for i in fit])
        bounds = np.quantile(sample, [0.01, 0.99], axis=(0, 2, 3))
        limits.append(bounds)
    for i, idx in enumerate(ids):
        picture(axes[i, 0], raw[idx])
        axes[i, 0].set_ylabel(meta.category[idx], fontsize=8)
        for j, (x, (lo, hi)) in enumerate(zip(features, limits)):
            channels = [64, 65, 66] if j < 2 else [127, 126, 125]
            rgb = x[idx, channels].astype(float)
            rgb = np.clip(
                (rgb - lo[:, None, None]) / (hi - lo)[:, None, None], 0, 1
            ).transpose(1, 2, 0)
            picture(axes[i, j + 1], rgb, interpolation="nearest")
    for a, t in zip(axes[0], ["Scene RGB", *[n for n, _ in source]]):
        a.set_title(t, size=10)
    table(
        OUT / "feature_gallery.csv", [dict(index=i, member=meta.member[i]) for i in ids]
    )
    save(fig, 3, "multiscale_features")


if __name__ == "__main__":
    setup()
    architecture()
    training()
    gains()
    anchors()
    cases()
    representations()
