"""Figures and paired statistics for the continuation/solver experiments."""
import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from course.common import table
from course.figures import (
    setup,
    page,
    picture,
    save,
    axes_title,
    TEAL,
    BLUE,
    AMBER,
    PLUM,
    GRAY,
    INK,
)
from experiment4.teacher_features import ROOT
from experiment4.train import BASE, data
from experiment4.visualize import CLASSES, tile

E = ROOT / "evaluation"
OLD = BASE / "outputs/evaluation"
ARMS = ["control", "time", "align", "combined"]
LABELS = ["Continue", "Time", "Alignment", "Both"]
STYLES = ["-", "--", ":", "-."]
COLORS = [GRAY, BLUE, AMBER, TEAL]
SAMPLERS = [
    "combined_12",
    "combined_midpoint6",
    "combined_strong",
    "combined_projected",
]
SAMPLER_NAMES = ["Euler · 1.6", "Midpoint · 1.6", "Euler · 3.0", "Projected · 3.0"]


def architecture():
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    fig.subplots_adjust(left=0.02, right=0.985, top=0.95, bottom=0.05)
    ax.set(xlim=(0, 12), ylim=(0, 4.35))
    ax.axis("off")

    def box(x, y, w, title, detail, fill, line):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                0.77,
                boxstyle="round,pad=.04,rounding_size=.08",
                fc=fill,
                ec=line,
                lw=1.1,
            )
        )
        ax.text(
            x + w / 2,
            y + 0.50,
            title,
            ha="center",
            va="center",
            weight="bold",
            fontsize=10,
        )
        ax.text(x + w / 2, y + 0.21, detail, ha="center", va="center", fontsize=8.4)

    def arrow(start, end):
        ax.add_patch(
            FancyArrowPatch(
                start, end, arrowstyle="-|>", mutation_scale=11, lw=1.2, color=GRAY
            )
        )

    source = ("#eaf1f7", BLUE)
    model = ("#e5f3ef", TEAL)
    aux = ("#f2eaf5", PLUM)
    loss = ("#faefde", AMBER)
    ax.text(
        0.1,
        4.10,
        "Two training signals for a single velocity field",
        fontsize=17,
        weight="bold",
    )
    box(0.12, 2.73, 1.45, "Clean RGB", "32 × 32", *source)
    box(2.15, 2.73, 1.90, "Interpolate", "noise, class and t", *source)
    box(4.75, 2.73, 2.08, "Flow U-Net", "8 × 8 context", *model)
    box(7.47, 2.73, 1.62, "RGB velocity", "32 × 32", *model)
    box(9.77, 2.73, 1.91, "Velocity MSE", "target x − noise", *loss)
    for a, b in [
        ((1.63, 3.115), (2.07, 3.115)),
        ((4.12, 3.115), (4.66, 3.115)),
        ((6.90, 3.115), (7.38, 3.115)),
        ((9.16, 3.115), (9.70, 3.115)),
    ]:
        arrow(a, b)
    box(0.12, 1.07, 2.08, "Frozen ResNet-18", "ImageNet · 128 × 128", *source)
    box(2.72, 1.07, 1.66, "Teacher tokens", "PCA-64 · 4 × 4", *aux)
    box(5.00, 1.07, 1.64, "Student tokens", "pool to 4 × 4", *model)
    box(7.16, 1.07, 2.02, "Masked adapter", "3 × 3 conv → 64", *aux)
    box(9.77, 1.07, 1.91, "Cosine loss", "4 hidden locations", *loss)
    arrow((0.86, 2.67), (0.86, 1.91))
    arrow((2.25, 1.455), (2.64, 1.455))
    arrow((5.82, 2.67), (5.82, 1.91))
    arrow((6.69, 1.455), (7.08, 1.455))
    arrow((9.24, 1.455), (9.70, 1.455))
    ax.plot([3.55, 3.55, 10.72], [1.01, 0.48, 0.48], c=PLUM, lw=1.2)
    arrow((10.72, 0.48), (10.72, 1.01))
    ax.text(
        7.02,
        0.26,
        "Clean target at the same spatial coordinates",
        ha="center",
        color=PLUM,
        fontsize=9,
    )
    ax.text(
        3.14, 3.73, "Uniform or logit-normal time", ha="center", color=BLUE, fontsize=9
    )
    ax.text(8.18, 2.14, "Training-only branch", ha="center", color=PLUM, fontsize=9)
    save(fig, 4, "research_architecture")


def plots_training(metrics):
    fig, axes = page(1, 3, size=(12.3, 3.75))
    for j, k in enumerate(ARMS):
        hist = pd.read_csv(ROOT / k / "history.csv")
        axes[0].plot(
            hist.total_epoch,
            hist.uniform_diagnostic_mse,
            label=LABELS[j],
            c=COLORS[j],
            ls=STYLES[j],
        )
        tr = hist.iloc[-1]
        axes[1].plot(
            np.arange(4),
            [tr[f"time_q{q}_mse"] for q in range(1, 5)],
            marker=["o", "s", "^", "D"][j],
            ms=4,
            ls=STYLES[j],
            c=COLORS[j],
        )
        if k in ["align", "combined"]:
            axes[2].plot(
                hist.total_epoch,
                hist.masked_cosine_loss,
                label=LABELS[j],
                c=COLORS[j],
                ls=STYLES[j],
            )
    axes_title(
        axes[0],
        "A   Fixed-noise error",
        "Total training epochs",
        "Uniform-time velocity MSE",
    )
    axes[0].legend(fontsize=8)
    axes[1].set_xticks(np.arange(4), ["0–.25", ".25–.5", ".5–.75", ".75–1"])
    axes_title(axes[1], "B   Final error by time", "Time interval", "Velocity MSE")
    axes_title(
        axes[2],
        "C   Predict hidden tokens",
        "Total training epochs",
        "Masked cosine loss",
    )
    axes[2].legend(fontsize=8)
    save(fig, 4, "research_training")
    fig, axes = page(1, 3, size=(11.7, 3.7))
    rows = [metrics.loc[k + "_12"] for k in ARMS]
    for ax, col, title, cmap in zip(
        axes,
        ["fid_test", "inception_score_mean", "agreement"],
        ["A   Test FID ↓", "B   Inception Score ↑", "C   Class agreement ↑"],
        ["Blues_r", "Purples", "BuGn"],
    ):
        values = np.array([r[col] for r in rows]).reshape(2, 2)
        if col == "agreement":
            values = values * 100
        low, high = values.min(), values.max()
        ax.imshow(
            values,
            cmap=cmap,
            aspect="auto",
            vmin=low - (high - low) * 0.35,
            vmax=high + (high - low) * 0.45,
        )
        for (i, j), score in np.ndenumerate(values):
            color = (
                "white"
                if (
                    (col == "fid_test" and score < low + (high - low) * 0.23)
                    or (col != "fid_test" and score > low + (high - low) * 0.78)
                )
                else INK
            )
            ax.text(
                j,
                i,
                (f"{score:.2f}%" if col == "agreement" else f"{score:.2f}"),
                ha="center",
                va="center",
                fontsize=15,
                weight="bold",
                color=color,
            )
        ax.set_xticks([0, 1], ["Uniform", "Logit-normal"])
        ax.set_yticks([0, 1], ["No adapter", "Masked alignment"])
        ax.set_title(title)
        ax.set_xlabel("Training time distribution")
    save(fig, 4, "research_factorial")


def sampler_fig(metrics):
    fig, axes = page(1, 3, size=(11.5, 3.5), sharey=True)
    yy = np.arange(4)
    for ax, col, title in zip(
        axes,
        ["fid_test", "kernel_inception_distance_mean", "agreement"],
        ["A   FID ↓", "B   KID × 1,000 ↓", "C   Class agreement ↑"],
    ):
        scale = (
            100 if col == "agreement" else (1000 if col.startswith("kernel") else 1.0)
        )
        for j, k in enumerate(SAMPLERS):
            val = metrics.loc[k, col] * scale
            ax.scatter(
                val,
                j,
                c=[TEAL, GRAY, BLUE, PLUM][j],
                marker=["o", "D", "s", "^"][j],
                s=57,
                zorder=4,
            )
        ax.set_yticks(yy, SAMPLER_NAMES)
        ax.set_title(title)
        ax.grid(axis="x")
        ax.set(
            xlabel="Agreement (%)"
            if col == "agreement"
            else ("KID × 1,000" if col.startswith("kernel") else "Test FID")
        )
    save(fig, 4, "research_samplers")


def color_distributions():
    true, _ = data(False)
    names = ["real", "combined_12", "combined_strong", "combined_projected"]
    labels = ["Test photos", "Guidance 1.6", "Guidance 3.0", "Projected 3.0"]
    values = []
    for name in names:
        images = (
            true.numpy()
            if name == "real"
            else np.load(E / name / "images.npy", mmap_mode="r")
        )
        frames = []
        for start in range(0, len(images), 200):
            batch = images[start : start + 200]
            chroma = (batch.max(axis=1) - batch.min(axis=1)).mean(axis=(1, 2)) / 255
            near_bound = ((batch <= 1) | (batch >= 254)).mean(axis=(1, 2, 3))
            rms = np.sqrt(
                np.square(batch.astype(np.float32) / 127.5 - 1).mean(axis=(1, 2, 3))
            )
            frames.append(
                pd.DataFrame(
                    dict(
                        scenario=name,
                        sample=np.arange(start, start + len(batch)),
                        mean_chroma=chroma,
                        near_bound=near_bound,
                        rms=rms,
                    )
                )
            )
        values += frames
    frame = pd.concat(values, ignore_index=True)
    frame.to_csv(E / "sample_color.csv", index=False)
    stats = frame.groupby("scenario", sort=False)[
        ["mean_chroma", "near_bound", "rms"]
    ].mean()
    stats.to_csv(E / "color_summary.csv")
    fig, axes = page(1, 3, size=(11.5, 3.6), sharey=True)
    for ax, key, title, xlabel in zip(
        axes,
        ["mean_chroma", "near_bound", "rms"],
        ["A   Pixel chroma", "B   Pixel values near the limits", "C   Pixel magnitude"],
        [
            "Mean channel range (0–1)",
            "Channel values at ≤1 or ≥254 (%)",
            "RMS in [−1, 1] space",
        ],
    ):
        for name, label, c, style in zip(
            names, labels, [GRAY, TEAL, BLUE, PLUM], ["--", "-", "-.", ":"]
        ):
            vals = np.sort(frame.loc[frame.scenario == name, key])
            ax.plot(
                vals * (100 if key == "near_bound" else 1),
                np.arange(1, len(vals) + 1) / len(vals),
                color=c,
                ls=style,
                label=label,
            )
        if key == "near_bound":
            q = frame.groupby("scenario")[key].quantile(0.99).max() * 100
            ax.set_xlim(-0.1, q)
        axes_title(ax, title, xlabel)
    axes[0].set_ylabel("Cumulative share of images")
    axes[2].legend(loc="lower right", fontsize=8)
    save(fig, 4, "research_color")


def comparisons(metrics):
    old = pd.read_csv(OLD / "conditional_diagnostics.csv")
    tests = {r: pd.read_csv(E / r / "per_sample.csv") for r in metrics.index}
    tests["previous"] = old
    rng = np.random.default_rng(816)
    contrasts = []
    for lhs, rhs in [
        ("previous", "control_12"),
        ("control_12", "time_12"),
        ("control_12", "align_12"),
        ("time_12", "combined_12"),
        ("align_12", "combined_12"),
        ("control_12", "combined_12"),
        ("combined_12", "extended_12"),
        ("combined_12", "combined_midpoint6"),
        ("combined_12", "combined_strong"),
        ("combined_strong", "combined_projected"),
        ("control_strong", "time_strong"),
        ("time_strong", "combined_strong"),
        ("align_strong", "combined_strong"),
        ("control_strong", "align_strong"),
        ("time_12", "time_strong"),
        ("align_12", "align_strong"),
        ("control_12", "control_strong"),
    ]:
        a = tests[lhs]
        b = tests[rhs]
        assert np.array_equal(a.condition, b.condition) and np.array_equal(
            a["sample"], b["sample"]
        )
        ac = a.condition.eq(a.probe_prediction)
        bc = b.condition.eq(b.probe_prediction)
        delta = bc.astype(int) - ac.astype(int)
        changes = np.zeros(2000)
        for k in range(10):
            part = delta[a.condition == k]
            counts = np.bincount(part + 1, minlength=3)
            new = rng.multinomial(len(part), counts / len(part), size=2000)
            changes += new[:, 2] - new[:, 0]
        l, u = np.quantile(changes / len(a), [0.025, 0.975])
        contrasts.append(
            dict(
                reference=lhs,
                method=rhs,
                gain=float(delta.mean()),
                lower95=l,
                upper95=u,
                corrected=int((delta > 0).sum()),
                lost=int((delta < 0).sum()),
                unchanged=int((delta == 0).sum()),
            )
        )
    table(E / "paired_agreement.csv", contrasts)
    classes = []
    for name, f in tests.items():
        for k in range(10):
            subset = f[f.condition == k]
            classes.append(
                dict(
                    scenario=name,
                    category=CLASSES[k],
                    agreement=float(subset.probe_prediction.eq(k).mean()),
                )
            )
    table(E / "class_agreement.csv", classes)
    class_frame = (
        pd.DataFrame(classes)
        .pivot(index="category", columns="scenario", values="agreement")
        .loc[CLASSES]
    )
    names = [
        "previous",
        "control_12",
        "time_12",
        "combined_12",
        "extended_12",
        "combined_strong",
        "combined_projected",
    ]
    label = [
        "20 epochs",
        "Continue",
        "Time",
        "Both",
        "Both + 12 ep.",
        "Both, CFG 3",
        "Both, projected",
    ]
    fig, ax = page(size=(9.1, 5.2))
    vals = 100 * class_frame[names].to_numpy().T
    im = ax.imshow(vals, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(np.arange(10), CLASSES, rotation=28, ha="right")
    ax.set_yticks(np.arange(len(names)), label)
    for (i, j), val in np.ndenumerate(vals):
        ax.text(
            j,
            i,
            f"{val:.1f}",
            ha="center",
            va="center",
            fontsize=9,
            color="white" if val > 65 else INK,
        )
    ax.set_title("Class agreement (%) · identical 1,000 noise draws per condition")
    fig.colorbar(im, ax=ax, label="Agreement (%)", shrink=0.8)
    save(fig, 4, "research_classes")
    return tests


def gallery(tests):
    figs = {
        k: np.load(E / k / "images.npy", mmap_mode="r")
        for k in (
            "control_12",
            "time_12",
            "combined_12",
            "extended_12",
            "combined_strong",
            "combined_projected",
        )
    }
    figs["previous"] = np.load(OLD / "flow_main_images.npy", mmap_mode="r")
    fig, axes = page(7, 1, size=(9.6, 7.8))
    for ax, k, name in zip(
        axes,
        [
            "previous",
            "control_12",
            "time_12",
            "combined_12",
            "extended_12",
            "combined_strong",
            "combined_projected",
        ],
        [
            "20 epochs",
            "Continue",
            "Time",
            "Both",
            "Both + 12 epochs",
            "Both, CFG 3",
            "Both, projected",
        ],
    ):
        picture(ax, tile(figs[k][:10], 10), interpolation="nearest")
        ax.set_ylabel(name, rotation=0, ha="right", va="center", labelpad=9, fontsize=9)
        if k == "previous":
            ax.set_xticks(np.arange(10) * 34 + 15.5, CLASSES, fontsize=8)
            ax.tick_params(axis="x", labeltop=True, labelbottom=False, length=0)
    save(fig, 4, "research_paired")
    # Fixed a priori classes that were least often recognised in the 20-epoch experiment.
    # In each class show the median margin-change case among paired wins and losses.
    prev = tests["control_12"]
    new = tests["combined_12"]
    correct_a = prev.probe_prediction == prev.condition
    correct_b = new.probe_prediction == new.condition
    diff = new.score_margin - prev.score_margin
    cases = []
    for k in (2, 7, 5):
        for category, mask in [
            ("Gained", ~correct_a & correct_b),
            ("Lost", correct_a & ~correct_b),
        ]:
            subset = diff[(prev.condition == k) & mask].sort_values()
            if len(subset):
                ix = int(subset.index[len(subset) // 2])
                cases.append(
                    dict(
                        sample=ix,
                        condition=CLASSES[k],
                        transition=category,
                        previous=CLASSES[prev.probe_prediction[ix]],
                        combined=CLASSES[new.probe_prediction[ix]],
                        value=float(diff[ix]),
                    )
                )
    table(E / "case_gallery.csv", cases)
    fig, axes = page(2, len(cases), size=(10.2, 3.65))
    for j, item in enumerate(cases):
        ix = item["sample"]
        for row, (key, field) in enumerate(
            [("control_12", "previous"), ("combined_12", "combined")]
        ):
            a = axes[row, j]
            picture(a, figs[key][ix].transpose(1, 2, 0), interpolation="nearest")
            if row == 0:
                a.set_title(f"{item['condition']} · {item['transition']}", fontsize=9)
            a.set_xlabel(item[field], fontsize=9)
            if j == 0:
                a.set_ylabel(["Continue", "Both"][row])
    save(fig, 4, "research_transitions")


def main():
    setup()
    metrics = pd.read_csv(E / "metrics.csv").set_index("scenario")
    architecture()
    plots_training(metrics)
    sampler_fig(metrics)
    color_distributions()
    gallery(comparisons(metrics))


if __name__ == "__main__":
    main()
