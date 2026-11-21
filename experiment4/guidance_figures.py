"""Visualize model-guidance training, momentum sweeps and paired CIFAR samples."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd

from course.common import table
from course.figures import (
    setup,
    page,
    axes_title,
    save,
    picture,
    TEAL,
    BLUE,
    AMBER,
    PLUM,
    GRAY,
    INK,
)
from experiment4.guidance_eval import CLASS, ROOT, REFERENCE
from experiment4.train import BASE, data

ARMS = {
    "parent": ("44-epoch parent", GRAY, "o"),
    "control": ("Velocity target", BLUE, "s"),
    "model_guidance": ("Posterior target", TEAL, "D"),
    "compute_control": ("Compute control", PLUM, "^"),
}


def diagram():
    fig, ax = plt.subplots(figsize=(12.2, 4))
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.03, top=0.96)
    ax.set(xlim=(0, 12), ylim=(0, 4))
    ax.axis("off")

    def box(x, y, w, h, title, subtitle, edge, face):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=.035,rounding_size=.08",
                linewidth=1.25,
                edgecolor=edge,
                facecolor=face,
            )
        )
        ax.text(
            x + w / 2,
            y + 0.65 * h,
            title,
            ha="center",
            va="center",
            weight="semibold",
            fontsize=10.4,
            color=INK,
        )
        ax.text(
            x + w / 2,
            y + 0.28 * h,
            subtitle,
            ha="center",
            va="center",
            fontsize=8.7,
            color=INK,
        )

    def arrow(a, b):
        ax.add_patch(
            FancyArrowPatch(
                a, b, arrowstyle="-|>", mutation_scale=12, color=GRAY, lw=1.2
            )
        )

    ax.text(
        0.1,
        3.64,
        "A    Change the regression target during training",
        fontsize=12,
        weight="semibold",
    )
    box(
        0.15,
        2.08,
        2.25,
        1.13,
        "Noisy CIFAR image",
        "plus time and class",
        BLUE,
        "#EEF4FA",
    )
    box(
        3.15,
        2.08,
        2.6,
        1.13,
        "EMA velocity difference",
        "conditional − null; stop gradient",
        PLUM,
        "#F5F0F8",
    )
    box(
        6.52,
        2.08,
        2.25,
        1.13,
        "Corrected target",
        "image − noise + 0.65 × difference",
        AMBER,
        "#FEF2E7",
    )
    box(
        9.48,
        2.08,
        2.3,
        1.13,
        "Velocity network",
        "learn target by squared error",
        TEAL,
        "#E8F4F2",
    )
    for x1, x2 in [(2.44, 3.08), (5.8, 6.46), (8.81, 9.43)]:
        arrow((x1, 2.65), (x2, 2.65))
    ax.text(
        0.1,
        1.61,
        "B    Extrapolate the velocity while sampling",
        fontsize=12,
        weight="semibold",
    )
    box(
        0.15,
        0.2,
        3.12,
        1,
        "Current velocity",
        "conditional or classifier-free guided",
        TEAL,
        "#E8F4F2",
    )
    box(
        4.22,
        0.2,
        3.1,
        1,
        "Velocity history",
        "exponential average of past steps",
        PLUM,
        "#F5F0F8",
    )
    box(
        8.35,
        0.2,
        3.4,
        1,
        "Euler update",
        "current + α × (current − history)",
        BLUE,
        "#EEF4FA",
    )
    arrow((3.31, 0.69), (4.16, 0.69))
    arrow((7.37, 0.69), (8.29, 0.69))
    save(fig, 4, "guidance_architecture")


def training():
    fig, axes = page(1, 3, size=(12.7, 3.8))
    for name in ["control", "model_guidance", "compute_control"]:
        d = pd.read_csv(ROOT / name / "history.csv")
        label, color, mark = ARMS[name]
        axes[0].plot(
            d.total_epoch, d.velocity_mse, label=label, c=color, marker=mark, ms=4
        )
        axes[2].plot(
            d.total_epoch,
            d.seconds.cumsum() / 60,
            label=label,
            c=color,
            marker=mark,
            ms=4,
        )
        if name == "model_guidance":
            axes[1].plot(
                d.total_epoch, d.correction_power * 1000, c=color, marker=mark, ms=4
            )
    axes_title(
        axes[0],
        "A   Uncorrected velocity error",
        "Total training epochs",
        "Mean squared error",
    )
    axes_title(
        axes[1],
        "B   Target correction",
        "Total training epochs",
        "Mean squared shift × 1,000",
    )
    axes_title(
        axes[2],
        "C   Measured continuation time",
        "Total training epochs",
        "Minutes on shared GPU",
    )
    axes[0].legend(fontsize=8)
    save(fig, 4, "guidance_training")


def sweep(cal):
    fig, axes = page(1, 2, size=(11.5, 4.2), sharex=True, sharey=True)
    for ax, (budget, sub) in zip(axes, cal.groupby("budget")):
        for arm, s in sub.groupby("arm"):
            label, color, mark = ARMS[arm]
            for _, line in s.groupby("cfg"):
                line = line.sort_values("alpha")
                ax.plot(
                    line.train_pfd,
                    line.train_probe_agreement * 100,
                    c=color,
                    alpha=0.42,
                    lw=1.3,
                )
            ax.scatter(
                s.train_pfd,
                s.train_probe_agreement * 100,
                marker=mark,
                c=color,
                s=32,
                label=label,
            )
        axes_title(
            ax,
            f"{budget} branch evaluations / image",
            "Training-reference projected FD ↓",
            "Condition agreement (%)" if ax is axes[0] else None,
        )
        ax.legend(loc="best", fontsize=8)
    save(fig, 4, "guidance_selection")


def metrics(m, locked):
    fig, ax = page(1, 2, size=(11.5, 4.1))
    prior = pd.read_csv(BASE / "outputs/research/evaluation/metrics.csv")
    for axis, (col, y) in zip(
        ax, [("fid_test", "Test FID ↓"), ("agreement", "Condition agreement (%) ↑")]
    ):
        factor = 100 if col == "agreement" else 1
        for arm, group in m.groupby("arm"):
            label, color, mark = ARMS[arm]
            # Slight horizontal offsets separate methods with exactly equal evaluation budgets.
            offset = {
                "parent": -0.7,
                "control": -0.24,
                "model_guidance": 0.24,
                "compute_control": 0.7,
            }[arm]
            axis.scatter(
                group.budget + offset,
                group[col] * factor,
                label=label,
                marker=mark,
                facecolors="none",
                edgecolors=color,
                s=60,
            )
        for budget in (8, 24):
            key = locked[f"nfe{budget}_quality"]
            row = m.set_index("scenario").loc[key]
            pos = (
                budget
                + {
                    "parent": -0.7,
                    "control": -0.24,
                    "model_guidance": 0.24,
                    "compute_control": 0.7,
                }[row.arm]
            )
            axis.scatter(
                pos, row[col] * factor, marker="*", s=130, c=ARMS[row.arm][1], zorder=4
            )
        reference = (
            prior.fid_test.min() if col == "fid_test" else 100 * prior.agreement.max()
        )
        axis.axhline(reference, color=GRAY, ls="--", lw=1.1)
        axis.text(
            0.03,
            reference,
            "Previous best",
            ha="left",
            va="bottom",
            transform=axis.get_yaxis_transform(),
            fontsize=8,
            color=GRAY,
        )
        axis.set_xticks([8, 24])
        axis.set_xlim(5, 27)
        axes_title(
            axis,
            "Test FID" if col == "fid_test" else "Requested class recognized",
            "Evaluations per image",
            y,
        )
    ax[0].legend(fontsize=8)
    save(fig, 4, "guidance_test")


def paired(locked):
    root = ROOT / "test"
    selected = locked["nfe8_quality"]
    idx = pd.read_csv(root / selected / "per_sample.csv")
    current = idx.probe_prediction.to_numpy()
    classes = idx.condition.to_numpy()
    old_dir = BASE / "outputs/research/evaluation/time_strong"
    previous = pd.read_csv(old_dir / "per_sample.csv").probe_prediction.to_numpy()
    new = np.load(root / selected / "images.npy", mmap_mode="r")
    old = np.load(old_dir / "images.npy", mmap_mode="r")
    cases = []
    perclass = []
    change = (current == classes).astype(int) - (previous == classes)
    bootstrap = np.random.default_rng(591)
    all_diffs = []
    for j in range(10):
        vals = change[classes == j]
        boot = vals[bootstrap.integers(len(vals), size=(2000, len(vals)))].mean(1)
        all_diffs.append(boot)
        perclass.append(
            dict(
                category=CLASS[j],
                previous=(previous[classes == j] == j).mean(),
                selected=(current[classes == j] == j).mean(),
                gain=(vals == 1).sum(),
                loss=(vals == -1).sum(),
                lo=np.quantile(boot, 0.025),
                hi=np.quantile(boot, 0.975),
            )
        )
    all_boot = np.mean(all_diffs, axis=0)
    diff = pd.DataFrame(perclass)
    table(ROOT / "paired_by_class.csv", perclass)
    table(
        ROOT / "paired_overall.csv",
        [
            dict(
                previous="time_strong",
                selected=selected,
                n=len(classes),
                gained=(change == 1).sum(),
                lost=(change == -1).sum(),
                delta=change.mean(),
                lo=np.quantile(all_boot, 0.025),
                hi=np.quantile(all_boot, 0.975),
            )
        ],
    )
    fig, ax = page(size=(8.2, 4.4))
    ids = np.arange(10)
    ax.hlines(ids, diff.lo * 100, diff.hi * 100, color="#AFB9BB", lw=2.8)
    ax.scatter((diff.selected - diff.previous) * 100, ids, c=TEAL, s=36, zorder=4)
    ax.axvline(0, c=GRAY, lw=1, ls="--")
    ax.set_yticks(ids, [n.capitalize() for n in CLASS])
    ax.invert_yaxis()
    axes_title(
        ax,
        "Same noise, same class, different generator",
        "Change in condition agreement (percentage points)",
    )
    save(fig, 4, "guidance_classes")
    fig, axes = page(2, 6, size=(11.5, 4))
    for k, j in enumerate([2, 3, 5, 7, 8, 9]):
        sign = 1 if k % 2 == 0 else -1
        ids = np.flatnonzero((classes == j) & (change == sign))
        if not len(ids):
            for a in axes[:, k]:
                a.axis("off")
            continue
        i = int(ids[0])
        cases.append(
            dict(
                index=i,
                category=CLASS[j],
                direction="gained" if sign == 1 else "lost",
                previous=CLASS[previous[i]],
                chosen=CLASS[current[i]],
            )
        )
        for r, (a, b, pred) in enumerate(
            [(axes[0, k], old, previous), (axes[1, k], new, current)]
        ):
            picture(
                a,
                b[i].transpose(1, 2, 0),
                CLASS[j].title() if r == 0 else None,
                interpolation="nearest",
            )
            a.set_xlabel(f"{CLASS[pred[i]]}  ·  #{i}", fontsize=8)
    axes[0, 0].set_ylabel("Prior · 24 evals", fontsize=9)
    axes[1, 0].set_ylabel("Selected · 8 evals", fontsize=9)
    table(ROOT / "paired_gallery.csv", cases)
    save(fig, 4, "guidance_pairs")


def matched(locked):
    pairs = [
        ("8-eval momentum", locked["nfe8_momentum_off"], locked["nfe8_quality"]),
        ("24-eval momentum", locked["nfe24_momentum_off"], locked["nfe24_quality"]),
        (
            "Posterior target; single branch",
            locked["single_raw_control"],
            locked["single_raw_model_guidance"],
        ),
        (
            "Posterior target; matched CFG",
            locked["matched_control_cfg"],
            locked["matched_model_guidance_cfg"],
        ),
        (
            "Posterior target; comparable training FLOPs",
            "compute_control_n24_w1_a0",
            locked["single_raw_model_guidance"],
        ),
    ]
    rows = []
    for label, a, b in pairs:
        before = pd.read_csv(ROOT / "test" / a / "per_sample.csv")
        after = pd.read_csv(ROOT / "test" / b / "per_sample.csv")
        assert np.array_equal(before.condition, after.condition)
        y = before.condition.to_numpy()
        delta = (after.probe_prediction.to_numpy() == y).astype(int) - (
            before.probe_prediction.to_numpy() == y
        )
        rng = np.random.default_rng(600)
        means = []
        for cl in range(10):
            vals = delta[y == cl]
            means.append(
                vals[rng.integers(0, len(vals), size=(2000, len(vals)))].mean(1)
            )
        change = np.mean(means, axis=0)
        rows.append(
            dict(
                contrast=label,
                previous=a,
                selected=b,
                n=len(y),
                gained=(delta == 1).sum(),
                lost=(delta == -1).sum(),
                delta=delta.mean(),
                lo=np.quantile(change, 0.025),
                hi=np.quantile(change, 0.975),
            )
        )
    table(ROOT / "paired_contrasts.csv", rows)


def samples_grid(locked):
    real, truth = data(False)
    inds = [int(np.flatnonzero(truth.numpy() == j)[0]) for j in range(10)]
    rows = [("Real test", real[inds].numpy())]
    old = np.load(
        BASE / "outputs/research/evaluation/time_strong/images.npy", mmap_mode="r"
    )
    rows.append(("Prior · 24 evals", old[:10]))
    for key, caption in [
        ("nfe8_quality", "Quality · 8 evals"),
        ("nfe24_condition", "Condition · 24 evals"),
    ]:
        img = np.load(ROOT / "test" / locked[key] / "images.npy", mmap_mode="r")
        rows.append((caption, img[:10]))
    fig, ax = page(4, 10, size=(13, 5.1))
    for i, (label, imgs) in enumerate(rows):
        ax[i, 0].set_ylabel(label, fontsize=8)
        for j in range(10):
            picture(
                ax[i, j],
                imgs[j].transpose(1, 2, 0),
                CLASS[j].title() if i == 0 else None,
                interpolation="nearest",
            )
    table(
        ROOT / "reference_gallery.csv",
        dict(category=CLASS, real_test_index=inds, generated_index=np.arange(10)),
    )
    save(fig, 4, "guidance_samples")


def main():
    setup()
    selection = ROOT / "selection"
    locked = json.loads((selection / "locked.json").read_text())
    m = pd.read_csv(ROOT / "test/metrics.csv")
    assert set(locked.values()) <= set(m.scenario)
    diagram()
    training()
    sweep(pd.read_csv(selection / "calibration.csv"))
    metrics(m, locked)
    paired(locked)
    matched(locked)
    samples_grid(locked)


if __name__ == "__main__":
    main()
