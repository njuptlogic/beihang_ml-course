"""Matched-solver quality, training diagnostics, and paired examples in the original pixel grid."""
import json
import numpy as np
import pandas as pd
import torch
from course.figures import (
    setup,
    page,
    save,
    picture,
    axes_title,
    BLUE,
    TEAL,
    AMBER,
    PLUM,
    GRAY,
)
from course.common import table
from experiment4.learned_solver import OUT, SPLITS, noise, initialize
from experiment4.learned_solver_evaluate import SCENARIOS
from experiment4.research_train import ROOT
from experiment4 import solver_residual as residual


@torch.inference_mode()
def transfer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = initialize(device)
    z, y = noise(*SPLITS["validation"])
    ref = (
        torch.from_numpy(np.load(OUT / "teacher_validation.npy")[:, -1].copy())
        .float()
        .clamp(-1, 1)
    )
    nets = {name: residual.load_variant(name, device) for name in residual.VARIANTS}
    rows = []
    for i in range(0, len(z), 25):
        inputs = residual.buffer(
            model, z[i : i + 25].to(device), y[i : i + 25].to(device)
        )
        outputs = {
            k: torch.stack([net(inputs) for net in ensemble]).mean(0)
            for k, ensemble in nets.items()
        }
        outputs["base"] = inputs[:, :3]
        for name, pred in outputs.items():
            errs = (
                (pred.cpu().clamp(-1, 1) - ref[i : i + len(pred)])
                .square()
                .flatten(1)
                .mean(1)
            )
            rows.extend(
                dict(
                    index=i + j,
                    method=name,
                    mse=v,
                    condition=(i + j) % 10,
                    device=str(device),
                )
                for j, v in enumerate(errs.tolist())
            )
    table(residual.DIRECTORY / "deployment_validation.csv", rows)


def figures():
    setup()
    root = OUT / "evaluation"
    scores = pd.read_csv(root / "metrics.csv").set_index("method")
    assert set(SCENARIOS) == set(scores.index)
    val = pd.read_csv(OUT / "validation.csv").set_index("method")
    old = json.loads((ROOT / "evaluation/time_strong/metrics.json").read_text())
    fig, ax = page(1, 3, size=(12.9, 4.2))
    groups = [
        ("Euler", ["euler_2", "euler_4", "euler_8"], BLUE, "o"),
        ("Two bands, midpoint teacher", ["bands_2", "bands_4", "bands_8"], TEAL, "D"),
        ("Two bands, Euler teacher", ["short_2", "short_4"], PLUM, "^"),
    ]
    for j, metric in enumerate(("fid_test", "agreement")):
        scale = 1 if j == 0 else 100
        for label, names, color, mark in groups:
            vals = scores.loc[names]
            ax[j].plot(
                vals.network_calls,
                scale * vals[metric],
                c=color,
                marker=mark,
                ms=5,
                label=label,
            )
        ax[j].scatter(
            4,
            scale * scores.loc["spatial_4", metric],
            c=AMBER,
            marker="X",
            s=78,
            label="Spatial residual",
        )
        ax[j].axhline(scale * old[metric], c=GRAY, ls="--", lw=1.1)
        ax[j].set_xticks([2, 4, 8])
        axes_title(
            ax[j],
            ["A   Distribution distance", "B   Condition agreement"][j],
            "Backbone calls per sample",
            ["CIFAR-10 test FID", "Probe agreement (%)"][j],
        )
    for fam, color, mark in [
        ("euler", BLUE, "o"),
        ("history", AMBER, "s"),
        ("bands", TEAL, "D"),
    ]:
        ax[2].plot(
            [2, 4, 8],
            val.loc[[f"{fam}_{n}" for n in [2, 4, 8]], "validation_mse"],
            c=color,
            marker=mark,
            label=fam.capitalize(),
        )
    ax[2].set_yscale("log")
    ax[2].set_xticks([2, 4, 8])
    axes_title(
        ax[2],
        "C   Matching the 48-call teacher",
        "Backbone calls per sample",
        "Clipped pixel MSE (log scale)",
    )
    ax[2].legend(fontsize=8)
    ax[1].legend(ncol=2, loc="upper center", bbox_to_anchor=(-0.04, -0.21), fontsize=8)
    save(fig, 4, "solver_quality_cost")
    fig, ax = page(1, 3, size=(12.7, 4))
    for b, c in [(0, TEAL), (1, PLUM)]:
        tab = np.full((4, 5), np.nan)
        for i, row in enumerate(json.loads((OUT / "bands_4.json").read_text())["rows"]):
            tab[i, : i + 2] = row[b]
        ax[b].imshow(tab, vmin=-1.15, vmax=1.15, cmap="RdBu", aspect="auto")
        for (i, j), v in np.ndenumerate(tab):
            if np.isfinite(v):
                ax[b].text(
                    j,
                    i,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    c="white" if abs(v) > 0.6 else "#15313D",
                    size=9,
                )
        ax[b].set_xticks(np.arange(5), ["Noise", "v₀", "v₁", "v₂", "v₃"])
        ax[b].set_yticks(range(4), range(1, 5))
        ax[b].set_ylabel("Learned step")
        ax[b].set_title(["A   Block averages", "B   Within-block detail"][b])
    for v, c in [("point", BLUE), ("spatial", TEAL)]:
        runs = np.array(
            [
                pd.read_csv(residual.DIRECTORY / f"{v}_{seed}.csv").validation_mse
                for seed in residual.SEEDS
            ]
        )
        xx = np.arange(1, runs.shape[1] + 1)
        ax[2].plot(xx, runs.mean(0), color=c, label=v.capitalize())
        ax[2].fill_between(xx, runs.min(0), runs.max(0), color=c, alpha=0.18)
    axes_title(
        ax[2], "C   Training the residual", "Epoch", "Held-out clipped pixel MSE"
    )
    ax[2].legend()
    save(fig, 4, "solver_coefficients")
    items = [
        ("Euler, 4", "euler_4"),
        ("Two bands, 4", "bands_4"),
        ("Spatial, 4", "spatial_4"),
        ("Short teacher, 4", "short_4"),
        ("Euler teacher, 12", None),
        ("Midpoint teacher, 48", "teacher"),
    ]
    classes = [
        "Airplane",
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
    fig, axs = page(len(items), 10, size=(12.2, 7.5))
    table(
        OUT / "gallery.csv",
        [
            dict(sample=i, condition=i, category=label)
            for i, label in enumerate(classes)
        ],
    )
    for k, (desc, name) in enumerate(items):
        arr = np.load(
            root / name / "images.npy"
            if name
            else ROOT / "evaluation/time_strong/images.npy",
            mmap_mode="r",
        )
        for j, a in enumerate(axs[k]):
            picture(
                a,
                arr[j].transpose(1, 2, 0),
                classes[j] if k == 0 else None,
                interpolation="nearest",
            )
        axs[k, 0].set_ylabel(desc, fontsize=8)
    save(fig, 4, "solver_matched_gallery")


if __name__ == "__main__":
    transfer()
    figures()
