"""Check which portion of the human density lands in forecast highest-density regions."""
import numpy as np
import pandas as pd
import torch

from course.common import configure, table
from course.figures import setup, page, save, axes_title, GRAY, TEAL, AMBER
from experiment3.train import BASE, resize
from experiment3.semantic_features import OUT

LEVELS = np.arange(1, 10, dtype=np.float64) / 10
MODELS = ("shared_prior", "previous", "selected")
LABELS = ("Shared prior", "Previous model", "Selected model")
COLORS = (GRAY, AMBER, TEAL)


def region_mass(target, prediction):
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    target = target / target.sum()
    prediction = prediction / prediction.sum()
    # Each density level receives one fractional boundary, shared by all tied pixels.
    values, inverse, counts = np.unique(
        prediction, return_inverse=True, return_counts=True
    )
    mass = (values * counts)[::-1]
    observed = np.bincount(inverse, weights=target)[::-1]
    domain = counts[::-1] / len(target)
    predicted_mass = np.r_[0, mass.cumsum()]
    return np.stack(
        [
            np.interp(LEVELS, predicted_mass, np.r_[0, x.cumsum()])
            for x in (observed, domain)
        ],
        axis=-1,
    )


def main():
    configure(42)
    setup()
    meta = (
        pd.read_csv(BASE / "data/images.csv")
        .query("split=='test'")
        .reset_index(drop=True)
    )
    targets = np.load(BASE / "data/targets.npy", mmap_mode="r")
    grids = [
        np.load(BASE / "outputs/prior.npy"),
        np.load(BASE / "outputs/ensemble/density.npy", mmap_mode="r"),
        np.load(OUT / "selected.npy", mmap_mode="r"),
    ]
    data = np.empty((len(MODELS), len(meta), len(LEVELS), 2))
    fine_shape = targets.shape[-2:]
    rows = []
    for j, (model, grid) in enumerate(zip(MODELS, grids)):
        for row in meta.itertuples():
            idx = 0 if len(grid) == 1 else row.index
            fine = resize(torch.from_numpy(grid[idx : idx + 1].copy()), fine_shape)[
                0, 0
            ].numpy()
            result = region_mass(targets[row.index], fine)
            data[j, row.Index] = result
            rows.extend(
                (row.index, row.category, model, level, obs, area)
                for level, (obs, area) in zip(LEVELS, result)
            )
    pd.DataFrame(
        rows, columns=["index", "category", "model", "level", "captured", "area"]
    ).to_csv(OUT / "region_by_image.csv", index=False)
    groups = meta.groupby("category").indices.values()
    rng = np.random.default_rng(617)
    draws = np.concatenate(
        [rng.choice(ids, (3000, len(ids)), replace=True) for ids in groups], axis=1
    )
    fig, axes = page(1, 2, size=(10.4, 4.0))
    summary, deltas = [], []
    for j, (name, label, color) in enumerate(zip(MODELS, LABELS, COLORS)):
        curve = data[j].mean(0)
        for k, level in enumerate(LEVELS):
            sample_means = data[j, draws, k, :].mean(1)
            intervals = np.quantile(sample_means, [0.025, 0.975], axis=0)
            summary.append(
                dict(
                    model=name,
                    level=level,
                    captured=curve[k, 0],
                    area=curve[k, 1],
                    captured_lo=intervals[0, 0],
                    captured_hi=intervals[1, 0],
                    area_lo=intervals[0, 1],
                    area_hi=intervals[1, 1],
                )
            )
        chunk = summary[-len(LEVELS) :]
        axes[0].plot(
            100 * LEVELS,
            100 * (curve[:, 0] - LEVELS),
            marker=("s", "o", "D")[j],
            ms=4,
            color=color,
            label=label,
        )
        axes[0].fill_between(
            100 * LEVELS,
            [100 * (r["captured_lo"] - r["level"]) for r in chunk],
            [100 * (r["captured_hi"] - r["level"]) for r in chunk],
            color=color,
            alpha=0.11,
        )
        axes[1].plot(
            100 * curve[:, 1],
            100 * curve[:, 0],
            color=color,
            ms=4,
            marker=("s", "o", "D")[j],
            label=label,
        )
    for k, level in enumerate(LEVELS):
        delta = data[2, :, k] - data[1, :, k]
        ci = np.quantile(delta[draws].mean(1), [0.025, 0.975], axis=0)
        for m, name in enumerate(("captured", "area")):
            deltas.append(
                dict(
                    level=level,
                    measure=name,
                    delta=delta[:, m].mean(),
                    lower=ci[0, m],
                    upper=ci[1, m],
                )
            )
    table(OUT / "region_summary.csv", summary)
    table(OUT / "region_paired.csv", deltas)
    axes[0].axhline(0, color="#304A5B", ls=":", lw=1)
    axes_title(
        axes[0],
        "A   Coverage of predicted regions",
        "Forecast mass (%)",
        "Observed minus forecast mass (pp)",
    )
    axes_title(
        axes[1],
        "B   Area needed to capture gaze",
        "Fraction of image area (%)",
        "Observed density captured (%)",
    )
    axes[0].legend()
    save(fig, 3, "semantic_coverage")


if __name__ == "__main__":
    main()
