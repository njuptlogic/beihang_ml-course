"""Compare both velocity parameterizations with the same reference set and noise."""
import numpy as np
import pandas as pd
from course.figures import setup, page, axes_title, picture, save, TEAL, BLUE
from experiment4.visualize import OUT, tile


def main():
    setup()
    alt = OUT / "preconditioned"
    records = []
    ref = np.load(OUT / "evaluation/reference_indices.npz")
    other = np.load(alt / "evaluation/reference_indices.npz")
    assert all(np.array_equal(ref[k], other[k]) for k in ref.files)
    colors = [BLUE, TEAL]
    fig, ax = page(2, 2, size=(11, 5.2), width_ratios=[1, 1.48], height_ratios=[1, 1])
    ax[1, 0].axis("off")
    for j, (d, name) in enumerate([(OUT, "Direct velocity"), (alt, "Preconditioned")]):
        a = pd.read_csv(d / "history.csv")
        metrics = (
            pd.read_csv(d / "evaluation/metrics.csv")
            .query("method=='flow_main'")
            .iloc[0]
            .to_dict()
        )
        metrics.update(
            parameterization=name,
            first_mse=a.velocity_mse.iloc[0],
            final_mse=a.velocity_mse.iloc[-1],
        )
        records.append(metrics)
        ax[0, 0].plot(
            a.epoch, a.velocity_mse, color=colors[j], ls=["-", "--"][j], label=name
        )
        im = np.load(d / "evaluation/flow_main_images.npy", mmap_mode="r")
        picture(ax[j, 1], tile(im[:20], 10), name, interpolation="nearest")
    axes_title(ax[0, 0], "A   Matched training runs", "Epoch", "Velocity MSE")
    ax[0, 0].legend()
    ax[1, 0].text(
        0.02,
        0.83,
        "Test FID ↓",
        fontsize=13,
        weight="bold",
        transform=ax[1, 0].transAxes,
    )
    for j, row in enumerate(records):
        ax[1, 0].text(
            0.02,
            0.58 - 0.22 * j,
            row["parameterization"],
            color=colors[j],
            transform=ax[1, 0].transAxes,
        )
        ax[1, 0].text(
            0.85,
            0.58 - 0.22 * j,
            f"{row['fid_test']:.2f}",
            weight="bold",
            ha="right",
            transform=ax[1, 0].transAxes,
        )
    save(fig, 4, "parameterization")
    pd.DataFrame(records).to_csv(OUT / "parameterization.csv", index=False)


if __name__ == "__main__":
    main()
