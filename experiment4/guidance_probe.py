"""Measure whether the extra supervision changes the conditional velocity on paired paths."""
import argparse
import numpy as np
import pandas as pd
import torch
from course.common import configure, table, dump
from course.figures import setup, page, axes_title, save, BLUE, PLUM, GRAY, TEAL
from experiment4.train import data
from experiment4.guidance_eval import ROOT, load_model

ARMS = {
    "parent": ("44-epoch parent", GRAY, "o"),
    "control": ("52-epoch control", BLUE, "s"),
    "model_guidance": ("52-epoch posterior target", TEAL, "D"),
    "compute_control": ("57-epoch control", PLUM, "^"),
}


@torch.inference_mode()
def probe(device):
    configure(42)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    gen = torch.Generator().manual_seed(761)
    data_x, targets = data(True)
    ids = torch.cat(
        [
            torch.where(targets == cl)[0][torch.randperm(5000, generator=gen)[:150]]
            for cl in range(10)
        ]
    )
    clean = data_x[ids].float() / 127.5 - 1
    labels = targets[ids]
    noise = torch.randn(clean.shape, generator=gen)
    dump(
        ROOT / "probe_protocol.json",
        dict(
            n=1500,
            per_class=150,
            seed=761,
            split="real CIFAR-10 training images",
            times=[0.125, 0.375, 0.625, 0.875],
            evaluation="paired clean images, base noise, labels and time across weights",
        ),
    )
    rows = []
    for arm in ARMS:
        model = load_model(arm, device)
        for t in (0.125, 0.375, 0.625, 0.875):
            for i in range(0, len(clean), 48):
                im = clean[i : i + 48].to(device)
                eps = noise[i : i + 48].to(device)
                y = labels[i : i + 48].to(device)
                state = torch.lerp(eps, im, t)
                cond, unc = model(
                    torch.cat((state, state)),
                    state.new_full((2 * len(state),), t),
                    torch.cat((y, torch.full_like(y, 10))),
                ).chunk(2)
                gap = (cond - unc).square().mean((1, 2, 3)).sqrt().cpu().numpy()
                error = (cond - (im - eps)).square().mean((1, 2, 3)).cpu().numpy()
                rows.extend(
                    dict(
                        arm=arm,
                        time=t,
                        train_index=ids[i + k].item(),
                        label=labels[i + k].item(),
                        rms_gap=a,
                        mse=b,
                    )
                    for k, (a, b) in enumerate(zip(gap, error))
                )
    table(ROOT / "probe_images.csv", rows)
    return pd.DataFrame(rows)


def figure(frame):
    setup()
    summary = frame.groupby(["arm", "time"], as_index=False)[["rms_gap", "mse"]].mean()
    table(ROOT / "probe_summary.csv", summary)
    fig, axs = page(1, 2, size=(10.7, 3.8), sharex=True)
    for arm, (label, color, marker) in ARMS.items():
        group = summary[summary.arm == arm]
        for ax, col in zip(axs, ["rms_gap", "mse"]):
            ax.plot(group.time, group[col], c=color, marker=marker, label=label)
    axes_title(
        axs[0],
        "A   Strength of class conditioning",
        "Flow time (noise → image)",
        "RMS conditional–null velocity gap",
    )
    axes_title(
        axs[1],
        "B   Velocity error under the original target",
        "Flow time (noise → image)",
        "MSE against image − noise",
    )
    axs[0].legend(fontsize=8)
    save(fig, 4, "guidance_velocity_probe")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    figure(probe(torch.device(args.device)))
