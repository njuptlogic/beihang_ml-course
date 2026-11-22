"""Plot fixed-rule error examples from the report reruns, with paired image identities."""
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from course.common import ROOT, table
from course.figures import setup, picture, save
from experiment2.spatial_features import get_views


def eye(root, out):
    p = pd.read_csv(out / "predictions.csv").query("split=='test'")
    cutoff = (
        pd.read_csv(out / "metrics.csv")
        .query("split=='test' and rank==18")
        .threshold.item()
    )
    fp = p[(p.category == "normal") & (p.rank_18 >= cutoff)].nlargest(2, "rank_18")
    fn = p[(p.category == "disease") & (p.rank_18 < cutoff)].nsmallest(2, "rank_18")
    cases = pd.concat([fp.assign(error="FP"), fn.assign(error="FN")])
    table(out / "cases.csv", cases)
    fig, axes = plt.subplots(2, 4, figsize=(9, 4.85), layout="constrained")
    with ZipFile(root / "data/raw/2-MedImage-TestSet.zip") as z:
        for j, r in enumerate(cases.itertuples()):
            blob = z.read(r.member)
            with Image.open(BytesIO(blob)) as im:
                raw = im.convert("RGB")
                raw.thumbnail((500, 500))
                picture(axes[0, j], raw)
            views, _ = get_views(blob)
            picture(axes[1, j], views[1])
            axes[0, j].set_title(
                f"{r.error} · {Path(r.member).stem}\np = {r.rank_18:.3f}", fontsize=9
            )
    axes[0, 0].set_ylabel("Original")
    axes[1, 0].set_ylabel("Model crop")
    save(fig, 2, "report_cases")


def gaze(root, out):
    p = pd.read_csv(out / "predictions.csv")
    p["row"] = np.arange(len(p))
    old = pd.read_csv(root / "outputs/multiscale/paired_images.csv")[
        ["index", "previous_cc"]
    ]
    p = p.merge(old, on="index", validate="one_to_one")
    p["delta_cc"] = p.cc - p.previous_cc
    cases = p.nsmallest(4, "delta_cc")
    table(out / "cases.csv", cases)
    grids = np.load(out / "densities.npy", mmap_mode="r")
    fig, axes = plt.subplots(3, 4, figsize=(11.4, 5.5), layout="constrained")
    with ZipFile(root / "data/raw/3-Saliency-TestSet.zip") as z:
        for j, r in enumerate(cases.itertuples()):
            with Image.open(BytesIO(z.read(r.member))) as im:
                picture(
                    axes[0, j],
                    im.convert("RGB").resize((320, 180), Image.Resampling.LANCZOS),
                )
            with Image.open(
                BytesIO(z.read(r.member.replace("/Stimuli/", "/FIXATIONMAPS/")))
            ) as im:
                gt = np.array(
                    im.convert("L").resize((320, 180), Image.Resampling.BOX),
                    dtype="float64",
                )
            pred = grids[r.row]
            for ax, d in [(axes[1, j], gt), (axes[2, j], pred)]:
                picture(
                    ax,
                    d / d.mean(),
                    cmap="magma",
                    vmin=0,
                    vmax=4,
                    interpolation="bilinear",
                    aspect="auto",
                )
            axes[0, j].set_title(
                f"{r.category}/{Path(r.member).stem}  ·  Δ {r.delta_cc:+.3f}",
                fontsize=9,
            )
    for a, name in zip(axes[:, 0], ["Scene", "Reference", "Prediction"]):
        a.set_ylabel(name)
    save(fig, 3, "report_cases")


def color(root, out):
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
    chosen = []
    for c in [2, 3, 5]:
        q = p[p.condition == c]
        for change, mask in [
            ("Gain", (q.selected == c) & (q.uncorrected != c)),
            ("Loss", (q.selected != c) & (q.uncorrected == c)),
        ]:
            chosen.append(q[mask].head(1).assign(change=change))
    cases = pd.concat(chosen)
    table(out / "cases.csv", cases)
    fig, axes = plt.subplots(2, len(cases), figsize=(11.2, 4), layout="constrained")
    for i, key in enumerate(["uncorrected", "selected"]):
        imgs = np.load(out / f"{key}_images.npy", mmap_mode="r")
        for j, r in enumerate(cases.itertuples()):
            ax = axes[i, j]
            picture(ax, imgs[r.sample].transpose(1, 2, 0), interpolation="nearest")
            ax.set_xlabel(f"Probe: {names[getattr(r,key)]}", fontsize=9)
            if i == 0:
                ax.set_title(
                    f"{names[r.condition]} {r.sample}\n{r.change}", fontsize=10
                )
        axes[i, 0].set_ylabel(["Original", "Corrected"][i])
    save(fig, 4, "report_cases")


if __name__ == "__main__":
    setup()
    for n, plot in [(2, eye), (3, gaze), (4, color)]:
        root = ROOT / f"experiment{n}"
        plot(root, root / "outputs/report_replay")
