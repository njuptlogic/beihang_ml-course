"""Evaluate the locked grouped readout and its landmark under acquisition changes."""
import json
from zipfile import ZipFile
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from course.common import configure, table
from course.figures import setup, page, axes_title, save, TEAL, AMBER, BLUE
from experiment2.foundation_stress import VARIANTS, NAMES, distort
from experiment2.spatial import features, probability, DESIGNS
from experiment2.spatial_features import OUT, get_views, encode
from experiment2.spatial_shapes import gram
from experiment2.retinal_features import backbone
from experiment2.train import BASE, scores


class Perturbed(Dataset):
    def __init__(self, meta, condition):
        self.rows = meta.to_dict("records")
        self.condition = condition
        self.zip = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        if self.zip is None:
            self.zip = ZipFile(BASE / "data/raw" / r["archive"])
        images, center = get_views(distort(self.zip.read(r["member"]), self.condition))
        return torch.from_numpy(images).permute(0, 3, 1, 2), center


def evaluate():
    configure(42)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = backbone("natural", dev)
    cfg = json.loads((OUT / "selection.json").read_text())
    name = cfg["design"]
    meta = pd.read_csv(BASE / "data/images.csv").query("split=='test'")
    y = meta.category.eq("disease").to_numpy()
    data = []
    stats = []
    for cond in VARIANTS:
        centers = []
        probs = []
        for imgs, c in DataLoader(Perturbed(meta, cond), num_workers=2, batch_size=4):
            bank = np.stack([encode(model, imgs[:, i], dev) for i in range(3)], axis=1)
            shape = (
                np.stack([gram(model, imgs[:, i], dev) for i in (1, 2)], axis=1)
                if "shape" in DESIGNS[name]
                else None
            )
            p, thr = probability(features(bank, name, shape), OUT / (name + ".npz"))
            probs.append(p)
            centers.append(c.numpy())
        c = np.concatenate(centers)
        p = np.concatenate(probs)
        if cond == "clean":
            c0, p0 = c, p
        dis = np.linalg.norm(c - c0, axis=1)
        frac = np.mean(dis > 0.10)
        stats.append(
            dict(
                condition=cond,
                threshold=thr,
                flips=int(np.sum((p >= thr) != (p0 >= thr))),
                center_shift_over_10pct=frac,
                mean_center_shift=dis.mean(),
                **scores(y, p, thr)
            )
        )
        for i, m, prob, pt, dt in zip(meta["index"], meta.member, p, c, dis):
            data.append(
                dict(
                    index=i,
                    member=m,
                    condition=cond,
                    probability=prob,
                    center_x=pt[0],
                    center_y=pt[1],
                    center_shift=dt,
                )
            )
        print(cond, stats[-1]["auc"], stats[-1]["accuracy"], flush=True)
    table(OUT / "stress_metrics.csv", stats)
    table(OUT / "stress_predictions.csv", data)
    # Reuse scores of the previously selected model, tested under the same perturbations.
    prev = (
        pd.read_csv(BASE / "outputs/foundation/stress_metrics.csv")
        .query("model == 'retina_triple'")
        .set_index("condition")
        .loc[list(VARIANTS)]
    )
    now = pd.DataFrame(stats)
    setup()
    fig, ax = page(1, 2, size=(11.7, 4.4))
    ix = np.arange(len(VARIANTS))
    for m, col, ls, label in [
        (prev, AMBER, "s--", "Prior RETFound"),
        (now, TEAL, "o-", "Selected DINO"),
    ]:
        ax[0].plot(ix, m.auc, ls, c=col, label=label)
    ax[0].set_xticks(ix, NAMES, rotation=12)
    axes_title(
        ax[0],
        "A  Ranking under the same perturbations",
        "Test input (250 images)",
        "AUROC",
    )
    ax[0].legend()
    ax[1].plot(ix, now.flips / len(y), "o-", c=TEAL, label="Decision changed")
    ax[1].plot(
        ix, now.center_shift_over_10pct, "D--", c=BLUE, label="Centre shifted by >0.10"
    )
    ax[1].set_xticks(ix, NAMES, rotation=12)
    ax[1].set_ylim(bottom=-0.012)
    axes_title(
        ax[1],
        "B  Which part of the pipeline moves?",
        "Test input (250 images)",
        "Fraction of test images",
    )
    ax[1].legend()
    save(fig, 2, "spatial_perturbations")


if __name__ == "__main__":
    evaluate()
