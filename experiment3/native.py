"""Reproject model densities and evaluate against original JPEG annotation pixels."""
import argparse
import runpy
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO
from contextlib import ExitStack
import numpy as np
import pandas as pd
from PIL import Image
import torch
from course.common import table, dump
from experiment3.train import BASE, compare, resize


@torch.inference_mode()
def main(out, device):
    meta = pd.read_csv(BASE / "data/images.csv")
    pred = torch.from_numpy(np.load(out / "ensemble/density.npy")).to(device)
    prior = torch.from_numpy(np.load(out / "prior.npy")).to(device)
    rows = []
    checks = []
    teacher_path = BASE.parent.parent / "Machine-Learning-Class-Assignment26/metric.py"
    teacher = None
    if teacher_path.is_file():
        teacher = runpy.run_path(str(teacher_path))
    with ExitStack() as stack:
        archives = {
            n: stack.enter_context(ZipFile(BASE / "data/raw" / n))
            for n in meta.archive.unique()
        }
        for r in meta.itertuples(index=False):
            path = r.member.replace("/Stimuli/", "/FIXATIONMAPS/")
            with Image.open(BytesIO(archives[r.archive].read(path))) as image:
                gt = np.array(image.convert("L"))
            target = torch.from_numpy(gt).to(device, dtype=torch.float32)[None, None]
            maps = resize(torch.cat((prior, pred[r.index : r.index + 1])), gt.shape)
            measures = compare(target.expand(2, -1, -1, -1), maps).cpu().numpy()
            for name, vals in zip(("spatial_prior", "ensemble"), measures):
                rows.append(
                    dict(
                        index=r.index,
                        member=r.member,
                        category=r.category,
                        split=r.split,
                        model=name,
                        cc=float(vals[0]),
                        kl=float(vals[1]),
                        sim=float(vals[2]),
                    )
                )
            if teacher and r.index in {0, 800, 1599, 1600, 1799, 1999}:
                qm = maps[1, 0].cpu().double().numpy()
                p = gt.astype(float)
                cc = teacher["calc_cc_score"](p, qm)
                kl = teacher["KLD"](p, qm)
                error = max(abs(cc - measures[1, 0]), abs(kl - measures[1, 1]))
                checks.append(
                    dict(
                        index=r.index,
                        teacher_cc=cc,
                        teacher_kl=kl,
                        max_error=float(error),
                    )
                )
                assert error < 2e-4
    table(out / "native_per_image.csv", rows)
    frame = pd.DataFrame(rows)
    combined = pd.concat(
        (frame, frame[frame.split != "test"].assign(split="train_full"))
    )
    combined.groupby(["model", "split"])[["cc", "kl", "sim"]].mean().to_csv(
        out / "native_metrics.csv"
    )
    frame.query("split=='test'").groupby(["model", "category"])[
        ["cc", "kl", "sim"]
    ].mean().to_csv(out / "native_categories.csv")
    dump(out / "teacher_metric_check.json", checks)
    print(pd.read_csv(out / "native_metrics.csv").to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=BASE / "outputs")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    main(a.output, a.device)
