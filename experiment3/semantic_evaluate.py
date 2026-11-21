"""Native-resolution paired comparison, including image-based scene priors and uncertainty."""
import argparse
from contextlib import ExitStack
from io import BytesIO
import json
import runpy
from zipfile import ZipFile
import numpy as np
import pandas as pd
from PIL import Image
import torch
from course.common import configure, dump, table
from experiment3.train import BASE, compare, resize
from experiment3.semantic_features import OUT
from experiment3.semantic_train import VARIANTS


def paired_summary(data):
    a = data.query("model=='selected'").set_index("index").sort_index()
    b = data.query("model=='previous'").set_index("index").loc[a.index]
    data_paired = a[
        ["member", "category", "split", "eccentricity", "centroid_radius"]
    ].copy()
    for metric in ("cc", "kl", "sim"):
        data_paired[f"new_{metric}"] = a[metric]
        data_paired[f"old_{metric}"] = b[metric]
        data_paired[f"delta_{metric}"] = a[metric] - b[metric]
    data_paired["information_gain_bits"] = -data_paired.delta_kl / np.log(2)
    data_paired.to_csv(OUT / "paired_images.csv")
    test = data_paired[data_paired.split == "test"].reset_index()
    groups = [g.index.to_numpy() for _, g in test.groupby("category")]
    rng = np.random.default_rng(118)
    values = test[
        [
            "delta_cc",
            "delta_kl",
            "delta_sim",
            "information_gain_bits",
            "new_cc",
            "new_kl",
            "new_sim",
        ]
    ]
    draws = np.concatenate(
        [rng.choice(g, size=(3000, len(g)), replace=True) for g in groups], axis=1
    )
    stats = values.to_numpy()[draws].mean(1)
    dump(
        OUT / "paired_intervals.json",
        {
            col: dict(
                point=values[col].mean(),
                ci95=np.quantile(stats[:, j], [0.025, 0.975]).tolist(),
            )
            for j, col in enumerate(values)
        },
    )
    cats = test.groupby("category")[list(values)].mean()
    cats["improved_cc"] = test.groupby("category").delta_cc.apply(
        lambda q: int((q > 0).sum())
    )
    cats.to_csv(OUT / "paired_categories.csv")


@torch.inference_mode()
def main(device):
    configure(42)
    meta = pd.read_csv(BASE / "data/images.csv")
    selected = json.loads((OUT / "selection.json").read_text())
    read = lambda n: np.load(OUT / f"{n}.npy", mmap_mode="r")
    swaps = ["swap_tokens", "swap_local", "swap_route"]
    variants = {s: read(s) for s in [*VARIANTS, *swaps]}
    variants["selected"] = read("selected")
    variants["routed_prior"] = read("routed_prior")
    variants["previous"] = np.load(BASE / "outputs/ensemble/density.npy", mmap_mode="r")
    shared = np.load(BASE / "outputs/prior.npy")
    w = selected["semantic_weight"]
    assert (
        np.max(
            np.abs(
                variants["selected"]
                - ((1 - w) * variants["previous"] + w * variants[selected["variant"]])
            )
        )
        < 1e-7
    )
    candidates = [
        "previous",
        "selected",
        "shared_prior",
        "routed_prior",
        *VARIANTS,
        *swaps,
    ]
    rows = []
    teacher_checks = []
    teacher_path = BASE.parent.parent / "Machine-Learning-Class-Assignment26/metric.py"
    teacher = runpy.run_path(str(teacher_path)) if teacher_path.is_file() else None
    with ExitStack() as st:
        archives = {
            n: st.enter_context(ZipFile(BASE / "data/raw" / n))
            for n in meta.archive.unique()
        }
        for row in meta.itertuples(index=False):
            gt_name = row.member.replace("/Stimuli/", "/FIXATIONMAPS/")
            with Image.open(BytesIO(archives[row.archive].read(gt_name))) as im:
                target = torch.from_numpy(np.array(im.convert("L"))).to(
                    device, dtype=torch.float32
                )[None, None]
            wh = target.shape[-2:]
            horizontal = target.sum((0, 1, 2))
            vertical = target.sum((0, 1, 3))
            u = torch.linspace(-1, 1, wh[1], device=device)
            v = torch.linspace(-1, 1, wh[0], device=device)
            horizontal /= horizontal.sum()
            vertical /= vertical.sum()
            rms = float(
                torch.sqrt(
                    (horizontal * u.square()).sum() + (vertical * v.square()).sum()
                )
            )
            centroid = float(
                torch.sqrt(
                    (horizontal * u).sum().square() + (vertical * v).sum().square()
                )
            )
            # Preserve the full 1,920 x 1,080 annotation grid. Use two maps at a time on 8 GB GPUs.
            names = candidates if row.split == "test" else ["previous", "selected"]
            for start in range(0, len(names), 2):
                group = names[start : start + 2]
                chunk = np.concatenate(
                    [
                        shared
                        if name == "shared_prior"
                        else variants[name][row.index : row.index + 1]
                        for name in group
                    ],
                    axis=0,
                )
                maps = resize(torch.as_tensor(chunk, device=device), wh)
                results = (
                    compare(target.expand(len(group), -1, -1, -1), maps).cpu().numpy()
                )
                for j, (name, result) in enumerate(zip(group, results)):
                    cc, kl, sim = result.tolist()
                    rows.append(
                        dict(
                            index=row.index,
                            member=row.member,
                            category=row.category,
                            split=row.split,
                            model=name,
                            cc=cc,
                            kl=kl,
                            sim=sim,
                            eccentricity=rms,
                            centroid_radius=centroid,
                        )
                    )
                    if (
                        teacher
                        and name == "selected"
                        and row.index in (37, 800, 1599, 1600, 1799, 1999)
                    ):
                        actual = target[0, 0].cpu().double().numpy()
                        pred = maps[j, 0].cpu().double().numpy()
                        tcc = teacher["calc_cc_score"](actual, pred)
                        tkl = teacher["KLD"](actual, pred)
                        err = max(abs(tcc - cc), abs(tkl - kl))
                        assert err < 0.0002
                        teacher_checks.append(
                            dict(
                                index=row.index,
                                teacher_cc=tcc,
                                teacher_kl=tkl,
                                max_error=err,
                            )
                        )
    table(OUT / "native_images.csv", rows)
    data = pd.DataFrame(rows)
    combined = pd.concat([data, data[data.split != "test"].assign(split="train_full")])
    combined.groupby(["split", "model"]).agg(
        n=("index", "size"), cc=("cc", "mean"), kl=("kl", "mean"), sim=("sim", "mean")
    ).to_csv(OUT / "native_metrics.csv")
    paired_summary(data)
    old = (
        pd.read_csv(BASE / "outputs/native_per_image.csv")
        .query("model=='ensemble'")
        .set_index("index")
        .sort_index()
    )
    check = data.query("model=='previous'").set_index("index").sort_index()
    error = float(
        np.abs(
            check[["cc", "kl", "sim"]].to_numpy() - old[["cc", "kl", "sim"]].to_numpy()
        ).max()
    )
    assert error < 2e-4
    dump(
        OUT / "native_checks.json",
        dict(teacher_checks=teacher_checks, old_replay_max_abs_error=error),
    )
    print(pd.read_csv(OUT / "native_metrics.csv").to_string(index=False), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    main(p.parse_args().device)
