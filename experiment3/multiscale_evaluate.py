"""Replay the released static comparator and score every selected density on the original map."""
import argparse
from contextlib import ExitStack
from io import BytesIO
import json, runpy
from zipfile import ZipFile
import numpy as np
import pandas as pd
from PIL import Image
import torch
from course.common import configure, dump, table
from experiment3.train import BASE, compare, resize
from experiment3.multiscale_train import OUT, PREV, CACHE


@torch.inference_mode()
def evaluate(device):
    configure(42)
    prev = np.load(PREV / "selected.npy", mmap_mode="r")
    cfg = json.loads((OUT / "backbone_selection.json").read_text())
    maps = {
        n: np.load(OUT / f"{n}.npy", mmap_mode="r")
        for n in [
            "selected",
            "dino_selected",
            "unisal",
            "cal_global",
            "cal_adaptive",
            *cfg["arms"],
        ]
    }
    unisal = np.load(CACHE / "unisal_logits.npy", mmap_mode="r")
    maps["previous"] = prev
    meta = pd.read_csv(BASE / "data/images.csv")
    allrows = []
    checks = []
    teacher = runpy.run_path(
        str(BASE.parent.parent / "Machine-Learning-Class-Assignment26/metric.py")
    )
    old = (
        pd.read_csv(PREV / "native_images.csv")
        .query("model=='selected'")
        .set_index("index")
    )
    with ExitStack() as st:
        archives = {
            a: st.enter_context(ZipFile(BASE / "data/raw" / a))
            for a in meta.archive.unique()
        }
        for r in meta.itertuples(index=False):
            with Image.open(
                BytesIO(
                    archives[r.archive].read(
                        r.member.replace("/Stimuli/", "/FIXATIONMAPS/")
                    )
                )
            ) as im:
                target = torch.as_tensor(np.array(im.convert("L")), device=device)[
                    None, None
                ].float()
            grid = target.shape[-2:]
            names = (
                ["previous", "selected", "unisal_native"]
                if r.split != "test"
                else [
                    "previous",
                    "selected",
                    "unisal_native",
                    "dino_selected",
                    "unisal",
                    "cal_global",
                    "cal_adaptive",
                    *cfg["arms"],
                ]
            )
            for name in names:
                arr = (unisal if name == "unisal_native" else maps[name])[
                    r.index : r.index + 1
                ]
                pred = resize(torch.from_numpy(np.array(arr)).to(device), grid)
                if name == "unisal_native":
                    pred = pred.flatten(1).softmax(1).reshape_as(pred)
                cc, kl, sim = compare(target, pred)[0].tolist()
                allrows.append(
                    dict(
                        index=r.index,
                        split=r.split,
                        category=r.category,
                        member=r.member,
                        model=name,
                        cc=cc,
                        kl=kl,
                        sim=sim,
                    )
                )
                if r.index in (37, 1200, 1600, 1999) and name in (
                    "selected",
                    "unisal_native",
                ):
                    a = target[0, 0].cpu().numpy().astype("float64")
                    b = pred[0, 0].cpu().numpy().astype("float64")
                    err = max(
                        abs(teacher["calc_cc_score"](a, b) - cc),
                        abs(teacher["KLD"](a, b) - kl),
                    )
                    assert err < 0.0002
                    checks.append(
                        dict(index=r.index, model=name, teacher_max_abs_error=err)
                    )
            if r.index % 200 == 0:
                print("Native", r.index, flush=True)
    frame = pd.DataFrame(allrows)
    frame.to_csv(OUT / "native_images.csv", index=False)
    fitfull = frame[frame.split != "test"].assign(split="train_full")
    pd.concat([frame, fitfull]).groupby(["split", "model"]).agg(
        n=("index", "size"), cc=("cc", "mean"), kl=("kl", "mean"), sim=("sim", "mean")
    ).to_csv(OUT / "native_metrics.csv")
    previous = frame.query("model=='previous'").set_index("index").sort_index()
    error = np.max(
        np.abs(
            previous[["cc", "kl", "sim"]].values
            - old.loc[previous.index, ["cc", "kl", "sim"]].values
        )
    )
    assert error < 0.0002
    dump(
        OUT / "teacher_checks.json",
        dict(checks=checks, previous_replay_max_error=float(error)),
    )
    test = frame[frame.split == "test"]
    oldt = test[test.model == "previous"].sort_values("index").reset_index(drop=True)
    rng = np.random.default_rng(569)
    draw = np.concatenate(
        [
            rng.choice(g.index, (4000, len(g)), replace=True)
            for _, g in oldt.groupby("category")
        ],
        1,
    )
    rows = []
    cases = oldt[["index", "category", "member"]].copy()
    values = ["cc", "kl", "sim"]
    for name, sub in test.groupby("model"):
        if name == "previous":
            continue
        sub = sub.set_index("index").loc[oldt["index"]]
        delta = sub[values].values - oldt[values].values
        for j, metric in enumerate(values):
            interval = np.quantile(delta[:, j][draw].mean(1), [0.025, 0.975])
            rows.append(
                dict(
                    model=name,
                    metric=metric,
                    point=delta[:, j].mean(),
                    lower=interval[0],
                    upper=interval[1],
                    improved=int(
                        (delta[:, j] * (1 if metric != "kl" else -1) > 0).sum()
                    ),
                    n=len(delta),
                )
            )
            if name == "selected":
                cases["previous_" + metric] = oldt[metric]
                cases["selected_" + metric] = sub[metric].values
                cases["delta_" + metric] = delta[:, j]
    table(OUT / "paired_intervals.csv", rows)
    cases.to_csv(OUT / "paired_images.csv", index=False)
    cases.groupby("category").agg(
        n=("index", "size"),
        delta_cc=("delta_cc", "mean"),
        delta_kl=("delta_kl", "mean"),
        delta_sim=("delta_sim", "mean"),
        cc_improved=("delta_cc", lambda v: int((v > 0).sum())),
    ).to_csv(OUT / "categories.csv")
    print(
        pd.read_csv(OUT / "native_metrics.csv")
        .query("split=='test'")
        .to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    evaluate(p.parse_args().device)
