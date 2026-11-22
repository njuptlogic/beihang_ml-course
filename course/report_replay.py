"""Reproduce the three selected models in separate output directories for the course reports."""
import argparse
from datetime import datetime
import hashlib
from io import BytesIO
import json
from pathlib import Path
import runpy
import subprocess
from time import perf_counter
from zipfile import ZipFile

import numpy as np
import pandas as pd
from PIL import Image
import torch
from threadpoolctl import threadpool_limits

from course.common import ROOT, configure, dump, table


def fingerprint(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def eye(out, device):
    from torch.utils.data import DataLoader
    from experiment2 import spatial as s
    from experiment2.spatial_features import Fundus, encode

    configure(42)
    meta = pd.read_csv(s.BASE / "data/images.csv")
    conf = json.loads((s.OUT / "selection.json").read_text())
    model, sha = s.backbone(conf["encoder"], device)
    batches = []
    for images, _ in DataLoader(Fundus(meta), batch_size=4, num_workers=2):
        batches.append(
            np.stack([encode(model, images[:, j], device) for j in range(3)], axis=1)
        )
    x = s.features(np.concatenate(batches), conf["design"])
    y, g, fit, folds = s.splits(meta)
    pred = meta[["index", "member", "category", "split"]].copy()
    metrics = []
    for rank in [0, conf["rank"]]:

        def solve(a, b):
            return next(
                s.train_solutions(
                    x[a],
                    y[a],
                    g[a],
                    x[b],
                    (rank,),
                    (conf["ridge"],),
                    (conf["weighting"],),
                )
            )

        oof = np.zeros(len(x))
        for a, b in folds:
            oof[b] = solve(a, b)[-1]
        calibration = s.calibrate(oof[fit], y[fit])
        threshold = s.oof_threshold(
            y[fit], g[fit], calibration.predict_proba(oof[fit, None])[:, 1]
        )
        _, coef, bias, score = solve(fit, np.arange(len(x)))
        s.save_model(out / f"rank_{rank}.npz", conf, coef, bias, calibration, threshold)
        prob = calibration.predict_proba(score[:, None])[:, 1]
        pred[f"rank_{rank}"] = prob
        for split, ids in meta.groupby("split").groups.items():
            metrics.append(
                dict(
                    rank=rank,
                    split=split,
                    n=len(ids),
                    threshold=threshold,
                    **s.scores(y[ids], prob[ids], threshold),
                )
            )
    table(out / "predictions.csv", pred)
    table(out / "metrics.csv", metrics)
    return dict(
        encoder_sha256=sha,
        images=len(x),
        views=3,
        fit=len(fit),
        heldout_folds=len(folds),
        configuration_sha256=fingerprint(s.OUT / "selection.json"),
        fit_backend="CPU float64 ridge after CUDA encoder",
    )


def gaze(out, device):
    from experiment3 import multiscale_train as m
    from experiment3.multiscale_predict import Predictor

    configure(42)
    ds = m.Dataset(device, out)
    cfg = json.loads((m.OUT / "backbone_selection.json").read_text())
    seeds = cfg["seeds"]
    for arm in cfg["weights"]:
        for seed in seeds:
            m.train(ds, arm, seed, 18, 20, out)
    meta = ds.meta.query("split=='test'").copy()
    del ds
    predictor = Predictor(device, out)
    teacher = runpy.run_path(
        str(ROOT.parent / "Machine-Learning-Class-Assignment26/metric.py")
    )
    assert meta.archive.nunique() == 1
    rows = []
    grids = []
    with ZipFile(m.BASE / "data/raw" / meta.archive.iloc[0]) as archive:
        for r in meta.itertuples(index=False):
            grid, pred = predictor.one(archive.read(r.member))
            grids.append(grid)
            with Image.open(
                BytesIO(archive.read(r.member.replace("/Stimuli/", "/FIXATIONMAPS/")))
            ) as im:
                target = np.array(im.convert("L"), dtype=np.float64)
            q = pred.astype(np.float64)
            rows.append(
                dict(
                    index=r.index,
                    member=r.member,
                    category=r.category,
                    cc=teacher["calc_cc_score"](target, q),
                    kl=teacher["KLD"](target, q),
                    sim=np.minimum(target / target.sum(), q / q.sum()).sum(),
                )
            )
            if len(rows) % 100 == 0:
                print("Native images:", len(rows), flush=True)
    raw = pd.DataFrame(rows)
    table(out / "predictions.csv", raw)
    np.save(out / "densities.npy", np.stack(grids))
    table(
        out / "categories.csv",
        raw.groupby("category")[["cc", "kl", "sim"]].mean().reset_index(),
    )
    table(
        out / "metrics.csv",
        [dict(n=len(rows), **raw[["cc", "kl", "sim"]].mean().to_dict())],
    )
    return dict(
        seeds=seeds,
        arms=list(cfg["weights"]),
        epochs_per_head=18,
        fit=1360,
        validation=240,
        teacher_sha256=fingerprint(
            ROOT.parent / "Machine-Learning-Class-Assignment26/metric.py"
        ),
        inference="Raw JPEG with freshly trained heads and original frozen encoders",
        weights={p.parent.name: fingerprint(p) for p in sorted(out.glob("*/best.pt"))},
    )


def color(out, device):
    from experiment4 import spectral_guidance as s
    from experiment4 import guidance_train as train
    from experiment4.model import FlowNet

    train.fit("model_guidance", device, out)
    configure(4300)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    path = out / "model_guidance/flow.pt"
    ck = torch.load(path, weights_only=True, map_location="cpu")
    assert len(ck["history"]) == 8
    model = FlowNet().to(device, memory_format=torch.channels_last).eval()
    model.load_state_dict(ck["ema"])
    base = s.original()
    ref = s.reference(device)
    selection = json.loads((s.OUT / "selection.json").read_text())["selected"]
    z, y = s.common_noise()
    real, _ = train.data(False)
    net = s.inception(device)
    real_features, _ = s.features(net, real, device)
    real_stats = s.fid_features_to_statistics(real_features)
    _, _, center, weight = s.extract_references()
    frame = pd.DataFrame({"sample": np.arange(len(z)), "condition": y.numpy()})
    rows = []
    for method, strength in [("uncorrected", 0.0), ("selected", selection["strength"])]:
        images = s.generate(model, base, z, y, device, strength, selection["kind"], ref)
        f, l = s.features(net, images, device)
        fid = s.fid_statistics_to_metric(
            s.fid_features_to_statistics(f), real_stats, False
        )["frechet_inception_distance"]
        kid = s.kid_features_to_metric(
            f,
            real_features,
            kid_subsets=50,
            kid_subset_size=1000,
            rng_seed=2020,
            verbose=False,
        )
        frame[method] = ((f.numpy() - center) @ weight).argmax(1)
        rows.append(
            dict(
                method=method,
                n=len(z),
                fid=fid,
                agreement=(frame[method] == frame.condition).mean(),
                **kid,
                **s.isc_features_to_metric(l),
            )
        )
        s.atomic_array(out / f"{method}_images.npy", images.numpy())
        print(method, rows[-1], flush=True)
    table(out / "metrics.csv", rows)
    table(out / "predictions.csv", frame)
    return dict(
        seed_train=train.SEED,
        seed_sample=109,
        parent_sha256=fingerprint(train.PARENT),
        ema_checkpoint_sha256=fingerprint(path),
        epochs=8,
        starting_epoch=44,
        fit=50000,
        euler_steps=base["steps"],
        strength=selection["strength"],
        training_reference="stored 50k-train radial powers",
        test_reference="fresh complete CIFAR-10 test RGB in Inception",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("experiment", type=int, choices=[2, 3, 4])
    p.add_argument(
        "--output",
        type=Path,
        help="New directory for a fresh fit, or a previous replay directory to resume",
    )
    a = p.parse_args()
    out = a.output or ROOT / f"experiment{a.experiment}/outputs/report_replay"
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    info = dict(
        started=datetime.now().astimezone().isoformat(),
        gpu=torch.cuda.get_device_name(device),
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        driver=subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
        ).strip(),
    )
    print(info, flush=True)
    start = perf_counter()
    with threadpool_limits(2):
        result = {2: eye, 3: gaze, 4: color}[a.experiment](out, device)
    torch.cuda.synchronize()
    dump(
        out / "run.json",
        dict(
            **info,
            completed=datetime.now().astimezone().isoformat(),
            invocation_seconds=perf_counter() - start,
            **result,
        ),
    )
    print((out / "metrics.csv").read_text(), flush=True)


if __name__ == "__main__":
    main()
