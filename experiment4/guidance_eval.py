"""Select flow-guidance settings on training references; audit the locked models at 10k."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
from torch_fidelity.metric_fid import (
    fid_features_to_statistics,
    fid_statistics_to_metric,
)
from torch_fidelity.metric_isc import isc_features_to_metric
from torch_fidelity.metric_kid import kid_features_to_metric

from course.common import configure, dump, table
from experiment4.evaluate import features
from experiment4.guidance_train import EPOCHS, PARENT, ROOT
from experiment4.model import FlowNet
from experiment4.research_evaluate import (
    REFERENCE,
    atomic_array,
    common_noise,
    extract_references,
)
from experiment4.research_train import file_hash
from experiment4.train import data

CALIBRATION_SEED = 1309
CALIBRATION_N = 1200
PCA_DIM = 192
BETA = 0.6
CLASS = [
    "airplane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def sources(arm):
    return PARENT if arm == "parent" else ROOT / arm / "flow.pt"


def load_model(arm, device):
    ck = torch.load(sources(arm), weights_only=True, map_location="cpu")
    if arm != "parent" and len(ck["history"]) != ck["config"]["epochs"]:
        raise ValueError(f"Training is incomplete for {arm}")
    model = FlowNet().to(device, memory_format=torch.channels_last).eval()
    model.load_state_dict(ck["ema"])
    return model


def inception(device):
    return (
        FeatureExtractorInceptionV3("inception-v3-compat", ["2048", "logits_unbiased"])
        .to(device)
        .eval()
    )


@torch.inference_mode()
def sample(model, z, label, steps, cfg, alpha, beta=BETA, step_transform=None):
    state = z.clone()
    previous = None
    for i in range(steps):
        t = z.new_full((len(z),), i / steps)
        with torch.autocast(z.device.type, dtype=torch.bfloat16, enabled=z.is_cuda):
            if cfg == 1:
                v = model(state, t, label)
            else:
                c, u = model(
                    torch.cat((state, state)),
                    torch.cat((t, t)),
                    torch.cat((label, torch.full_like(label, 10))),
                ).chunk(2)
                v = u + cfg * (c - u)
        v = v.float()
        if previous is None:
            previous = v
        state = state + (v + alpha * (v - previous)) / steps
        if step_transform is not None:
            state = step_transform(state, (i + 1) / steps)
        previous = beta * previous + (1 - beta) * v
    return ((state.clamp(-1, 1) + 1) * 127.5).round().byte()


def generate(model, z, y, row, device):
    results = []
    for zi, yi in zip(z.split(64), y.split(64)):
        results.append(
            sample(
                model,
                zi.to(device),
                yi.to(device),
                int(row["steps"]),
                row["cfg"],
                row["alpha"],
            ).cpu()
        )
    return torch.cat(results)


def protocol():
    result = []
    for arm, scales in [
        ("parent", [3.0]),
        ("control", [2.5, 3.5, 4.5]),
        ("model_guidance", [1.75, 2.5, 3.5]),
    ]:
        for budget in (8, 24):
            for cfg in scales:
                for alpha in (0, 0.4, 0.9):
                    result.append(
                        dict(
                            arm=arm,
                            cfg=cfg,
                            steps=budget // 2,
                            alpha=alpha,
                            budget=budget,
                        )
                    )
            # Single conditional branch can take twice as many time steps.
            if arm != "parent":
                for alpha in (0, 0.4, 0.9):
                    result.append(
                        dict(arm=arm, cfg=1.0, steps=budget, alpha=alpha, budget=budget)
                    )
    for r in result:
        r["scenario"] = f'{r["arm"]}_n{r["steps"]}_w{r["cfg"]:g}_a{r["alpha"]:g}'
    return result


def training_reference():
    real = np.load(REFERENCE / "real_train_features.npy").astype(np.float64)
    ids = np.load(REFERENCE / "reference_indices.npz")["train"]
    _, labels = data(True)
    center = real.mean(0)
    X = real - center
    cov = X.T @ X
    eigval, Q = np.linalg.eigh(cov / (len(X) - 1))
    ix = np.arange(-PCA_DIM, 0)
    Y = np.eye(10)[labels.numpy()[ids]] - 0.1
    cov.flat[:: len(cov) + 1] += 100
    probe = np.linalg.solve(cov, X.T @ Y)
    return center, Q[:, ix], eigval[ix], probe


def projected_fd(generated, values):
    mu = generated.mean(0)
    scale = np.sqrt(values)
    cov = np.cov(generated, rowvar=False)
    cross = scale[:, None] * cov * scale[None, :]
    return float(
        mu @ mu
        + np.trace(cov)
        + values.sum()
        - 2 * np.sqrt(np.maximum(np.linalg.eigvalsh(cross), 0)).sum()
    )


def calibrate(device, arms):
    root = ROOT / "selection"
    root.mkdir(parents=True, exist_ok=True)
    planned = protocol()
    contract = dict(
        noise_seed=CALIBRATION_SEED,
        samples=CALIBRATION_N,
        training_reference=10000,
        projection=f"leading {PCA_DIM} real training Inception features (PCA)",
        beta=BETA,
        criteria="Minimum projected FD per branch-evaluation budget; for the secondary condition objective maximize probe agreement among settings within 12% of minimum FD. No official test images or final-seed noise used for selection.",
        candidates=planned,
    )
    cp = root / "protocol.json"
    if cp.exists() and json.loads(cp.read_text()) != contract:
        raise ValueError("Saved selection contract differs")
    dump(cp, contract)
    z = torch.randn(
        CALIBRATION_N,
        3,
        32,
        32,
        generator=torch.Generator().manual_seed(CALIBRATION_SEED),
    )
    y = torch.arange(CALIBRATION_N) % 10
    center, Q, var, probe = training_reference()
    feature_net = inception(device)
    rows = []
    for arm in arms:
        model = load_model(arm, device)
        digest = file_hash(sources(arm))
        for row in (r for r in planned if r["arm"] == arm):
            file = root / (row["scenario"] + ".json")
            if file.exists():
                if json.loads(file.read_text())["checkpoint_sha256"] != digest:
                    raise ValueError("stale calibration checkpoint")
                continue
            image = generate(model, z, y, row, device)
            feat, _ = features(feature_net, image, device)
            f = feat.numpy().astype(np.float64) - center
            value = dict(
                **row,
                train_pfd=projected_fd(f @ Q, var),
                train_probe_agreement=float(
                    ((f @ probe).argmax(1) == y.numpy()).mean()
                ),
                checkpoint_sha256=digest,
            )
            dump(file, value)
            print(
                "Calibrated",
                row["scenario"],
                round(value["train_pfd"], 2),
                value["train_probe_agreement"],
                flush=True,
            )
    for row in planned:
        fp = root / (row["scenario"] + ".json")
        if not fp.exists():
            return
        rows.append(json.loads(fp.read_text()))
    frame = pd.DataFrame(rows)
    table(root / "calibration.csv", rows)
    selected = {}
    for budget, sub in frame.groupby("budget"):
        name = f"nfe{budget}"
        best = sub.sort_values(["train_pfd", "scenario"]).iloc[0]
        selected[name + "_quality"] = best.scenario
        eligible = sub[sub.train_pfd <= 1.12 * best.train_pfd]
        selected[name + "_condition"] = (
            eligible.sort_values(
                ["train_probe_agreement", "train_pfd"], ascending=[False, True]
            )
            .iloc[0]
            .scenario
        )
        for arm in ("control", "model_guidance"):
            s = sub[sub.arm == arm]
            selected[f"{name}_{arm}_single"] = (
                s[s.cfg == 1].sort_values("train_pfd").iloc[0].scenario
            )
        no_mom = sub[(sub.arm == best.arm) & (sub.cfg == best.cfg) & (sub.alpha == 0)]
        selected[name + "_momentum_off"] = no_mom.iloc[0].scenario
    for arm in ("parent", "control", "model_guidance"):
        if arm == "parent":
            selected["matched_parent_cfg"] = (
                frame.query("arm==@arm and budget==24 and cfg==3 and alpha==0")
                .iloc[0]
                .scenario
            )
        if arm != "parent":
            selected["matched_" + arm + "_cfg"] = (
                frame.query("arm==@arm and budget==24 and cfg==3.5 and alpha==0")
                .iloc[0]
                .scenario
            )
            selected["single_raw_" + arm] = (
                frame.query("arm==@arm and budget==24 and cfg==1 and alpha==0")
                .iloc[0]
                .scenario
            )
    dump(root / "locked.json", selected)
    print("Locked", selected, flush=True)


def validation_free_controls():
    arms = []
    for cfg, steps in [(1.0, 24), (3.5, 12)]:
        name = f"compute_control_n{steps}_w{cfg:g}_a0"
        arms.append(
            dict(
                arm="compute_control",
                cfg=cfg,
                steps=steps,
                alpha=0.0,
                budget=24,
                scenario=name,
                checkpoint_sha256=file_hash(sources("compute_control")),
            )
        )
    return arms


def score(folder, row, refs, stats, center, probe, y, sec):
    name = row["scenario"]
    feat = torch.from_numpy(np.load(folder / "features.npy"))
    logits = torch.from_numpy(np.load(folder / "logits.npy"))
    G = fid_features_to_statistics(feat)
    fids = {
        k: fid_statistics_to_metric(G, s, False)["frechet_inception_distance"]
        for k, s in stats.items()
    }
    kid = kid_features_to_metric(
        feat,
        refs["test"],
        kid_subsets=50,
        kid_subset_size=1000,
        rng_seed=2020,
        verbose=False,
    )
    pred = ((feat.numpy() - center) @ probe).argmax(1)
    table(
        folder / "per_sample.csv",
        dict(sample=np.arange(len(y)), condition=y.numpy(), probe_prediction=pred),
    )
    results = dict(
        **row,
        **isc_features_to_metric(logits),
        **kid,
        noise_seed=109,
        samples=len(y),
        sample_seconds=sec,
        fid_test=fids["test"],
        fid_train=fids["train"],
        agreement=np.mean(pred == y.numpy()),
    )
    dump(folder / "metrics.json", results)
    print(
        "Test",
        name,
        round(results["fid_test"], 3),
        round(results["agreement"], 4),
        flush=True,
    )


def evaluate(device, compute_only=False):
    sel = ROOT / "selection"
    locked = json.loads((sel / "locked.json").read_text())
    refs, stats, center, probe = extract_references()
    z, y = common_noise()
    encoder = inception(device)
    out = ROOT / "test"
    out.mkdir(exist_ok=True)
    if compute_only:
        rows = validation_free_controls()
        dump(ROOT / "compute_control_protocol.json", rows)
    else:
        main_roles = (
            "nfe8_quality",
            "nfe8_condition",
            "nfe24_quality",
            "nfe24_condition",
        )
        names = [locked[key] for key in main_roles] + list(locked.values())
        rows = [
            json.loads((sel / f"{name}.json").read_text())
            for name in dict.fromkeys(names)
        ]
    # GPU generation and Inception encoding overlap only with the previous
    # collection's CPU metrics. The worker reads feature banks off disk.
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = None
        for row in rows:
            name = row["scenario"]
            if row["checkpoint_sha256"] != file_hash(sources(row["arm"])):
                raise ValueError("Changed model after selection")
            folder = out / name
            folder.mkdir(exist_ok=True)
            if (folder / "metrics.json").exists():
                continue
            meta = folder / "sampling.json"
            if meta.exists() and (folder / "images.npy").exists():
                old = json.loads(meta.read_text())
                if (
                    old["scenario"] != name
                    or old["checkpoint_sha256"] != row["checkpoint_sha256"]
                ):
                    raise ValueError("Cached image protocol differs")
                sec = old["seconds"]
                image = torch.from_numpy(np.load(folder / "images.npy"))
            else:
                model = load_model(row["arm"], device)
                tic = perf_counter()
                image = generate(model, z, y, row, device)
                sec = perf_counter() - tic
                atomic_array(folder / "images.npy", image.numpy())
                dump(
                    meta,
                    dict(
                        scenario=name,
                        checkpoint_sha256=row["checkpoint_sha256"],
                        seconds=sec,
                    ),
                )
            if not all((folder / n).exists() for n in ("features.npy", "logits.npy")):
                feat, logits = features(encoder, image, device)
                atomic_array(folder / "features.npy", feat.numpy())
                atomic_array(folder / "logits.npy", logits.numpy())
                del feat, logits
            del image
            if pending is not None:
                pending.result()
            pending = executor.submit(
                score, folder, row, refs, stats, center, probe, y, sec
            )
        if pending is not None:
            pending.result()
    table(
        out / "metrics.csv",
        [json.loads(p.read_text()) for p in sorted(out.glob("*/metrics.json"))],
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["calibrate", "test", "compute-test"])
    p.add_argument("--arms", nargs="+", default=["parent", "control", "model_guidance"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    configure(42)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    with threadpool_limits(3):
        if args.stage == "calibrate":
            calibrate(torch.device(args.device), args.arms)
        else:
            evaluate(torch.device(args.device), args.stage == "compute-test")
