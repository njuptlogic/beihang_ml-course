"""Evaluate locked trajectory samplers on paired CIFAR-10 reference features."""
import argparse
import json
from time import perf_counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from threadpoolctl import threadpool_limits
from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
from torch_fidelity.metric_fid import (
    fid_features_to_statistics,
    fid_statistics_to_metric,
)
from torch_fidelity.metric_isc import isc_features_to_metric
from torch_fidelity.metric_kid import kid_features_to_metric

from course.common import dump, table
from experiment4.learned_solver import (
    OUT,
    GUIDANCE,
    TEACHER_STEPS,
    initialize,
    noise,
    parameters,
    pixels,
    sample_raw,
    teacher_path,
)
from experiment4.research_evaluate import atomic_array, common_noise, extract_references
from experiment4.research_train import file_hash
from experiment4 import solver_residual as residual
from experiment4 import solver_target

SPATIAL = {v + "_4": v for v in residual.VARIANTS}
SHORT = {f"short_{b}" for b in solver_target.BUDGETS}

SCENARIOS = (
    "euler_4",
    "bands_4",
    "history_4",
    "euler_2",
    "bands_2",
    "euler_8",
    "bands_8",
    "teacher",
    *SPATIAL,
    "short_2",
    "short_4",
)


@torch.inference_mode()
def generate(model, z, y, name, device, batch):
    out = []
    if name in SHORT:
        k = int(name.split("_")[1])
        pars = (
            k,
            "bands",
            json.loads((solver_target.DIR / f"{name}.json").read_text())["rows"],
        )
    else:
        pars = None if name == "teacher" or name in SPATIAL else parameters(name)
    heads = residual.load_variant(SPATIAL[name], device) if name in SPATIAL else []
    for zz, yy in zip(z.split(batch), y.split(batch)):
        zz, yy = zz.to(device), yy.to(device)
        if name == "teacher":
            img = teacher_path(model, zz, yy)[:, -1]
        elif heads:
            inputs = residual.buffer(model, zz, yy)
            img = torch.stack([head(inputs) for head in heads]).mean(0)
        else:
            img = sample_raw(model, zz, yy, *pars)
        out.append(pixels(img).cpu())
    return torch.cat(out)


@torch.inference_mode()
def main(names, device, batch):
    model = initialize(device)
    selection = json.loads((OUT / "selection.json").read_text())
    z, y = common_noise()
    refs, ref_stats, center, weight = extract_references()
    inception = (
        FeatureExtractorInceptionV3("inception-v3-compat", ["2048", "logits_unbiased"])
        .eval()
        .to(device)
    )
    root = OUT / "evaluation"
    root.mkdir(exist_ok=True)
    for name in names:
        folder = root / name
        folder.mkdir(exist_ok=True)
        cp = OUT / f"{name}.json"
        src = file_hash(cp) if cp.exists() else None
        if name in SPATIAL:
            src = {
                str(seed): file_hash(residual.DIRECTORY / f"{SPATIAL[name]}_{seed}.pt")
                for seed in residual.SEEDS
            }
            src["base"] = file_hash(OUT / f"{residual.PARENT}.json")
        if name in SHORT:
            src = file_hash(solver_target.DIR / f"{name}.json")
        config = dict(
            method=name,
            learned_coefficients_sha256=src,
            protocol_sha256=file_hash(OUT / "protocol.json"),
            noise_seed=109,
            n=10000,
            guidance=GUIDANCE,
            network_calls=(
                2 * TEACHER_STEPS if name == "teacher" else int(name.rsplit("_", 1)[1])
            ),
            selected=name in selection.values(),
            generation_batch=batch,
        )
        if name in SPATIAL:
            choice = json.loads((residual.DIRECTORY / "selection.json").read_text())[
                "method"
            ]
            config["selected"] = SPATIAL[name] == choice
            config["residual_protocol_sha256"] = file_hash(
                residual.DIRECTORY / "protocol.json"
            )
        if name in SHORT:
            config["target_protocol_sha256"] = file_hash(
                solver_target.DIR / "protocol.json"
            )
        if (folder / "config.json").exists() and json.loads(
            (folder / "config.json").read_text()
        ) != config:
            raise ValueError(f"Stale evaluation in {folder}")
        dump(folder / "config.json", config)
        if (folder / "metrics.json").is_file():
            continue
        if not (folder / "images.npy").is_file():
            tic = perf_counter()
            images = generate(model, z, y, name, device, batch)
            dump(folder / "timing.json", dict(seconds=perf_counter() - tic))
            atomic_array(folder / "images.npy", images.numpy())
        else:
            images = torch.from_numpy(np.load(folder / "images.npy"))
        if not all((folder / f"{f}.npy").is_file() for f in ("features", "logits")):
            logits = []
            bank = []
            for x in images.split(25):
                f, l = inception(x.to(device))
                bank.append(f.cpu())
                logits.append(l.cpu())
            f, l = torch.cat(bank), torch.cat(logits)
            atomic_array(folder / "features.npy", f.numpy())
            atomic_array(folder / "logits.npy", l.numpy())
        else:
            f, l = (
                torch.from_numpy(np.load(folder / f"{s}.npy"))
                for s in ("features", "logits")
            )
        stats = fid_features_to_statistics(f)
        fids = {
            f"fid_{s}": fid_statistics_to_metric(stats, v, False)[
                "frechet_inception_distance"
            ]
            for s, v in ref_stats.items()
        }
        kid = kid_features_to_metric(
            f,
            refs["test"],
            kid_subsets=50,
            kid_subset_size=1000,
            rng_seed=2020,
            verbose=False,
        )
        pred = ((f.numpy() - center) @ weight).argmax(1)
        table(
            folder / "per_sample.csv",
            dict(sample=np.arange(len(y)), condition=y.numpy(), probe_prediction=pred),
        )
        row = dict(
            **config,
            **fids,
            **kid,
            **isc_features_to_metric(l),
            agreement=float(np.mean(y.numpy() == pred)),
        )
        dump(folder / "metrics.json", row)
        print(name, row["fid_test"], row["agreement"], flush=True)
    table(
        root / "metrics.csv",
        [json.loads(p.read_text()) for p in sorted(root.glob("*/metrics.json"))],
    )


@torch.inference_mode()
def verify(device):
    model = initialize(device)
    z, y = common_noise()
    scores = pd.read_csv(OUT / "evaluation/metrics.csv").set_index("method")
    refs, stats, center, weight = extract_references()
    checks = {}
    for name in SCENARIOS:
        folder = OUT / "evaluation" / name
        cfg = json.loads((folder / "config.json").read_text())
        f = torch.from_numpy(np.load(folder / "features.npy"))
        stored = np.load(folder / "images.npy", mmap_mode="r")
        b = cfg["generation_batch"]
        fresh = generate(model, z[:b], y[:b], name, device, b).numpy()
        assert np.array_equal(stored[:b], fresh)
        case = pd.read_csv(folder / "per_sample.csv")
        assert np.array_equal(case.condition, y.numpy())
        assert np.array_equal(
            case.probe_prediction, ((f.numpy() - center) @ weight).argmax(1)
        )
        fid = fid_statistics_to_metric(
            fid_features_to_statistics(f), stats["test"], False
        )["frechet_inception_distance"]
        assert abs(fid - scores.loc[name, "fid_test"]) < 0.001
        assert np.isclose(
            (case.probe_prediction == case.condition).mean(),
            scores.loc[name, "agreement"],
        )
        checks[name] = dict(
            regenerated_images=b,
            pixel_exact=True,
            n=len(case),
            recomputed_fid=fid,
            sha256=file_hash(folder / "images.npy"),
        )
    dump(OUT / "verification.json", checks)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["evaluate", "verify", "sample"])
    p.add_argument("--method", choices=SCENARIOS, default="spatial_4")
    p.add_argument("--output", type=Path)
    p.add_argument("--class-id", type=int, choices=range(10), default=8)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--methods", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch", type=int, default=50)
    a = p.parse_args()
    with threadpool_limits(2):
        if a.action == "evaluate":
            main(a.methods, torch.device(a.device), a.batch)
        elif a.action == "sample":
            if a.output is None:
                p.error("Provide --output for single-image generation")
            device = torch.device(a.device)
            model = initialize(device)
            z, _ = noise(1, a.seed)
            img = generate(model, z, torch.tensor([a.class_id]), a.method, device, 1)[0]
            a.output.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(img.permute(1, 2, 0).numpy()).save(a.output)
        else:
            verify(torch.device(a.device))
