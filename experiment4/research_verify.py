"""Reconstruct each continuation sampler and check saved metrics against its feature bank."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from PIL import Image
from threadpoolctl import threadpool_limits
import torch
from torch_fidelity.metric_fid import (
    fid_features_to_statistics,
    fid_statistics_to_metric,
)
from torch_fidelity.metric_kid import kid_features_to_metric
from torch_fidelity.metric_isc import isc_features_to_metric
from course.common import configure, dump
from experiment4.model import FlowNet
from experiment4.research_evaluate import (
    SCENARIOS,
    common_noise,
    integrate,
    metadata,
    extract_references,
)
from experiment4.teacher_features import ROOT


@torch.inference_mode()
def verify(root, device, names):
    configure(42)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    model = FlowNet().eval().to(device, memory_format=torch.channels_last)
    noise, labels = common_noise()
    with threadpool_limits(4):
        refs, reference_stats, center, weight = extract_references()
    path = root / "verification.json"
    report = json.loads(path.read_text()) if path.exists() else {}
    for name in names:
        folder = root / "evaluation" / name
        ck, config = metadata(root, name)
        saved = json.loads((folder / "metrics.json").read_text())
        assert saved["checkpoint_sha256"] == config["checkpoint_sha256"]
        model.load_state_dict(ck["ema"])
        fresh = (
            integrate(
                model,
                noise[:100].to(device),
                labels[:100].to(device),
                config["solver"],
                config["steps"],
                config["guide"],
            )
            .cpu()
            .numpy()
        )
        arr = np.load(folder / "images.npy", mmap_mode="r")
        delta = np.abs(fresh.astype(np.int16) - arr[:100].astype(np.int16))
        assert arr.shape == (10000, 3, 32, 32) and arr.dtype == np.uint8
        assert delta.mean() < 0.05 and delta.max() <= 3
        bank = np.load(folder / "features.npy")
        assert bank.shape == (10000, 2048) and np.isfinite(bank).all()
        frame = pd.read_csv(folder / "per_sample.csv")
        with threadpool_limits(4):
            pred = ((bank - center) @ weight).argmax(1)
            features = torch.from_numpy(bank)
            fid = fid_statistics_to_metric(
                fid_features_to_statistics(features), reference_stats["test"], False
            )["frechet_inception_distance"]
            kid = kid_features_to_metric(
                features,
                refs["test"],
                kid_subsets=50,
                kid_subset_size=1000,
                rng_seed=2020,
                verbose=False,
            )["kernel_inception_distance_mean"]
        assert (
            abs(fid - saved["fid_test"]) < 1e-4
            and abs(kid - saved["kernel_inception_distance_mean"]) < 1e-6
        )
        assert np.array_equal(frame.probe_prediction, pred)
        assert np.array_equal(frame.condition, labels.numpy())
        agreement = float((pred == labels.numpy()).mean())
        assert abs(agreement - saved["agreement"]) < 1e-8
        isc = isc_features_to_metric(torch.from_numpy(np.load(folder / "logits.npy")))[
            "inception_score_mean"
        ]
        assert abs(isc - saved["inception_score_mean"]) < 1e-8
        report[name] = dict(
            n=len(arr),
            checkpoint_sha256=config["checkpoint_sha256"],
            resampled=100,
            mean_abs_pixel_error=float(delta.mean()),
            max_abs_pixel_error=int(delta.max()),
            recomputed_is=isc,
            recomputed_agreement=agreement,
            recomputed_test_fid=fid,
            recomputed_test_kid=kid,
        )
        dump(path, report)
        print("Verified", name, flush=True)
    print(
        "Verified",
        len(report),
        "checkpoints/samplers against 10,000 saved predictions and Inception logits each.",
    )


def generate(root, device, name, category, seed, destination):
    configure(42)
    model = FlowNet().eval().to(device, memory_format=torch.channels_last)
    ck, config = metadata(root, name)
    model.load_state_dict(ck["ema"])
    noise = torch.randn(1, 3, 32, 32, generator=torch.Generator().manual_seed(seed)).to(
        device
    )
    image = integrate(
        model,
        noise,
        torch.tensor([category], device=device),
        config["solver"],
        config["steps"],
        config["guide"],
    )[0]
    destination.parent.mkdir(exist_ok=True, parents=True)
    Image.fromarray(image.permute(1, 2, 0).cpu().numpy()).save(destination)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    p.add_argument("--out", type=Path, default=ROOT)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--generate",
        type=Path,
        help="Instead of a full audit, generate one fresh PNG at this path.",
    )
    p.add_argument("--scenario", choices=SCENARIOS, default="combined_12")
    p.add_argument("--seed", type=int, default=810)
    p.add_argument("--class-id", type=int, choices=range(10), default=8)
    a = p.parse_args()
    if a.generate:
        generate(
            a.out, torch.device(a.device), a.scenario, a.class_id, a.seed, a.generate
        )
    else:
        verify(a.out, torch.device(a.device), a.scenarios)
