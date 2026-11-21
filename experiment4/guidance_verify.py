"""Reconstruct conditional-flow samples and recompute metrics from the full feature banks."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from threadpoolctl import threadpool_limits
from torch_fidelity.metric_isc import isc_features_to_metric
from course.research_results import covariance_root, audit_fid

from course.common import configure, dump
from experiment4.guidance_eval import ROOT, load_model, sample, sources
from experiment4.research_evaluate import common_noise, extract_references
from experiment4.research_train import file_hash


def row_for(name):
    selection = ROOT / "selection"
    locked = json.loads((selection / "locked.json").read_text())
    name = locked.get(name, name)
    return json.loads((ROOT / "test" / name / "metrics.json").read_text())


def unbiased_kid(fake, real, device):
    rng = np.random.RandomState(2020)
    dim = fake.shape[1]
    m = 1000
    samples = []
    for _ in range(50):
        a = fake[rng.choice(len(fake), m, replace=False)].to(device)
        b = real[rng.choice(len(real), m, replace=False)].to(device)
        aa = (1 + (a @ a.T) / dim) ** 3
        bb = (1 + (b @ b.T) / dim) ** 3
        ab = (1 + (a @ b.T) / dim) ** 3
        term = (
            aa.sum(dtype=torch.float64)
            - aa.diag().sum(dtype=torch.float64)
            + bb.sum(dtype=torch.float64)
            - bb.diag().sum(dtype=torch.float64)
        ) / (m * (m - 1))
        samples.append(term - 2 * ab.mean(dtype=torch.float64))
    return float(torch.stack(samples).mean().cpu())


def verify(device, selected=None):
    z, y = common_noise()
    refs, _, center, weight = extract_references()
    ref = covariance_root(refs["test"].numpy())
    destination = ROOT / "verification.json"
    output = (
        json.loads(destination.read_text()) if selected and destination.exists() else {}
    )
    for folder in sorted((ROOT / "test").iterdir()):
        if not folder.is_dir() or (selected and folder.name not in selected):
            continue
        r = row_for(folder.name)
        model = load_model(r["arm"], device)
        assert r["checkpoint_sha256"] == file_hash(sources(r["arm"]))
        fresh = (
            sample(
                model,
                z[:64].to(device),
                y[:64].to(device),
                r["steps"],
                r["cfg"],
                r["alpha"],
            )
            .cpu()
            .numpy()
        )
        images = np.load(folder / "images.npy", mmap_mode="r")
        assert images.shape == (10000, 3, 32, 32) and images.dtype == np.uint8
        assert np.array_equal(images[:64], fresh)
        f = torch.from_numpy(np.load(folder / "features.npy"))
        assert f.shape == (10000, 2048) and torch.isfinite(f).all()
        scores = (f.numpy() - center) @ weight
        observed = pd.read_csv(folder / "per_sample.csv")
        assert np.array_equal(observed["sample"], np.arange(10000))
        assert np.array_equal(observed.condition, y.numpy())
        assert np.array_equal(observed.probe_prediction, scores.argmax(1))
        agreement = np.mean(scores.argmax(1) == y.numpy())
        fid = audit_fid(f.numpy(), ref)
        kid = unbiased_kid(f, refs["test"], device)
        inc = isc_features_to_metric(torch.from_numpy(np.load(folder / "logits.npy")))[
            "inception_score_mean"
        ]
        actual = np.array([fid, kid, inc, agreement])
        expected = np.array(
            [
                r["fid_test"],
                r["kernel_inception_distance_mean"],
                r["inception_score_mean"],
                r["agreement"],
            ]
        )
        np.testing.assert_allclose(actual[1:], expected[1:], atol=2e-6, rtol=1e-7)
        assert abs(actual[0] - expected[0]) < 0.005
        output[folder.name] = dict(
            images=10000,
            regenerated_identical=64,
            fd=fid,
            is_mean=inc,
            kid=kid,
            agreement=agreement,
        )
        print("Verified", folder.name, flush=True)
    dump(ROOT / "verification.json", output)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--generate",
        type=Path,
        help="Save a new 32×32 image instead of running the audit.",
    )
    p.add_argument("--verify-methods", nargs="+")
    p.add_argument("--method", default="nfe24_quality")
    p.add_argument("--class-id", type=int, choices=range(10), default=8)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    configure(42)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    d = torch.device(args.device)
    with threadpool_limits(3):
        if args.generate:
            row = row_for(args.method)
            model = load_model(row["arm"], d)
            noise = torch.randn(
                1, 3, 32, 32, generator=torch.Generator().manual_seed(args.seed)
            ).to(d)
            im = sample(
                model,
                noise,
                torch.tensor([args.class_id], device=d),
                row["steps"],
                row["cfg"],
                row["alpha"],
            )
            args.generate.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(im[0].permute(1, 2, 0).cpu().numpy()).save(args.generate)
        else:
            verify(d, args.verify_methods)


if __name__ == "__main__":
    main()
