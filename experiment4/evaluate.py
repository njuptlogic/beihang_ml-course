"""Paired sampling experiment and standard torch-fidelity Inception metrics."""
import argparse
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from torch.nn import functional as F
from threadpoolctl import threadpool_limits
from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
from torch_fidelity.metric_fid import (
    fid_features_to_statistics,
    fid_statistics_to_metric,
)
from torch_fidelity.metric_isc import isc_features_to_metric
from torch_fidelity.metric_kid import kid_features_to_metric
from course.common import configure, dump, table
from experiment4.train import BASE, data
from experiment4.model import make_model, sample

CONFIGS = {
    "gaussian": None,
    "flow_main": ("ema", 12, 1.6, (0.0, 1.0)),
    "four_steps": ("ema", 4, 1.6, (0.0, 1.0)),
    "no_guidance": ("ema", 12, 1.0, (0.0, 1.0)),
    "interval": ("ema", 12, 1.6, (0.25, 0.75)),
    "no_ema": ("model", 12, 1.6, (0.0, 1.0)),
}


def gaussian_fit_sample(x, y, labels, gen):
    # Per-class full covariance on the 8x8 RGB grid, fitted to all training images.
    small = F.avg_pool2d(x.float(), 4).flatten(1)
    generated = torch.empty(len(labels), 3, 32, 32)
    for c in range(10):
        cls = small[y == c]
        mean = cls.mean(0)
        cov = torch.cov(cls.T)
        root = torch.linalg.cholesky(cov + 1e-3 * torch.eye(len(mean)))
        count = (labels == c).sum().item()
        v = torch.randn(count, 192, generator=gen) @ root.T + mean
        generated[labels == c] = F.interpolate(
            v.reshape(-1, 3, 8, 8), scale_factor=4, mode="bilinear", align_corners=False
        )
    return generated.round().clamp(0, 255).byte()


@torch.inference_mode()
def features(model, images, device):
    pooled = []
    logits = []
    for batch in images.split(40):
        f, l = model(batch.to(device))
        pooled.append(f.cpu())
        logits.append(l.cpu())
    return torch.cat(pooled), torch.cat(logits)


def main(out, n, device, methods):
    if n < 10 or n % 10 or n > 10000:
        raise ValueError("Choose a positive multiple of ten up to 10,000.")
    configure(42)
    # Metric extraction uses full FP32, including matrix multiplications.
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    ck = torch.load(out / "flow.pt", weights_only=True, map_location="cpu")
    if len(ck["history"]) != ck["config"]["epochs"]:
        raise ValueError("Training has not reached the specified final epoch.")
    model = (
        make_model(ck["config"].get("parameterization", "direct"))
        .to(device, memory_format=torch.channels_last)
        .eval()
    )
    extractor = (
        FeatureExtractorInceptionV3("inception-v3-compat", ["2048", "logits_unbiased"])
        .eval()
        .to(device)
    )
    x, y = data(True)
    test, yt = data(False)
    rng = np.random.default_rng(103)

    def strat(labels):
        return np.sort(
            np.concatenate(
                [
                    rng.choice(np.where(labels.numpy() == k)[0], n // 10, replace=False)
                    for k in range(10)
                ]
            )
        )

    train_idx, test_idx = strat(y), strat(yt)
    evaldir = out / "evaluation"
    evaldir.mkdir(exist_ok=True)
    np.savez(evaldir / "reference_indices.npz", train=train_idx, test=test_idx)
    ref = {}
    for name, im in (("train", x[train_idx]), ("test", test[test_idx])):
        feat, logits = features(extractor, im, device)
        ref[name] = (feat, fid_features_to_statistics(feat))
        np.save(evaldir / f"real_{name}_features.npy", feat.numpy())
        if name == "test":
            true_is = isc_features_to_metric(logits)
    # Train-vs-test at the same sample count supplies a finite-sample reference.
    with threadpool_limits(5):
        real_fid = fid_statistics_to_metric(ref["train"][1], ref["test"][1], False)
    dump(
        evaldir / "reference.json",
        {**true_is, **real_fid, "n": n, "noise_seed": 109, "reference_seed": 103},
    )
    labels = torch.arange(n) % 10
    gen = torch.Generator().manual_seed(109)
    noise = torch.randn(n, 3, 32, 32, generator=gen)
    rows = []
    for name in methods:
        cfg = CONFIGS[name]
        start = perf_counter()
        if cfg is None:
            imgs = gaussian_fit_sample(x, y, labels, torch.Generator().manual_seed(109))
            steps = 0
            guidance = 0
            nfe = 0
        else:
            which, steps, guidance, interval = cfg
            model.load_state_dict(ck[which])
            imgs = torch.cat(
                [
                    sample(
                        model, z.to(device), c.to(device), steps, guidance, interval
                    ).cpu()
                    for z, c in zip(noise.split(100), labels.split(100))
                ]
            )
            nfe = steps + sum(
                interval[0] <= i / steps < interval[1] and guidance != 1
                for i in range(steps)
            )
        seconds = perf_counter() - start
        print("Scoring", name, len(imgs), flush=True)
        np.save(evaldir / f"{name}_images.npy", imgs.numpy())
        feat, logits = features(extractor, imgs, device)
        # Keep the primary feature bank for the independent reference-neighbour analysis.
        if name == "flow_main":
            np.save(evaldir / "flow_main_features.npy", feat.numpy())
        stat = fid_features_to_statistics(feat)
        isc = isc_features_to_metric(logits)
        with threadpool_limits(5):
            fid_train = fid_statistics_to_metric(stat, ref["train"][1], False)
            fid_test = fid_statistics_to_metric(stat, ref["test"][1], False)
            kid = kid_features_to_metric(
                feat,
                ref["test"][0],
                kid_subsets=50,
                kid_subset_size=min(1000, n),
                rng_seed=2020,
                verbose=False,
            )
        row = dict(
            method=name,
            n=n,
            steps=steps,
            guidance=guidance,
            evaluations_per_image=nfe,
            generation_seconds=seconds,
            fid_train=fid_train["frechet_inception_distance"],
            fid_test=fid_test["frechet_inception_distance"],
            **isc,
            **kid,
        )
        rows.append(row)
        table(evaldir / "metrics.csv", rows)
        print(row, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=BASE / "outputs")
    p.add_argument("--samples", type=int, default=10000)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--methods", nargs="+", choices=CONFIGS, default=list(CONFIGS))
    a = p.parse_args()
    main(a.output, a.samples, a.device, a.methods)
