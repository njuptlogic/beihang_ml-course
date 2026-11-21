"""Full matched-reference evaluation for the conditional flow continuation branches."""
import argparse
from pathlib import Path
import json
from time import perf_counter
import numpy as np
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
from experiment4.research_train import ARMS, EPOCHS, ROOT, file_hash
from experiment4.model import FlowNet, sample
from experiment4.train import BASE, data

SCENARIOS = {f"{arm}_12": (arm, "euler", 12) for arm in ARMS}
SCENARIOS["combined_midpoint6"] = ("combined", "midpoint", 6)
for arm in ("control", "time", "align", "combined"):
    SCENARIOS[f"{arm}_strong"] = (arm, "euler", 12)
SCENARIOS["combined_projected"] = ("combined", "projected_euler", 12)
REFERENCE = BASE / "outputs/evaluation"
N = 10000


@torch.inference_mode()
def integrate(model, z, y, solver, steps, guide=1.6):
    if solver == "euler":
        return sample(model, z, y, steps, guide)

    def velocity(state, time):
        t = state.new_full((2 * len(state),), time)
        labels = torch.cat((y, torch.full_like(y, 10)))
        with torch.autocast(
            state.device.type, dtype=torch.bfloat16, enabled=state.is_cuda
        ):
            c, u = model(torch.cat((state, state)), t, labels).chunk(2)
            if solver == "midpoint":
                return (u + guide * (c - u)).float()
        c, u = c.float(), u.float()
        delta = c - u
        # In a straight flow x_clean = x_t + (1-t) v. The noise scale cancels
        # when the clean-space orthogonal guidance increment is mapped back to v.
        denoised = state + (1 - time) * c
        dims = (1, 2, 3)
        parallel = (delta * denoised).sum(dims, keepdim=True) / denoised.square().sum(
            dims, keepdim=True
        ).clamp_min(1e-8)
        return c + (guide - 1) * (delta - parallel * denoised)

    x = z.clone()
    for k in range(steps):
        v = velocity(x, k / steps)
        if solver == "midpoint":
            v = velocity(x + v / (2 * steps), (k + 0.5) / steps)
        x = x + v / steps
    return ((x.clamp(-1, 1) + 1) * 127.5).round().byte()


def metadata(root, name):
    arm, solver, steps = SCENARIOS[name]
    path = root / arm / "flow.pt"
    ck = torch.load(path, weights_only=True, map_location="cpu")
    if len(ck["history"]) != EPOCHS:
        raise ValueError(f"Complete {arm} before evaluating {name}")
    return ck, dict(
        scenario=name,
        arm=arm,
        solver=solver,
        steps=steps,
        evaluations_per_image=2 * steps * (2 if solver == "midpoint" else 1),
        samples=N,
        noise_seed=109,
        guide=3.0 if name.endswith(("_strong", "_projected")) else 1.6,
        epoch=ck["history"][-1]["total_epoch"],
        checkpoint_sha256=file_hash(path),
        inception="torch-fidelity 0.3.0 inception-v3-compat FP32",
    )


def common_noise():
    return (
        torch.randn(N, 3, 32, 32, generator=torch.Generator().manual_seed(109)),
        torch.arange(N) % 10,
    )


def extract_references():
    indices = np.load(REFERENCE / "reference_indices.npz")
    assert np.array_equal(indices["test"], np.arange(N))
    train_images, train_labels = data(True)
    del train_images
    test_images, test_labels = data(False)
    del test_images
    refs = {
        s: torch.from_numpy(np.load(REFERENCE / f"real_{s}_features.npy"))
        for s in ("train", "test")
    }
    assert all(x.shape == (N, 2048) for x in refs.values())
    f = refs["train"].numpy()
    center = f.mean(0)
    a = f - center
    label = train_labels.numpy()[indices["train"]]
    assert np.all(np.bincount(label) == 1000)
    target = np.eye(10, dtype=np.float32)[label] - 0.1
    mat = a.T @ a
    mat.flat[:: len(mat) + 1] += 100.0
    weight = np.linalg.solve(mat, a.T @ target)
    accuracy = ((refs["test"].numpy() - center) @ weight).argmax(
        1
    ) == test_labels.numpy()[indices["test"]]
    if abs(accuracy.mean() - 0.8875) > 1e-4:
        raise ValueError("Linear probe disagrees with the original evaluation")
    return (
        refs,
        {s: fid_features_to_statistics(x) for s, x in refs.items()},
        center,
        weight,
    )


@torch.inference_mode()
def evaluate(root, names, device):
    configure(42)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    with threadpool_limits(4):
        refs, stats, center, weight = extract_references()
    noise, labels = common_noise()
    model = FlowNet().eval().to(device, memory_format=torch.channels_last)
    inception = (
        FeatureExtractorInceptionV3("inception-v3-compat", ["2048", "logits_unbiased"])
        .eval()
        .to(device)
    )
    out = root / "evaluation"
    out.mkdir(exist_ok=True)
    for name in names:
        ck, config = metadata(root, name)
        folder = out / name
        folder.mkdir(exist_ok=True)
        manifest = folder / "config.json"
        if manifest.exists() and json.loads(manifest.read_text()) != config:
            raise ValueError(f"Stale results in {folder}. Use a new run directory.")
        dump(manifest, config)
        if (folder / "metrics.json").exists():
            continue
        model.load_state_dict(ck["ema"])
        del ck
        if not (folder / "images.npy").exists():
            start = perf_counter()
            im = torch.cat(
                [
                    integrate(
                        model,
                        z.to(device),
                        y.to(device),
                        config["solver"],
                        config["steps"],
                        config["guide"],
                    ).cpu()
                    for z, y in zip(noise.split(100), labels.split(100))
                ]
            )
            elapsed = perf_counter() - start
            dump(folder / "timing.json", dict(generation_seconds=elapsed))
            atomic_array(folder / "images.npy", im.numpy())
        else:
            im = torch.from_numpy(np.load(folder / "images.npy"))
        print("Extracting features", name, flush=True)
        if (
            not (folder / "logits.npy").exists()
            or not (folder / "features.npy").exists()
        ):
            feat, logits = features(inception, im, device)
            atomic_array(folder / "features.npy", feat.numpy())
            atomic_array(folder / "logits.npy", logits.numpy())
        else:
            feat = torch.from_numpy(np.load(folder / "features.npy"))
            logits = torch.from_numpy(np.load(folder / "logits.npy"))
        with threadpool_limits(4):
            local_stats = fid_features_to_statistics(feat)
            fids = {
                s: fid_statistics_to_metric(local_stats, stats[s], False)[
                    "frechet_inception_distance"
                ]
                for s in stats
            }
            kid = kid_features_to_metric(
                feat,
                refs["test"],
                kid_subsets=50,
                kid_subset_size=1000,
                rng_seed=2020,
                verbose=False,
            )
            probe = (feat.numpy() - center) @ weight
        pred = probe.argmax(1)
        label = labels.numpy()
        other = probe.copy()
        other[np.arange(N), label] = -np.inf
        diagnostics = dict(
            sample=np.arange(N),
            condition=label,
            probe_prediction=pred,
            score_margin=probe[np.arange(N), label] - other.max(1),
        )
        table(folder / "per_sample.csv", diagnostics)
        row = {
            **config,
            **json.loads((folder / "timing.json").read_text()),
            "fid_train": float(fids["train"]),
            "fid_test": float(fids["test"]),
            **isc_features_to_metric(logits),
            **kid,
            "agreement": float((pred == label).mean()),
        }
        dump(folder / "metrics.json", row)
        print("Evaluated", name, row["fid_test"], row["agreement"], flush=True)
    table(
        out / "metrics.csv",
        [json.loads(p.read_text()) for p in sorted(out.glob("*/metrics.json"))],
    )


def atomic_array(path, arr):
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, arr)
    tmp.replace(path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    p.add_argument("--out", type=Path, default=ROOT)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    evaluate(a.out, a.scenarios, torch.device(a.device))
