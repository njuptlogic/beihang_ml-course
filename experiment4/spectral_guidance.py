"""Test radial spectral correction on the train-selected eight-branch CIFAR generator."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from threadpoolctl import threadpool_limits
from torch_fidelity.metric_fid import (
    fid_features_to_statistics,
    fid_statistics_to_metric,
)
from torch_fidelity.metric_isc import isc_features_to_metric
from torch_fidelity.metric_kid import kid_features_to_metric
from course.common import configure, dump, table
from experiment4.evaluate import features
from experiment4.guidance_eval import (
    ROOT,
    load_model,
    sample,
    training_reference,
    projected_fd,
    inception,
    sources,
)
from experiment4.research_train import file_hash
from experiment4.research_evaluate import atomic_array, common_noise, extract_references
from experiment4.train import data
from experiment4.learned_solver import noise
from course.figures import setup, page, axes_title, save, picture, BLUE, TEAL, AMBER
from matplotlib import pyplot as plt

OUT = ROOT.parent / "spectral_guidance"
STRENGTHS = (0, 0.05, 0.15, 0.35, 0.65)
BINS = 12
VAL_SEED = 91724
VAL_N = 1600


def original():
    locked = json.loads((ROOT / "selection/locked.json").read_text())["nfe8_quality"]
    return json.loads((ROOT / "selection" / (locked + ".json")).read_text())


def indices(device):
    f = torch.fft.fftfreq(32, device=device)
    rad = torch.sqrt(f[:, None].square() + f[None, :].square())
    bins = torch.clamp(torch.ceil(rad / (0.5**0.5) * BINS).long(), 1, BINS) - 1
    mask = torch.nn.functional.one_hot(bins, num_classes=BINS).permute(2, 0, 1).float()
    mask[:, 0, 0] = 0
    return bins, mask.flatten(1)


def power(spectrum, mask):
    energy = spectrum.abs().square().mean(1).flatten(1)
    return energy @ mask.T / mask.sum(1)


def reference(device):
    bins, mask = indices(device)
    path = OUT / "class_power.npy"
    if not path.exists():
        imgs, labels = data(True)
        sums = torch.zeros(10, BINS, device=device, dtype=torch.float64)
        for x, y in zip(imgs.split(200), labels.split(200)):
            f = torch.fft.fft2(
                x.to(device, dtype=torch.float32) / 127.5 - 1, norm="ortho"
            )
            sums.index_add_(0, y.to(device), power(f, mask).double())
        atomic_array(
            path,
            (sums / torch.bincount(labels).to(device)[:, None]).float().cpu().numpy(),
        )
    target = torch.from_numpy(np.load(path)).to(device)
    return bins, mask, target


def transform(bins, mask, ref, y, strength, kind):
    rgb = ref[y] if kind == "class" else ref.mean(0, keepdim=True)

    def apply(x, t):
        f = torch.fft.fft2(x, norm="ortho")
        observed = power(f, mask)
        # For x_t = (1-t)z + t*x_data and an orthonormal FFT, white-noise power is one.
        goal = t * t * rgb + (1 - t) ** 2
        gain = ((goal + 1e-6) / (observed + 1e-6)).pow(0.5 * strength)
        scale = gain[:, bins]
        scale[:, 0, 0] = 1
        return torch.fft.ifft2(f * scale[:, None], norm="ortho").real.contiguous()

    return apply


@torch.inference_mode()
def generate(model, base, z, y, device, strength, kind, ref):
    imgs = []
    for zz, yy in zip(z.split(64), y.split(64)):
        yy = yy.to(device)
        fn = transform(*ref, yy, strength, kind) if strength > 0 else None
        imgs.append(
            sample(
                model,
                zz.to(device),
                yy,
                base["steps"],
                base["cfg"],
                base["alpha"],
                step_transform=fn,
            ).cpu()
        )
    return torch.cat(imgs)


def init(device):
    configure(4300)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    OUT.mkdir(exist_ok=True)
    base = original()
    assert base["checkpoint_sha256"] == file_hash(sources(base["arm"]))
    contract = dict(
        base=base["scenario"],
        checkpoint_sha256=base["checkpoint_sha256"],
        bins=BINS,
        strengths=list(STRENGTHS),
        reference="50k train only, class or pooled radial power, preserve DC",
        validation_seed=VAL_SEED,
        n_validation=VAL_N,
        criteria="minimum training-reference projected FD (192 dimensions), ties prefer smaller strength",
        test_arms="locked overall plus best nonzero candidate if zero wins; fixed starting point is the archived uncorrected method",
        inspiration="Kim & Kim, Correcting Guided Diffusion Trajectories with Spectral Alignment, arXiv:2610.02753, 2026-10-02",
    )
    p = OUT / "protocol.json"
    if p.exists() and json.loads(p.read_text()) != contract:
        raise ValueError("Unexpected protocol change")
    dump(p, contract)
    return base, load_model(base["arm"], device), reference(device)


def score_train(device):
    base, model, ref = init(device)
    center, Q, var, probe = training_reference()
    z, y = noise(VAL_N, VAL_SEED)
    feature_net = inception(device)
    rows = []
    for strength in STRENGTHS:
        for kind in ("global", "class") if strength else ("global",):
            name = f"{kind}_{strength:g}"
            artifact = OUT / (name + ".json")
            if artifact.exists():
                rows.append(json.loads(artifact.read_text()))
                continue
            imgs = generate(model, base, z, y, device, strength, kind, ref)
            f, _ = features(feature_net, imgs, device)
            f = f.numpy() - center
            spec = power(
                torch.fft.fft2(imgs.float() / 127.5 - 1, norm="ortho"), ref[1].cpu()
            )
            deviation = (
                (spec.reshape(-1, 10, BINS).mean(0).log() - ref[2].cpu().log())
                .square()
                .mean()
                .item()
            )
            row = dict(
                name=name,
                strength=strength,
                kind=kind,
                train_fd=projected_fd(f @ Q, var),
                agreement=float(np.mean((f @ probe).argmax(1) == y.numpy())),
                log_power_mse=deviation,
            )
            dump(artifact, row)
            rows.append(row)
            print(row, flush=True)
    table(OUT / "sweep.csv", rows)
    chosen = min(rows, key=lambda r: (r["train_fd"], r["strength"]))
    other = min(
        (r for r in rows if r["strength"] > 0),
        key=lambda r: (r["train_fd"], r["strength"]),
    )
    dump(OUT / "selection.json", dict(selected=chosen, best_corrected=other))


def score_test(device):
    base, model, ref = init(device)
    selected = json.loads((OUT / "selection.json").read_text())
    method = selected["best_corrected"]
    z, y = common_noise()
    refdir = ROOT / "test" / base["scenario"]
    refs, stats, center, weight = extract_references()
    extractor = inception(device)
    path = OUT / "test"
    path.mkdir(exist_ok=True)
    imgs = generate(model, base, z, y, device, method["strength"], method["kind"], ref)
    atomic_array(path / "images.npy", imgs.numpy())
    f, l = features(extractor, imgs, device)
    atomic_array(path / "features.npy", f.numpy())
    atomic_array(path / "logits.npy", l.numpy())
    fstat = fid_features_to_statistics(f)
    fids = {
        f"fid_{s}": fid_statistics_to_metric(fstat, v, False)[
            "frechet_inception_distance"
        ]
        for s, v in stats.items()
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
    oldcase = pd.read_csv(refdir / "per_sample.csv")
    assert np.array_equal(y.numpy(), oldcase.condition)
    table(
        path / "per_sample.csv",
        dict(
            sample=np.arange(len(y)),
            condition=y.numpy(),
            probe_prediction=pred,
            base_prediction=oldcase.probe_prediction,
        ),
    )
    raw = generate(model, base, z[:64], y[:64], device, 0, "global", ref).numpy()
    assert np.array_equal(raw, np.load(refdir / "images.npy", mmap_mode="r")[:64])
    raw = generate(
        model, base, z[:64], y[:64], device, method["strength"], method["kind"], ref
    ).numpy()
    assert np.array_equal(raw, imgs[:64].numpy())
    row = dict(
        name=method["name"],
        kind=method["kind"],
        strength=method["strength"],
        calibration_probe_agreement=method["agreement"],
        train_fd=method["train_fd"],
        base=base["scenario"],
        **fids,
        **kid,
        **isc_features_to_metric(l),
        agreement=float(np.mean(pred == y.numpy())),
        n=10000,
        budget=8,
        noise_seed=109,
        checkpoint_sha256=base["checkpoint_sha256"],
        base_pixel_exact=64,
        corrected_pixel_exact=64,
        selected=method["name"] == selected["selected"]["name"],
    )
    dump(path / "metrics.json", row)
    basem = json.loads((refdir / "metrics.json").read_text())
    table(
        OUT / "paired_metrics.csv",
        [
            dict(
                method=name,
                **{
                    k: r[k]
                    for k in (
                        "fid_test",
                        "inception_score_mean",
                        "agreement",
                        "kernel_inception_distance_mean",
                    )
                },
            )
            for name, r in [("uncorrected", basem), (method["name"], row)]
        ],
    )
    print(row, flush=True)


def draw():
    setup()
    data = pd.read_csv(OUT / "sweep.csv")
    fig, ax = page(1, 2, size=(9.6, 3.6))
    for kind, c, m in [("global", BLUE, "o"), ("class", TEAL, "D")]:
        s = data[(data.kind == kind) | (data.strength == 0)].sort_values("strength")
        for j, target in enumerate(["train_fd", "log_power_mse"]):
            ax[j].plot(
                s.strength, s[target], color=c, marker=m, label=kind.capitalize()
            )
    axes_title(
        ax[0],
        "A   Selection on training references",
        "Correction strength",
        "Projected feature distance",
    )
    axes_title(
        ax[1],
        "B   Radial spectra on calibration noise",
        "Correction strength",
        "Log-power MSE vs. training classes",
    )
    ax[0].legend()
    save(fig, 4, "spectral_guidance")
    radial_profiles()


def radial_profiles():
    _, mask = indices("cpu")
    target = np.load(OUT / "class_power.npy").mean(0)
    selected = json.loads((OUT / "test/metrics.json").read_text())
    folders = [ROOT / "test" / selected["base"], OUT / "test"]
    labels = ["Without correction", "Spectral correction"]
    records = []
    gallery = []
    fig = plt.figure(figsize=(11.2, 4.0), layout="constrained")
    grids = fig.add_gridspec(2, 4, width_ratios=[3.0, 1, 1, 1])
    ax = fig.add_subplot(grids[:, 0])
    radius = (np.arange(BINS) + 0.5) * np.sqrt(0.5) / BINS
    ax.plot(radius, target, "--", color=BLUE, label="Real training images (50,000)")
    for i, (label, folder, color) in enumerate(zip(labels, folders, [AMBER, TEAL])):
        images = np.load(folder / "images.npy", mmap_mode="r")
        totals = np.zeros(BINS, dtype=np.float64)
        for start in range(0, len(images), 200):
            x = torch.tensor(images[start : start + 200]).float() / 127.5 - 1
            totals += (
                power(torch.fft.fft2(x, norm="ortho"), mask).double().sum(0).numpy()
            )
        observed = totals / len(images)
        ax.plot(
            radius,
            observed,
            marker=["o", "D"][i],
            markersize=3,
            color=color,
            label=label + " (10,000)",
        )
        records.extend(
            dict(series=label, bin=k + 1, radius=r, mean_power=p, n=len(images))
            for k, (r, p) in enumerate(zip(radius, observed))
        )
        for col, (index, name) in enumerate([(2, "Bird"), (4, "Deer"), (8, "Ship")]):
            panel = fig.add_subplot(grids[i, col + 1])
            picture(panel, images[index].transpose(1, 2, 0), interpolation="nearest")
            if i == 0:
                panel.set_title(name, fontsize=10)
            if col == 0:
                panel.set_ylabel(["Original", "Corrected"][i])
            gallery.append(
                dict(
                    series=label,
                    sample=index,
                    condition=index % 10,
                    class_name=name,
                    noise_seed=selected["noise_seed"],
                )
            )
    ax.set_yscale("log")
    axes_title(
        ax,
        "Output power by spatial frequency",
        "Radial frequency (cycles / pixel)",
        "Mean RGB power per Fourier coefficient",
    )
    ax.legend(loc="upper right", fontsize=8)
    records.extend(
        dict(series="Real training images", bin=k + 1, radius=r, mean_power=p, n=50000)
        for k, (r, p) in enumerate(zip(radius, target))
    )
    table(OUT / "profiles.csv", records)
    table(OUT / "profiles_gallery.csv", gallery)
    save(fig, 4, "spectral_profiles")


def repeat(device):
    """Use a new generated sample of 10,000 without changing the weights or selection."""
    base, model, ref = init(device)
    selected = json.loads((OUT / "selection.json").read_text())["selected"]
    path = OUT / "repeat"
    path.mkdir(exist_ok=True)
    seed = 27619
    z, y = noise(10000, seed)
    extractor = inception(device)
    refs, stats, center, weight = extract_references()
    lock = dict(
        seed=seed,
        n=10000,
        selection_sha256=file_hash(OUT / "selection.json"),
        base=base["scenario"],
        cohort="same complete 10k real test set; fresh balanced synthetic noise",
    )
    dump(path / "protocol.json", lock)
    results = []
    frame = dict(sample=np.arange(len(y)), condition=y.numpy())
    for name, strength, kind in [
        ("uncorrected", 0, "global"),
        ("selected", selected["strength"], selected["kind"]),
    ]:
        folder = path / name
        folder.mkdir(exist_ok=True)
        if (
            not (folder / "features.npy").is_file()
            or not (folder / "logits.npy").is_file()
        ):
            imgs = generate(model, base, z, y, device, strength, kind, ref)
            f, l = features(extractor, imgs, device)
            atomic_array(folder / "images.npy", imgs.numpy())
            atomic_array(folder / "features.npy", f.numpy())
            atomic_array(folder / "logits.npy", l.numpy())
        else:
            f, l = [
                torch.from_numpy(np.load(folder / f"{s}.npy"))
                for s in ("features", "logits")
            ]
        frame[name] = ((f.numpy() - center) @ weight).argmax(1)
        fid = fid_statistics_to_metric(
            fid_features_to_statistics(f), stats["test"], False
        )["frechet_inception_distance"]
        kid = kid_features_to_metric(
            f,
            refs["test"],
            kid_subsets=50,
            kid_subset_size=1000,
            rng_seed=2020,
            verbose=False,
        )
        row = dict(
            method=name,
            n=10000,
            noise_seed=seed,
            fid_test=fid,
            agreement=float((frame[name] == y.numpy()).mean()),
            **kid,
            **isc_features_to_metric(l),
        )
        results.append(row)
        print(row, flush=True)
    table(path / "per_sample.csv", frame)
    table(path / "metrics.csv", results)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["select", "test", "figure", "repeat", "sample"])
    p.add_argument("--output", type=Path)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--class-id", type=int, choices=range(10), default=8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    with threadpool_limits(2):
        if a.action == "select":
            score_train(torch.device(a.device))
        elif a.action == "test":
            score_test(torch.device(a.device))
        elif a.action == "repeat":
            repeat(torch.device(a.device))
        elif a.action == "sample":
            if a.output is None:
                p.error("Provide --output for a single sample")
            device = torch.device(a.device)
            base, net, ref = init(device)
            selected = json.loads((OUT / "selection.json").read_text())["selected"]
            z, _ = noise(1, a.seed)
            img = generate(
                net,
                base,
                z,
                torch.tensor([a.class_id]),
                device,
                selected["strength"],
                selected["kind"],
                ref,
            )[0]
            a.output.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(img.permute(1, 2, 0).numpy()).save(a.output)
        else:
            draw()
