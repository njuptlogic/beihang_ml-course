"""Disentangle marginal image fidelity from class-conditional coverage in a second feature space."""
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from threadpoolctl import threadpool_limits
from torchvision.datasets import CIFAR10

from course.common import configure, dump, table
from experiment3.semantic_features import COMMIT, WEIGHT_SHA256, load_encoder
from experiment4.train import BASE

OUT = BASE / "outputs/research/semantic_evaluation"
REFERENCE = BASE / "outputs/evaluation"
NEW = BASE / "outputs/research/evaluation"
SOURCES = {
    "test": None,
    "train": None,
    "old": REFERENCE / "flow_main_images.npy",
    "control": NEW / "control_12/images.npy",
    "time": NEW / "time_12/images.npy",
    "align": NEW / "align_12/images.npy",
    "combined": NEW / "combined_12/images.npy",
    "strong": NEW / "combined_strong/images.npy",
    "control_strong": NEW / "control_strong/images.npy",
    "time_strong": NEW / "time_strong/images.npy",
    "align_strong": NEW / "align_strong/images.npy",
    "extended": NEW / "extended_12/images.npy",
}
DISPLAY = {
    "train": "Real vs. real",
    "old": "Original",
    "control": "Continuation",
    "time": "Time sampling",
    "align": "Alignment",
    "combined": "Time + alignment",
    "strong": "Stronger guidance",
    "extended": "Longer training",
    "permuted": "Permuted labels",
}
CLASSES = [
    "Airplane",
    "Car",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck",
]
NEIGHBOURS = (3, 5, 10)


def images_and_labels(name):
    if name in {"train", "test"}:
        indices = np.load(REFERENCE / "reference_indices.npz")[name]
        dataset = CIFAR10(str(BASE / "data"), train=name == "train")
        images = dataset.data[indices].transpose(0, 3, 1, 2).copy()
        return images, np.asarray(dataset.targets)[indices]
    arr = np.load(SOURCES[name], mmap_mode="r")
    if arr.shape != (10000, 3, 32, 32) or arr.dtype != np.uint8:
        raise ValueError(
            f"Unexpected image collection for {name}: {arr.shape}, {arr.dtype}"
        )
    rows_path = (
        REFERENCE / "conditional_diagnostics.csv"
        if name == "old"
        else SOURCES[name].parent / "per_sample.csv"
    )
    rows = pd.read_csv(rows_path, usecols=["sample", "condition"])
    if not np.array_equal(rows["sample"], np.arange(len(arr))) or not np.array_equal(
        rows["condition"], np.arange(len(arr)) % 10
    ):
        raise ValueError(
            f"Image order or class conditions differ from the paired protocol: {rows_path}"
        )
    return arr, rows["condition"].to_numpy()


@torch.inference_mode()
def extract(names, device, batch_size):
    configure(42)
    torch.set_float32_matmul_precision("highest")
    if batch_size < 1:
        raise ValueError("Positive batch size required")
    OUT.mkdir(parents=True, exist_ok=True)
    model = None
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]
    for name in names:
        imgs, labels = images_and_labels(name)
        digest = hashlib.sha256(np.ascontiguousarray(imgs)).hexdigest()
        metadata = dict(
            source=name,
            source_pixels_sha256=digest,
            backbone="DINOv2-S/14-reg4",
            backbone_commit=COMMIT,
            weights_sha256=WEIGHT_SHA256,
            resize=224,
            representation="final CLS",
            interpolation="bicubic, align_corners=False, antialias=True, clamp to [0,1]",
            amp="fp16" if torch.device(device).type == "cuda" else "fp32",
            n=len(imgs),
        )
        meta_path = OUT / f"{name}.json"
        target = OUT / f"{name}.npz"
        if (
            meta_path.exists()
            and target.exists()
            and json.loads(meta_path.read_text()) == metadata
        ):
            with np.load(target) as saved:
                if saved["features"].shape == (len(imgs), 384) and np.array_equal(
                    saved["labels"], labels
                ):
                    print("Cached", name, flush=True)
                    continue
        if model is None:
            model, _ = load_encoder(device)
        feats = []
        for i in range(0, len(imgs), batch_size):
            x = (
                torch.from_numpy(imgs[i : i + batch_size].copy()).to(
                    device, dtype=torch.float32
                )
                / 255
            )
            x = F.interpolate(
                x, (224, 224), mode="bicubic", align_corners=False, antialias=True
            ).clamp_(0, 1)
            with torch.autocast(
                torch.device(device).type,
                dtype=torch.float16,
                enabled=torch.device(device).type == "cuda",
            ):
                token = model((x - mean) / std)
            feats.append(token.float().cpu().numpy())
            if i > 0 and i % (batch_size * 125) == 0:
                print(name, i, "/", len(imgs), flush=True)
        features = np.concatenate(feats)
        if features.shape != (len(imgs), 384) or not np.isfinite(features).all():
            raise ValueError(f"Invalid features for {name}")
        tmp = target.with_suffix(".tmp.npz")
        np.savez(tmp, features=features, labels=labels)
        tmp.replace(target)
        dump(meta_path, metadata)
        print("Encoded", name, len(features), flush=True)


def frechet_distance(real, other):
    """Stable trace-square-root evaluation using symmetric positive semidefinite covariances."""
    real = real.astype(np.float64)
    other = other.astype(np.float64)
    a, b = np.cov(real, rowvar=False), np.cov(other, rowvar=False)
    eig, vec = np.linalg.eigh(a)
    root = (vec * np.sqrt(np.maximum(eig, 0))) @ vec.T
    prod = root @ b @ root
    cross = np.sqrt(np.maximum(np.linalg.eigvalsh((prod + prod.T) / 2), 0)).sum()
    delta = real.mean(0) - other.mean(0)
    return float(max(0, delta @ delta + np.trace(a) + np.trace(b) - 2 * cross))


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def same_set_radii(x):
    """Return squared chord distance to kth neighbour on the unit sphere, excluding self."""
    radii = []
    cols = np.asarray(NEIGHBOURS) - 1
    for i in range(0, len(x), 256):
        other = x[i : i + 256]
        dist = np.maximum(0, 2 - 2 * (other @ x.T))
        j = np.arange(len(other))
        dist[j, i + j] = np.inf
        radii.append(np.partition(dist, cols, axis=1)[:, cols])
    return np.vstack(radii)


def compare_neighbourhoods(real, radii, other, other_radii):
    r = len(real)
    s = len(other)
    k = len(NEIGHBOURS)
    precision = np.zeros((s, k), dtype=bool)
    recall = np.empty((r, k), dtype=bool)
    membership = np.zeros(k, dtype=np.int64)
    nearest = np.empty(r, dtype=int)
    nearest_distance = np.empty(r)
    for i in range(0, r, 256):
        dist = np.maximum(0, 2 - 2 * (real[i : i + 256] @ other.T))
        nearest[i : i + len(dist)] = np.argmin(dist, axis=1)
        nearest_distance[i : i + len(dist)] = np.min(dist, axis=1)
        for j in range(k):
            inside = dist < radii[i : i + len(dist), j, None]
            precision[:, j] |= inside.any(0)
            recall[i : i + len(dist), j] = (dist < other_radii[:, j]).any(1)
            membership[j] += inside.sum()
    covered = nearest_distance[:, None] < radii
    stats = [
        dict(
            k=k,
            precision=precision[:, j].mean(),
            recall=recall[:, j].mean(),
            density=membership[j] / (k * s),
            coverage=covered[:, j].mean(),
        )
        for j, k in enumerate(NEIGHBOURS)
    ]
    radius_ratio = np.divide(
        np.sqrt(nearest_distance),
        np.sqrt(radii[:, 1]),
        out=np.full(r, np.inf),
        where=radii[:, 1] > 0,
    )
    return stats, nearest, radius_ratio


def summarize(names):
    with np.load(OUT / "test.npz") as z:
        real, y = z["features"], z["labels"]
    with np.load(OUT / "train.npz") as z:
        train, tr_y = z["features"], z["labels"]
    ref = unit(real)
    real_groups = {j: np.flatnonzero(y == j) for j in range(10)}
    ref_radii = {j: same_set_radii(ref[idx]) for j, idx in real_groups.items()}
    full_radius = same_set_radii(ref)
    centered = train - train.mean(0)
    gram = centered.T @ centered
    gram.flat[:: len(gram) + 1] += 100
    probe = np.linalg.solve(
        gram, centered.T @ (np.eye(10, dtype=np.float32)[tr_y] - 0.1)
    )

    def classify(f):
        return ((f - train.mean(0)) @ probe).argmax(1)

    dump(
        OUT / "probe.json",
        dict(
            training_reference="10,000 stratified real training images",
            ridge=100,
            real_test_accuracy=float(np.mean(classify(real) == y)),
        ),
    )
    summaries = []
    byclass = []
    individuals = []
    # The negative control keeps all combined images and changes only their condition labels.
    if "combined" in names:
        names = [*names, "permuted"]
    for name in names:
        with np.load(OUT / f"{'combined' if name=='permuted' else name}.npz") as z:
            feat, labels = z["features"], z["labels"]
        if name == "permuted":
            labels = np.random.default_rng(571).permutation(labels)
        pred = classify(feat)
        other = unit(feat)
        overall, _, _ = compare_neighbourhoods(
            ref, full_radius, other, same_set_radii(other)
        )
        fd = frechet_distance(real, feat)
        summaries.extend(
            dict(source=name, fd=fd, agreement=(pred == labels).mean(), **row)
            for row in overall
        )
        for c, ids in real_groups.items():
            idx = np.flatnonzero(labels == c)
            assert len(ids) == len(idx) == 1000
            stats, nearest, radius_ratio = compare_neighbourhoods(
                ref[ids], ref_radii[c], other[idx], same_set_radii(other[idx])
            )
            byclass.extend(
                dict(
                    source=name,
                    category=CLASSES[c],
                    agreement=(pred[idx] == c).mean(),
                    **row,
                )
                for row in stats
            )
            individuals.extend(
                dict(
                    source=name,
                    real_reference_row=int(i),
                    category=CLASSES[c],
                    nearest_source_row=int(j),
                    distance_over_radius=float(v),
                    covered_k5=bool(v < 1),
                )
                for i, j, v in zip(ids, idx[nearest], radius_ratio)
            )
        print("Measured", name, flush=True)
    table(OUT / "metrics.csv", summaries)
    table(OUT / "by_class.csv", byclass)
    table(OUT / "nearest.csv", individuals)


def plot():
    from course.figures import (
        setup,
        page,
        axes_title,
        save,
        TEAL,
        BLUE,
        AMBER,
        GRAY,
        PLUM,
        INK,
    )

    setup()
    all_metrics = pd.read_csv(OUT / "metrics.csv").query("k==5").set_index("source")
    cls = pd.read_csv(OUT / "by_class.csv").query("k==5")
    order = [n for n in DISPLAY if n in all_metrics.index]
    colors = {
        "train": GRAY,
        "old": AMBER,
        "control": BLUE,
        "time": "#67731C",
        "align": "#705D49",
        "combined": TEAL,
        "strong": "#C15470",
        "extended": PLUM,
        "permuted": "#AA617F",
    }
    fig, ax = page(1, 2, size=(12, 5.0), width_ratios=[1, 1.18])
    yy = np.arange(len(order))
    for i, n in enumerate(order):
        sub = cls[cls.source == n]
        ax[0].plot(
            [all_metrics.loc[n, "coverage"], sub.coverage.mean()],
            [i, i],
            c=colors[n],
            lw=2,
        )
        ax[0].scatter(
            all_metrics.loc[n, "coverage"],
            i,
            facecolors="white",
            edgecolors=colors[n],
            s=60,
        )
        ax[0].scatter(sub.coverage.mean(), i, c=colors[n], marker="D", s=40)
    ax[0].set_yticks(yy, [DISPLAY[n] for n in order])
    ax[0].invert_yaxis()
    ax[0].scatter([], [], facecolor="white", edgecolor=INK, label="Pooled classes")
    ax[0].scatter([], [], marker="D", color=INK, label="Within class")
    axes_title(ax[0], "A   Coverage of real test images", "Fraction covered (k = 5)")
    ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    subset = [
        name for name in ["old", "control", "combined", "extended"] if name in order
    ]
    for n, mark in zip(subset, ("o", "s", "D", "^")):
        vals = cls[cls.source == n].set_index("category").loc[CLASSES, "coverage"]
        ax[1].plot(
            vals,
            np.arange(10),
            marker=mark,
            c=colors[n],
            ls="none",
            ms=6,
            label=DISPLAY[n],
        )
    ax[1].set_yticks(np.arange(10), CLASSES)
    ax[1].invert_yaxis()
    axes_title(
        ax[1],
        "B   Coverage within each class",
        "Fraction of 1,000 real test images covered",
    )
    ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    ax[0].set_xlim(-0.02, 1.02)
    upper = cls.loc[cls.source.isin(subset), "coverage"].max()
    ax[1].set_xlim(0, min(1, np.ceil((upper + 0.025) * 10) / 10))
    save(fig, 4, "research_semantic_coverage")


def neighbours_figure():
    from course.figures import setup, page, picture, save

    setup()
    rows = pd.read_csv(OUT / "nearest.csv")
    ref, _ = images_and_labels("test")
    ctrl = rows[rows.source == "control"].set_index("real_reference_row").sort_index()
    alt = rows[rows.source == "combined"].set_index("real_reference_row").sort_index()
    assert ctrl.index.equals(alt.index)
    arrays = {n: images_and_labels(n)[0] for n in ("control", "combined")}
    fig, axes = page(3, 4, size=(9, 6.2))
    examples = []
    for j, (name, gain) in enumerate(
        [("Bird", True), ("Cat", True), ("Ship", False), ("Truck", False)]
    ):
        # First official test index matching the stated condition, without visual selection.
        mask = (ctrl.category == name) & (~ctrl.covered_k5) & (alt.covered_k5 == gain)
        if not mask.any():
            for a in axes[:, j]:
                a.axis("off")
            axes[0, j].set_title(f"{name}: no match", size=9)
            continue
        i = int(ctrl.index[mask][0])
        picture(
            axes[0, j],
            ref[i].transpose(1, 2, 0),
            f"{name} · {'gained' if gain else 'missed'}",
            interpolation="nearest",
        )
        axes[0, j].set_xlabel(f"Test #{i}", fontsize=9)
        for row, (n, records) in enumerate([("control", ctrl), ("combined", alt)], 1):
            index = int(records.loc[i, "nearest_source_row"])
            picture(
                axes[row, j],
                arrays[n][index].transpose(1, 2, 0),
                interpolation="nearest",
            )
            ratio = records.loc[i, "distance_over_radius"]
            axes[row, j].set_xlabel(f"d / r = {ratio:.4f}", fontsize=9)
            examples.append(
                dict(
                    category=name,
                    case="gained" if gain else "missed",
                    test_reference_row=i,
                    generator=n,
                    generated_sample=index,
                    distance_over_radius=ratio,
                )
            )
    for a, label in zip(axes[:, 0], ["Real test", "Continuation", "Time + alignment"]):
        a.set_ylabel(label, fontsize=9)
    table(OUT / "gallery.csv", examples)
    save(fig, 4, "research_coverage_neighbours")


def guidance_figure():
    from course.figures import setup, page, axes_title, save, BLUE, TEAL

    setup()
    arms = ["control", "time", "align", "combined"]
    stronger = ["control_strong", "time_strong", "align_strong", "strong"]
    names = ["Continuation", "Time sampling", "Alignment", "Both"]
    stats = pd.read_csv(OUT / "metrics.csv").query("k == 5").set_index("source")
    groups = pd.read_csv(OUT / "by_class.csv").query("k == 5")
    stats["class_coverage"] = groups.groupby("source").coverage.mean()
    fig, axes = page(1, 3, size=(12, 3.9))
    for j, (column, title, xlabel) in enumerate(
        [
            ("fd", "A   DINO Fréchet distance", "Distance (lower is better)"),
            ("agreement", "B   Condition agreement", "DINO probe agreement (%)"),
            ("class_coverage", "C   Within-class coverage", "Real images covered (%)"),
        ]
    ):
        ax = axes[j]
        scale = 1 if column == "fd" else 100
        lo = stats.loc[arms, column].to_numpy() * scale
        hi = stats.loc[stronger, column].to_numpy() * scale
        y = np.arange(len(arms))
        ax.hlines(
            y, np.minimum(lo, hi), np.maximum(lo, hi), color="#ABB5BB", linewidth=2
        )
        ax.scatter(lo, y, c=BLUE, s=52, label="Guidance 1.6", zorder=3)
        ax.scatter(hi, y, c=TEAL, marker="D", s=47, label="Guidance 3.0", zorder=3)
        ax.set_yticks(y, names if j == 0 else [])
        ax.set_ylim(len(arms) - 0.55, -0.45)
        ax.margins(x=0.16)
        axes_title(ax, title, xlabel)
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=2)
    save(fig, 4, "research_guidance_coverage")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["extract", "measure", "figure"])
    p.add_argument("collections", nargs="*")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    if a.collections and a.action != "extract":
        p.error(
            "Only extract accepts collection names. measure recomputes the complete comparison."
        )
    if unknown := set(a.collections) - set(SOURCES):
        p.error(f"Unknown collection(s): {sorted(unknown)}")
    with threadpool_limits(3):
        if a.action == "extract":
            extract(a.collections or list(SOURCES), a.device, a.batch_size)
        elif a.action == "measure":
            summarize([n for n in SOURCES if n != "test"])
        else:
            plot()
            neighbours_figure()
            guidance_figure()
