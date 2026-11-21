"""Cross-check the training-selected CIFAR settings in the previous DINO feature space."""
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from threadpoolctl import threadpool_limits

from course.common import configure, dump, table
from course.figures import setup, page, save, axes_title, TEAL, BLUE, GRAY, PLUM, AMBER
from experiment3.semantic_features import load_encoder, COMMIT, WEIGHT_SHA256
from experiment4.guidance_eval import ROOT, CLASS
from experiment4.semantic_evaluation import (
    OUT as REFS,
    frechet_distance,
    unit,
    same_set_radii,
    compare_neighbourhoods,
)


@torch.inference_mode()
def extract(folder, model, device):
    imgs = np.load(folder / "images.npy", mmap_mode="r")
    rows = pd.read_csv(folder / "per_sample.csv")
    assert np.array_equal(rows["sample"], np.arange(10000)) and np.array_equal(
        rows.condition, np.arange(10000) % 10
    )
    digest = hashlib.sha256(np.ascontiguousarray(imgs)).hexdigest()
    cache = folder / "dino.npz"
    meta = folder / "dino.json"
    config = dict(
        n=10000,
        backbone="DINOv2-S/14-reg4",
        commit=COMMIT,
        weights_sha256=WEIGHT_SHA256,
        source_pixels_sha256=digest,
        resize=224,
        representation="final CLS",
        interpolation="bicubic, align_corners=False, antialias=True, clamp to [0,1]",
        amp="fp16" if device.type == "cuda" else "fp32",
    )
    if meta.exists() and json.loads(meta.read_text()) == config and cache.exists():
        with np.load(cache) as z:
            feats = z["features"]
            assert feats.shape == (10000, 384) and np.array_equal(
                z["labels"], rows.condition.to_numpy()
            )
            return feats
    outputs = []
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]
    for batch in range(0, len(imgs), 16):
        x = torch.from_numpy(imgs[batch : batch + 16].copy()).float().to(device) / 255.0
        x = F.interpolate(
            x, (224, 224), mode="bicubic", align_corners=False, antialias=True
        ).clamp_(0, 1)
        with torch.autocast(
            device.type, dtype=torch.float16, enabled=device.type == "cuda"
        ):
            token = model((x - mean) / std)
        outputs.append(token.float().cpu().numpy())
    feats = np.concatenate(outputs)
    tmp = folder / "dino.tmp.npz"
    np.savez(tmp, features=feats, labels=rows.condition.to_numpy())
    tmp.replace(cache)
    dump(meta, config)
    return feats


def summarize(device):
    configure(42)
    torch.set_float32_matmul_precision("highest")
    locked = json.loads((ROOT / "selection/locked.json").read_text())
    names = dict.fromkeys(
        locked[key]
        for key in [
            "nfe8_quality",
            "nfe8_condition",
            "nfe24_quality",
            "nfe24_condition",
        ]
    )
    train = np.load(REFS / "train.npz")
    test = np.load(REFS / "test.npz")
    x, y = train["features"], train["labels"]
    real, truth = test["features"], test["labels"]
    mean = x.mean(0)
    gram = (x - mean).T @ (x - mean)
    gram.flat[:: len(gram) + 1] += 100
    w = np.linalg.solve(gram, (x - mean).T @ (np.eye(10, dtype=np.float32)[y] - 0.1))
    assert np.isclose((((real - mean) @ w).argmax(1) == truth).mean(), 0.9682)
    unit_real = unit(real)
    radii = same_set_radii(unit_real)
    class_radii = [same_set_radii(unit_real[truth == j]) for j in range(10)]
    model, _ = load_encoder(str(device))
    rows = []
    class_rows = []
    for name in names:
        folder = ROOT / "test" / name
        f = extract(folder, model, device)
        y = np.arange(10000) % 10
        decisions = ((f - mean) @ w).argmax(1)
        other = unit(f)
        stats, _, _ = compare_neighbourhoods(
            unit_real, radii, other, same_set_radii(other)
        )
        cc = []
        for j in range(10):
            b = other[y == j]
            a = unit_real[truth == j]
            subset, _, _ = compare_neighbourhoods(
                a, class_radii[j], b, same_set_radii(b)
            )
            coverage = subset[1]["coverage"]
            cc.append(coverage)
            class_rows.append(
                dict(
                    scenario=name,
                    category=CLASS[j],
                    coverage_k5=coverage,
                    agreement=(decisions[y == j] == j).mean(),
                )
            )
        rows.append(
            dict(
                scenario=name,
                fd=frechet_distance(real, f),
                agreement=(decisions == y).mean(),
                pooled_coverage=stats[1]["coverage"],
                within_coverage=np.mean(cc),
            )
        )
        print(name, rows[-1], flush=True)
    table(ROOT / "semantic_metrics.csv", rows)
    table(ROOT / "semantic_by_class.csv", class_rows)
    return pd.DataFrame(rows), locked


def visualize(rows, locked):
    baseline = (
        pd.read_csv(REFS / "metrics.csv")
        .query("k == 5")
        .set_index("source")
        .loc["time_strong"]
    )
    groups = pd.read_csv(REFS / "by_class.csv").query(
        "k == 5 and source == 'time_strong'"
    )
    series = [
        dict(
            label="32 ep · strong CFG",
            fd=baseline.fd,
            agreement=baseline.agreement,
            within_coverage=groups.coverage.mean(),
        )
    ]
    shown = set()
    for budget in (8, 24):
        for role in ("quality", "condition"):
            key = locked[f"nfe{budget}_{role}"]
            if key in shown:
                continue
            shown.add(key)
            both = locked[f"nfe{budget}_quality"] == locked[f"nfe{budget}_condition"]
            text = f"{budget} evals · " + ("both choices" if both else role)
            series.append(
                dict(label=text, **rows.set_index("scenario").loc[key].to_dict())
            )
    frame = pd.DataFrame(series)
    setup()
    fig, ax = page(1, 3, size=(11.7, 3.5))
    for i, (axis, (col, title)) in enumerate(
        zip(
            ax,
            [
                ("fd", "DINO distance ↓"),
                ("agreement", "Condition agreement ↑"),
                ("within_coverage", "Within-class coverage ↑"),
            ],
        )
    ):
        scale = 1 if i == 0 else 100
        values = frame[col] * scale
        colors = [GRAY, BLUE, TEAL, PLUM, AMBER][: len(frame)]
        axis.scatter(values, np.arange(len(frame)), c=colors, s=58)
        axis.set_yticks(np.arange(len(frame)), frame.label if i == 0 else [])
        axis.set_ylim(len(frame) - 0.55, -0.45)
        axes_title(axis, title, "Distance" if i == 0 else "Percent")
        axis.margins(x=0.14)
    save(fig, 4, "guidance_semantics")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    with threadpool_limits(3):
        result, locked = summarize(torch.device(args.device))
    visualize(result, locked)
