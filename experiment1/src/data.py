"""直接从教师提供的 ZIP 读取 BMP，标签严格取自文件名首字符。"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import zipfile

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_archive(path: Path, expected_count: int | None = None):
    images, labels, names = [], [], []
    with zipfile.ZipFile(path) as archive:
        members = sorted(n for n in archive.namelist()
                         if n.lower().endswith(".bmp") and not n.startswith("__MACOSX/"))
        if expected_count is not None and len(members) != expected_count:
            raise ValueError(f"{path.name}: 期望 {expected_count} 张 BMP，实际 {len(members)}")
        if not members:
            raise ValueError(f"{path} 中没有 BMP 图像")
        for name in members:
            filename = PurePosixPath(name.replace("\\", "/")).name
            if filename[0] not in "0123456789":
                raise ValueError(f"文件名首字符不是合法标签: {name}")
            with Image.open(io.BytesIO(archive.read(name))) as im:
                arr = np.asarray(im.convert("L"), dtype=np.uint8).copy()
            if arr.shape != (28, 28):
                raise ValueError(f"{name}: 预期 28×28，实际 {arr.shape}，请核实数据来源")
            images.append(arr)
            labels.append(int(filename[0]))
            names.append(name)
    y = np.asarray(labels, dtype=np.int64)
    if expected_count and set(y.tolist()) != set(range(10)):
        raise ValueError("数据集必须包含全部 10 类")
    return np.stack(images), y, np.asarray(names)


def prepare_data(seed: int = 42, val_fraction: float = 0.1):
    raw = ROOT / "data/raw"
    cache = ROOT / "data/processed"
    cache.mkdir(parents=True, exist_ok=True)
    archives = {s: raw / f"1-Digit-{s.title()}Set.zip" for s in ("train", "test")}
    for path in archives.values():
        if not path.exists():
            raise FileNotFoundError(f"缺少课程数据: {path}；请先运行 python download_data.py")
    digests = {s: sha256(p) for s, p in archives.items()}
    manifest_path = cache / "manifest.json"
    cache_path = cache / "digits.npz"
    if manifest_path.exists() and cache_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (manifest.get("archive_sha256") == digests and manifest.get("seed") == seed
                and manifest.get("val_fraction") == val_fraction):
            with np.load(cache_path, allow_pickle=False) as data:
                return {k: data[k] for k in data.files}, manifest

    data = {}
    for split, count in (("train", 60000), ("test", 10000)):
        x, y, names = read_archive(archives[split], count)
        data[f"{split}_x"], data[f"{split}_y"], data[f"{split}_names"] = x, y, names
    fit, val = train_test_split(np.arange(len(data["train_y"])), test_size=val_fraction,
                                random_state=seed, stratify=data["train_y"])
    data["fit_indices"], data["val_indices"] = np.sort(fit), np.sort(val)
    # 只用参与优化的子集计算标准化参数。
    pixels = data["train_x"][fit].astype(np.float32) / 255.0
    mean, std = float(pixels.mean()), float(pixels.std())
    del pixels
    hashes = {s: [hashlib.sha256(im.tobytes()).hexdigest() for im in data[f"{s}_x"]]
              for s in ("train", "test")}
    fit_hashes = {hashes["train"][i] for i in fit}
    train_hashes = set(hashes["train"])
    counts = lambda y: np.bincount(y, minlength=10).tolist()
    manifest = {
        "source": "教师 README 中的北航网盘原始 ZIP；没有使用 torchvision MNIST 替代",
        "source_url": "https://bhpan.buaa.edu.cn/link/AAB14093C4DF1F4085922390C07C87498F",
        "archive_sha256": digests, "seed": seed, "val_fraction": val_fraction,
        "image_shape": [28, 28], "mean": mean, "std": std,
        "counts": {"train_full": 60000, "fit": len(fit), "validation": len(val), "test": 10000},
        "class_counts": {"train_full": counts(data["train_y"]),
                         "fit": counts(data["train_y"][fit]),
                         "validation": counts(data["train_y"][val]),
                         "test": counts(data["test_y"])},
        "pixel_overlap": {
            "validation_samples_identical_to_fit": sum(hashes["train"][i] in fit_hashes for i in val),
            "test_samples_identical_to_train": sum(h in train_hashes for h in hashes["test"]),
            "train_duplicate_pixels": len(hashes["train"]) - len(train_hashes),
            "test_duplicate_pixels": len(hashes["test"]) - len(set(hashes["test"])),
        },
    }
    np.savez_compressed(cache_path, **data)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return data, manifest
