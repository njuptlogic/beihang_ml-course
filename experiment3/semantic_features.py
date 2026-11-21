"""Freeze the official DINOv2-S/14 register model, then compress dense features on fit images."""
import argparse
import hashlib
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from io import BytesIO
from pathlib import Path
import sys
import tempfile
from zipfile import ZipFile

import numpy as np
import pandas as pd
from PIL import Image
import requests
import torch
from threadpoolctl import threadpool_limits
from course.common import configure, dump
from experiment3.train import BASE

COMMIT = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
WEIGHT_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth"
WEIGHT_SHA256 = "f433177089a681826f849f194ece3bb48f4d63fb38d32fc837e3dc7a4e5641fb"
WEIGHT_BYTES = 88291785
BACKBONES = BASE / "data/backbones"
CACHE = BASE / "data/semantic"
OUT = BASE / "outputs/semantic"
BLOCKS = (5, 11)
SIZE = (448, 252)
PATCH_GRID = (18, 32)
RANK = 64


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def download(url, path):
    with requests.get(url, stream=True, timeout=(25, 120)) as response:
        response.raise_for_status()
        tmp = path.with_suffix(".part")
        with tmp.open("wb") as f:
            for chunk in response.iter_content(1 << 20):
                f.write(chunk)
    tmp.replace(path)


def download_weight(path):
    # This host supports ranges. A dropped connection then costs a single chunk.
    step = 512 * 1024

    def fetch(start):
        end = min(WEIGHT_BYTES, start + step) - 1
        expected = f"bytes {start}-{end}/{WEIGHT_BYTES}"
        for attempt in range(5):
            try:
                with requests.get(
                    WEIGHT_URL,
                    headers={"Range": f"bytes={start}-{end}"},
                    timeout=(15, 45),
                ) as response:
                    response.raise_for_status()
                    if response.headers.get("Content-Range") != expected:
                        raise OSError(f"Missing expected range {expected}")
                    part = response.content
                    if len(part) != end - start + 1:
                        raise OSError(f"Incomplete range {expected}")
                    return part
            except (requests.RequestException, OSError):
                if attempt == 4:
                    raise

    with tempfile.TemporaryDirectory(dir=path.parent) as directory:
        tmp = Path(directory) / "weight.pth"
        with ThreadPoolExecutor(max_workers=8) as pool, tmp.open("wb") as f:
            for part in pool.map(fetch, range(0, WEIGHT_BYTES, step)):
                f.write(part)
        if digest(tmp) != WEIGHT_SHA256:
            raise OSError("Downloaded DINOv2 weights failed SHA-256 verification")
        tmp.replace(path)


def load_encoder(device):
    BACKBONES.mkdir(parents=True, exist_ok=True)
    source = BACKBONES / ("dinov2-" + COMMIT)
    if not (source / "dinov2/hub/backbones.py").is_file():
        archive = BACKBONES / "dinov2.zip"
        download(
            f"https://codeload.github.com/facebookresearch/dinov2/zip/{COMMIT}", archive
        )
        with ZipFile(archive) as z:
            assert {p.split("/")[0] for p in z.namelist()} == {source.name}
            z.extractall(BACKBONES)
    weight = BACKBONES / WEIGHT_URL.split("/")[-1]
    if not weight.is_file() or digest(weight) != WEIGHT_SHA256:
        download_weight(weight)
    sys.path.insert(0, str(source))
    from dinov2.hub.backbones import dinov2_vits14_reg

    model = dinov2_vits14_reg(pretrained=False)
    state = torch.load(weight, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.eval().to(device), weight


def pixels(blob):
    with Image.open(BytesIO(blob)) as image:
        return np.asarray(image.convert("RGB").resize(SIZE, Image.Resampling.LANCZOS))


@torch.inference_mode()
def encode(model, imgs, device):
    batch = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).to(device) / 255.0
    batch = (
        batch - batch.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
    ) / batch.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
    with torch.autocast(
        device_type=torch.device(device).type,
        dtype=torch.float16,
        enabled=device == "cuda",
    ):
        layers = model.get_intermediate_layers(
            batch, n=list(BLOCKS), reshape=True, return_class_token=True
        )
    return [feat.float().cpu().numpy() for feat, cls in layers], layers[-1][
        1
    ].float().cpu().numpy()


def reduce_features(raw, fit_rng, fit, block, out, n):
    # Identical number of patch samples per fitting image, independent of its category.
    drawn = fit_rng.integers(0, 18 * 32, size=(len(fit), 24))
    sampled = (
        np.stack([raw[idx].reshape(384, -1)[:, pos].T for idx, pos in zip(fit, drawn)])
        .reshape(-1, 384)
        .astype("float64")
    )
    mean = sampled.mean(0)
    sampled -= mean
    with threadpool_limits(2):
        cov = (sampled.T @ sampled) / (len(sampled) - 1)
        eigenvalue, eigenvector = np.linalg.eigh(cov)
        # Fix signs for stable serialized transforms.
        U = eigenvector[:, -RANK:][:, ::-1]
        U *= np.sign(U[np.argmax(np.abs(U), axis=0), np.arange(RANK)])
        proj = (U / np.sqrt(eigenvalue[-RANK:][::-1] + 1e-5)).T.astype("float32")
        mean = mean.astype("float32")
        feat = np.lib.format.open_memmap(
            out / f"block{block+1}.tmp",
            mode="w+",
            dtype="float16",
            shape=(n, RANK, *PATCH_GRID),
        )
        for start in range(0, n, 12):
            batch = raw[start : start + 12].astype("float32").reshape(-1, 384, 18 * 32)
            feat[start : start + len(batch)] = (
                (proj @ (batch - mean[None, :, None])).reshape(-1, RANK, *PATCH_GRID)
            ).astype("float16")
        feat.flush()
        (out / f"block{block+1}.tmp").replace(out / f"block{block+1}.npy")
    return mean, proj, float(eigenvalue[-RANK:].sum() / eigenvalue.sum())


def extract(device, batch_size):
    configure(42)
    CACHE.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    model, weight = load_encoder(device)
    meta = pd.read_csv(BASE / "data/images.csv")
    cls = np.empty((len(meta), 384), dtype="float32")
    meta_rows = list(meta.itertuples(index=False))
    with tempfile.TemporaryDirectory(dir=CACHE) as work, ExitStack() as stack:
        raw = [
            np.lib.format.open_memmap(
                Path(work) / f"{b}.npy",
                mode="w+",
                dtype="float16",
                shape=(len(meta), 384, *PATCH_GRID),
            )
            for b in BLOCKS
        ]
        archives = {
            n: stack.enter_context(ZipFile(BASE / "data/raw" / n))
            for n in meta.archive.unique()
        }
        for start in range(0, len(meta), batch_size):
            rows = meta_rows[start : start + batch_size]
            imgs = [pixels(archives[r.archive].read(r.member)) for r in rows]
            maps, tokens = encode(model, imgs, device)
            cls[start : start + len(rows)] = tokens
            for dest, m in zip(raw, maps):
                dest[start : start + len(rows)] = m.astype("float16")
            if start % 200 == 0:
                print(f"Encoded {start}/{len(meta)}", flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        rng = np.random.default_rng(458)
        fit = meta.index[meta.split == "fit"].to_numpy()
        vectors = {}
        variance = []
        for b, array in zip(BLOCKS, raw):
            mean, projection, fraction = reduce_features(
                array, rng, fit, b, CACHE, len(meta)
            )
            vectors[f"mean{b+1}"] = mean
            vectors[f"proj{b+1}"] = projection
            variance.append(fraction)
    np.savez(OUT / "pca.npz", **vectors)
    np.save(CACHE / "cls.npy", cls)
    dump(
        OUT / "backbone.json",
        dict(
            source=f"https://github.com/facebookresearch/dinov2/tree/{COMMIT}",
            url=WEIGHT_URL,
            sha256=digest(weight),
            rank=RANK,
            fit_images=len(fit),
            patches_per_fit_image=24,
            variance_retained=variance,
            raw_size=SIZE,
            blocks=[b + 1 for b in BLOCKS],
            device=device,
            batch_size=batch_size,
        ),
    )
    print("Semantic cache complete", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=5)
    args = parser.parse_args()
    extract(args.device, args.batch_size)
