"""Official Meta PE-Core-T16 visual features for saliency fusion, independent of the DINO encoder."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib, sys, time
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import pandas as pd
import torch
import requests
from torch.utils.data import DataLoader
from course.common import configure, dump
from experiment3.multiscale_features import CACHE, OUT, BASE, Images

REPO = "facebook/PE-Core-T16-384"
REV = "8b4722741a516c18e7d311f158f13c600fea6c78"
COMMIT = "3e352cca660658d4b5c90f42a7808b11469e4c66"
FILENAME = "PE-Core-T16-384.pt"
TOTAL = 278284008
SHA = "868f89f066b7c1ae51add02d2cb7b925fd051076282773f2ec3c71359fbdaf3b"
STORE = BASE / "data/backbones/perception"
WEIGHT = STORE / FILENAME


def digest(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


CONFIGS = {
    "core": (REPO, REV, FILENAME, TOTAL, SHA),
    "spatial": (
        "facebook/PE-Spatial-T16-512",
        "78e209188fb04f428092a5f506a22af24298e659",
        "PE-Spatial-T16-512.pt",
        22781083,
        "6bed3e60e22f7af691d51b87fde8b29bf7c0c3d23a905446e6be45fb0c95df6e",
    ),
}
SIZE = (512, 288)
SOURCE_FILES = (
    "LICENSE.PE",
    "core/vision_encoder/pe.py",
    "core/vision_encoder/config.py",
    "core/vision_encoder/rope.py",
)


def download(name="core"):
    repo, rev, filename, total, sha = CONFIGS[name]
    weight = STORE / filename
    STORE.mkdir(exist_ok=True, parents=True)
    if weight.exists() and digest(weight) == sha:
        return
    parts = STORE / ("chunks" if name == "core" else "spatial_chunks")
    parts.mkdir(exist_ok=True)
    unit = 1024 * 1024
    url = f"https://huggingface.co/{repo}/resolve/{rev}/{filename}"

    def fetch(start):
        end = min(total, start + unit) - 1
        path = parts / str(start)
        if path.is_file() and path.stat().st_size == end - start + 1:
            return
        for attempt in range(8):
            try:
                r = requests.get(
                    url + f"?download=true&slice={start}-{end}&retry={attempt}",
                    headers={"Range": f"bytes={start}-{end}"},
                    timeout=(15, 45),
                )
                r.raise_for_status()
                if (
                    r.headers.get("Content-Range") != f"bytes {start}-{end}/{total}"
                    or len(r.content) != end - start + 1
                ):
                    raise IOError("invalid byte range")
                tmp = path.with_suffix(".part")
                tmp.write_bytes(r.content)
                tmp.replace(path)
                return
            except (requests.RequestException, OSError):
                if attempt == 7:
                    raise
                time.sleep(min(3, attempt * 0.5))

    offsets = range(0, total, unit)
    with ThreadPoolExecutor(max_workers=14) as pool:
        for i, _ in enumerate(pool.map(fetch, offsets)):
            if i % 25 == 0:
                print("Weight chunks", i + 1, len(offsets), flush=True)
    tmp = weight.with_suffix(".part")
    with tmp.open("wb") as out:
        for start in offsets:
            out.write((parts / str(start)).read_bytes())
    assert digest(tmp) == sha
    tmp.replace(weight)
    for start in offsets:
        (parts / str(start)).unlink()
    parts.rmdir()
    print("Weight verified", sha, flush=True)


def load(name, device):
    download(name)
    for n in SOURCE_FILES:
        path = STORE / n
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            url = f"https://raw.githubusercontent.com/facebookresearch/perception_models/{COMMIT}/{n}"
            r = requests.get(url, timeout=(15, 60))
            r.raise_for_status()
            path.write_bytes(r.content)
    sys.path.insert(0, str(STORE))
    from core.vision_encoder.pe import VisionTransformer

    repo, _, filename, _, _ = CONFIGS[name]
    model = VisionTransformer.from_config(repo.split("/")[-1], pretrained=False)
    sd = torch.load(STORE / filename, map_location="cpu", weights_only=True)
    if "state_dict" in sd:
        sd = sd["state_dict"]
    if any(k.startswith("visual.") for k in sd):
        sd = {k[len("visual.") :]: v for k, v in sd.items() if k.startswith("visual.")}
    model.load_state_dict(sd, strict=True)
    return model.eval().to(device)


@torch.inference_mode()
def encode(model, imgs, device):
    x = torch.as_tensor(imgs, device=device).permute(0, 3, 1, 2).float() / 127.5 - 1.0
    saved = []

    def record(module, args, out):
        saved.append(out[:, 1:].detach())

    handles = [
        model.transformer.resblocks[i].register_forward_hook(record) for i in (5, 10)
    ]
    try:
        with torch.autocast(
            torch.device(device).type,
            dtype=torch.float16,
            enabled=str(device).startswith("cuda"),
        ):
            model.forward_features(x, layer_idx=10)
    finally:
        for h in handles:
            h.remove()
    return torch.stack(saved, dim=1).half()


def extract(name, device):
    configure(42)
    m = load(name, device)
    meta = pd.read_csv(BASE / "data/images.csv")
    shape = (len(meta), 2, 18 * 32, 192)
    rawfile = CACHE / f"pe_{name}_raw.tmp"
    raw = np.lib.format.open_memmap(rawfile, mode="w+", dtype="float16", shape=shape)
    k = 0
    for imgs in DataLoader(Images(meta, SIZE), batch_size=5, num_workers=2):
        raw[k : k + len(imgs)] = encode(m, imgs, device).cpu().numpy()
        k += len(imgs)
        if k % 400 == 0:
            print(name, k, flush=True)
    raw.flush()
    del m
    fit = meta.index[meta.split == "fit"].to_numpy()
    rng = np.random.default_rng(569)
    patch = rng.integers(0, 576, (len(fit), 24))
    basis = {}
    out_file = CACHE / f"pe_{name}.tmp"
    out = np.lib.format.open_memmap(
        out_file, mode="w+", dtype="float16", shape=(len(meta), 128, 18, 32)
    )
    var = []
    for layer in range(2):
        x = raw[fit[:, None], layer, patch].reshape(-1, 192).astype("float64")
        mu = x.mean(0)
        x -= mu
        e, v = np.linalg.eigh(x.T @ x / (len(x) - 1))
        proj = (v[:, -64:] / np.sqrt(e[-64:] + 1e-5)).astype("float32")
        mu = mu.astype("float32")
        basis[f"mean{layer}"] = mu
        basis[f"proj{layer}"] = proj
        var.append(float(e[-64:].sum() / e.sum()))
        for i in range(0, len(meta), 12):
            part = raw[i : i + 12, layer].astype("float32")
            out[i : i + len(part), layer * 64 : (layer + 1) * 64] = (
                ((part - mu) @ proj).transpose(0, 2, 1).reshape(-1, 64, 18, 32)
            )
    out.flush()
    del raw, out
    out_file.replace(CACHE / f"pe_{name}.npy")
    rawfile.unlink()
    np.savez(OUT / f"pe_{name}_pca.npz", **basis)
    dump(
        OUT / f"pe_{name}.json",
        dict(
            weights=CONFIGS[name][0],
            revision=CONFIGS[name][1],
            sha256=CONFIGS[name][-1],
            code_commit=COMMIT,
            size=SIZE,
            blocks=[6, 11],
            retained=var,
            fit_images=len(fit),
            patches_per_fit=24,
        ),
    )
    print("PE ready", name, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["download", "extract"])
    p.add_argument("--name", choices=CONFIGS, default="core")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    download(a.name) if a.action == "download" else extract(a.name, a.device)
