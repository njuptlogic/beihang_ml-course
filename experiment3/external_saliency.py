"""Run the published UNISAL static SALICON branch on exactly the course images."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import requests
import torch
from course.common import configure, dump
from experiment3.multiscale_features import BASE, OUT, Images
from experiment3.train import resize

COMMIT = "7f8ca1a7ecb3dbc1bf7fca837cb6186a85ac27d9"
ROOT = BASE / "data/backbones" / ("unisal-" + COMMIT)
WEIGHT = "training_runs/pretrained_unisal/weights_best.pth"
SIZE = (
    416,
    256,
)  # Official FolderImageDataset get_optimal_out_size for the course 16:9 images.


def fetch():
    manifest = json.loads((BASE / "unisal_manifest.json").read_text())
    base = f"https://raw.githubusercontent.com/rdroste/unisal/{COMMIT}/"

    def digest(b):
        return hashlib.sha1(f"blob {len(b)}\0".encode() + b).hexdigest()

    def request_range(name, start, end, total):
        for attempt in range(7):
            try:
                url = base + name
                hdr = {}
                if total > 300000:
                    url += f"?range={start}-{end}"
                    hdr["Range"] = f"bytes={start}-{end}"
                r = requests.get(url, headers=hdr, timeout=(10, 30))
                r.raise_for_status()
                data = r.content
                if len(data) == total and end - start + 1 != total:
                    data = data[start : end + 1]
                if len(data) != end - start + 1:
                    raise OSError("truncated response")
                return data
            except (requests.RequestException, OSError):
                if attempt == 6:
                    raise
                time.sleep(0.4 * (attempt + 1))

    missing = []
    for name, info in manifest.items():
        path = ROOT / name
        if not path.exists() or digest(path.read_bytes()) != info["sha"]:
            missing.append(name)
    with ThreadPoolExecutor(max_workers=12) as pool:
        jobs = {}
        for name in missing:
            size = manifest[name]["size"]
            step = 256 * 1024
            jobs[name] = [
                pool.submit(request_range, name, i, min(i + step, size) - 1, size)
                for i in range(0, size, step)
            ]
        for name in missing:
            data = b"".join(j.result() for j in jobs[name])
            assert digest(data) == manifest[name]["sha"], name
            path = ROOT / name
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".part")
            tmp.write_bytes(data)
            tmp.replace(path)
            print("Ready", name, flush=True)


def load(device):
    import types, importlib

    fetch()
    # Upstream package __init__ imports its training CLI with unrelated legacy dependencies.
    # Create the package namespace so the original model and its relative imports load directly.
    if "unisal" not in sys.modules:
        pkg = types.ModuleType("unisal")
        pkg.__path__ = [str(ROOT / "unisal")]
        sys.modules["unisal"] = pkg
    module = importlib.import_module("unisal.model")
    folder = ROOT / "training_runs/pretrained_unisal"
    cfg = json.loads((folder / "UNISAL.json").read_text())
    cfg["cnn_cfg"] = {**cfg.get("cnn_cfg", {}), "pretrained": False}
    cfg["verbose"] = 0
    net = module.UNISAL(**cfg)
    weights = torch.load(ROOT / WEIGHT, map_location="cpu", weights_only=True)
    net.load_state_dict(weights, strict=True)
    return net.eval().to(device)


@torch.inference_mode()
def infer(model, rgb, device, return_logits=False):
    x = torch.as_tensor(rgb, device=device).permute(0, 3, 1, 2).float() / 255.0
    x = (x - x.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]) / x.new_tensor(
        [0.229, 0.224, 0.225]
    )[None, :, None, None]
    # Original network returns a spatial log-probability; downsample probabilities only after its output-space smoothing.
    logp = model(
        x[:, None], target_size=(SIZE[1], SIZE[0]), source="SALICON", static=True
    )[:, 0]
    p = resize(logp, (45, 80)).flatten(1).softmax(1).reshape(-1, 1, 45, 80)
    return (p, logp) if return_logits else p


def extract(device):
    configure(42)
    model = load(device)
    meta = pd.read_csv(BASE / "data/images.csv")
    parts = []
    tmp = BASE / "data/multiscale/unisal_logits.partial"
    bank = np.lib.format.open_memmap(
        tmp, mode="w+", shape=(len(meta), 1, SIZE[1], SIZE[0]), dtype="float32"
    )
    k = 0
    for img in torch.utils.data.DataLoader(
        Images(meta, SIZE), batch_size=2, num_workers=2
    ):
        p, logs = infer(model, img, device, return_logits=True)
        parts.append(p.cpu().numpy())
        bank[k : k + len(img)] = logs.cpu().numpy()
        k += len(img)
        if k % 200 == 0:
            print(k, flush=True)
    assert k == len(meta)
    bank.flush()
    del bank
    tmp.replace(tmp.with_name("unisal_logits.npy"))
    maps = np.concatenate(parts)
    assert maps.shape == (2000, 1, 45, 80)
    np.save(OUT / "unisal.npy", maps)
    dump(
        OUT / "unisal.json",
        dict(
            source="https://github.com/rdroste/unisal",
            commit=COMMIT,
            weight=WEIGHT,
            sha256=hashlib.sha256((ROOT / WEIGHT).read_bytes()).hexdigest(),
            configuration="static single image, SALICON branch, 416x256, frozen published weights",
            published_training_domains=[
                "DHF1K",
                "Hollywood-2",
                "UCF Sports",
                "SALICON",
            ],
        ),
    )
    print("Exported", maps.shape, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("download", "predict"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    fetch() if a.action == "download" else extract(a.device)
