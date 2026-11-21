"""Frozen DINO features at coarse and fine fields of view, projected into the same fit-only PCA basis."""
import argparse
from io import BytesIO
from zipfile import ZipFile
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from course.common import configure, dump
from experiment3.semantic_features import (
    OUT as PREV,
    BASE,
    load_encoder,
    BLOCKS,
    WEIGHT_SHA256,
)

CACHE = BASE / "data/multiscale"
OUT = BASE / "outputs/multiscale"
SCALES = {"fine": (672, 378), "coarse": (224, 126)}


def pixels(blob, size):
    with Image.open(BytesIO(blob)) as im:
        return np.array(im.convert("RGB").resize(size, Image.Resampling.LANCZOS))


class Images(Dataset):
    def __init__(self, meta, size):
        self.rows = list(meta.itertuples(index=False))
        self.size = size
        self.zips = {}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        if r.archive not in self.zips:
            self.zips[r.archive] = ZipFile(BASE / "data/raw" / r.archive)
        return pixels(self.zips[r.archive].read(r.member), self.size)


def pca(device):
    with np.load(PREV / "pca.npz") as z:
        return [
            (
                torch.tensor(z[f"mean{b+1}"], device=device)[None, :, None, None],
                torch.tensor(z[f"proj{b+1}"], device=device)[:, :, None, None],
            )
            for b in BLOCKS
        ]


@torch.inference_mode()
def embed(model, imgs, device, basis):
    x = torch.as_tensor(imgs, device=device).permute(0, 3, 1, 2).float() / 255
    x = (x - x.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]) / x.new_tensor(
        [0.229, 0.224, 0.225]
    )[None, :, None, None]
    with torch.autocast(
        torch.device(device).type,
        dtype=torch.float16,
        enabled=str(device).startswith("cuda"),
    ):
        layers = model.get_intermediate_layers(x, n=list(BLOCKS), reshape=True)
    return torch.cat(
        [
            torch.nn.functional.conv2d(layer.float() - mu, proj)
            for layer, (mu, proj) in zip(layers, basis)
        ],
        1,
    ).half()


def extract(scales, device, batch):
    configure(42)
    torch.set_float32_matmul_precision("highest")
    CACHE.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True, parents=True)
    meta = pd.read_csv(BASE / "data/images.csv")
    model, _ = load_encoder(device)
    basis = pca(device)
    for name in scales:
        dest = CACHE / f"{name}.npy"
        size = SCALES[name]
        shape = (len(meta), 128, size[1] // 14, size[0] // 14)
        if dest.exists():
            assert np.load(dest, mmap_mode="r").shape == shape
            continue
        tmp = dest.with_suffix(".partial")
        arr = np.lib.format.open_memmap(tmp, mode="w+", shape=shape, dtype="float16")
        start = 0
        for batch_rgb in DataLoader(
            Images(meta, size), batch_size=batch, num_workers=2, shuffle=False
        ):
            arr[start : start + len(batch_rgb)] = (
                embed(model, batch_rgb, device, basis).cpu().numpy()
            )
            start += len(batch_rgb)
            if start % 200 < batch:
                print(name, start, flush=True)
        assert start == len(meta)
        arr.flush()
        del arr
        tmp.replace(dest)
        dump(
            OUT / f"{name}_features.json",
            dict(
                sha256=WEIGHT_SHA256,
                size=size,
                blocks=[6, 12],
                basis="frozen previous fit-only 448x252 PCA weights",
                dimensions=shape,
                dtype="float16",
            ),
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scales", nargs="+", choices=SCALES, default=["fine", "coarse"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch", type=int, default=3)
    a = p.parse_args()
    extract(a.scales, a.device, a.batch)
