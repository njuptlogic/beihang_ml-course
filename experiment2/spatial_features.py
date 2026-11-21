"""Read global and target-centred retinal descriptors from the frozen ViT-S/14."""
import argparse
from io import BytesIO
from zipfile import ZipFile

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter
import torch
from torch.utils.data import DataLoader, Dataset

from course.common import configure, dump
from experiment2.retinal_features import backbone, SIDE
from experiment2.train import BASE

CACHE = BASE / "data/spatial"
OUT = BASE / "outputs/spatial"
VIEWS = ("field", "local", "balanced")
POOLS = ("cls", "mean", "core", "rim")


def get_views(blob):
    with Image.open(BytesIO(blob)) as original:
        image = original.convert("RGB")
    small = (
        np.asarray(image.resize((96, 96), Image.Resampling.BILINEAR), dtype=np.float32)
        / 255
    )
    green = small[:, :, 1]
    red = small[:, :, 0]
    smooth = gaussian_filter(green, 2.6)
    # A bright area that rises above its surroundings. The full horizontal field
    # is eligible because lateral discs can touch the edge of a square photo.
    score = (
        smooth - 0.72 * gaussian_filter(green, 12) + 0.12 * gaussian_filter(red, 2.6)
    )
    eligible = (np.arange(96)[None, :] >= 3) & (np.arange(96)[None, :] <= 92)
    eligible = (
        eligible & (np.arange(96)[:, None] >= 12) & (np.arange(96)[:, None] <= 83)
    )
    yy, xx = np.unravel_index(np.where(eligible, score, -np.inf).argmax(), score.shape)
    cx, cy = (xx + 0.5) / 96, (yy + 0.5) / 96
    w, h = image.size
    # Reflect-pad the short field beyond the photographic border, so the centre
    # of every local descriptor refers to the detected location.
    side = round(0.42 * min(w, h))
    x0, y0 = round(cx * w - side / 2), round(cy * h - side / 2)
    x1, y1 = x0 + side, y0 + side
    interior = np.array(image.crop((max(0, x0), max(0, y0), min(w, x1), min(h, y1))))
    local = np.pad(
        interior,
        ((max(0, -y0), max(0, y1 - h)), (max(0, -x0), max(0, x1 - w)), (0, 0)),
        mode="reflect",
    )
    local = np.asarray(
        Image.fromarray(local).resize((SIDE, SIDE), Image.Resampling.BILINEAR)
    )
    lo, hi = np.percentile(local[::4, ::4], [2, 98], axis=(0, 1))
    balanced = (
        np.clip((local - lo) / (hi - lo).clip(1) * 255, 0, 255).round().astype("uint8")
    )
    field = np.asarray(image.resize((SIDE, SIDE), Image.Resampling.BILINEAR))
    views = np.stack([field, local, balanced])
    if cx < 0.5:
        views = np.ascontiguousarray(views[:, :, ::-1])
    return views, np.array([cx, cy], dtype="float32")


class Fundus(Dataset):
    def __init__(self, meta):
        self.rows = meta.to_dict("records")
        self.archives = {}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        name = r["archive"]
        if name not in self.archives:
            self.archives[name] = ZipFile(BASE / "data/raw" / name)
        imgs, center = get_views(self.archives[name].read(r["member"]))
        return torch.from_numpy(imgs).permute(0, 3, 1, 2), center


@torch.inference_mode()
def encode(model, x, device, normalization="natural"):
    x = x.to(device, dtype=torch.float32) / 255
    if normalization == "natural":
        x = (
            x - x.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        ) / x.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
    else:
        x = (x - 0.5) * 2
    with torch.autocast(
        device.type, dtype=torch.float16, enabled=device.type == "cuda"
    ):
        h = model.forward_features(x)
    h = h.float()
    patch = h[:, model.num_prefix_tokens :].reshape(-1, 28, 28, 384)
    c = (torch.arange(28, device=device) + 0.5 - 14) / 28
    radius = (c[None, :].square() + c[:, None].square()).sqrt()
    return (
        torch.stack(
            [
                h[:, 0],
                patch.mean((1, 2)),
                patch[:, radius < 0.23].mean(1),
                patch[:, (radius >= 0.23) & (radius < 0.42)].mean(1),
            ],
            dim=1,
        )
        .cpu()
        .numpy()
    )


def extract(name, device, batch):
    configure(42)
    device = torch.device(device)
    enc, sha = backbone(name, device)
    meta = pd.read_csv(BASE / "data/images.csv")
    loader = DataLoader(Fundus(meta), batch_size=batch, num_workers=2)
    CACHE.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True, parents=True)
    banks = []
    centers = []
    for k, (imgs, center) in enumerate(loader):
        banks.append(
            np.stack([encode(enc, imgs[:, j], device, name) for j in range(3)], 1)
        )
        centers.append(center.numpy())
        if k % 45 == 0:
            print(name, k * batch, len(meta), flush=True)
    target = CACHE / f"{name}.npz"
    tmp = target.with_suffix(".tmp.npz")
    np.savez(
        tmp,
        embeddings=np.concatenate(banks),
        center=np.concatenate(centers),
        index=meta["index"].to_numpy(),
    )
    tmp.replace(target)
    dump(
        CACHE / f"{name}.json",
        dict(
            backbone=name,
            sha256=sha,
            views=VIEWS,
            pools=POOLS,
            shape=list(np.concatenate(banks).shape),
            input_resolution=SIDE,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("name", choices=("natural", "retina"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch", default=4, type=int)
    args = p.parse_args()
    extract(args.name, args.device, args.batch)
