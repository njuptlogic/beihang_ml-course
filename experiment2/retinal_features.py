"""Extract paired views from original JPEGs with RETFound-Green and DINOv2-reg."""
import argparse
import hashlib
from io import BytesIO
from zipfile import ZipFile

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
import requests
import timm
import torch
from torch.utils.data import DataLoader, Dataset
from timm.models.vision_transformer import checkpoint_filter_fn
from course.common import configure, dump
from experiment2.train import BASE

CACHE = BASE / "data/foundation"
RET = "retfoundgreen_statedict.pth"
DINO = "dinov2_vits14_reg4_pretrain.pth"
WEIGHTS = {
    "retina": (
        RET,
        "https://github.com/justinengelmann/RETFound_Green/releases/download/v0.1/"
        + RET,
        "431de5dbc1bebbb32f60e2c0bcf8daa4f8bcbf06f7cb1e1dc97ec589713942e1",
    ),
    "natural": (
        DINO,
        "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/" + DINO,
        "f433177089a681826f849f194ece3bb48f4d63fb38d32fc837e3dc7a4e5641fb",
    ),
}
SIDE = 392
VIEWS = ("field", "crop", "balanced_crop")


def weight_path(name):
    filename, url, expected = WEIGHTS[name]
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / filename
    if not path.exists():
        tmp = path.with_suffix(".part")
        with requests.get(url, stream=True, timeout=(15, 120)) as response, tmp.open(
            "wb"
        ) as handle:
            response.raise_for_status()
            for chunk in response.iter_content(1 << 20):
                handle.write(chunk)
        if expected and hashlib.sha256(tmp.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Unexpected checksum for {filename}")
        tmp.replace(path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected and sha != expected:
        raise ValueError(f"Unexpected checksum for {filename}")
    return path, sha


def backbone(name, device):
    path, sha = weight_path(name)
    model = timm.create_model(
        "vit_small_patch14_reg4_dinov2", img_size=SIDE, num_classes=0
    )
    state = torch.load(path, map_location="cpu", weights_only=True)
    if name == "natural":
        state = checkpoint_filter_fn(state, model)
    model.load_state_dict(state, strict=True)
    model.global_pool = "avg"
    return model.to(device).eval(), sha


def views(blob):
    with Image.open(BytesIO(blob)) as im:
        im = im.convert("RGB")
        width, height = im.size
        field = im.resize((SIDE, SIDE), Image.Resampling.BILINEAR)
        small = im.resize((64, 64), Image.Resampling.BILINEAR).filter(
            ImageFilter.GaussianBlur(3)
        )
        green = np.asarray(small)[:, :, 1].astype(float)
        # The optic disc is often bright. Avoid the outer 20% and softly favour central locations.
        yy, xx = np.mgrid[:64, :64]
        xx = (xx + 0.5) / 64
        yy = (yy + 0.5) / 64
        score = green / 255 - 0.2 * ((xx - 0.5) ** 2 + (yy - 0.5) ** 2)
        score[(xx < 0.2) | (xx > 0.8) | (yy < 0.2) | (yy > 0.8)] = -np.inf
        row, col = np.unravel_index(score.argmax(), score.shape)
        cx, cy = float(xx[row, col]), float(yy[row, col])
        side = round(0.55 * min(width, height))
        x0 = int(np.clip(round(cx * width - side / 2), 0, width - side))
        y0 = int(np.clip(round(cy * height - side / 2), 0, height - side))
        crop = np.array(
            im.crop((x0, y0, x0 + side, y0 + side)).resize(
                (SIDE, SIDE), Image.Resampling.BILINEAR
            )
        )
    # Use the per-view channel range; fixed quantiles keep the transform independent of labels.
    lo, hi = np.percentile(crop[::4, ::4], (2, 98), axis=(0, 1))
    balanced = (
        np.clip((crop - lo) * 255 / np.maximum(hi - lo, 1), 0, 255)
        .round()
        .astype(np.uint8)
    )
    box = np.array(
        [x0 / width, y0 / height, (x0 + side) / width, (y0 + side) / height],
        dtype=np.float32,
    )
    return np.stack((np.asarray(field), crop, balanced)), box


class RetinaImages(Dataset):
    def __init__(self, meta):
        self.rows = meta.to_dict("records")
        self.archives = {}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        name = row["archive"]
        if name not in self.archives:
            self.archives[name] = ZipFile(BASE / "data/raw" / name)
        im, box = views(self.archives[name].read(row["member"]))
        return torch.from_numpy(im).permute(0, 3, 1, 2), torch.from_numpy(box)


@torch.inference_mode()
def encode(model, imgs, name, device):
    x = imgs.to(device).float().div_(255)
    if name == "retina":
        x = x.sub_(0.5).mul_(2.0)
    else:
        x = (
            x - x.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        ) / x.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
    with torch.autocast(
        device.type, dtype=torch.float16, enabled=device.type == "cuda"
    ):
        return model(x).float().cpu().numpy()


def extract(name, device, batch):
    configure(42)
    device = torch.device(device)
    meta = pd.read_csv(BASE / "data/images.csv")
    model, sha = backbone(name, device)
    ds = RetinaImages(meta)
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=2)
    keep = VIEWS if name == "retina" else ("field",)
    features = {k: [] for k in keep}
    boxes = []
    for i, (inputs, box) in enumerate(loader):
        for j, k in enumerate(keep):
            features[k].append(encode(model, inputs[:, j], name, device))
        boxes.append(box.numpy())
        if i % 50 == 0:
            print(name, i * batch, len(ds), flush=True)
    tmp = CACHE / f"{name}.tmp.npz"
    np.savez(
        tmp,
        **{k: np.concatenate(v) for k, v in features.items()},
        boxes=np.concatenate(boxes),
        index=meta["index"].to_numpy(),
    )
    tmp.replace(CACHE / f"{name}.npz")
    dump(
        CACHE / f"{name}.json",
        dict(
            weight=WEIGHTS[name][0],
            url=WEIGHTS[name][1],
            sha256=sha,
            timm=timm.__version__,
            resolution=SIDE,
            parameters=sum(p.numel() for p in model.parameters()),
            images=len(ds),
            views=keep,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("backbone", choices=WEIGHTS)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch", type=int, default=6)
    a = p.parse_args()
    extract(a.backbone, a.device, a.batch)
