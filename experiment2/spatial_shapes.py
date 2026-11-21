"""Geometric descriptors of the relative content of local retinal patches."""
import argparse

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from experiment2.spatial_features import Fundus, CACHE
from experiment2.retinal_features import backbone
from experiment2.train import BASE
from course.common import configure, dump


@torch.inference_mode()
def gram(model, image, device, normalization="natural"):
    x = image.to(device, dtype=torch.float32) / 255
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
    grid = (
        h[:, model.num_prefix_tokens :]
        .float()
        .reshape(-1, 28, 28, 384)
        .permute(0, 3, 1, 2)
    )
    pooled = F.avg_pool2d(grid, 4).flatten(2).transpose(1, 2)
    u = F.normalize(pooled, dim=-1)
    similarity = u @ u.transpose(1, 2)
    tri = torch.triu_indices(49, 49, offset=1, device=device)
    return similarity[:, tri[0], tri[1]].cpu().numpy()


def extract(name, device, batch):
    configure(42)
    device = torch.device(device)
    model, sha = backbone(name, device)
    meta = pd.read_csv(BASE / "data/images.csv")
    loader = DataLoader(Fundus(meta), batch_size=batch, num_workers=2)
    result = []
    for k, (imgs, _) in enumerate(loader):
        result.append(
            np.stack([gram(model, imgs[:, j], device, name) for j in (1, 2)], 1)
        )
        if k % 60 == 0:
            print("Gram", name, k * batch, flush=True)
    tmp = CACHE / f"{name}_gram.tmp.npz"
    np.savez(tmp, gram=np.concatenate(result), index=meta["index"].to_numpy())
    tmp.replace(CACHE / f"{name}_gram.npz")
    dump(
        CACHE / f"{name}_gram.json",
        dict(
            backbone=name,
            sha256=sha,
            order="strict upper triangle of row-major 7x7 cosine self-similarity; local RGB then balanced",
            pairs=1176,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("name", choices=["natural", "retina"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch", default=4, type=int)
    a = p.parse_args()
    extract(a.name, a.device, a.batch)
