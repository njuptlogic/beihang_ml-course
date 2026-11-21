"""Predict a native-resolution gaze density using the validation-selected mixture."""
import argparse
import json
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import pandas as pd
from PIL import Image
import torch
from scipy.special import softmax
from torchvision.models import resnet18, ResNet18_Weights
from course.common import configure, dump
from course.predict import normalized, gaze
from course.prepare import resize as make_rgb
from experiment3.train import BASE, resize
from experiment3.semantic_features import (
    OUT,
    BLOCKS,
    encode,
    pixels,
    load_encoder,
    PATCH_GRID,
)
from experiment3.semantic_train import SemanticHead


class Predictor:
    def __init__(self, device):
        self.device = device
        self.cfg = json.loads((OUT / "selection.json").read_text())
        self.variant = self.cfg["variant"]
        self.w = self.cfg["semantic_weight"]
        self.dino, _ = load_encoder(device)
        self.pca = dict(np.load(OUT / "pca.npz"))
        self.router = dict(np.load(OUT / "router.npz"))
        self.models = []
        for seed in self.cfg["seeds"]:
            ck = torch.load(
                OUT / f"{self.variant}_{seed}/best.pt",
                map_location=device,
                weights_only=True,
            )
            model = SemanticHead(self.variant).eval().to(device)
            model.load_state_dict(ck["model"])
            self.models.append(model)
        if "local" in self.variant:
            self.cnn = (
                resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval().to(device)
            )

    @torch.inference_mode()
    def predict(self, blob):
        device = self.device
        raw, cls = encode(self.dino, [pixels(blob)], device)
        features = []
        for b, r in zip(BLOCKS, raw):
            mean = self.pca[f"mean{b+1}"]
            proj = self.pca[f"proj{b+1}"]
            r = r.astype("float16").astype("float32").reshape(384, -1)
            f = (proj @ (r - mean[:, None])).reshape(1, -1, *PATCH_GRID)
            features.append(torch.tensor(f.astype("float16"), device=device).float())
        route = self.router
        probs = softmax(
            ((cls - route["mean"]) / route["std"]) @ route["coef"].T
            + route["intercept"],
            axis=1,
        )
        prior = route["shared"]
        if "routed" in self.variant:
            prior = (probs.astype("float32") @ route["bank"]).reshape(1, 1, 24, 40)
        prior = torch.as_tensor(prior, device=device)
        local = None
        if "local" in self.variant:
            im, _ = make_rgb(blob, "RGB", (320, 192))
            x = normalized(im, device)
            c = self.cnn
            local = (
                c.layer2(c.layer1(c.maxpool(c.relu(c.bn1(c.conv1(x)))))).half().float()
            )
        grid = torch.stack([m(*features, prior, local) for m in self.models]).mean(0)[
            0, 0
        ]
        if self.w < 1.0:
            previous, _ = gaze(blob, BASE / "outputs", device)
            native_grid = torch.as_tensor(previous[0, 0], device=device)
            grid = self.w * grid + (1 - self.w) * native_grid
        with Image.open(BytesIO(blob)) as im:
            width, height = im.size
        full = resize(grid[None, None], (height, width))[0, 0]
        return grid.cpu().numpy(), (full / full.sum()).cpu().numpy(), probs[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--verify", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    configure(42)
    if not a.verify and (a.image is None or a.output is None):
        p.error("Supply --image and --output, or --verify.")
    model = Predictor(a.device)
    if a.verify:
        truth = np.load(OUT / "selected.npy", mmap_mode="r")
        meta = pd.read_csv(BASE / "data/images.csv")
        rows = []
        for idx in (0, 819, 1600, 1999):
            r = meta.iloc[idx]
            with ZipFile(BASE / "data/raw" / r.archive) as z:
                pred, full, prob = model.predict(z.read(r.member))
            tv = float(np.abs(pred - truth[idx, 0]).sum() / 2)
            mass = float(full.sum())
            assert tv < 0.003 and abs(mass - 1) < 2e-5
            rows.append(
                dict(
                    index=idx, member=r.member, total_variation=tv, probability_sum=mass
                )
            )
        dump(OUT / "inference_check.json", rows)
        print(rows)
    if a.image is not None and a.output is not None:
        _, full, _ = model.predict(a.image.read_bytes())
        a.output.parent.mkdir(exist_ok=True, parents=True)
        if a.output.suffix == ".npy":
            np.save(a.output, full)
        else:
            Image.fromarray(np.rint(full / full.max() * 255).astype("uint8")).save(
                a.output
            )


if __name__ == "__main__":
    main()
