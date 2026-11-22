"""Rebuild the selected density from RGB with frozen encoders and saved course readouts."""
import argparse, json
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision.models import resnet18, ResNet18_Weights
from course.common import configure, dump
from course.predict import normalized
from course.prepare import resize as small_rgb
from experiment3.train import BASE, resize
from experiment3.multiscale_features import (
    OUT,
    SCALES,
    embed,
    pca,
    pixels,
    load_encoder,
)
from experiment3.semantic_features import (
    encode as mid_encode,
    pixels as mid_pixels,
    OUT as PREV,
)
from experiment3.semantic_predict import Predictor as Previous
from experiment3.multiscale_train import load_head, OUTPUT_GRID
from experiment3.anchor_calibration import Calibrator
from experiment3 import external_saliency as ex
from experiment3 import perception_features as pe


class Predictor:
    def __init__(self, device, head_directory=OUT):
        self.device = device
        self.cfg = json.loads((OUT / "selection.json").read_text())
        self.mix = self.cfg["weights"]
        self.backbone = json.loads((OUT / "backbone_selection.json").read_text())
        self.sub = self.backbone["weights"] if "dino_selected" in self.mix else {}
        self.models = {
            arm: [load_head(arm, s, device, head_directory) for s in self.backbone["seeds"]]
            for arm in self.sub
            if arm != "previous"
        }
        self.basis = dict(np.load(head_directory / "basis.npz"))
        self.pca0 = dict(np.load(PREV / "pca.npz"))
        self.whiten = pca(device)
        self.anchor = (
            ex.load(device) if any(n != "dino_selected" for n in self.mix) else None
        )
        self.dino, _ = load_encoder(device)
        self.local = (
            resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval().to(device)
            if self.models
            else None
        )
        self.cal = {}
        for name in self.mix:
            if name.startswith("cal_"):
                adaptive = name == "cal_adaptive"
                self.cal[name] = []
                for seed in [17, 42, 73] if adaptive else [17]:
                    m = Calibrator(adaptive).to(device).eval()
                    m.load_state_dict(
                        torch.load(
                            OUT / f"{name}_{seed}/best.pt",
                            map_location=device,
                            weights_only=True,
                        )["model"]
                    )
                    self.cal[name].append(m)
        self.previous = (
            Previous(device)
            if any(k in self.sub for k in ("previous", "residual"))
            else None
        )
        pe_names = set(k[3:] for k in self.models if k in ("pe_core", "pe_spatial"))
        if "dual" in self.models:
            pe_names.add("spatial")
        self.perception = {
            k: (pe.load(k, device), dict(np.load(OUT / f"pe_{k}_pca.npz")))
            for k in pe_names
        }

    @torch.inference_mode()
    def one(self, blob):
        device = self.device
        features, cls = mid_encode(self.dino, [mid_pixels(blob)], device)
        ctx = torch.tensor(
            (cls - self.basis["cls_mean"]) @ self.basis["cls_matrix"], device=device
        )
        prior = torch.as_tensor(self.basis["prior"], device=device)
        p = 0
        if self.sub:
            if self.previous:
                old = torch.as_tensor(self.previous.predict(blob)[0], device=device)[
                    None, None
                ]
                prev = resize(old, OUTPUT_GRID)
                p = self.sub.get("previous", 0) * prev / prev.sum()
            else:
                old = None
            if self.models:
                tensors = []
                for i, block in enumerate((6, 12)):
                    raw = (
                        features[i].astype("float16").astype("float32").reshape(384, -1)
                    )
                    data = (
                        self.pca0[f"proj{block}"]
                        @ (raw - self.pca0[f"mean{block}"][:, None])
                    ).reshape(1, 64, 18, 32)
                    tensors.append(
                        torch.as_tensor(data.astype("float16"), device=device).float()
                    )
                mid = torch.cat(tensors, 1)
                im, _ = small_rgb(blob, "RGB", (320, 192))
                c = self.local
                x = normalized(im, device)
                texture = (
                    c.layer2(c.layer1(c.maxpool(c.relu(c.bn1(c.conv1(x))))))
                    .half()
                    .float()
                )
                need = {
                    v
                    for k, v in [("coarse", SCALES["coarse"]), ("fine", SCALES["fine"])]
                    if any(m[0].multiscale for m in self.models.values())
                }
                scales = {
                    size: embed(
                        self.dino, np.stack([pixels(blob, size)]), device, self.whiten
                    ).float()
                    for size in need
                }
                pe_features = {}
                for k, (encoder, basis) in self.perception.items():
                    raw = (
                        pe.encode(encoder, np.stack([pixels(blob, pe.SIZE)]), device)
                        .float()
                        .cpu()
                        .numpy()
                    )
                    part = [
                        ((raw[0, i] - basis[f"mean{i}"]) @ basis[f"proj{i}"]).T.reshape(
                            64, 18, 32
                        )
                        for i in range(2)
                    ]
                    pe_features[k] = torch.tensor(
                        np.concatenate(part).astype("float16"), device=device
                    ).float()[None]
                for arm, heads in self.models.items():
                    m = (
                        pe_features[arm[3:]]
                        if arm in ("pe_core", "pe_spatial")
                        else mid
                    )
                    coarse = (
                        pe_features["spatial"]
                        if arm == "dual"
                        else scales.get(SCALES["coarse"])
                    )
                    fine = scales.get(SCALES["fine"])
                    p = p + self.sub[arm] * torch.stack(
                        [h(m, texture, coarse, fine, ctx, prior, old) for h in heads]
                    ).mean(0)
            p = p * self.mix["dino_selected"]
        if self.anchor:
            anchor = ex.infer(self.anchor, np.stack([pixels(blob, ex.SIZE)]), device)
            for name, w in self.mix.items():
                if name == "unisal":
                    p = p + w * anchor
                if name in self.cal:
                    p = p + w * torch.stack(
                        [m(anchor, ctx, prior) for m in self.cal[name]]
                    ).mean(0)
        with Image.open(BytesIO(blob)) as im:
            width, height = im.size
        full = resize(p, (height, width))[0, 0]
        return p[0, 0].cpu().numpy(), (full / full.sum()).cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--verify", action="store_true")
    p.add_argument("--image", type=Path)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    configure(42)
    pred = Predictor(a.device)
    if a.verify:
        meta = pd.read_csv(BASE / "data/images.csv")
        truth = np.load(OUT / "selected.npy", mmap_mode="r")
        rows = []
        for idx in (0, 819, 1600, 1999):
            r = meta.iloc[idx]
            with ZipFile(BASE / "data/raw" / r.archive) as z:
                small, full = pred.one(z.read(r.member))
            tv = float(np.abs(small - truth[idx, 0]).sum() / 2)
            mass = float(full.sum())
            assert tv < 0.004 and abs(mass - 1) < 2e-5, (idx, tv, mass)
            rows.append(dict(index=idx, member=r.member, total_variation=tv, mass=mass))
        dump(OUT / "inference_check.json", rows)
        print(rows)
    if a.image and a.output:
        _, full = pred.one(a.image.read_bytes())
        a.output.parent.mkdir(exist_ok=True, parents=True)
        if a.output.suffix == ".npy":
            np.save(a.output, full)
        else:
            Image.fromarray(np.rint(255 * full / full.max()).astype("uint8")).save(
                a.output
            )
    elif not a.verify:
        p.error("Provide --verify or --image and --output.")


if __name__ == "__main__":
    main()
