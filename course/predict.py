"""Load trained weights for one retinal photograph, one gaze map, or a new CIFAR sample."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torchvision.models import resnet18, ResNet18_Weights
from course.common import ROOT, configure
from course.prepare import resize


@torch.inference_mode()
def retina(blob, out, device):
    from experiment2.followup import local_contrast

    info = json.loads((out / "selection.json").read_text())
    rgb, _ = resize(blob, "RGB", (224, 224))
    if info["variant"] == "local_contrast":
        rgb = local_contrast(rgb)
    x = normalized(rgb, device)
    scores = []
    for seed in info["seeds"]:
        ck = torch.load(
            out / f"{info['variant']}_{seed}/best.pt",
            map_location=device,
            weights_only=True,
        )
        model = resnet18(weights=None)
        model.fc = torch.nn.Linear(512, 1)
        model.load_state_dict(ck["model"])
        model.to(device, memory_format=torch.channels_last).eval()
        with torch.autocast(
            torch.device(device).type, dtype=torch.bfloat16, enabled=device == "cuda"
        ):
            scores.append(float(model(x).float().sigmoid().item()))
    p = float(np.mean(scores))
    return dict(
        probability=p,
        threshold=info["threshold"],
        label="disease" if p >= info["threshold"] else "normal",
    )


def normalized(rgb, device):
    x = torch.from_numpy(rgb.copy()).to(device).permute(2, 0, 1)[None] / 255.0
    return (
        x - x.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
    ) / x.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]


@torch.inference_mode()
def gaze(blob, out, device):
    from experiment3.train import DensityHead, resize as up

    info = json.loads((out / "selection.json").read_text())
    rgb, (w, h) = resize(blob, "RGB", (320, 192))
    enc = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval().to(device)
    x = enc.layer1(enc.maxpool(enc.relu(enc.bn1(enc.conv1(normalized(rgb, device))))))
    stages = []
    for layer in (enc.layer2, enc.layer3, enc.layer4):
        x = layer(x)
        stages.append(x.half().float())
    densities = []
    for seed in info["seeds"]:
        ck = torch.load(
            out / f"{info['variant']}_{seed}/best.pt",
            map_location=device,
            weights_only=True,
        )
        model = DensityHead(info["variant"], torch.zeros(1, 1, 24, 40)).to(device)
        model.load_state_dict(ck["model"])
        densities.append(model.eval()(stages))
    grid = torch.stack(densities).mean(0)
    result = up(grid, (h, w))[0, 0]
    return grid.cpu().numpy(), (result / result.sum()).cpu().numpy()


@torch.inference_mode()
def generate(out, seed, category, device, steps=12, guidance=1.6):
    from experiment4.model import make_model, sample

    ck = torch.load(out / "flow.pt", map_location=device, weights_only=True)
    model = (
        make_model(ck["config"].get("parameterization", "direct"))
        .to(device, memory_format=torch.channels_last)
        .eval()
    )
    model.load_state_dict(ck["ema"])
    noise = torch.randn(1, 3, 32, 32, generator=torch.Generator().manual_seed(seed)).to(
        device
    )
    return (
        sample(model, noise, torch.tensor([category], device=device), steps, guidance)[
            0
        ]
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    sub = p.add_subparsers(dest="task", required=True)
    r = sub.add_parser("retina")
    r.add_argument("image", type=Path)
    r.add_argument("--run", type=Path, default=ROOT / "experiment2/outputs/followup")
    g = sub.add_parser("gaze")
    g.add_argument("image", type=Path)
    g.add_argument("output", type=Path)
    g.add_argument("--run", type=Path, default=ROOT / "experiment3/outputs")
    c = sub.add_parser("generate")
    c.add_argument("output", type=Path)
    c.add_argument("--run", type=Path, default=ROOT / "experiment4/outputs")
    c.add_argument("--seed", type=int, default=719)
    c.add_argument("--class-id", type=int, choices=range(10), default=0)
    c.add_argument("--steps", type=int, default=12)
    a = p.parse_args()
    configure(42)
    if a.task == "retina":
        print(json.dumps(retina(a.image.read_bytes(), a.run, a.device), indent=2))
        return
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.task == "gaze":
        _, array = gaze(a.image.read_bytes(), a.run, a.device)
        if a.output.suffix == ".npy":
            np.save(a.output, array)
            return
        array = np.round(255 * array / array.max()).astype("uint8")
    else:
        if a.steps < 1:
            p.error("--steps must be positive")
        array = generate(a.run, a.seed, a.class_id, a.device, a.steps)
    Image.fromarray(array).save(a.output)


if __name__ == "__main__":
    main()
