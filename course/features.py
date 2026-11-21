"""Extract frozen ImageNet ResNet-18 stages, preserving spatial coordinates."""
import argparse
import numpy as np
import torch
from torchvision.models import resnet18, ResNet18_Weights
from course.common import ROOT, configure


@torch.inference_mode()
def extract(number, device):
    configure(42)
    root = ROOT / f"experiment{number}" / "data"
    images = np.load(root / "images.npy", mmap_mode="r")
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval().to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]
    files = {}
    for start in range(0, len(images), 16):
        batch = (
            torch.from_numpy(images[start : start + 16].copy())
            .permute(0, 3, 1, 2)
            .to(device)
            / 255
        )
        x = model.maxpool(model.relu(model.bn1(model.conv1((batch - mean) / std))))
        for stage, layer in enumerate(
            [model.layer1, model.layer2, model.layer3, model.layer4], 1
        ):
            x = layer(x)
            if stage >= (3 if number == 2 else 2):
                feat = x
                if number == 2 and stage == 3:
                    feat = torch.nn.functional.avg_pool2d(feat, 2)
                arr = feat.half().cpu().numpy()
                if stage not in files:
                    files[stage] = np.lib.format.open_memmap(
                        root / f"stage{stage}.tmp",
                        mode="w+",
                        dtype="float16",
                        shape=(len(images), *arr.shape[1:]),
                    )
                files[stage][start : start + len(arr)] = arr
    for stage, f in files.items():
        f.flush()
        (root / f"stage{stage}.tmp").replace(root / f"stage{stage}.npy")
    print("Features ready", number, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("experiment", type=int, choices=[2, 3])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    extract(a.experiment, a.device)
