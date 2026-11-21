"""Cache compact spatial targets for original and flipped CIFAR training images."""
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18
from course.common import configure, dump
from experiment4.train import BASE, data

ROOT = BASE / "outputs/research"


@torch.inference_mode()
def build_teacher(out, device):
    configure(344)
    out.mkdir(parents=True, exist_ok=True)
    images, _ = data(True)
    network = torch.nn.Sequential(
        *list(resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).children())[:-2]
    ).eval()
    network.to(device, memory_format=torch.channels_last)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def encode(batch):
        rgb = F.interpolate(
            batch.to(device).float() / 255,
            size=(128, 128),
            mode="bilinear",
            align_corners=False,
        )
        with torch.autocast(
            device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            feat = network((rgb - mean) / std)
        return feat.float().permute(0, 2, 3, 1)

    ids = torch.randperm(len(images), generator=torch.Generator().manual_seed(344))[
        :2048
    ]
    fit = torch.cat([encode(images[idx]).cpu() for idx in ids.split(40)]).reshape(
        -1, 512
    )
    center = fit.mean(0)
    centered = fit - center
    eigen, basis = torch.linalg.eigh(centered.T @ centered / (len(fit) - 1))
    basis = basis[:, -64:].flip(1).contiguous()
    saved = {"mean": center, "basis": basis, "fit_indices": ids}
    torch.save(saved, out / "teacher_projection.pt")
    retained = float(eigen[-64:].sum() / eigen.sum())
    center, basis = center.to(device), basis.to(device)
    tmp = out / "teacher.tmp.npy"
    bank = np.lib.format.open_memmap(
        tmp, mode="w+", dtype=np.float16, shape=(len(images), 2, 64, 4, 4)
    )
    for i in range(0, len(images), 40):
        batch = images[i : i + 40]
        for flip in range(2):
            features = encode(batch.flip(-1) if flip else batch)
            projected = F.normalize((features - center) @ basis, dim=-1)
            bank[i : i + len(batch), flip] = projected.permute(0, 3, 1, 2).cpu().numpy()
        if i % 10000 == 0:
            print(f"{i}/{len(images)}", flush=True)
    bank.flush()
    del bank
    tmp.replace(out / "teacher.npy")
    dump(
        out / "teacher.json",
        dict(
            teacher="torchvision ResNet18_Weights.IMAGENET1K_V1",
            features="layer4 output",
            resolution=128,
            grid=4,
            views=["original", "horizontal_flip"],
            n=50000,
            pca_dim=64,
            pca_fit=2048,
            pca_seed=344,
            explained_variance=retained,
        ),
    )
    print("Cached teacher targets", retained, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    build_teacher(args.out, torch.device(args.device))
