"""Spatial decomposition of the linear classification logit for four test cases."""
import json
import numpy as np
import pandas as pd
import torch
from torchvision.models import resnet18
from course.common import configure, table
from course.predict import normalized
from course.figures import setup, page, picture, save
from experiment2.train import BASE


@torch.inference_mode()
def main():
    configure(42)
    setup()
    base = BASE / "outputs"
    out = base / "followup"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(out / "raw_42/best.pt", map_location=device, weights_only=True)
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(512, 1)
    model.load_state_dict(ck["model"])
    model.eval().to(device)
    meta = pd.read_csv(BASE / "data/images.csv")
    im = np.load(BASE / "data/images.npy", mmap_mode="r")
    scores = np.load(out / "ensemble.npy")
    y = meta.category.eq("disease").to_numpy()
    ids = np.flatnonzero(meta.split.eq("test"))
    threshold = json.loads((out / "selection.json").read_text())["threshold"]
    pred = scores >= threshold
    examples = []
    maps = []
    rows = []
    for name, label, decision in [
        ("True positive", 1, 1),
        ("False positive", 0, 1),
        ("True negative", 0, 0),
        ("False negative", 1, 0),
    ]:
        candidates = ids[(y[ids] == label) & (pred[ids] == decision)]
        i = candidates[
            np.argmax(scores[candidates]) if decision else np.argmin(scores[candidates])
        ]
        x = normalized(im[i], device)
        feat = model.maxpool(model.relu(model.bn1(model.conv1(x))))
        feat = model.layer4(model.layer3(model.layer2(model.layer1(feat))))
        cam = (feat * model.fc.weight[:, :, None, None]).sum(1)[0]
        logit = model.fc(model.avgpool(feat).flatten(1)).item()
        assert abs(float(cam.mean() + model.fc.bias) - logit) < 1e-4
        maps.append(cam.cpu().numpy())
        examples.append((name, i))
        rows.append(
            dict(
                kind=name,
                member=meta.member[i],
                ensemble_probability=float(scores[i]),
                single_model_logit=logit,
                bias=float(model.fc.bias),
                mean_spatial_evidence=float(cam.mean()),
            )
        )
    table(out / "spatial_evidence.csv", rows)
    fig, ax = page(2, 4, size=(9.6, 4.8))
    v = np.abs(np.stack(maps)).max()
    for j, (name, i) in enumerate(examples):
        picture(ax[0, j], im[i], name)
        picture(
            ax[1, j], maps[j], cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest"
        )
        ax[1, j].set_xlabel(meta.member[i].split("/")[-1].split(".")[0], fontsize=8)
    ax[0, 0].set_ylabel("RGB image")
    ax[1, 0].set_ylabel("7 × 7 evidence")
    fig.colorbar(
        ax[1, 0].images[0], ax=ax[1, :], shrink=0.72, label="Local logit contribution"
    )
    save(fig, 2, "spatial_evidence")


if __name__ == "__main__":
    main()
