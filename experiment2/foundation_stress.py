"""Fixed-threshold perturbation study on the same 250 test images."""
import json
from io import BytesIO
from zipfile import ZipFile
import joblib
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18

from course.common import configure, table
from course.figures import BLUE, TEAL, GRAY, axes_title, setup, page, save
from course.prepare import resize
from experiment2.retinal_features import VIEWS, views, backbone, encode
from experiment2.foundation import FEATURES, OUT, probability
from experiment2.train import BASE, scores

VARIANTS = ("clean", "dim", "gamma", "coarse", "jpeg")
NAMES = ("Original", "Gain × 0.65", "Gamma 1.6", "96 px", "JPEG Q20")


def distort(blob, condition):
    if condition == "clean":
        return blob
    with Image.open(BytesIO(blob)) as source:
        image = source.convert("RGB")
        if condition == "coarse":
            image = image.resize((96, 96), Image.Resampling.BILINEAR).resize(
                source.size, Image.Resampling.BILINEAR
            )
        elif condition == "dim":
            image = image.point([round(i * 0.65) for i in range(256)] * 3)
        elif condition == "gamma":
            image = image.point([round(255 * (i / 255) ** 1.6) for i in range(256)] * 3)
        stream = BytesIO()
        if condition == "jpeg":
            image.save(stream, format="JPEG", quality=20)
        else:
            image.save(stream, format="BMP")
        return stream.getvalue()


class StressViews(Dataset):
    def __init__(self, meta, condition):
        self.rows = meta.to_dict("records")
        self.condition = condition
        self.zip = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        if self.zip is None:
            self.zip = ZipFile(BASE / "data/raw" / row["archive"])
        b = distort(self.zip.read(row["member"]), self.condition)
        imgs, _ = views(b)
        rgb, _ = resize(b, "RGB", (224, 224))
        return torch.from_numpy(imgs).permute(0, 3, 1, 2), torch.from_numpy(
            rgb.copy()
        ).permute(2, 0, 1)


@torch.inference_mode()
def evaluate():
    configure(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = pd.read_csv(BASE / "data/images.csv").query("split=='test'")
    y = meta.category.eq("disease").to_numpy()
    name = json.loads((OUT / "selection.json").read_text())["feature"]
    est, cfg = joblib.load(OUT / f"{name}.joblib")
    spec = FEATURES[name]
    kind = spec[0].split("_", 1)[0]
    if kind == "resnet":
        raise ValueError("This comparison is defined for a selected foundation model.")
    feature, _ = backbone(kind, device)
    old_dir = BASE / "outputs/followup"
    old_cfg = json.loads((old_dir / "selection.json").read_text())
    assert old_cfg["variant"] == "raw"
    old_models = []
    for seed in old_cfg["seeds"]:
        model = resnet18(weights=None)
        model.fc = torch.nn.Linear(512, 1)
        model.load_state_dict(
            torch.load(
                old_dir / f"raw_{seed}/best.pt", map_location="cpu", weights_only=True
            )["model"]
        )
        old_models.append(model.eval().to(device, memory_format=torch.channels_last))
    preds = {}
    rows = []
    details = []
    for condition in VARIANTS:
        old = []
        new = []
        dl = DataLoader(StressViews(meta, condition), batch_size=6, num_workers=2)
        for im, rgb in dl:
            x = np.concatenate(
                [
                    encode(
                        feature, im[:, VIEWS.index(k.split("_", 1)[1])], kind, device
                    )
                    for k in spec
                ],
                axis=1,
            )
            new.append(probability(est, cfg, x))
            rgb = rgb.to(device) / 255
            rgb = (
                rgb - rgb.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
            ) / rgb.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
            with torch.autocast(
                device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                old.append(
                    torch.stack(
                        [m(rgb).flatten().float().sigmoid() for m in old_models]
                    )
                    .mean(0)
                    .cpu()
                    .numpy()
                )
        preds[condition] = (np.concatenate(old), np.concatenate(new))
        for j, (model, threshold) in enumerate(
            (("previous_resnet", old_cfg["threshold"]), (name, cfg["threshold"]))
        ):
            p = preds[condition][j]
            clean = preds["clean"][j]
            rows.append(
                dict(
                    condition=condition,
                    model=model,
                    flips=int(np.sum((p >= threshold) != (clean >= threshold))),
                    mean_abs_score_change=np.abs(p - clean).mean(),
                    **scores(y, p, threshold),
                )
            )
            for idx, prob in zip(meta.index, p):
                details.append(
                    dict(
                        index=int(idx),
                        condition=condition,
                        model=model,
                        probability=prob,
                        threshold=threshold,
                    )
                )
        print(condition, flush=True)
    table(OUT / "stress_metrics.csv", rows)
    table(OUT / "stress_predictions.csv", details)
    original = pd.read_csv(OUT / "predictions.csv").query("split=='test'")
    for j, model in enumerate(("previous_resnet", name)):
        errors = abs(original[model].to_numpy() - preds["clean"][j])
        assert errors.max() < 0.01, (model, errors.max())
    mat = pd.DataFrame(rows)
    setup()
    fig, axes = page(1, 3, size=(12.3, 3.7))
    for m, lab, c, marker in (
        ("previous_resnet", "Earlier ensemble", BLUE, "o"),
        (name, "Selected probe", TEAL, "s"),
    ):
        subset = mat[mat.model == m].set_index("condition").loc[list(VARIANTS)]
        for a, key, t, yl in zip(
            axes,
            ["auc", "balanced_accuracy", "flips"],
            ["A   Ranking", "B   Operating point", "C   Decision stability"],
            ["AUROC", "Balanced accuracy", "Changed labels / 250"],
        ):
            ax_x = np.arange(len(VARIANTS))
            a.plot(
                ax_x,
                subset[key],
                marker=marker,
                color=c,
                label=lab,
                ms=5,
                ls="--" if m == "previous_resnet" else "-",
            )
            a.set_xticks(ax_x, NAMES, rotation=24, ha="right")
            axes_title(a, t, y=yl)
    axes[0].legend(fontsize=8)
    save(fig, 2, "foundation_stress")


if __name__ == "__main__":
    evaluate()
