"""Exploratory follow-up after the frozen-head experiment: image size and illumination."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from course.common import configure, table, batches, dump
from experiment2.train import BASE, scores, choose_threshold, SEEDS


def local_contrast(im):
    bg = np.asarray(Image.fromarray(im).filter(ImageFilter.GaussianBlur(8))).astype(
        np.float32
    )
    return np.clip(128 + 3 * (im.astype(np.float32) - bg), 0, 255).astype(np.uint8)


def main(out, device):
    df = pd.read_csv(BASE / "data/images.csv")
    out.mkdir(parents=True, exist_ok=True)
    fit = df.index[df.split.eq("fit")].to_numpy()
    val = df.index[df.split.eq("validation")].to_numpy()
    test = df.index[df.split.eq("test")].to_numpy()
    parts = {"fit": fit, "validation": val, "train_full": np.r_[fit, val], "test": test}
    y = df.category.eq("disease").astype(int).to_numpy()
    rows = []

    def save_result(name, seed, p, epoch):
        thr = choose_threshold(y[val], p[val])
        for split, ids in parts.items():
            rows.append(
                dict(
                    model=name,
                    seed=seed,
                    epoch=epoch,
                    split=split,
                    threshold=thr,
                    **scores(y[ids], p[ids], thr),
                )
            )
        table(out / "metrics.csv", rows)
        return thr

    fitframe = df.loc[fit].assign(label=y[fit])
    sizes = fitframe.groupby(["width", "height"]).label.agg(["sum", "count"])
    prior = y[fit].mean()
    probs = np.asarray(
        [
            (sizes.loc[(w, h), "sum"] + 2 * prior) / (sizes.loc[(w, h), "count"] + 2)
            if (w, h) in sizes.index
            else prior
            for w, h in zip(df.width, df.height)
        ]
    )
    save_result("size_only", 0, probs, 0)
    np.save(out / "size_predictions.npy", probs)
    images = np.load(BASE / "data/images.npy")
    for variant in ("raw", "local_contrast"):
        inputs = (
            images
            if variant == "raw"
            else np.stack([local_contrast(im) for im in images])
        )
        rgb = torch.from_numpy(inputs).permute(0, 3, 1, 2).to(device)
        target = torch.tensor(y, device=device, dtype=torch.float32)
        mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]
        for seed in SEEDS:
            configure(seed)
            run = out / f"{variant}_{seed}"
            run.mkdir(exist_ok=True)
            model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            for module in (model.conv1, model.bn1, model.layer1, model.layer2):
                module.requires_grad_(False)
            model.fc = nn.Linear(512, 1)
            model.to(device, memory_format=torch.channels_last)
            opt = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=0.00018,
                weight_decay=0.015,
            )
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 10, eta_min=0.00002)
            history = []
            best = -1

            def logits(ids, augment):
                x = rgb[ids].float() / 255.0
                if augment:
                    x = torch.where(
                        torch.rand(len(x), 1, 1, 1, device=device) < 0.5, x.flip(-1), x
                    )
                    x = (
                        x * (0.85 + 0.30 * torch.rand(len(x), 1, 1, 1, device=device))
                    ).clamp(0, 1)
                with torch.autocast(
                    torch.device(device).type,
                    dtype=torch.bfloat16,
                    enabled=device == "cuda",
                ):
                    return model((x - mean) / std).flatten().float()

            @torch.inference_mode()
            def pred(indices):
                model.eval()
                return torch.cat(
                    [logits(i, False).sigmoid().cpu() for i in batches(indices, 48)]
                ).numpy()

            for ep in range(1, 11):
                model.train()
                for mod in (model.bn1, model.layer1, model.layer2):
                    mod.eval()
                total = 0
                for ids in batches(fit, 32, True):
                    loss = F.binary_cross_entropy_with_logits(
                        logits(ids, True), target[ids]
                    )
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    total += loss.detach() * len(ids)
                pv = pred(val)
                auc = scores(y[val], pv, 0.5)["auc"]
                history.append(
                    dict(epoch=ep, training_bce=float(total) / len(fit), val_auc=auc)
                )
                if auc > best:
                    best = auc
                    torch.save(
                        dict(
                            model=model.state_dict(),
                            epoch=ep,
                            seed=seed,
                            variant=variant,
                        ),
                        run / "best.pt",
                    )
                sched.step()
            ck = torch.load(run / "best.pt", map_location=device, weights_only=True)
            model.load_state_dict(ck["model"])
            p = pred(np.arange(len(df)))
            thr = save_result(variant, seed, p, ck["epoch"])
            table(run / "history.csv", history)
            np.save(run / "probability.npy", p)
            print(variant, seed, best, flush=True)
    frame = pd.DataFrame(rows)
    which = (
        frame.query("split=='validation' and model!='size_only'")
        .groupby("model")
        .auc.mean()
        .idxmax()
    )
    p = np.mean([np.load(out / f"{which}_{s}/probability.npy") for s in SEEDS], axis=0)
    thr = save_result("followup_ensemble", -1, p, 0)
    np.save(out / "ensemble.npy", p)
    dump(out / "selection.json", dict(variant=which, threshold=thr, seeds=SEEDS))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=BASE / "outputs/followup")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    main(a.output, a.device)
