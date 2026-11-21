"""Round-trip trained weights from original files and recompute summary statistics."""
import json
from zipfile import ZipFile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from course.common import ROOT, configure, dump
from course.predict import retina, gaze
from experiment2.train import scores
from experiment4.model import make_model, sample


def main():
    configure(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for number, chosen in ((2, (0, 821, 1639, 1888)), (3, (0, 819, 1600, 1999))):
        base = ROOT / f"experiment{number}"
        out = base / "outputs"
        meta = pd.read_csv(base / "data/images.csv")
        target = np.load(
            out / ("followup/ensemble.npy" if number == 2 else "ensemble/density.npy"),
            mmap_mode="r",
        )
        checks = []
        for idx in chosen:
            r = meta.iloc[idx]
            with ZipFile(base / "data/raw" / r.archive) as z:
                blob = z.read(r.member)
            if number == 2:
                pred = retina(blob, out / "followup", device)["probability"]
            else:
                pred, _ = gaze(blob, out, device)
                pred = pred[0]
            err = float(np.max(np.abs(pred - target[idx])))
            # Single-image and larger-batch BF16 convolution may round differently.
            check = {"index": idx, "member": r.member, "max_abs_error": err}
            if number == 2:
                cutoff = json.loads((out / "followup/selection.json").read_text())[
                    "threshold"
                ]
                assert err < 0.005 and (pred >= cutoff) == (target[idx] >= cutoff)
            else:
                tv = float(np.abs(pred - target[idx]).sum() / 2)
                check["total_variation"] = tv
                assert tv < 0.001
            checks.append(check)
        if number == 2:
            idx = meta.index[meta.split.eq("test")].to_numpy()
            t = json.loads((out / "followup/selection.json").read_text())["threshold"]
            y = meta.loc[idx, "category"].eq("disease").to_numpy()
            metric = scores(y, target[idx], t)
            ref = (
                pd.read_csv(out / "followup/metrics.csv")
                .query("model=='followup_ensemble' and split=='test'")
                .iloc[0]
            )
            for k, v in metric.items():
                assert abs(v - ref[k]) < 1e-7
        else:
            ref = (
                pd.read_csv(out / "native_metrics.csv")
                .query("model=='ensemble' and split=='test'")
                .iloc[0]
            )
            images = pd.read_csv(out / "native_per_image.csv").query(
                "model=='ensemble' and split=='test'"
            )
            metric = images[["cc", "kl", "sim"]].mean().to_dict()
            for k, v in metric.items():
                assert abs(v - ref[k]) < 1e-7
        dump(
            out / "verification.json",
            dict(raw_image_weight_checks=checks, recomputed_test=metric, device=device),
        )
    base = ROOT / "experiment4/outputs"
    for out in (base, base / "preconditioned"):
        ck = torch.load(out / "flow.pt", map_location=device, weights_only=True)
        model = make_model(ck["config"].get("parameterization", "direct")).to(
            device, memory_format=torch.channels_last
        )
        model.load_state_dict(ck["ema"])
        model.eval()
        seed = ck["config"]["snapshot_seed"]
        ep = ck["config"]["epochs"]
        assert len(ck["history"]) == ep
        noise = torch.randn(
            50,
            3,
            32,
            32,
            device=device,
            generator=torch.Generator(device=device).manual_seed(seed),
        )
        imgs = (
            sample(model, noise, torch.arange(10, device=device).repeat_interleave(5))
            .cpu()
            .numpy()
        )
        truth = np.load(out / f"epoch_{ep:02}.npy")
        errors = np.abs(imgs.astype(int) - truth.astype(int))
        assert errors.mean() < 0.1
        dump(
            out / "verification.json",
            dict(
                epoch=ep,
                samples=50,
                noise_seed=seed,
                mean_abs_pixel_error=float(errors.mean()),
                max_abs_pixel_error=int(errors.max()),
                device=device,
            ),
        )
    print("Verified image-to-output paths and course summary metrics.")


if __name__ == "__main__":
    main()
