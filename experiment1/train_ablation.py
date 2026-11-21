"""固定划分的 2×2 对照。选模只读验证集，最终评测见 analyze_ablation.py。"""
import argparse
import copy
import csv
import json
import math
import platform
import time
from pathlib import Path

from run_experiment import seed_everything, tensor_images, validate
from src.data import ROOT, prepare_data
from src.models import build_model
from src.training_monitor import add_monitor_args, monitor_run, start_dashboard
from src.views import sample_affine, validation_views
import numpy as np
import torch
from torch.nn import functional as F


def affine_batch(x, generator, black):
    """逐图在原图坐标系抽取采样网格，越界位置填黑色。"""
    n, _, h, w = x.shape
    r = 2 * torch.rand((n, 4), generator=generator, device=x.device) - 1
    angle = r[:, 0] * math.radians(12)
    scale = 1 + 0.1 * r[:, 1]
    c, s = scale * angle.cos(), scale * angle.sin()
    theta = torch.stack((c, -s, 4 * r[:, 2] / w, s, c, 4 * r[:, 3] / h), 1)
    return sample_affine(x, theta.reshape(n, 2, 3), black)


def train_pair(data, manifest, seed, augmented, args):
    seed_everything(seed)
    device = args.device
    model = build_model("cnn").to(device)
    shadow = copy.deepcopy(model).requires_grad_(False)
    rng = torch.Generator(device=device).manual_seed(seed + 10000)
    prefix = "affine" if augmented else "plain"
    folder = args.output / f"seed{seed}"
    folder.mkdir(exist_ok=True)
    fit, val = data["fit_indices"], data["val_indices"]
    x = tensor_images(data["train_x"][fit], manifest, device)
    y = torch.from_numpy(data["train_y"][fit]).to(device)
    vx = tensor_images(data["train_x"][val], manifest, device)
    vy = torch.from_numpy(data["train_y"][val]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-5)
    branches = {prefix: model, prefix + "_ema": shadow}
    best = {name: (-1, -float("inf")) for name in branches}
    black = -manifest["mean"] / manifest["std"]
    ema_params, live_params = list(shadow.parameters()), list(model.parameters())
    batches = math.ceil(len(y) / 128)
    with (monitor_run(args.output, f"seed{seed}/{prefix}", getattr(args, "log_every", 50), ema=True) as monitor,
          (folder / f"{prefix}_history.csv").open("w", newline="") as f):
        writer = None
        for epoch in range(1, args.epochs + 1):
            start = time.perf_counter()
            model.train()
            loss_sum = torch.zeros((), device=device)
            correct = torch.zeros((), device=device)
            lr = optimizer.param_groups[0]["lr"]
            order = torch.randperm(len(y), device=device)
            for batch, ids in enumerate(order.split(128), 1):
                xb, yb = x[ids], y[ids]
                if augmented:
                    xb = affine_batch(xb, rng, black)
                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = F.cross_entropy(logits, yb)
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    torch._foreach_lerp_(ema_params, live_params, 1 - args.decay)
                loss_sum += loss.detach() * len(ids)
                batch_correct = (logits.detach().argmax(1) == yb).sum()
                correct += batch_correct
                monitor.batch(epoch, batch, batches, loss.detach(), batch_correct, len(ids), lr)
            row = dict(epoch=epoch, lr=lr, train_loss=loss_sum.item() / len(y),
                       train_accuracy=correct.item() / len(y))
            for name, branch in branches.items():
                loss, acc = validate(branch, vx, vy)
                split = "val" if name == prefix else "ema_val"
                row[split + "_loss"] = loss
                row[split + "_accuracy"] = acc
                if (acc, -loss) > best[name]:
                    best[name] = (acc, -loss)
                    torch.save(dict(model_name="cnn", state_dict=branch.state_dict(),
                                    seed=seed, split_seed=42, epoch=epoch, variant=name,
                                    mean=manifest["mean"], std=manifest["std"],
                                    validation_accuracy=acc, validation_loss=loss,
                                    archive_sha256=manifest["archive_sha256"]), folder / f"{name}.pt")
                for condition, views in validation_views(vx, black):
                    scores = [validate(branch, view, vy) for view in views]
                    view_loss, view_acc = np.mean(scores, axis=0)
                    row[f"{split}_{condition}_loss"] = view_loss
                    row[f"{split}_{condition}_accuracy"] = view_acc
                    monitor.perturbation(epoch, split, condition, view_loss, view_acc, acc)
            row["seconds"] = time.perf_counter() - start
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            f.flush()
            monitor.epoch(row)
            schedule.step()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/ablation")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--decay", type=float, default=.995)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", choices=["cpu", "cuda"])
    add_monitor_args(parser)
    args = parser.parse_args()
    if args.log_every < 1:
        parser.error("--log-every 必须为正整数")
    if args.epochs < 1 or not 0 <= args.decay < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("请使用正轮数、互异种子以及 [0,1) 内的 decay")
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        parser.error("请指定空输出目录")
    if not args.no_dashboard:
        start_dashboard(args.output)
    config = dict(seeds=args.seeds, split_seed=42, epochs=args.epochs, decay=args.decay, batch_size=128,
                  python=platform.python_version(), torch=str(torch.__version__), cuda=torch.version.cuda,
                  device=args.device, gpu=torch.cuda.get_device_name() if args.device == "cuda" else None)
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    torch.set_num_threads(4)
    data, manifest = prepare_data(42)
    for seed in args.seeds:
        for augmented in [False, True]:
            train_pair(data, manifest, seed, augmented, args)
    # 先按跨种子验证准确率的平均值锁定方案，同分时取平均验证损失更低者。
    scores = {}
    for name in ["plain", "plain_ema", "affine", "affine_ema"]:
        ckpts = [torch.load(args.output / f"seed{s}/{name}.pt", map_location="cpu", weights_only=True)
                 for s in args.seeds]
        scores[name] = dict(accuracy=float(np.mean([c["validation_accuracy"] for c in ckpts])),
                            loss=float(np.mean([c["validation_loss"] for c in ckpts])))
    winner = max(scores, key=lambda n: (scores[n]["accuracy"], -scores[n]["loss"]))
    (args.output / "selection.json").write_text(json.dumps(dict(
        scores=scores, winner=winner, representative_seed=args.seeds[0]), indent=2))


if __name__ == "__main__":
    main()
