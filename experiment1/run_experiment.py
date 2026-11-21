"""一条命令完成数据校验、两个模型的训练、最终评估与图表导出。"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch
from torch import nn

from src.data import ROOT, prepare_data
from src.models import build_model
from src.training_monitor import add_monitor_args, monitor_run, start_dashboard
from src.evaluation import (classification_metrics, plot_confusion, plot_dataset,
                            plot_failures, plot_history, predict_probabilities, save_predictions)


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def tensor_images(x, manifest, device):
    t = torch.from_numpy(x.astype(np.float32)).unsqueeze(1).to(device) / 255.0
    return (t - manifest["mean"]) / manifest["std"]


@torch.inference_mode()
def validate(model, x, y):
    model.eval()
    total_loss = torch.zeros((), device=x.device)
    correct = torch.zeros((), device=x.device)
    for xb, yb in zip(x.split(512), y.split(512)):
        logits = model(xb)
        total_loss += nn.functional.cross_entropy(logits, yb, reduction="sum")
        correct += (logits.argmax(1) == yb).sum()
    return total_loss.item() / len(y), correct.item() / len(y)


def train(name, epochs, args, data, manifest, device, output):
    seed_everything(args.seed)
    folder = output / name; folder.mkdir()
    model = build_model(name).to(device)
    fit, val = data["fit_indices"], data["val_indices"]
    x = tensor_images(data["train_x"][fit], manifest, device)
    y = torch.from_numpy(data["train_y"][fit]).to(device)
    vx = tensor_images(data["train_x"][val], manifest, device)
    vy = torch.from_numpy(data["train_y"][val]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    history, best = [], (-1.0, float("-inf"))
    batches = math.ceil(len(y) / args.batch_size)
    started = time.perf_counter()
    with monitor_run(output, f"seed{args.seed}/{name}", getattr(args, "log_every", 50)) as monitor:
        for epoch in range(1, epochs + 1):
            epoch_start = time.perf_counter()
            model.train()
            order = torch.randperm(len(y), device=device)
            loss_sum = torch.zeros((), device=device)
            correct = torch.zeros((), device=device)
            learning_rate = optimizer.param_groups[0]["lr"]
            for batch, ids in enumerate(order.split(args.batch_size), 1):
                xb, yb = x[ids], y[ids]
                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = nn.functional.cross_entropy(logits, yb)
                loss.backward(); optimizer.step()
                loss_sum += loss.detach() * len(ids)
                batch_correct = (logits.detach().argmax(1) == yb).sum()
                correct += batch_correct
                monitor.batch(epoch, batch, batches, loss.detach(), batch_correct, len(ids), learning_rate)
            val_loss, val_accuracy = validate(model, vx, vy)
            row = {"epoch": epoch, "lr": learning_rate, "train_loss": loss_sum.item() / len(y),
                   "train_accuracy": correct.item() / len(y), "val_loss": val_loss,
                   "val_accuracy": val_accuracy, "seconds": time.perf_counter() - epoch_start}
            history.append(row)
            # 只依靠验证集选择 checkpoint；准确率相同时选择验证损失更小者。
            if (val_accuracy, -val_loss) > best:
                best = (val_accuracy, -val_loss)
                torch.save({"model_name": name, "state_dict": model.state_dict(), "epoch": epoch,
                            "mean": manifest["mean"], "std": manifest["std"], "seed": args.seed,
                            "validation_accuracy": val_accuracy, "validation_loss": val_loss,
                            "archive_sha256": manifest["archive_sha256"]}, folder / "best.pt")
            scheduler.step()
            with (folder / "history.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row)); writer.writeheader(); writer.writerows(history)
            monitor.epoch(row)
    seconds = time.perf_counter() - started
    plot_history(history, folder / "learning_curves.png", f"{name.upper()} learning curves")
    return {"parameters": sum(p.numel() for p in model.parameters()), "epochs": epochs,
            "training_seconds": seconds, "history": history, "architecture": str(model)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/main")
    parser.add_argument("--models", nargs="+", choices=["mlp", "cnn"], default=["mlp", "cnn"])
    parser.add_argument("--mlp-epochs", type=int, default=15)
    parser.add_argument("--cnn-epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    add_monitor_args(parser)
    args = parser.parse_args()
    if args.log_every < 1:
        parser.error("--log-every 必须为正整数")
    if min(args.mlp_epochs, args.cnn_epochs, args.batch_size) < 1:
        parser.error("训练轮数和 batch size 必须为正数")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error(f"输出目录非空，为避免覆盖实验请换用 --output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if not args.no_dashboard:
        start_dashboard(output)
    torch.set_num_threads(4)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    data, manifest = prepare_data(args.seed)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    config = vars(args).copy(); config["output"] = str(output); config["device"] = str(device)
    config.update({"started_at": datetime.now().astimezone().isoformat(), "python": platform.python_version(),
                   "torch": str(torch.__version__), "cuda": torch.version.cuda,
                   "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                   "optimizer": "Adam", "weight_decay": 1e-4,
                   "scheduler": "CosineAnnealingLR", "minimum_lr": 1e-5,
                   "augmentation": "none", "deterministic_algorithms": True})
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output / "data_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_dataset(data, manifest, output)
    summary = {"config": config, "data": manifest, "models": {}}
    for name in args.models:
        summary["models"][name] = train(name, getattr(args, f"{name}_epochs"), args,
                                         data, manifest, device, output)
    # 所有模型完成训练和验证集选择后，才开始读取测试标签做性能评估。
    print("训练与 checkpoint 选择完成，开始最终训练集/验证集/测试集评价。", flush=True)
    for name, result in summary["models"].items():
        folder = output / name
        checkpoint = torch.load(folder / "best.pt", map_location=device, weights_only=True)
        model = build_model(name).to(device); model.load_state_dict(checkpoint["state_dict"])
        result["best_epoch"] = checkpoint["epoch"]
        result["metrics"] = {}
        for split in ("train", "test"):
            probabilities = predict_probabilities(model, tensor_images(data[f"{split}_x"], manifest, device))
            y = data[f"{split}_y"]
            result["metrics"]["train_full" if split == "train" else "test"] = classification_metrics(y, probabilities)
            save_predictions(folder / f"{split}_predictions.csv", data[f"{split}_names"], y, probabilities)
            if split == "train":
                for key, ids in (("fit", data["fit_indices"]), ("validation", data["val_indices"])):
                    result["metrics"][key] = classification_metrics(y[ids], probabilities[ids])
            else:
                np.savez_compressed(folder / "test_probabilities.npz", probabilities=probabilities, labels=y)
                plot_confusion(result["metrics"]["test"]["confusion_matrix"],
                               folder / "confusion_matrix.png", f"{name.upper()} test confusion matrix")
                result["failure_examples"] = plot_failures(data["test_x"], y, probabilities,
                    data["test_names"], folder / "failure_examples.png")
        (folder / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"{name.upper()} best epoch={result['best_epoch']}, "
              f"test accuracy={result['metrics']['test']['accuracy']:.4%}", flush=True)
    summary["completed_at"] = datetime.now().astimezone().isoformat()
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"全部结果保存在 {output}", flush=True)


if __name__ == "__main__":
    main()
