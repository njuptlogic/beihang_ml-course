"""锁定验证方案后，导出逐种子结果、同图检验、表征和错误分析。"""
import argparse
import csv
import json
from pathlib import Path

from run_experiment import seed_everything, tensor_images
from train_ablation import affine_batch
from src.data import ROOT, prepare_data
from src.evaluation import classification_metrics, predict_probabilities, save_predictions
from src.models import build_model
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest
import torch
from torch.nn import functional as F


VARIANTS = {"plain": "CNN", "plain_ema": "CNN + EMA", "affine": "CNN + affine",
            "affine_ema": "CNN + affine + EMA"}
COLORS = ["#546678", "#337a9d", "#d6863a", "#127e67"]


def write_rows(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)


def paired(base, new, y):
    a, b = base.argmax(1) == y, new.argmax(1) == y
    fixed, broken = int((~a & b).sum()), int((a & ~b).sum())
    # 对 -1, 0, +1 三种同图差值计数重抽样，等价于有放回抽取 N 个样本编号。
    counts = np.array([fixed, broken, len(y) - fixed - broken])
    draws = np.random.default_rng(2026).multinomial(len(y), counts / len(y), size=20000)
    delta = (draws[:, 0] - draws[:, 1]) * 100 / len(y)
    lo, hi = np.quantile(delta, [.025, .975])
    return dict(fixed=fixed, broken=broken, both_wrong=int((~a & ~b).sum()),
                both_correct=int((a & b).sum()), delta_pp=(fixed-broken)*100/len(y),
                ci_low_pp=lo, ci_high_pp=hi,
                mcnemar_p=binomtest(fixed, fixed + broken).pvalue if fixed + broken else 1.)


def finish(fig, out, name):
    fig.tight_layout()
    fig.savefig(out / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def plot_ablation(rows, seeds, out):
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4), sharey=True)
    for ax, key, title in zip(axes, ["validation", "test"], ["Validation", "Test"]):
        matrix = np.array([[r[key] * 100 for r in rows if r["seed"] == s] for s in seeds])
        for i, vals in enumerate(matrix):
            ax.plot(range(4), vals, ".--", lw=1, alpha=.7, label=f"seed {seeds[i]}")
        ax.plot(range(4), matrix.mean(0), "ko-", label="Mean", lw=2)
        ax.set(xticks=range(4), xticklabels=["CNN", "+ EMA", "+ affine", "+ both"], title=title)
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Accuracy (%)")
    axes[1].legend(fontsize=8)
    finish(fig, out, "ablation")


def plot_learning(folder, out):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), sharex=True)
    for row, stem in enumerate(["plain", "affine"]):
        h = np.genfromtxt(folder / f"{stem}_history.csv", delimiter=",", names=True)
        for col, metric in enumerate(["loss", "accuracy"]):
            ax = axes[row, col]
            for key, name, color in zip(["train_", "val_", "ema_val_"],
                                         ["Online train", "Validation", "Validation (EMA)"], COLORS):
                v = h[key + metric] * (100 if metric == "accuracy" else 1)
                ax.plot(h["epoch"], v, label=name, color=color, lw=1.8)
            ax.set_ylabel(("Clean inputs" if row == 0 else "Affine training") + "\n" + ("Cross-entropy" if col == 0 else "Accuracy (%)"))
            if col == 0:
                ax.set_yscale("log")
            ax.grid(alpha=.2)
            if row == 1:
                ax.set_xlabel("Epoch")
    axes[0, 1].legend(fontsize=8)
    finish(fig, out, "learning")


@torch.inference_mode()
def plot_features(model, x, y, names, p, selected, out):
    wrong = selected[0]
    right = np.flatnonzero((y == y[wrong]) & (p.argmax(1) == y))[0]
    fig, axes = plt.subplots(4, 7, figsize=(11, 6))
    values = {}
    for i, idx in enumerate([right, wrong]):
        z = x[idx:idx+1]
        for layer, op in enumerate(model):
            z = op(z)
            if layer not in (1, 8):
                continue
            stage = 0 if layer == 1 else 1
            row = axes[i * 2 + stage]
            f = z[0].cpu().numpy()
            values[f"sample{idx}_relu{layer}"] = f
            row[0].imshow(x[idx, 0].cpu(), cmap="gray")
            row[0].set_title(f"{Path(names[idx]).name}\nTrue {y[idx]}, pred {p[idx].argmax()}", fontsize=8)
            # 所有图均取固定通道 0..5，每张图在本层共用色限。
            for ch, ax in enumerate(row[1:]):
                ax.imshow(f[ch], cmap="viridis", vmin=0, vmax=max(f[:6].max(), 1e-9))
                ax.set_title(f"ReLU {1 if stage == 0 else 4} / ch {ch}", fontsize=8)
            for ax in row:
                ax.axis("off")
            if layer == 8:
                break
    np.savez_compressed(out / "feature_maps.npz", **values)
    finish(fig, out, "features")


def plot_changes(data, base, new, out):
    y, names = data["test_y"], data["test_names"]
    a, b = base.argmax(1), new.argmax(1)
    fig, axes = plt.subplots(2, 4, figsize=(9, 4.8))
    records = []
    for row, (mask, label) in enumerate([((a != y) & (b == y), "Corrected"),
                                          ((a == y) & (b != y), "New error")]):
        ids = np.flatnonzero(mask)
        for ax in axes[row]:
            ax.axis("off")
        for ax, j in zip(axes[row], ids[:4]):
            ax.imshow(data["test_x"][j], cmap="gray", vmin=0, vmax=255)
            ax.set_title(f"{label} | {Path(names[j]).name}\nTruth {y[j]}: {a[j]} → {b[j]}", fontsize=8)
        for j in ids:
            records.append(dict(filename=names[j], truth=y[j], baseline=a[j], selected=b[j], change=label))
    if records:
        write_rows(out / "changed_decisions.csv", records)
    finish(fig, out, "changed_decisions")


def plot_diagnostics(data, manifest, p, base, details, out):
    y, images, names = data["test_y"], data["test_x"], data["test_names"]
    pred, conf = p.argmax(1), p.max(1)
    wrong = np.flatnonzero(pred != y)
    ranked = sorted(wrong, key=lambda j: -conf[j])
    selected, seen = [], set()
    for j in ranked:
        pair = (y[j], pred[j])
        if pair not in seen:
            selected.append(j)
            seen.add(pair)
        if len(selected) == 6:
            break
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.6))
    for ax in axes.flat:
        ax.axis("off")
    for ax, idx in zip(axes.flat, selected):
        ax.imshow(images[idx], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_title(f"{Path(names[idx]).name}  |  {y[idx]} → {pred[idx]}\np(pred) = {conf[idx]:.5f}", fontsize=9)
    finish(fig, out, "failures")
    matrix = np.array(details["confusion_matrix"])
    norm = matrix / matrix.sum(1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(7.3, 6.3))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=100)
    for i, j in np.ndindex(10, 10):
        if matrix[i, j]:
            ax.text(j, i, f"{norm[i,j]:.1f}", ha="center", va="center", fontsize=8,
                    color="white" if i == j else "#264560")
    ax.set(xticks=range(10), yticks=range(10), xlabel="Prediction", ylabel="True digit")
    fig.colorbar(im, ax=ax, label="Within-class share (%)", fraction=.048)
    finish(fig, out, "confusion")
    fig, ax = plt.subplots(figsize=(8, 3.3))
    for pb, label, mark in [(base, "Baseline", "o"), (p, "Selected", "s")]:
        recall = [100*np.mean(pb.argmax(1)[y == i] == i) for i in range(10)]
        ax.plot(range(10), recall, marker=mark, label=label)
    ax.set(xticks=range(10), xlabel="Digit", ylabel="Test recall (%)")
    ax.grid(alpha=.2)
    ax.legend()
    finish(fig, out, "recall")
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for ids, label in [(pred == y, "Correct"), (pred != y, "Errors")]:
        ax.hist(conf[ids], bins=np.linspace(0, 1, 21), weights=np.ones(ids.sum()) / max(1, ids.sum()),
                alpha=.6, label=f"{label} (n={ids.sum()})")
    ax.set(xlabel="Maximum predicted probability", ylabel="Fraction within group")
    ax.legend()
    finish(fig, out, "confidence")
    g = torch.Generator().manual_seed(42)
    ids = [data["fit_indices"][np.flatnonzero(data["train_y"][data["fit_indices"]] == k)[0]] for k in [2, 4, 6, 9]]
    x = tensor_images(data["train_x"][ids], manifest, "cpu")
    fig, axes = plt.subplots(4, 5, figsize=(6, 4.8))
    for j in range(5):
        z = x if j == 0 else affine_batch(x, g, -manifest["mean"] / manifest["std"])
        for i, ax in enumerate(axes[:, j]):
            ax.imshow(z[i, 0], cmap="gray", vmin=-manifest["mean"]/manifest["std"],
                      vmax=(1-manifest["mean"])/manifest["std"])
            ax.axis("off")
        axes[0,j].set_title("Original" if j == 0 else f"Draw {j}", fontsize=9)
    finish(fig, out, "augmentation")
    return selected


@torch.inference_mode()
def shift_sweep(models, x, y, black, out):
    rows = []
    for d in [0, 1, 2, 3]:
        for dx, dy in ([(0,0)] if d == 0 else [(-d,0),(d,0),(0,-d),(0,d)]):
            pad = F.pad(x, (d, d, d, d), value=black)
            shifted = pad[..., d-dy:d-dy+x.shape[-2], d-dx:d-dx+x.shape[-1]]
            for name, model in models.items():
                p = predict_probabilities(model, shifted)
                rows.append(dict(variant=name, pixels=d, dx=dx, dy=dy, accuracy=np.mean(p.argmax(1) == y)))
    write_rows(out / "shift_validation.csv", rows)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for i, name in enumerate(models):
        vals = [np.mean([r["accuracy"] for r in rows if r["variant"] == name and r["pixels"] == d]) for d in range(4)]
        ax.plot(range(4), 100*np.array(vals), "o-", color=COLORS[i], label=VARIANTS[name])
    ax.set(xticks=range(4), xlabel="Integer translation (pixels)", ylabel="Validation accuracy (%)")
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    finish(fig, out, "shifts")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "outputs/ablation")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    run = args.run
    cfg = json.loads((run / "config.json").read_text())
    lock = json.loads((run / "selection.json").read_text())
    winner, rep = lock["winner"], lock["representative_seed"]
    out = run / "analysis"
    out.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    seed_everything(42)
    data, manifest = prepare_data(cfg["split_seed"])
    y, labels = data["test_y"], data["train_y"]
    train = tensor_images(data["train_x"], manifest, args.device)
    test = tensor_images(data["test_x"], manifest, args.device)
    fit, val = data["fit_indices"], data["val_indices"]
    rows, pairs, models = [], [], {}
    for seed in cfg["seeds"]:
        probs = {}
        for name in VARIANTS:
            c = torch.load(run / f"seed{seed}/{name}.pt", map_location=args.device, weights_only=True)
            if c["archive_sha256"] != manifest["archive_sha256"]:
                raise RuntimeError("当前原始数据与 checkpoint 不符")
            model = build_model("cnn").to(args.device)
            model.load_state_dict(c["state_dict"])
            p = predict_probabilities(model, test)
            pt = predict_probabilities(model, train)
            probs[name] = p
            tright = pt.argmax(1) == labels
            metrics = classification_metrics(y, p)
            top2 = (np.argsort(p, axis=1)[:, -2:] == y[:, None]).any(1).mean()
            rows.append(dict(seed=seed, variant=name, epoch=c["epoch"], fit=tright[fit].mean(),
                             validation=tright[val].mean(), train_full=tright.mean(),
                             test=metrics["accuracy"], errors=metrics["errors"], top2=top2,
                             nll=metrics["cross_entropy"], macro_f1=metrics["macro_f1"]))
            if seed == rep:
                models[name] = model
                if name == winner:
                    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
                    save_predictions(out / "test_predictions.csv", data["test_names"], y, p)
                    save_predictions(out / "train_predictions.csv", data["train_names"], labels, pt)
        np.savez_compressed(out / f"seed{seed}_test.npz", names=data["test_names"], labels=y, **probs)
        for name in list(VARIANTS)[1:]:
            pairs.append(dict(seed=seed, comparator="plain", variant=name, **paired(probs["plain"], probs[name], y)))
        if seed == rep:
            primary, baseline = probs[winner], probs["plain"]
    write_rows(out / "runs.csv", rows)
    write_rows(out / "paired.csv", pairs)
    plot_changes(data, baseline, primary, out)
    plot_ablation(rows, cfg["seeds"], out)
    plot_learning(run / f"seed{rep}", out)
    details = classification_metrics(y, primary)
    selected = plot_diagnostics(data, manifest, primary, baseline, details, out)
    if selected:
        plot_features(models[winner], test, y, data["test_names"], primary, selected, out)
    shift_sweep(models, train[val], labels[val], -manifest["mean"]/manifest["std"], out)
    print(f"{winner}, reference seed {rep}. Wrote {out}")


if __name__ == "__main__":
    main()
