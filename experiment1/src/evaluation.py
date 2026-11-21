"""评价指标、逐图预测与可视化；测试集仅在训练和模型选择结束后调用。"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

plt.rcParams.update({"font.size": 10, "figure.dpi": 130, "savefig.dpi": 180,
                     "axes.spines.top": False, "axes.spines.right": False})


@torch.inference_mode()
def predict_probabilities(model, x, batch_size=512):
    model.eval()
    return torch.cat([model(batch).softmax(1).cpu() for batch in x.split(batch_size)]).numpy()


def classification_metrics(y, probabilities):
    pred = probabilities.argmax(1)
    report = classification_report(y, pred, labels=list(range(10)), output_dict=True, zero_division=0)
    loss = -np.log(np.maximum(probabilities[np.arange(len(y)), y], 1e-30)).mean()
    return {
        "count": len(y), "correct": int((pred == y).sum()), "errors": int((pred != y).sum()),
        "accuracy": float(accuracy_score(y, pred)), "cross_entropy": float(loss),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y, pred, labels=list(range(10))).tolist(),
    }


def save_predictions(path, names, y, probabilities):
    pred = probabilities.argmax(1)
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "true_label", "predicted_label", "confidence", "true_class_probability"])
        for i, name in enumerate(names):
            writer.writerow([name, int(y[i]), int(pred[i]), float(probabilities[i, pred[i]]),
                             float(probabilities[i, y[i]])])


def plot_confusion(matrix, path, title):
    matrix = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(7.2, 6))
    im = ax.imshow(matrix, cmap="Blues")
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if matrix[i, j] > matrix.max() * .5 else "#17324d")
    ax.set(xticks=range(10), yticks=range(10), xlabel="Predicted digit", ylabel="True digit", title=title)
    fig.colorbar(im, ax=ax, fraction=.046)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_failures(x, y, probabilities, names, path, count=24):
    pred = probabilities.argmax(1)
    wrong = np.flatnonzero(pred != y)
    wrong = wrong[np.argsort(-probabilities[wrong, pred[wrong]], kind="stable")][:count]
    rows = max(1, int(np.ceil(len(wrong) / 6)))
    fig, axes = plt.subplots(rows, 6, figsize=(12, rows * 2.2), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    selected = []
    for ax, idx in zip(axes.flat, wrong):
        ax.imshow(x[idx], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        filename = Path(names[idx]).name
        ax.set_title(f"{filename}\nTrue {y[idx]} / Pred {pred[idx]}\nConfidence {probabilities[idx, pred[idx]]:.3f}", fontsize=8)
        selected.append({"index": int(idx), "filename": str(names[idx]), "true": int(y[idx]),
                         "predicted": int(pred[idx]), "confidence": float(probabilities[idx, pred[idx]])})
    if not len(wrong):
        axes[0, 0].text(0, .5, "No classification errors")
    fig.suptitle("Test errors ranked by prediction confidence", fontsize=14)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return selected


def plot_history(history, path, title):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    epochs = [h["epoch"] for h in history]
    for key, label, color in (("train", "Training (online, dropout enabled)", "#2378a6"),
                               ("val", "Validation (eval mode)", "#d27523")):
        axes[0].plot(epochs, [h[f"{key}_loss"] for h in history], label=label, color=color)
        axes[1].plot(epochs, [h[f"{key}_accuracy"] * 100 for h in history], label=label, color=color)
    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy")
    axes[1].set(xlabel="Epoch", ylabel="Accuracy (%)")
    for ax in axes:
        ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_dataset(data, manifest, out):
    fig, axes = plt.subplots(2, 10, figsize=(12, 3.2))
    for label in range(10):
        ids = np.flatnonzero(data["train_y"] == label)[:2]
        for row, idx in enumerate(ids):
            axes[row, label].imshow(data["train_x"][idx], cmap="gray", vmin=0, vmax=255)
            axes[row, label].axis("off")
            if row == 0: axes[row, label].set_title(str(label))
    fig.suptitle("Examples from the teacher's BMP training archive")
    fig.tight_layout(); fig.savefig(out / "data_examples.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    digits = np.arange(10)
    for offset, key, label in ((-.25, "fit", "Optimization subset"), (0, "validation", "Validation"),
                                (.25, "test", "Test")):
        ax.bar(digits + offset, manifest["class_counts"][key], .25, label=label)
    ax.set(xticks=digits, xlabel="Digit", ylabel="Number of images", title="Class distribution")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "class_distribution.png"); plt.close(fig)
