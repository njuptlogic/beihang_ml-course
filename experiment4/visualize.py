"""Render CIFAR-10 examples and measure conditional semantics with a held-out probe."""
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from course.figures import (
    TEAL,
    PLUM,
    BLUE,
    AMBER,
    INK,
    GRAY,
    setup,
    architecture,
    page,
    save,
    axes_title,
    picture,
)
from course.common import table, dump
from experiment4.train import BASE, data

OUT = BASE / "outputs"
E = OUT / "evaluation"
CLASSES = [
    "Airplane",
    "Car",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck",
]
NAMES = {
    "gaussian": "Gaussian",
    "flow_main": "Full flow",
    "four_steps": "4 steps",
    "no_guidance": "No CFG",
    "interval": "Middle CFG",
    "no_ema": "No EMA",
}


def tile(items, cols, gap=2):
    items = np.asarray(items).transpose(0, 2, 3, 1)
    h, w = items.shape[1:3]
    rows = (len(items) + cols - 1) // cols
    canvas = np.full(
        (rows * h + (rows - 1) * gap, cols * w + (cols - 1) * gap, 3),
        248,
        dtype=np.uint8,
    )
    for k, img in enumerate(items):
        row, col = divmod(k, cols)
        r = row * (h + gap)
        c = col * (w + gap)
        canvas[r : r + h, c : c + w] = img
    return canvas


def probe(trainf, testf, genf, y, yt, labels):
    # Ridge regression on centered features. Single fixed regularization, fitted on real train only.
    mean = trainf.mean(0)
    a = trainf - mean
    b = np.eye(10, dtype="float32")[y] - 0.1
    gram = a.T @ a
    gram.flat[:: len(gram) + 1] += 100.0
    weights = np.linalg.solve(gram, a.T @ b)
    real = (testf - mean) @ weights
    gen = (genf - mean) @ weights
    real_correct = real.argmax(1) == yt
    pred = gen.argmax(1)
    wrong = gen.copy()
    wrong[np.arange(len(gen)), labels] = -np.inf
    margin = gen[np.arange(len(gen)), labels] - wrong.max(1)
    train_norm = trainf / np.maximum(
        np.linalg.norm(trainf, axis=1, keepdims=True), 1e-8
    )
    gn = genf / np.maximum(np.linalg.norm(genf, axis=1, keepdims=True), 1e-8)
    nearest = []
    sim = []
    for start in range(0, len(gn), 256):
        dots = gn[start : start + 256] @ train_norm.T
        nearest.extend(dots.argmax(1))
        sim.extend(dots.max(1))
    return real_correct, pred, margin, np.array(nearest), np.array(sim)


def radial_spectrum(images):
    radius = np.rint(
        np.hypot(*np.meshgrid(np.fft.fftfreq(32) * 32, np.fft.fftfreq(32) * 32))
    ).astype(int)
    accum = np.zeros((32, 32))
    for start in range(0, len(images), 200):
        x = images[start : start + 200].astype(np.float32) / 255
        luma = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
        luma -= luma.mean((1, 2), keepdims=True)
        accum += (np.abs(np.fft.fft2(luma) / (32 * 32)) ** 2).sum(0)
    accum /= len(images)
    return np.array([accum[radius == k].mean() for k in range(1, 17)])


def main():
    setup()
    architecture(4)
    real_train, yt = data(True)
    real_test, ys = data(False)
    real_train = real_train.numpy()
    real_test = real_test.numpy()
    yt = yt.numpy()
    ys = ys.numpy()
    indices = np.load(E / "reference_indices.npz")
    train_ids = indices["train"]
    test_ids = indices["test"]
    imgs = {k: np.load(E / f"{k}_images.npy", mmap_mode="r") for k in NAMES}
    synth = imgs["flow_main"]
    labels = np.arange(len(synth)) % 10
    met = pd.read_csv(E / "metrics.csv").set_index("method")
    trainf = np.load(E / "real_train_features.npy")
    testf = np.load(E / "real_test_features.npy")
    genf = np.load(E / "flow_main_features.npy")
    with threadpool_limits(4):
        correct, pred, margin, near, sim = probe(
            trainf, testf, genf, yt[train_ids], ys[test_ids], labels
        )
    table(
        E / "conditional_diagnostics.csv",
        dict(
            sample=np.arange(len(synth)),
            condition=labels,
            probe_prediction=pred,
            score_margin=margin,
            nearest_training_index=train_ids[near],
            cosine_similarity=sim,
        ),
    )
    classrows = [
        dict(
            category=name,
            test_accuracy=correct[ys[test_ids] == j].mean(),
            generated_agreement=(pred[labels == j] == j).mean(),
        )
        for j, name in enumerate(CLASSES)
    ]
    table(E / "class_agreement.csv", classrows)
    dump(
        E / "probe.json",
        dict(
            real_accuracy=float(correct.mean()),
            generated_agreement=float((pred == labels).mean()),
            ridge=100.0,
        ),
    )
    fig, axes = page(10, 2, size=(8.4, 8.6), width_ratios=(4, 6))
    for j, name in enumerate(CLASSES):
        real = real_test[test_ids[ys[test_ids] == j][:4]]
        generated = synth[labels == j][:6]
        picture(
            axes[j, 0],
            tile(real, 4),
            "Real test images" if j == 0 else None,
            interpolation="nearest",
        )
        picture(
            axes[j, 1],
            tile(generated, 6),
            "Generated · fixed class" if j == 0 else None,
            interpolation="nearest",
        )
        axes[j, 0].set_ylabel(
            name, rotation=0, ha="right", va="center", labelpad=8, fontsize=9
        )
    save(fig, 4, "gallery")
    fig, axes = page(5, 1, size=(9.0, 5.9))
    for a, ep in zip(axes, [1, 3, 6, 12, 20]):
        samples = np.load(OUT / f"epoch_{ep:02}.npy")
        picture(a, tile(samples[::5], 10), interpolation="nearest")
        a.set_ylabel(f"Epoch {ep}", rotation=0, ha="right", va="center")
        if ep == 1:
            a.set_xticks(np.arange(10) * 34 + 15.5, CLASSES, fontsize=8)
            a.tick_params(axis="x", labeltop=True, labelbottom=False, length=0)
    save(fig, 4, "progression")
    hist = pd.read_csv(OUT / "history.csv")
    fig, axes = page(1, 2, size=(10.2, 3.65))
    axes[0].plot(hist.epoch, hist.velocity_mse, c=TEAL, marker="o", ms=3)
    axes_title(axes[0], "A   Velocity regression", "Epoch", "Mean squared error")
    axes[1].plot(hist.epoch, hist.lr * 1000, c=PLUM, marker="s", ms=3)
    axes_title(axes[1], "B   Learning rate", "Epoch", "Learning rate × 1,000")
    save(fig, 4, "training")
    fig, axes = page(6, 1, size=(9.1, 6.35))
    for a, (key, label) in zip(axes, NAMES.items()):
        picture(a, tile(imgs[key][:10], 10), interpolation="nearest")
        a.set_ylabel(label, rotation=0, ha="right", va="center")
        if key == "gaussian":
            a.set_xticks(np.arange(10) * 34 + 15.5, CLASSES, fontsize=8)
            a.tick_params(axis="x", labeltop=True, labelbottom=False, length=0)
    save(fig, 4, "paired_samples")
    fig, axes = page(1, 2, size=(10.6, 4.0))
    sub = met.drop("gaussian").sort_values("fid_test", ascending=False)
    y = np.arange(len(sub))
    for ax, col, title, lab in zip(
        axes,
        ["fid_test", "inception_score_mean"],
        ["A   Distance to test images ↓", "B   Inception Score ↑"],
        ["FID (10,000 vs 10,000)", "IS (10 shuffled splits)"],
    ):
        errs = sub.inception_score_std if col == "inception_score_mean" else None
        ax.errorbar(sub[col], y, xerr=errs, fmt="o", color=TEAL, capsize=3, ms=7)
        ax.set_yticks(y, [NAMES[k] for k in sub.index])
        axes_title(ax, title, lab)
    save(fig, 4, "ablation")
    fig, axes = page(1, 2, size=(10.5, 4.35))
    v = np.array([r["test_accuracy"] for r in classrows])
    g = np.array([r["generated_agreement"] for r in classrows])
    y = np.arange(10)
    axes[0].hlines(y, np.minimum(g, v), np.maximum(g, v), color="#B3C6C8")
    axes[0].scatter(v, y, color=BLUE, marker="s", label="Real test")
    axes[0].scatter(g, y, color=AMBER, marker="o", label="Generated")
    axes[0].set_yticks(y, CLASSES)
    axes[0].set_xlim(0, 1)
    axes_title(axes[0], "A   Linear probe", "Fraction matching the class")
    axes[0].legend(loc="lower left", fontsize=8)
    freq = np.arange(1, 17)
    spectra = []
    for name, im, color, ls in [
        ("Real test", real_test[test_ids], INK, "-"),
        ("Full flow", synth, TEAL, "--"),
        ("Gaussian", imgs["gaussian"], AMBER, "-."),
    ]:
        spec = radial_spectrum(im)
        axes[1].semilogy(freq, spec, c=color, ls=ls, label=name)
        spectra.extend(
            dict(source=name, cycles=int(k), power=float(s)) for k, s in zip(freq, spec)
        )
    axes_title(
        axes[1],
        "B   Spatial frequency",
        "Cycles per image",
        "Mean luminance power (log scale)",
    )
    axes[1].legend()
    table(E / "spectra.csv", spectra)
    save(fig, 4, "diagnostics")
    fig, axes = page(3, 6, size=(10.6, 5.45))
    cases = []
    # Show six classes in the CIFAR label order, one disagreeing probe example per class.
    for j in range(6):
        ids = np.flatnonzero((labels == j) & (pred != j))
        i = ids[np.argmin(margin[ids])]
        rid = train_ids[near[i]]
        same = np.flatnonzero(ys[test_ids] == j)[0]
        for row, img in enumerate(
            [synth[i], real_train[rid], real_test[test_ids[same]]]
        ):
            a = axes[row, j]
            picture(
                a,
                img.transpose(1, 2, 0),
                CLASSES[j] if row == 0 else None,
                interpolation="nearest",
            )
            if j == 0:
                a.set_ylabel(
                    ["Generated", "Nearest real", "Target example"][row], fontsize=9
                )
        axes[0, j].set_xlabel(f"Probe: {CLASSES[pred[i]]}", fontsize=8)
        axes[1, j].set_xlabel(f"{CLASSES[yt[rid]]} · {sim[i]:.2f}", fontsize=8)
        cases.append(
            dict(
                sample=int(i),
                condition=CLASSES[j],
                probe=CLASSES[pred[i]],
                nearest_class=CLASSES[yt[rid]],
                nearest_index=int(rid),
                cosine=float(sim[i]),
            )
        )
    table(E / "case_gallery.csv", cases)
    save(fig, 4, "failures")


if __name__ == "__main__":
    main()
