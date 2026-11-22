"""Draw the training and inference paths actually used by the three selected models."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from course.figures import setup, save, INK, TEAL, BLUE


def diagram(n):
    specs = {
        2: (
            "Fundus classification",
            "Grouped fit",
            ["RGB fundus", "Frozen ViT-S/14", "Width subspace", "Linear readout"],
            [
                "1,393 fit images\n4 held-width folds",
                "3 views × 392²\nCLS + region means",
                "Class-wise means\nSVD, rank ≤ 18",
                "Ridge 0.5\nout-of-fold calibration",
            ],
            "One image",
            [
                "Find bright region",
                "Encode three views",
                "Pool 10 vectors",
                "Predict disease",
            ],
            [
                "Short side × 0.42\nmirror and align",
                "Field, raw local,\ncontrast-stretched local",
                "10 × 384 features\nnormalized by block",
                "Fold projection into w\nthreshold 0.6515",
            ],
        ),
        3: (
            "Fixation-density prediction",
            "Fit readouts",
            [
                "Frozen feature banks",
                "Two density heads",
                "Three seeds",
                "Validation blend",
            ],
            [
                "DINO / PE-Spatial\nResNet texture",
                "Three-scale DINO\nDINO + spatial PE",
                "17 · 42 · 73\n18 epochs each",
                "240 labelled images\nmaximize mean CC",
            ],
            "One scene",
            [
                "Multi-scale RGB",
                "Dense features",
                "Density estimates",
                "Restore resolution",
            ],
            [
                "DINO widths\n224 / 448 / 672",
                "DINO patch tokens\nPE + local texture",
                "0.45 pyramid + 0.45 dual\n+ 0.10 UNISAL",
                "45 × 80 → native\nunit probability mass",
            ],
        ),
        4: (
            "Four-step conditional image generation",
            "Velocity training",
            [
                "CIFAR-10 + noise",
                "Conditional teacher",
                "Velocity target",
                "Update U-Net",
            ],
            [
                "50,000 train images\nx(t) = (1−t)z + tx",
                "EMA network\nclass and null outputs",
                "x − z + 0.65 ×\nclass velocity difference",
                "Start at epoch 44\n8 epochs · AdamW",
            ],
            "Four updates",
            [
                "Noise + label",
                "Conditional velocity",
                "Euler + momentum",
                "Spectral correction",
            ],
            [
                "3 × 32 × 32\n10 target classes",
                "Class + null outputs\nCFG scale 1.75",
                "History decay 0.6\nextrapolation 0.9",
                "12 radial bands\nstrength 0.35 per step",
            ],
        ),
    }
    title, l1, t1, s1, l2, t2, s2 = specs[n]
    fig, ax = plt.subplots(figsize=(11.8, 4.25))
    fig.subplots_adjust(left=0.015, right=0.99, bottom=0.02, top=0.97)
    ax.set(xlim=(0, 12), ylim=(0, 4.3))
    ax.axis("off")
    ax.text(0.24, 4.03, title, fontsize=16, weight="bold", color=INK)
    for y, label, titles, sub, color, fill in [
        (2.85, l1, t1, s1, BLUE, "#EFF4F9"),
        (1.01, l2, t2, s2, TEAL, "#EAF4F1"),
    ]:
        ax.text(0.25, y + 0.61, label.upper(), color=color, weight="bold", fontsize=9)
        for j, (name, detail) in enumerate(zip(titles, sub)):
            x = 0.25 + 2.94 * j
            width = 2.6
            ax.add_patch(
                FancyBboxPatch(
                    (x, y - 0.65),
                    width,
                    1.05,
                    boxstyle="round,pad=0.04,rounding_size=0.10",
                    fc=fill,
                    ec=color,
                    lw=1.05,
                )
            )
            ax.text(
                x + 0.13,
                y + 0.13,
                name,
                fontsize=11,
                color=INK,
                weight="semibold",
                va="center",
            )
            ax.text(
                x + 0.13,
                y - 0.29,
                detail,
                fontsize=9,
                color=INK,
                va="center",
                linespacing=1.4,
            )
            if j < 3:
                ax.add_patch(
                    FancyArrowPatch(
                        (x + width + 0.03, y - 0.15),
                        (x + 2.90, y - 0.15),
                        arrowstyle="-|>",
                        mutation_scale=11,
                        color=color,
                        lw=1.05,
                    )
                )
    save(fig, n, "report_pipeline")


if __name__ == "__main__":
    setup()
    for number in [2, 3, 4]:
        diagram(number)
