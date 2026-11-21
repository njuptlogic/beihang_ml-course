"""Shared print styles and vector diagrams used in the three reports."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from course.common import ROOT

TEAL = "#087E83"
BLUE = "#3B67A4"
AMBER = "#BD5F29"
PLUM = "#76569A"
INK = "#182D3C"
GRAY = "#677887"
COLORS = [BLUE, AMBER, PLUM, TEAL]
STYLE = ["-", "--", "-.", ":"]


def setup():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.labelcolor": INK,
            "text.color": INK,
            "axes.edgecolor": "#CAD2D5",
            "xtick.color": "#465D68",
            "ytick.color": "#465D68",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "grid.color": "#DCE3E5",
            "grid.linewidth": 0.7,
            "lines.linewidth": 2,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def page(nrows=1, ncols=1, size=(10, 4), **kw):
    return plt.subplots(nrows, ncols, figsize=size, layout="constrained", **kw)


def save(fig, number, name):
    directory = ROOT / f"experiment{number}" / "figures"
    directory.mkdir(exist_ok=True)
    fig.savefig(directory / f"{name}.png", dpi=210)
    fig.savefig(directory / f"{name}.svg", metadata={"Date": None})
    plt.close(fig)


def axes_title(ax, title, x=None, y=None):
    ax.set_title(title, pad=10)
    if x:
        ax.set_xlabel(x)
    if y:
        ax.set_ylabel(y)
    ax.grid(axis="y")


def picture(ax, im, title=None, **kw):
    ax.imshow(im, **kw)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, fontsize=9, pad=3)


def architecture(n):
    setup()
    fig, ax = plt.subplots(figsize=(12.4, 4.3))
    fig.subplots_adjust(left=0.015, right=0.985, bottom=0.03, top=0.98)
    ax.set(xlim=(0, 12), ylim=(0, 4.15))
    ax.axis("off")
    palettes = {
        "source": ("#EAF1F7", "#457294"),
        "model": ("#E2F2EE", "#227D75"),
        "mix": ("#F3EDF7", "#806194"),
        "loss": ("#FCEDDF", "#AB663B"),
    }
    nodes = {}

    def box(id, x, y, w, h, title, note, kind):
        fill, line = palettes[kind]
        nodes[id] = (x, y, w, h)
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=.018,rounding_size=.10",
                facecolor=fill,
                edgecolor=line,
                lw=1.25,
                zorder=3,
            )
        )
        ax.text(
            x + w / 2,
            y + h * 0.65,
            title,
            ha="center",
            va="center",
            weight="bold",
            size=10.3,
            zorder=4,
        )
        ax.text(
            x + w / 2,
            y + h * 0.30,
            note,
            ha="center",
            va="center",
            size=8.2,
            color="#405761",
            zorder=4,
        )

    def edge(a, b, side="right", entry="left", dash=False):
        def anchor(k, side):
            x, y, w, h = nodes[k]
            return {
                "left": (x, y + h / 2),
                "right": (x + w, y + h / 2),
                "top": (x + w / 2, y + h),
                "bottom": (x + w / 2, y),
            }[side]

        p = anchor(a, side)
        q = anchor(b, entry)
        ax.add_patch(
            FancyArrowPatch(
                p,
                q,
                arrowstyle="-|>",
                mutation_scale=13,
                lw=1.15,
                color=GRAY,
                linestyle="--" if dash else "-",
                shrinkA=2,
                shrinkB=2,
                zorder=2,
            )
        )

    titles = {
        2: "Retinal image classification",
        3: "A spatial density model of human gaze",
        4: "Class-conditional rectified flow",
    }
    ax.text(0.10, 3.93, titles[n], fontsize=18, weight="bold")
    if n in (2, 3):
        box(
            "rgb",
            0.12,
            1.65,
            1.28,
            0.88,
            "RGB image",
            "224 × 224" if n == 2 else "320 × 192",
            "source",
        )
        box("enc", 1.76, 1.65, 1.62, 0.88, "ResNet-18", "frozen encoder", "model")
        edge("rgb", "enc")
        if n == 2:
            box("mid", 3.9, 2.54, 1.66, 0.72, "Stage 3", "pool → 256 × 7²", "model")
            box("deep", 3.9, 0.96, 1.66, 0.72, "Stage 4", "512 × 7²", "model")
            box("fuse", 6.10, 1.65, 1.63, 0.88, "Concatenate", "1 × 1 conv → 64", "mix")
            box(
                "pool",
                8.22,
                1.65,
                1.64,
                0.88,
                "Hybrid pooling",
                "mean + attention",
                "mix",
            )
            box(
                "out",
                10.40,
                1.65,
                1.46,
                0.88,
                "Disease score",
                "sigmoid(logit)",
                "source",
            )
            for key in ("mid", "deep"):
                edge("enc", key)
                edge(key, "fuse")
            edge("fuse", "pool")
            edge("pool", "out")
            caption = "Training: binary cross-entropy on image labels.     Operating point: Youden threshold selected on validation images."
        else:
            for id, y, label, note in [
                ("a", 2.72, "Stage 2", "128 × 24 × 40"),
                ("b", 1.69, "Stage 3", "256 × 12 × 20"),
                ("c", 0.67, "Stage 4", "512 × 6 × 10"),
            ]:
                box(id, 3.78, y, 1.66, 0.72, label, note, "model")
                edge("enc", id)
            box(
                "fuse", 5.91, 1.65, 1.72, 0.88, "Align & fuse", "3 × 16 channels", "mix"
            )
            for id in ("a", "b", "c"):
                edge(id, "fuse")
            box(
                "prior",
                8.10,
                2.83,
                1.74,
                0.70,
                "Spatial prior",
                "fit-set log density",
                "loss",
            )
            box(
                "read",
                8.10,
                1.65,
                1.74,
                0.88,
                "Log-density",
                "content + α log π",
                "mix",
            )
            box(
                "out",
                10.30,
                1.65,
                1.55,
                0.88,
                "Spatial softmax",
                "24 × 40 density",
                "source",
            )
            edge("fuse", "read")
            edge("prior", "read", "bottom", "top")
            edge("read", "out")
            caption = "Training: KL(target ∥ prediction) + 0.3 (1 − CC).     Evaluation: bilinear projection to the original fixation-map grid."
    else:
        box("noise", 0.12, 2.51, 1.55, 0.73, "Gaussian noise", "z ∼ N(0, I)", "source")
        box("rgb", 0.12, 0.95, 1.55, 0.73, "CIFAR-10", "x ∈ [−1, 1]", "source")
        box("mix", 2.16, 1.65, 1.72, 0.88, "Interpolate", "(1 − t) z + t x", "mix")
        edge("noise", "mix")
        edge("rgb", "mix")
        box(
            "enc",
            4.37,
            1.65,
            1.66,
            0.88,
            "Pack & encode",
            "12 × 16² → 48 × 16²",
            "model",
        )
        box(
            "mid",
            6.45,
            1.65,
            1.73,
            0.88,
            "Self-attention",
            "96 × 8² · 4 heads",
            "model",
        )
        box("out", 8.65, 1.65, 1.47, 0.88, "Skip decoder", "velocity vθ", "model")
        box("loss", 10.48, 1.65, 1.36, 0.88, "MSE loss", "target x − z", "loss")
        box("cond", 6.45, 2.87, 1.73, 0.69, "Time + class", "FiLM in each block", "mix")
        edge("mix", "enc")
        edge("enc", "mid")
        edge("mid", "out")
        edge("out", "loss", dash=True)
        edge("cond", "mid", "bottom", "top")
        ax.plot([5.20, 5.20, 9.38, 9.38], [1.65, 1.11, 1.11, 1.35], c=GRAY, lw=1.1)
        ax.annotate(
            "",
            xy=(9.38, 1.65),
            xytext=(9.38, 1.32),
            arrowprops=dict(arrowstyle="-|>", color=GRAY),
        )
        ax.text(7.29, 0.88, "16 × 16 skip features", ha="center", fontsize=8.5)
        caption = "Sampling: start with fresh noise, evaluate the learned velocity field, and integrate from t = 0 to t = 1."
    ax.plot([0.1, 11.85], [0.52, 0.52], color="#D7E0E1", lw=1)
    ax.text(0.12, 0.22, caption, size=10.2, color="#455D69")
    save(fig, n, "architecture")
