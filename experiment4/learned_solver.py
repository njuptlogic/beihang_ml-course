"""Distill a frozen velocity field into a few-step multistep rule using synthetic trajectories."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from course.common import configure, dump, table
from experiment4.model import FlowNet
from experiment4.research_evaluate import atomic_array, metadata
from experiment4.research_train import ROOT

OUT = ROOT.parent / "learned_solver"
BUDGETS = (2, 4, 8)
FAMILIES = ("euler", "history", "bands")
SPLITS = {"fit": (1200, 72101), "validation": (400, 72102)}
TEACHER_STEPS = 24
GUIDANCE = 3.0
RIDGE = 1e-4


def initialize(device):
    configure(72100)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    checkpoint, previous = metadata(ROOT, "time_strong")
    model = (
        FlowNet()
        .eval()
        .requires_grad_(False)
        .to(device, memory_format=torch.channels_last)
    )
    model.load_state_dict(checkpoint["ema"])
    protocol = dict(
        parent=previous["checkpoint_sha256"],
        epoch=32,
        guidance=GUIDANCE,
        teacher="midpoint",
        teacher_steps=TEACHER_STEPS,
        teacher_network_calls=2 * TEACHER_STEPS,
        synthetic_splits=SPLITS,
        budgets=BUDGETS,
        families=FAMILIES,
        ridge=RIDGE,
        input_noise="independent CPU standard normal; condition = row % 10",
        primary_budget=4,
        selection="lowest mean final clipped-image MSE vs midpoint on validation at each fixed budget",
        evaluation="paired seed 109; 10k balanced class conditions vs all 10k CIFAR-10 test images",
    )
    OUT.mkdir(exist_ok=True)
    expected = json.loads(json.dumps(protocol))
    p = OUT / "protocol.json"
    if p.exists() and json.loads(p.read_text()) != expected:
        raise ValueError(
            "The frozen sampler or protocol changed; use a separate output folder"
        )
    if not p.exists():
        dump(p, expected)
    return model


def noise(n, seed):
    return (
        torch.randn(n, 3, 32, 32, generator=torch.Generator().manual_seed(seed)),
        torch.arange(n) % 10,
    )


def velocity(model, x, y, t):
    times = x.new_full((2 * len(y),), t)
    labels = torch.cat((y, torch.full_like(y, 10)))
    with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.is_cuda):
        c, u = model(torch.cat((x, x)), times, labels).chunk(2)
        return (u + GUIDANCE * (c - u)).float()


@torch.inference_mode()
def teacher_path(model, z, y):
    x = z.clone()
    snaps = []
    for i in range(TEACHER_STEPS):
        h = 1 / TEACHER_STEPS
        v = velocity(model, x, y, i * h)
        x = x + h * velocity(model, x + 0.5 * h * v, y, (i + 0.5) * h)
        if (i + 1) % (TEACHER_STEPS // max(BUDGETS)) == 0:
            snaps.append(x.cpu())
    return torch.stack(snaps, 1)


def pixels(x):
    return ((x.clamp(-1, 1) + 1) * 127.5).round().byte()


def bands(x, family):
    if family == "history":
        return [x]
    low = F.interpolate(F.avg_pool2d(x, 4), scale_factor=4, mode="nearest")
    return [low, x - low]


def combine(history, coefficients, family):
    result = torch.zeros_like(history[0])
    for j, item in enumerate(history):
        for b, band in enumerate(bands(item, family)):
            result = result + coefficients[b][j] * band
    return result


@torch.inference_mode()
def sample_raw(model, z, y, budget, family, rows=None):
    x = z
    history = [z]
    for i in range(budget):
        v = velocity(model, x, y, i / budget)
        if family == "euler":
            x = x + v / budget
        else:
            history.append(v)
            x = combine(history, rows[i], family)
    return x


def archive_teacher(model, device, batch):
    for split, (n, seed) in SPLITS.items():
        path = OUT / f"teacher_{split}.npy"
        if path.exists():
            if np.load(path, mmap_mode="r").shape != (n, max(BUDGETS), 3, 32, 32):
                raise ValueError(f"Incomplete trajectory bank: {path}")
            continue
        z, y = noise(n, seed)
        chunks = []
        for zz, yy in zip(z.split(batch), y.split(batch)):
            chunks.append(teacher_path(model, zz.to(device), yy.to(device)).half())
        atomic_array(path, torch.cat(chunks).numpy())
        print("Teacher trajectories", split, n, flush=True)


@torch.inference_mode()
def fit_family(model, device, budget, family, batch, bank_path=None):
    # Fit each stage on its own current student states, keeping all previous stages fixed.
    n, seed = SPLITS["fit"]
    z, y = noise(n, seed)
    teacher = np.load(bank_path or OUT / "teacher_fit.npy", mmap_mode="r")
    if teacher.shape[1] % budget:
        raise ValueError("Teacher times must include all student step boundaries")
    history = [z]
    x = z
    rows = []
    trace = []
    for i in range(budget):
        velocities = [
            velocity(model, xx.to(device), yy.to(device), i / budget).cpu()
            for xx, yy in zip(x.split(batch), y.split(batch))
        ]
        history.append(torch.cat(velocities))
        target = torch.from_numpy(
            teacher[:, (i + 1) * teacher.shape[1] // budget - 1].astype("float32")
        )
        nvar = len(history)
        gram = np.zeros((1 if family == "history" else 2, nvar, nvar))
        rhs = np.zeros(gram.shape[:2])
        for start in range(0, n, batch):
            sl = slice(start, start + batch)
            variables = [bands(v[sl], family) for v in history]
            truths = bands(target[sl], family)
            for b, truth in enumerate(truths):
                A = torch.stack([p[b].reshape(-1) for p in variables])
                gram[b] += (A @ A.T).numpy()
                rhs[b] += (A @ truth.reshape(-1)).numpy()
        gram /= target.numel()
        rhs /= target.numel()
        prior = np.array([1.0] + [1 / budget] * (nvar - 1))
        coeff = [
            np.linalg.solve(g + RIDGE * np.eye(nvar), r + RIDGE * prior).tolist()
            for g, r in zip(gram, rhs)
        ]
        rows.append(coeff)
        x = torch.cat(
            [
                combine([v[j : j + batch] for v in history], coeff, family)
                for j in range(0, n, batch)
            ]
        )
        trace.append(dict(step=i + 1, fit_mse=float((x - target).square().mean())))
    return rows, trace


@torch.inference_mode()
def train(model, device, batch):
    archive_teacher(model, device, batch)
    n, seed = SPLITS["validation"]
    z, y = noise(n, seed)
    reference = torch.from_numpy(
        np.load(OUT / "teacher_validation.npy")[:, -1].astype("float32")
    )
    summaries, individuals, traces, selections = [], [], [], {}
    for budget in BUDGETS:
        for family in FAMILIES:
            name = f"{family}_{budget}"
            cp = OUT / f"{name}.json"
            if family == "euler":
                rows, trace = None, []
            elif cp.exists():
                ck = json.loads(cp.read_text())
                rows, trace = ck["rows"], ck["training_trace"]
            else:
                rows, trace = fit_family(model, device, budget, family, batch)
                dump(
                    cp,
                    dict(budget=budget, family=family, rows=rows, training_trace=trace),
                )
            traces.extend(dict(method=name, **d) for d in trace)
            x = torch.cat(
                [
                    sample_raw(
                        model, zz.to(device), yy.to(device), budget, family, rows
                    ).cpu()
                    for zz, yy in zip(z.split(batch), y.split(batch))
                ]
            )
            errors = (
                (x.clamp(-1, 1) - reference.clamp(-1, 1)).square().flatten(1).mean(1)
            )
            grad_error = (
                (
                    (x[:, :, :, 1:] - x[:, :, :, :-1])
                    - (reference[:, :, :, 1:] - reference[:, :, :, :-1])
                )
                .square()
                .flatten(1)
                .mean(1)
            )
            vals = dict(
                method=name,
                family=family,
                budget=budget,
                free_coefficients=sum(len(a) for row in (rows or []) for a in row),
                validation_mse=errors.mean().item(),
                horizontal_gradient_mse=grad_error.mean().item(),
            )
            summaries.append(vals)
            individuals.extend(
                dict(method=name, index=j, condition=j % 10, clipped_mse=e)
                for j, e in enumerate(errors.tolist())
            )
            print(vals, flush=True)
        selections[str(budget)] = min(
            (v for v in summaries if v["budget"] == budget),
            key=lambda v: v["validation_mse"],
        )["method"]
    table(OUT / "validation.csv", summaries)
    table(OUT / "validation_images.csv", individuals)
    table(OUT / "stage_fit.csv", traces)
    dump(OUT / "selection.json", selections)


def parameters(method):
    family, k = method.split("_")
    k = int(k)
    if k not in BUDGETS or family not in FAMILIES:
        raise ValueError(method)
    rows = (
        None
        if family == "euler"
        else json.loads((OUT / f"{method}.json").read_text())["rows"]
    )
    return k, family, rows


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["teacher", "train", "sample"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch", type=int, default=40)
    p.add_argument("--budget", type=int, choices=BUDGETS, default=4)
    p.add_argument("--class-id", type=int, choices=range(10), default=8)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    device = torch.device(a.device)
    model = initialize(device)
    if a.action == "teacher":
        archive_teacher(model, device, a.batch)
    elif a.action == "train":
        train(model, device, a.batch)
    else:
        if a.output is None:
            p.error("Specify --output for a sample")
        name = json.loads((OUT / "selection.json").read_text())[str(a.budget)]
        z, _ = noise(1, a.seed)
        k, f, r = parameters(name)
        pred = sample_raw(
            model, z.to(device), torch.tensor([a.class_id], device=device), k, f, r
        )
        a.output.parent.mkdir(exist_ok=True, parents=True)
        Image.fromarray(pixels(pred)[0].permute(1, 2, 0).cpu().numpy()).save(a.output)
