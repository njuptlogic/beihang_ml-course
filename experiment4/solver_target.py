"""Compare trajectory distillation from the 12-step deployed flow with the 48-call ODE teacher."""
import argparse
import json

import numpy as np
import torch

from course.common import dump, table
from experiment4.learned_solver import (
    OUT,
    SPLITS,
    fit_family,
    initialize,
    noise,
    sample_raw,
    velocity,
)
from experiment4.research_evaluate import atomic_array

DIR = OUT / "short_teacher"
BUDGETS = (2, 4)


@torch.inference_mode()
def train(device):
    model = initialize(device)
    DIR.mkdir(exist_ok=True)
    cfg = dict(
        teacher="provided 32-epoch time model, 12 Euler calls, guidance 3",
        fit_seed=72101,
        validation_seed=72102,
        budgets=list(BUDGETS),
        family="bands",
        check="paired MSE to 12-step Euler target",
        compare_to="matched 48-call midpoint target; two-band design unchanged",
        chronology="follow-up to four-step trajectory fit; fixed family and budgets before its 10k scoring",
    )
    p = DIR / "protocol.json"
    if p.exists() and json.loads(p.read_text()) != cfg:
        raise ValueError("Changed distillation protocol")
    dump(p, cfg)
    for split, (n, seed) in SPLITS.items():
        bank = DIR / f"{split}.npy"
        if bank.exists():
            continue
        z, y = noise(n, seed)
        targets = []
        for zz, yy in zip(z.split(32), y.split(32)):
            x = zz.to(device)
            yy = yy.to(device)
            states = []
            for i in range(12):
                x = x + velocity(model, x, yy, i / 12) / 12
                if (i + 1) % 3 == 0:
                    states.append(x.cpu().half())
            targets.append(torch.stack(states, 1))
        atomic_array(bank, torch.cat(targets).numpy())
    z, y = noise(*SPLITS["validation"])
    raw = torch.from_numpy(np.load(DIR / "validation.npy")[:, -1]).float().clamp(-1, 1)
    metrics = []
    for b in BUDGETS:
        rows, trace = fit_family(model, device, b, "bands", 32, DIR / "fit.npy")
        dump(DIR / f"short_{b}.json", dict(rows=rows, training_trace=trace))
        for teacher, coeff in (
            ("euler", None),
            ("midpoint48", json.loads((OUT / f"bands_{b}.json").read_text())["rows"]),
            ("euler12", rows),
        ):
            preds = torch.cat(
                [
                    sample_raw(
                        model,
                        zz.to(device),
                        yy.to(device),
                        b,
                        "bands" if coeff else "euler",
                        coeff,
                    ).cpu()
                    for zz, yy in zip(z.split(32), y.split(32))
                ]
            )
            val = (preds.clamp(-1, 1) - raw).square().mean().item()
            metrics.append(
                dict(teacher=teacher, student_calls=b, validation_mse_to_euler12=val)
            )
            print(metrics[-1], flush=True)
    table(DIR / "validation.csv", metrics)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    train(torch.device(p.parse_args().device))
