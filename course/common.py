"""Shared paths, seeded splits, and result writers."""
from pathlib import Path
import json
import random
import numpy as np
import torch
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPLIT_SEED = 42


def configure(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("high")


def dump(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    temp.replace(path)


def table(path, rows):
    frame = pd.DataFrame(rows)
    temp = path.with_suffix(".tmp")
    frame.to_csv(temp, index=False)
    temp.replace(path)


def batches(indices, size, shuffle=False):
    indices = np.asarray(indices)
    if not indices.flags.writeable:
        indices = indices.copy()
    if shuffle:
        indices = np.random.permutation(indices)
    for start in range(0, len(indices), size):
        yield indices[start : start + size]
