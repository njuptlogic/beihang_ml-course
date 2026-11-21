"""在固定的训练方案上用验证集选整数平移读出，保存可直接推理的代表权重。"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from analyze_ablation import write_rows, finish
from run_experiment import seed_everything, tensor_images
from src.data import ROOT, prepare_data
from src.views import inference_model, translated_views


@torch.inference_mode()
def score_views(model, x, y, black):
    """每种半径共享中心及邻域 logits，再计算不同混合系数的验证指标。"""
    logits = torch.cat([model(b) for b in x.split(512)])
    rows = []
    for radius, mixes in [(0, [0.]), (1, [.5, .8, 1.]), (2, [.5, .8, 1.])]:
        nearby = logits
        if radius:
            nearby = torch.stack([torch.cat([model(b) for b in v.split(512)])
                                  for v in translated_views(x, radius, black)]).mean(0)
        for mix in mixes:
            pred = torch.lerp(logits, nearby, mix)
            loss = torch.nn.functional.cross_entropy(pred, y).item()
            accuracy = int((pred.argmax(1) == y).sum()) / len(y)
            rows.append(dict(radius=radius, mix=mix, accuracy=accuracy, nll=loss))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "outputs/ablation")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    run, device = args.run, args.device
    out = run / "views"
    out.mkdir(exist_ok=True)
    lock = json.loads((run / "selection.json").read_text())
    cfg = json.loads((run / "config.json").read_text())
    winner, rep, seeds = lock["winner"], lock["representative_seed"], cfg["seeds"]
    torch.set_num_threads(4)
    seed_everything(42)
    data, man = prepare_data(cfg["split_seed"])
    val = data["val_indices"]
    x = tensor_images(data["train_x"][val], man, device)
    labels = data["train_y"]
    vy = torch.tensor(labels[val], device=device)
    black = -man["mean"]/man["std"]

    def load(seed, name):
        c = torch.load(run / f"seed{seed}/{name}.pt", map_location="cpu", weights_only=True)
        if man["archive_sha256"] != c["archive_sha256"]:
            raise RuntimeError("数据集与权重不匹配")
        return c

    grid = []
    for seed in seeds:
        m = inference_model(load(seed, winner)).to(device)
        for r in score_views(m, x, vy, black):
            grid.append(dict(seed=seed, **r))
    write_rows(out / "validation_grid.csv", grid)
    configs = list(dict.fromkeys((r["radius"], r["mix"]) for r in grid))
    def score(params, key):
        return np.mean([r[key] for r in grid if (r["radius"], r["mix"]) == params])
    radius, mix = max(configs, key=lambda p: (score(p, "accuracy"), -score(p, "nll")))
    policy = dict(training=winner, radius=radius, mix=mix, representative_seed=rep)
    (out / "selection.json").write_text(json.dumps(policy, indent=2))
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for d in (1, 2):
        xx = [0., .5, .8, 1.]
        ax.plot(xx, [100 * score((0, 0) if v == 0 else (d, v), "accuracy") for v in xx], "o-", label=f"{d} px")
    ax.set(xlabel="Weight on four translated views (total)", ylabel="Mean validation accuracy (%)")
    ax.grid(alpha=.2)
    ax.legend()
    finish(fig, out, "sensitivity")

    checkpoint = load(rep, winner)
    checkpoint.update(view_radius=radius, view_mix=mix)
    torch.save(checkpoint, out / "selected.pt")
    print(policy)


if __name__ == "__main__":
    main()
