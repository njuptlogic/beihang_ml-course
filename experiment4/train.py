"""Train rectified flow from scratch; resume at completed epoch boundaries."""
import argparse
from copy import deepcopy
from time import perf_counter
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from course.common import ROOT, configure, table, dump
from experiment4.model import make_model, sample

BASE = ROOT / "experiment4"


def data(train):
    d = CIFAR10(str(BASE / "data"), train=train, download=True)
    return torch.from_numpy(d.data).permute(0, 3, 1, 2), torch.tensor(d.targets)


def main(epochs, batch, seed, out, parameterization="direct"):
    configure(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, y = data(True)
    x, y = x.to(device), y.to(device)
    out.mkdir(parents=True, exist_ok=True)
    model = make_model(parameterization).to(device, memory_format=torch.channels_last)
    ema = deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=0.0007, weight_decay=0.01)
    gen = torch.Generator(device=device).manual_seed(719)
    noise = torch.randn(50, 3, 32, 32, device=device, generator=gen)
    labels = torch.arange(10, device=device).repeat_interleave(5)
    cfg = dict(
        epochs=epochs,
        batch=batch,
        seed=seed,
        n_parameters=sum(p.numel() for p in model.parameters()),
        lr=0.0007,
        ema=0.995,
        label_dropout=0.12,
        snapshot_seed=719,
    )
    if parameterization != "direct":
        cfg["parameterization"] = parameterization
    history = []
    last = out / "flow.pt"
    if last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=True)
        if ck["config"] != cfg:
            raise ValueError(
                "Use a new output directory for a different configuration."
            )
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["optim"])
        history = ck["history"]
        torch.set_rng_state(ck["cpu_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(ck["gpu_rng"])
    dump(out / "config.json", cfg)
    for ep in range(len(history) + 1, epochs + 1):
        start = perf_counter()
        lr = 0.0007 * (0.15 + 0.85 * (1 + np.cos(np.pi * (ep - 1) / epochs)) / 2)
        opt.param_groups[0]["lr"] = float(lr)
        total = torch.zeros((), device=device)
        for idx in torch.randperm(len(x), device=device).split(batch):
            target = x[idx].float() / 127.5 - 1
            target = torch.where(
                torch.rand(len(idx), 1, 1, 1, device=device) < 0.5,
                target.flip(-1),
                target,
            )
            z = torch.randn_like(target)
            t = torch.rand(len(idx), device=device)
            xt = torch.lerp(z, target, t[:, None, None, None])
            ys = torch.where(torch.rand(len(idx), device=device) < 0.12, 10, y[idx])
            with torch.autocast(
                device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                v = model(xt, t, ys)
            loss = F.mse_loss(v.float(), target - z)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            with torch.no_grad():
                torch._foreach_lerp_(
                    list(ema.parameters()), list(model.parameters()), 0.005
                )
            total += loss.detach() * len(idx)
        row = dict(
            epoch=ep,
            velocity_mse=total.item() / len(x),
            lr=float(lr),
            seconds=perf_counter() - start,
        )
        history.append(row)
        print(row, flush=True)
        if ep in {1, 3, 6, 12, epochs}:
            np.save(
                out / f"epoch_{ep:02}.npy", sample(ema, noise, labels).cpu().numpy()
            )
        tmp = out / "flow.tmp"
        torch.save(
            dict(
                config=cfg,
                model=model.state_dict(),
                ema=ema.state_dict(),
                optim=opt.state_dict(),
                history=history,
                cpu_rng=torch.get_rng_state(),
                gpu_rng=torch.cuda.get_rng_state() if device.type == "cuda" else None,
            ),
            tmp,
        )
        tmp.replace(last)
        table(out / "history.csv", history)
    print("Finished training", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=type(BASE), default=BASE / "outputs")
    p.add_argument(
        "--parameterization", choices=["direct", "preconditioned"], default="direct"
    )
    a = p.parse_args()
    main(a.epochs, a.batch, a.seed, a.output, a.parameterization)
