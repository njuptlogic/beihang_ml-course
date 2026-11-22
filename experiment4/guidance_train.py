"""Paired CIFAR-10 continuations with optional condition-posterior target correction."""
import argparse
from copy import deepcopy
import math
from time import perf_counter

import torch
import torch.nn.functional as F

from course.common import configure, dump, table
from experiment4.model import FlowNet
from experiment4.research_train import file_hash, quantile_time
from experiment4.train import BASE, data

ROOT = BASE / "outputs/guidance"
PARENT = BASE / "outputs/research/extended/flow.pt"
EPOCHS = 8
BATCH = 128
SEED = 1843


@torch.no_grad()
def correction(ema, state, time, label, target, config):
    # Equation 20 in Tang et al.; unconditional items keep the original velocity.
    cond, uncond = (
        ema(
            torch.cat((state, state)),
            torch.cat((time, time)),
            torch.cat((label, torch.full_like(label, 10))),
        )
        .float()
        .chunk(2)
    )
    return target + config["correction_strength"] * (cond - uncond)


def fit(name, device, output_root=ROOT):
    configure(SEED)
    folder = output_root / name
    folder.mkdir(parents=True, exist_ok=True)
    model = FlowNet().to(device, memory_format=torch.channels_last)
    ema = deepcopy(model).eval().requires_grad_(False)
    parent = torch.load(PARENT, weights_only=True, map_location="cpu")
    model.load_state_dict(parent["model"])
    ema.load_state_dict(parent["ema"])
    del parent
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00014, weight_decay=0.01)
    x, y = data(True)
    x, y = x.to(device), y.to(device)
    n_epochs = 13 if name == "compute_control" else EPOCHS
    config = dict(
        arm=name,
        epochs=n_epochs,
        parent_epochs=44,
        seed=SEED,
        parent_sha256=file_hash(PARENT),
        batch=BATCH,
        ema=0.995,
        learning_rate_start=0.00014,
        learning_rate_end=0.00006,
        drop_label=0.12,
        time_law="logit_normal_0_1",
        generator_parameters=sum(p.numel() for p in model.parameters()),
    )
    if name == "model_guidance":
        config.update(correction_strength=0.65)
    history = []
    checkpoint = folder / "flow.pt"
    if checkpoint.exists():
        ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if ck["config"] != config:
            raise ValueError(f"Run has changed in {folder}")
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        optimizer.load_state_dict(ck["optim"])
        history = ck["history"]
        del ck
    dump(folder / "config.json", config)
    for epoch in range(len(history) + 1, n_epochs + 1):
        lr = (
            0.00006
            + 0.00008 * (1 + math.cos(math.pi * (epoch - 1) / (n_epochs - 1))) / 2
        )
        optimizer.param_groups[0]["lr"] = lr
        gen = torch.Generator(device=device).manual_seed(SEED * 100 + epoch)
        sums = torch.zeros(3, device=device)
        start = perf_counter()
        for ids in torch.randperm(len(x), device=device, generator=gen).split(BATCH):
            image = x[ids].float() / 127.5 - 1
            n = len(ids)
            flips = torch.rand(n, device=device, generator=gen) < 0.5
            image = torch.where(flips[:, None, None, None], image.flip(-1), image)
            noise = torch.randn(image.shape, device=device, generator=gen)
            time = quantile_time(torch.rand(n, device=device, generator=gen), True)
            state = torch.lerp(noise, image, time[:, None, None, None])
            label = torch.where(
                torch.rand(n, device=device, generator=gen) < 0.12, 10, y[ids]
            )
            target = image - noise
            corrected = target
            with torch.autocast(
                device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                if name == "model_guidance":
                    corrected = correction(ema, state, time, label, target, config)
                v = model(state, time, label)
            loss = F.mse_loss(v.float(), corrected)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            with torch.no_grad():
                torch._foreach_lerp_(
                    list(ema.parameters()), list(model.parameters()), 0.005
                )
                sums += (
                    torch.stack(
                        [
                            loss,
                            F.mse_loss(v.float(), target),
                            F.mse_loss(corrected, target),
                        ]
                    )
                    * n
                )
        values = (sums / len(x)).tolist()
        history.append(
            dict(
                epoch=epoch,
                total_epoch=44 + epoch,
                loss=values[0],
                velocity_mse=values[1],
                correction_power=values[2],
                learning_rate=lr,
                seconds=perf_counter() - start,
            )
        )
        tmp = checkpoint.with_suffix(".tmp")
        torch.save(
            dict(
                model=model.state_dict(),
                ema=ema.state_dict(),
                optim=optimizer.state_dict(),
                config=config,
                history=history,
            ),
            tmp,
        )
        tmp.replace(checkpoint)
        table(folder / "history.csv", history)
        print(name, history[-1], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("arm", choices=["control", "model_guidance", "compute_control"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    fit(args.arm, torch.device(args.device))
