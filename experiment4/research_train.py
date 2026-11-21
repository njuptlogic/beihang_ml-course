"""Paired continuation experiments on time sampling and masked semantic alignment."""
import argparse
import hashlib
import math
from copy import deepcopy
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from course.common import configure, dump, table
from experiment4.model import FlowNet, sample
from experiment4.teacher_features import ROOT
from experiment4.train import BASE, data

# Every 12-epoch branch starts from the same baseline. The extended branch uses
# exactly the first 12 epochs of the combined branch, then starts a second schedule.
ARMS = {
    "control": (False, False),
    "time": (True, False),
    "align": (False, True),
    "combined": (True, True),
    "extended": (True, True),
}
SEED = 812
EPOCHS = 12
BATCH = 128


class MaskedAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 96, 1, 1))
        self.readout = nn.Sequential(
            nn.Conv2d(96, 96, 3, padding=1), nn.SiLU(), nn.Conv2d(96, 64, 1)
        )

    def forward(self, context, teacher, random):
        pooled = F.avg_pool2d(context, 2)
        # Select exactly four distinct tokens, including corner and edge sites.
        hidden = random.argsort(dim=1)[:, :4]
        mask = (
            torch.zeros_like(random, dtype=torch.bool)
            .scatter_(1, hidden, True)
            .reshape(-1, 1, 4, 4)
        )
        prediction = self.readout(
            torch.where(mask, self.mask_token.to(pooled.dtype), pooled)
        )
        loss = 1 - F.cosine_similarity(prediction.float(), teacher.float(), dim=1)
        return (loss * mask[:, 0]).sum() / (4 * len(pooled))


def quantile_time(u, concentrated):
    if concentrated:
        return torch.sigmoid(
            math.sqrt(2) * torch.erfinv(2 * u.clamp(1e-6, 1 - 1e-6) - 1)
        )
    return u


def file_hash(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def fit(arm, root, device):
    configure(SEED)
    use_time, align = ARMS[arm]
    out = root / arm
    out.mkdir(parents=True, exist_ok=True)
    extended = arm == "extended"
    start_file = root / "combined/flow.pt" if extended else BASE / "outputs/flow.pt"
    previous = torch.load(start_file, map_location="cpu", weights_only=True)
    x, y = data(True)
    x, y = x.to(device), y.to(device)
    model = FlowNet().to(device, memory_format=torch.channels_last)
    ema = deepcopy(model).eval().requires_grad_(False)
    model.load_state_dict(previous["model"])
    ema.load_state_dict(previous["ema"])
    adapter = (
        MaskedAdapter().to(device, memory_format=torch.channels_last) if align else None
    )
    if extended:
        if len(previous["history"]) != EPOCHS:
            raise ValueError("First complete the matched combined run.")
        adapter.load_state_dict(previous["adapter"])
    trainable = list(model.parameters())
    if align:
        trainable += list(adapter.parameters())
    opt = torch.optim.AdamW(trainable, lr=0.00022, weight_decay=0.01)
    teacher = None
    if align:
        teacher = torch.from_numpy(np.load(root / "teacher.npy")).to(device)
    ckfile = out / "flow.pt"
    base_lr = 0.00014 if extended else 0.00022
    first_epoch = 32 if extended else 20
    cfg = dict(
        arm=arm,
        epochs=EPOCHS,
        batch=BATCH,
        seed=SEED,
        base_epochs=first_epoch,
        parent_sha256=file_hash(start_file),
        time_law="logit_normal_mu0_sigma1" if use_time else "uniform",
        learning_rate=base_lr,
        minimum_lr=0.00008,
        align_weight=0.12 if align else 0.0,
        mask_tokens=4 if align else 0,
        ema=0.995,
        drop_label=0.12,
        parameters_generator=sum(p.numel() for p in model.parameters()),
        parameters_adapter=sum(p.numel() for p in adapter.parameters()) if align else 0,
    )
    history = []
    if ckfile.is_file():
        ck = torch.load(ckfile, map_location="cpu", weights_only=True)
        if ck["config"] != cfg:
            raise ValueError(f"Run settings differ from existing checkpoint {ckfile}")
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["optim"])
        if align:
            adapter.load_state_dict(ck["adapter"])
        history = ck["history"]
    dump(out / "config.json", cfg)
    snap_gen = torch.Generator(device=device).manual_seed(719)
    snapshot = torch.randn(50, 3, 32, 32, generator=snap_gen, device=device)
    snap_labels = torch.arange(10, device=device).repeat_interleave(5)
    # A common, uniform-time MSE diagnostic, measured on fixed training images.
    diag_gen = torch.Generator(device=device).manual_seed(213)
    dx = x[::194].float() / 127.5 - 1
    dy = y[::194]
    dz = torch.randn(dx.shape, device=device, generator=diag_gen)
    dt = torch.linspace(0.002, 0.998, len(dx), device=device)
    dxt = torch.lerp(dz, dx, dt[:, None, None, None])
    for ep in range(len(history) + 1, EPOCHS + 1):
        start = perf_counter()
        lr = (
            0.00008
            + (base_lr - 0.00008)
            * (1 + math.cos(math.pi * (ep - 1) / (EPOCHS - 1)))
            / 2
        )
        opt.param_groups[0]["lr"] = lr
        # Restartable per-epoch streams ensure the two factors receive identical
        # images, flips, Gaussian noise, dropout flags and time quantiles.
        gen = torch.Generator(device=device).manual_seed(SEED * 100 + first_epoch + ep)
        mask_gen = torch.Generator(device=device).manual_seed(7090 + first_epoch + ep)
        sums = torch.zeros(2, device=device)
        for ix in torch.randperm(len(x), device=device, generator=gen).split(BATCH):
            n = len(ix)
            flip = torch.rand(n, device=device, generator=gen) < 0.5
            clean = x[ix].float() / 127.5 - 1
            clean = torch.where(flip[:, None, None, None], clean.flip(-1), clean)
            noise = torch.randn(clean.shape, device=device, generator=gen)
            t = quantile_time(torch.rand(n, device=device, generator=gen), use_time)
            mid = torch.lerp(noise, clean, t[:, None, None, None])
            label = torch.where(
                torch.rand(n, device=device, generator=gen) < 0.12, 10, y[ix]
            )
            with torch.autocast(
                device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                velocity, context = model(mid, t, label, return_features=True)
                aux = velocity.new_zeros((), dtype=torch.float32)
                if align:
                    aux = adapter(
                        context,
                        teacher[ix, flip.long()],
                        torch.rand(n, 16, device=device, generator=mask_gen),
                    )
            mse = F.mse_loss(velocity.float(), clean - noise)
            loss = mse + cfg["align_weight"] * aux
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            with torch.no_grad():
                torch._foreach_lerp_(
                    list(ema.parameters()), list(model.parameters()), 0.005
                )
                sums += torch.stack((mse, aux)) * n
        with torch.inference_mode(), torch.autocast(
            device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            predicted = torch.cat(
                [
                    ema(a, b, c)
                    for a, b, c in zip(dxt.split(64), dt.split(64), dy.split(64))
                ]
            )
        diagnostic = (predicted.float() - (dx - dz)).square().mean((1, 2, 3))
        values = (sums / len(x)).tolist()
        row = dict(
            epoch=ep,
            total_epoch=first_epoch + ep,
            velocity_mse=values[0],
            masked_cosine_loss=values[1],
            uniform_diagnostic_mse=diagnostic.mean().item(),
            lr=lr,
            seconds=perf_counter() - start,
        )
        for i in range(4):
            row[f"time_q{i+1}_mse"] = (
                diagnostic[(dt >= i / 4) & (dt < (i + 1) / 4)].mean().item()
            )
        history.append(row)
        if ep in (1, 4, 8, EPOCHS):
            samples = sample(ema, snapshot, snap_labels)
            np.save(out / f"epoch_{ep:02}.npy", samples.cpu().numpy())
        tmp = out / "flow.tmp"
        torch.save(
            dict(
                model=model.state_dict(),
                ema=ema.state_dict(),
                adapter=adapter.state_dict() if align else None,
                optim=opt.state_dict(),
                config=cfg,
                history=history,
            ),
            tmp,
        )
        tmp.replace(ckfile)
        table(out / "history.csv", history)
        print(arm, row, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--out", type=Path, default=ROOT)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    a = parser.parse_args()
    for name in a.arms:
        fit(name, a.out, torch.device(a.device))
