"""Scale-pyramid, local feature resampling and fixation scatter on the fixed gaze split."""
import argparse
import json
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from torch import nn
from torch.nn import functional as F
from course.common import batches, configure, dump, table
from experiment3.train import BASE, compare, density, resize
from experiment3.semantic_features import CACHE as MID, OUT as PREV
from experiment3.multiscale_features import OUT, CACHE

OUTPUT_GRID = (45, 80)
FEATURE_GRID = (27, 48)
ARMS = (
    "middle",
    "pyramid",
    "guided",
    "scatter",
    "full",
    "residual",
    "pe_core",
    "pe_spatial",
    "dual",
)
SEEDS = (17, 42, 73)
EPOCHS = 18


def block(c, d, k):
    return nn.Sequential(
        nn.Conv2d(c, d, k, padding=k // 2), nn.GroupNorm(4, d), nn.SiLU()
    )


def gaussian_kernel(sigma):
    k = int(3 * sigma + 0.5)
    t = torch.arange(-k, k + 1, dtype=torch.float32)
    w = torch.exp(-0.5 * (t / sigma).square())
    w /= w.sum()
    return w[:, None] * w[None, :]


class Head(nn.Module):
    def __init__(self, arm):
        super().__init__()
        self.arm = arm
        self.multiscale = arm not in ("middle", "pe_core", "pe_spatial")
        self.guided = arm in ("guided", "full", "residual", "dual")
        self.scatter = arm in ("scatter", "full", "residual", "dual")
        self.residual = arm == "residual"
        self.tokens = block(128, 32, 1)
        self.local = block(128, 16, 1)
        if self.multiscale:
            self.scales = block(32 * 3, 32, 1)
        if arm == "dual":
            self.pe_proj = block(128, 32, 1)
        if self.guided:
            self.query = nn.Conv2d(16, 9, 1)
            nn.init.zeros_(self.query.weight)
            nn.init.zeros_(self.query.bias)
            self.key = nn.Conv2d(32, 1, 1)
            self.alpha = nn.Parameter(torch.tensor(-0.5))
        self.read = nn.Sequential(
            block(48, 48, 3), nn.Dropout2d(0.1), block(48, 24, 3), nn.Conv2d(24, 1, 1)
        )
        nn.init.zeros_(self.read[-1].weight)
        nn.init.zeros_(self.read[-1].bias)
        self.prior_gain = nn.Parameter(torch.tensor(0.0 if self.residual else 1.0))
        if self.scatter:
            self.scene = nn.Sequential(
                nn.Linear(48, 32), nn.SiLU(), nn.Dropout(0.15), nn.Linear(32, 10)
            )
            nn.init.zeros_(self.scene[-1].weight)
            nn.init.zeros_(self.scene[-1].bias)
            for i, sigma in enumerate((1.5, 3.25)):
                kernel = gaussian_kernel(sigma)
                self.register_buffer(f"blur{i}", kernel[None, None])

    def forward(self, mid, local, coarse, fine, ctx, prior, old, mirror=False):
        m = resize(self.tokens(mid), FEATURE_GRID)
        if self.multiscale:
            m = self.scales(
                torch.cat(
                    [
                        resize(
                            (self.pe_proj if self.arm == "dual" else self.tokens)(
                                coarse
                            ),
                            FEATURE_GRID,
                        ),
                        m,
                        self.tokens(fine),
                    ],
                    1,
                )
            )
        l = resize(self.local(local), FEATURE_GRID)
        if self.guided:
            padded = F.pad(m, (1, 1, 1, 1), "replicate")
            b, c, h, w = m.shape
            keys = F.unfold(self.key(padded), 3).reshape(b, 9, h, w)
            weights = (keys + self.query(l)).softmax(1)
            lifted = sum(
                weights[:, i : i + 1]
                * padded[:, :, i // 3 : i // 3 + h, i % 3 : i % 3 + w]
                for i in range(9)
            )
            m = m + self.alpha.sigmoid() * (lifted - m)
        raw = resize(self.read(torch.cat([m, l], 1)), OUTPUT_GRID).float()
        base = prior.clamp_min(1e-10).log()
        base = base - base.mean((-2, -1), keepdim=True)
        raw = raw + self.prior_gain * base
        if self.residual:
            raw = raw + resize(old, OUTPUT_GRID).clamp_min(1e-12).log()
        if self.scatter:
            controls = self.scene(ctx.float()).float()
            y = torch.linspace(-1, 1, raw.shape[-2], device=raw.device)[None, :, None]
            x = torch.linspace(-1, 1, raw.shape[-1], device=raw.device)[None, None, :]
            x = x.expand(1, y.shape[1], -1)
            y = y.expand_as(x)
            if mirror:
                x = -x
            shapes = torch.cat(
                [x, y, x.square(), y.square(), x * y, (x.square() + y.square()).sqrt()],
                0,
            )
            raw = (
                raw * controls[:, 0, None, None, None].clamp(-0.6, 0.6).exp()
                + torch.einsum("bc,chw->bhw", controls[:, 1:7], shapes)[:, None]
            )
        d = raw.flatten(1).softmax(1).view(-1, 1, *OUTPUT_GRID)
        if self.scatter:
            weights = (controls[:, 7:] + d.new_tensor([1.0, 0.0, -0.5])).softmax(1)
            pieces = [d]
            for kernel in (self.blur0, self.blur1):
                r = kernel.shape[-1] // 2
                smooth = F.conv2d(F.pad(d, (r, r, r, r), "reflect"), kernel)
                pieces.append(smooth / smooth.sum((-2, -1), keepdim=True))
            d = (torch.cat(pieces, 1) * weights[:, :, None, None]).sum(1, keepdim=True)
        return d


def context(fit, labels):
    features = np.load(MID / "cls.npy")
    mean = features[fit].mean(0)
    x = (features[fit] - mean).astype("float64")
    cov = x.T @ x / (len(fit) - 1)
    with threadpool_limits(1):
        eig, U = np.linalg.eigh(cov)
    mat = (U[:, -48:] / np.sqrt(eig[-48:] + 1e-5)).astype("float32")
    c = (features - mean) @ mat
    low = F.interpolate(labels[fit], size=OUTPUT_GRID, mode="area")
    prior = density(low).mean(0).reshape(1, 1, *OUTPUT_GRID)
    with tempfile.NamedTemporaryFile(dir=OUT, suffix=".npz", delete=False) as tmp:
        np.savez(tmp, cls_mean=mean, cls_matrix=mat, prior=prior.numpy(), fit=fit)
    Path(tmp.name).replace(OUT / "basis.npz")
    return c, prior


class Dataset:
    def __init__(self, device):
        self.device = device
        self.meta = pd.read_csv(BASE / "data/images.csv")
        self.fit = self.meta.index[self.meta.split == "fit"].to_numpy(copy=True)
        self.val = self.meta.index[self.meta.split == "validation"].to_numpy(copy=True)
        self.allids = self.meta.index.to_numpy(copy=True)
        self.gt = torch.from_numpy(np.load(BASE / "data/targets.npy"))[:, None].float()
        self.targets = F.interpolate(self.gt, size=OUTPUT_GRID, mode="area")
        self.ctx, self.prior = context(self.fit, self.gt)
        self.prior = self.prior.to(device)
        self.features = {
            key: np.load(p, mmap_mode="r")
            for key, p in {
                "mid": MID / "block6.npy",
                "deep": MID / "block12.npy",
                "local": BASE / "data/stage2.npy",
                "coarse": CACHE / "coarse.npy",
                "fine": CACHE / "fine.npy",
                "pe_core": CACHE / "pe_core.npy",
                "pe_spatial": CACHE / "pe_spatial.npy",
                "old": PREV / "selected.npy",
            }.items()
            if p.exists()
        }

    def forward(self, model, ids, flip=False):
        def read(name):
            a = torch.from_numpy(self.features[name][ids]).to(
                self.device, dtype=torch.float32
            )
            return a.flip(-1) if flip else a

        mid = (
            read(model.arm)
            if model.arm in ("pe_core", "pe_spatial")
            else torch.cat([read("mid"), read("deep")], 1)
        )
        c = torch.tensor(self.ctx[ids], device=self.device)
        prior = self.prior.flip(-1) if flip else self.prior
        return model(
            mid,
            read("local"),
            read("pe_spatial" if model.arm == "dual" else "coarse")
            if model.multiscale
            else None,
            read("fine") if model.multiscale else None,
            c,
            prior,
            read("old") if model.residual else None,
            mirror=flip,
        )

    @torch.inference_mode()
    def predict(self, model, ids, batch=22):
        model.eval()
        return torch.cat([self.forward(model, b).cpu() for b in batches(ids, batch)])

    def score(self, maps, ids):
        return (
            torch.cat(
                [
                    compare(
                        self.gt[ids[i : i + 24]],
                        resize(maps[i : i + 24], self.gt.shape[-2:]),
                    )
                    for i in range(0, len(ids), 24)
                ]
            )
            .mean(0)
            .tolist()
        )


def train(ds, arm, seed, epochs, batch):
    configure(seed)
    run = OUT / f"{arm}_{seed}"
    run.mkdir(exist_ok=True)
    model = Head(arm).to(ds.device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.0011, weight_decay=0.012)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=0.00006)
    best = -np.inf
    history = []
    if (run / "validation.npy").exists() and (run / "best.pt").exists():
        return
    for epoch in range(1, epochs + 1):
        total = 0.0
        model.train()
        for ids in batches(ds.fit, batch, shuffle=True):
            flip = np.random.rand() < 0.5
            pred = ds.forward(model, ids, flip)
            target = ds.targets[ids].to(ds.device)
            target = target.flip(-1) if flip else target
            cc, kl, sim = compare(target, pred).mean(0)
            loss = 0.8 * kl + 0.8 * (1 - cc) + 0.25 * (1 - sim)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 4)
            opt.step()
            total += loss.item() * len(ids)
        pred = ds.predict(model, ds.val)
        cc, kl, sim = ds.score(pred, ds.val)
        history.append(
            dict(epoch=epoch, loss=total / len(ds.fit), cc=cc, kl=kl, sim=sim)
        )
        if cc > best:
            best = cc
            torch.save(
                dict(
                    arm=arm,
                    seed=seed,
                    epoch=epoch,
                    params=sum(p.numel() for p in model.parameters()),
                    model=model.state_dict(),
                ),
                run / "best.pt",
            )
            np.save(run / "validation_tmp.npy", pred.numpy())
        sched.step()
        table(run / "history.csv", history)
        if epoch % 5 == 0 or epoch == epochs:
            print(arm, seed, epoch, round(cc, 6), flush=True)
    (run / "validation_tmp.npy").replace(run / "validation.npy")


def select(ds, arms, seeds):
    variants = {}
    rows = []
    prev = resize(torch.from_numpy(np.array(ds.features["old"][ds.val])), OUTPUT_GRID)
    prev = prev / prev.sum((-2, -1), keepdim=True)

    def assess(name, maps, weights):
        cc, kl, sim = ds.score(maps, ds.val)
        rows.append(dict(name=name, weights=json.dumps(weights), cc=cc, kl=kl, sim=sim))
        return cc

    variants["previous"] = prev
    assess("previous", prev, {"previous": 1})
    for arm in arms:
        maps = [
            torch.from_numpy(np.load(OUT / f"{arm}_{s}/validation.npy")) for s in seeds
        ]
        variants[arm] = torch.stack(maps).mean(0)
        assess(arm, variants[arm], {arm: 1})
    # Select a sparse ensemble on a fixed grid; same validation set and criterion for all arms.
    for i, a in enumerate(variants):
        for b in list(variants)[i + 1 :]:
            for w in (0.25, 0.5, 0.75):
                assess(
                    f"{a}+{b}",
                    w * variants[a] + (1 - w) * variants[b],
                    {a: w, b: 1 - w},
                )
    if len(arms) > 1:
        weights = {a: 1 / len(arms) for a in arms}
        assess("equal_all", sum(weights[a] * variants[a] for a in arms), weights)
    table(OUT / "backbone_validation_search.csv", rows)
    best = max(rows, key=lambda r: r["cc"])
    best.update(
        weights=json.loads(best["weights"]),
        seeds=seeds,
        arms=arms,
        criterion="validation mean per-image CC on 90x160; no test labels",
    )
    dump(OUT / "backbone_selection.json", best)
    print("LOCKED", best, flush=True)


def load_head(arm, seed, device):
    ck = torch.load(
        OUT / f"{arm}_{seed}/best.pt", map_location=device, weights_only=True
    )
    m = Head(arm).to(device).eval()
    m.load_state_dict(ck["model"])
    return m


def export(ds):
    conf = json.loads((OUT / "backbone_selection.json").read_text())
    selected = np.zeros((len(ds.meta), 1, *OUTPUT_GRID), "float32")
    for arm in ["previous", *conf["arms"]]:
        if arm == "previous":
            pred = resize(torch.tensor(np.array(ds.features["old"])), OUTPUT_GRID)
            pred = pred / pred.sum((-2, -1), keepdim=True)
        else:
            pred = sum(
                ds.predict(load_head(arm, s, ds.device), ds.allids)
                for s in conf["seeds"]
            ) / len(conf["seeds"])
            np.save(OUT / f"{arm}.npy", pred.numpy())
        selected += conf["weights"].get(arm, 0) * pred.numpy()
    assert np.max(np.abs(selected.sum((1, 2, 3)) - 1)) < 1e-5
    np.save(OUT / "dino_selected.npy", selected)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["train", "select", "export"])
    p.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    p.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch", type=int, default=20)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    configure(42)
    OUT.mkdir(exist_ok=True, parents=True)
    ds = Dataset(a.device)
    if a.action == "train":
        for seed in a.seeds:
            for arm in a.arms:
                train(ds, arm, seed, a.epochs, a.batch)
    elif a.action == "select":
        select(ds, a.arms, a.seeds)
    else:
        export(ds)
