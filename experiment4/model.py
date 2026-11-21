"""A class-conditional velocity field on a losslessly packed 16x16 grid."""
import math
import torch
from torch import nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, c_in, c_out, emb=128):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.GroupNorm(8, c_in), nn.SiLU(), nn.Conv2d(c_in, c_out, 3, padding=1)
        )
        self.affine = nn.Linear(emb, 2 * c_out)
        self.norm = nn.GroupNorm(8, c_out)
        self.out_conv = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x, emb):
        h = self.in_conv(x)
        scale, bias = self.affine(emb)[:, :, None, None].chunk(2, 1)
        h = self.norm(h) * (1 + scale) + bias
        return self.skip(x) + self.out_conv(F.silu(h))


class FlowNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer(
            "freq", torch.exp(-math.log(10000) * torch.arange(32) / 31)
        )
        self.time = nn.Sequential(nn.Linear(64, 128), nn.SiLU(), nn.Linear(128, 128))
        self.label = nn.Embedding(11, 128)
        self.stem = nn.Conv2d(12, 48, 3, padding=1)
        self.enc = Block(48, 48)
        self.down = nn.Conv2d(48, 96, 3, 2, 1)
        self.mid1 = Block(96, 96)
        self.attn_norm = nn.GroupNorm(8, 96)
        self.qkv = nn.Conv2d(96, 288, 1)
        self.proj = nn.Conv2d(96, 96, 1)
        self.mid2 = Block(96, 96)
        self.up = nn.Conv2d(96, 48, 3, padding=1)
        self.dec = Block(96, 48)
        self.head = nn.Sequential(
            nn.GroupNorm(8, 48), nn.SiLU(), nn.Conv2d(48, 12, 3, padding=1)
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x, t, y, return_features=False):
        angle = t[:, None] * 1000 * self.freq[None]
        emb = self.time(torch.cat((angle.sin(), angle.cos()), 1)) + self.label(y)
        skip = self.enc(self.stem(F.pixel_unshuffle(x, 2)), emb)
        h = self.mid1(self.down(skip), emb)
        b, c, hh, ww = h.shape
        q, k, v = (
            self.qkv(self.attn_norm(h))
            .reshape(b, 3, 4, 24, hh * ww)
            .permute(1, 0, 2, 4, 3)
            .unbind()
        )
        att = F.scaled_dot_product_attention(
            q.contiguous(), k.contiguous(), v.contiguous()
        )
        context = h + self.proj(att.transpose(2, 3).reshape(b, c, hh, ww))
        h = self.up(
            F.interpolate(self.mid2(context, emb), scale_factor=2, mode="nearest")
        )
        velocity = F.pixel_shuffle(self.head(self.dec(torch.cat((skip, h), 1), emb)), 2)
        return (velocity, context) if return_features else velocity


class PreconditionedFlow(FlowNet):
    """Separate the scalar linear predictor from the learnable velocity residual."""

    def forward(self, x, t, y):
        time = t[:, None, None, None]
        sigma = 0.5
        variance = time.square() * sigma**2 + (1 - time).square()
        skip = (time * sigma**2 - (1 - time)) / variance
        scale = variance.rsqrt()
        residual = super().forward(x * scale, t, y)
        return skip * x + sigma * scale * residual.float()


def make_model(parameterization):
    return {"direct": FlowNet, "preconditioned": PreconditionedFlow}[parameterization]()


@torch.inference_mode()
def sample(model, noise, labels, steps=12, guidance=1.6, interval=(0.0, 1.0)):
    x = noise.clone()
    for i in range(steps):
        t = x.new_full((len(x),), i / steps)
        with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.is_cuda):
            if guidance == 1 or not interval[0] <= i / steps < interval[1]:
                v = model(x, t, labels)
            else:
                cond, uncond = model(
                    torch.cat((x, x)),
                    torch.cat((t, t)),
                    torch.cat((labels, torch.full_like(labels, 10))),
                ).chunk(2)
                v = uncond + guidance * (cond - uncond)
        x = x + v.float() / steps
    return ((x.clamp(-1, 1) + 1) * 127.5).round().byte()
