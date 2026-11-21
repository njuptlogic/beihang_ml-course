"""验证与推理的几何变换，图外填原始黑色。"""
import math

import torch
from torch import nn
from torch.nn import functional as F


def translated_views(x, radius, black):
    h, w = x.shape[-2:]
    padded = F.pad(x, (radius,) * 4, value=black)
    for dx, dy in [(-radius, 0), (radius, 0), (0, -radius), (0, radius)]:
        a, b = radius - dy, radius - dx
        yield padded[..., a:a+h, b:b+w]


def sample_affine(x, theta, black):
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x - black, grid, align_corners=False) + black


def fixed_affine_views(x, settings, black):
    for degrees, scale in settings:
        angle = math.radians(degrees)
        c, s = scale * math.cos(angle), scale * math.sin(angle)
        theta = x.new_tensor([[c, -s, 0], [s, c, 0]]).expand(len(x), -1, -1)
        yield sample_affine(x, theta, black)


def validation_views(x, black):
    """每种条件都从原图出发，逐视图评估，再对该条件取均值。"""
    yield "shift_3px", translated_views(x, 3, black)
    yield "rotate_12deg", fixed_affine_views(x, [(-12, 1), (12, 1)], black)
    yield "grid_scale_10pct", fixed_affine_views(x, [(0, .9), (0, 1.1)], black)


class ShiftAverage(nn.Module):
    def __init__(self, backbone, radius, mix, black):
        super().__init__()
        self.backbone = backbone
        self.radius = radius
        self.mix = mix
        self.black = black

    def forward(self, x):
        center = self.backbone(x)
        if self.mix == 0:
            return center
        neighbors = [self.backbone(v) for v in translated_views(x, self.radius, self.black)]
        return torch.lerp(center, torch.stack(neighbors).mean(0), self.mix)


def inference_model(checkpoint):
    from .models import build_model
    model = build_model(checkpoint["model_name"])
    model.load_state_dict(checkpoint["state_dict"])
    if checkpoint.get("view_mix", 0):
        model = ShiftAverage(model, checkpoint["view_radius"], checkpoint["view_mix"],
                             -checkpoint["mean"] / checkpoint["std"])
    return model.eval()
