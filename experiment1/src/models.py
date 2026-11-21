"""从零训练的 MLP 基线与 CNN 主模型；输出 logits，不提前做 softmax。"""
from torch import nn


def build_model(name: str, input_dim: tuple[int, int, int] = (1, 28, 28),
                num_classes: int = 10) -> nn.Module:
    """input_dim 为 (C,H,W)。默认结构和旧权重的层序号兼容。"""
    channels, height, width = input_dim
    if min(channels, height, width, num_classes) < 1:
        raise ValueError("输入尺寸和类别数必须为正")
    if name == "mlp":
        return nn.Sequential(
            nn.Flatten(), nn.Linear(channels * height * width, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, num_classes),
        )
    if name == "cnn":
        if min(height, width) < 4:
            raise ValueError("两次池化要求输入高宽至少为 4")
        return nn.Sequential(
            nn.Conv2d(channels, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64 * (height // 4) * (width // 4), 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )
    raise ValueError(f"未知模型: {name}")
