"""用已训练模型识别单张数字图像。输入默认应为黑底白字。"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch

from src.data import ROOT
from src.views import inference_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "outputs/main/cnn/best.pt")
    parser.add_argument("--invert", action="store_true", help="白底黑字图像可使用此选项反色")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = inference_model(checkpoint)
    with Image.open(args.image) as image:
        image = image.convert("L")
        if args.invert: image = ImageOps.invert(image)
        if image.size != (28, 28): image = image.resize((28, 28), Image.Resampling.BILINEAR)
        x = torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).reshape(1, 1, 28, 28) / 255
    x = (x - checkpoint["mean"]) / checkpoint["std"]
    with torch.inference_mode(): probabilities = model(x).softmax(1)[0]
    print(json.dumps({"image": str(args.image), "prediction": int(probabilities.argmax()),
                      "probabilities": {str(i): float(v) for i, v in enumerate(probabilities)}}, indent=2))


if __name__ == "__main__":
    main()
