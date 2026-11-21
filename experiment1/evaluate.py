"""独立重载 checkpoint 并复核测试指标，无需重新训练。"""
import argparse
import json
from pathlib import Path

from run_experiment import tensor_images
from src.data import ROOT, prepare_data
from src.evaluation import classification_metrics, predict_probabilities
from src.views import inference_model
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "outputs/main/cnn/best.pt")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, help="可选：另存 JSON 评价结果")
    args = parser.parse_args()
    torch.set_num_threads(4)
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    data, manifest = prepare_data(seed=checkpoint.get("split_seed", checkpoint["seed"]))
    if checkpoint["archive_sha256"] != manifest["archive_sha256"]:
        raise RuntimeError("当前原始数据与训练 checkpoint 的数据校验值不一致")
    model = inference_model(checkpoint).to(args.device)
    normalizer = {"mean": checkpoint["mean"], "std": checkpoint["std"]}
    probabilities = predict_probabilities(model, tensor_images(data["test_x"], normalizer, args.device))
    result = classification_metrics(data["test_y"], probabilities)
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
