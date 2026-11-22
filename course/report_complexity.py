"""Measure the main neural blocks at their reporting resolutions (one multiply-add = two FLOPs)."""
import torch
from torch.utils.flop_counter import FlopCounterMode
from course.common import ROOT, table
from experiment2.retinal_features import backbone
from experiment3.multiscale_train import Head
from experiment3.semantic_features import load_encoder
from experiment4.model import FlowNet


def measure(module, call):
    with torch.no_grad(), FlopCounterMode(display=False) as mode:
        call()
    return dict(
        parameters=sum(p.numel() for p in module.parameters()),
        counted_flops=mode.get_total_flops(),
    )


def main():
    torch.set_num_threads(2)
    device = torch.device("cpu")
    vit, _ = backbone("natural", device)
    report(2, [dict(module="DINOv2", input="392x392", **measure(
        vit, lambda: vit.forward_features(torch.zeros(1, 3, 392, 392))
    ))])
    encoder, _ = load_encoder(device)
    gaze = []
    for height, width in [(126, 224), (252, 448), (378, 672)]:
        def forward():
            encoder.get_intermediate_layers(
                torch.zeros(1, 3, height, width), [5, 11], return_class_token=True
            )

        gaze.append(
            dict(
                module="DINOv2", input=f"{width}x{height}", **measure(encoder, forward)
            )
        )
    for arm in ["pyramid", "dual"]:
        head = Head(arm).eval()
        inputs = [
            torch.zeros(1, 128, 18, 32),
            torch.zeros(1, 128, 24, 40),
            torch.zeros(
                1, 128, 18 if arm == "dual" else 9, 32 if arm == "dual" else 16
            ),
            torch.zeros(1, 128, 27, 48),
            torch.zeros(1, 48),
            torch.full((1, 1, 45, 80), 1 / 3600),
            None,
        ]
        gaze.append(
            dict(
                module=arm,
                input="one density head",
                **measure(head, lambda: head(*inputs)),
            )
        )
    report(3, gaze)
    flow = FlowNet().eval()
    params = [
        dict(
            module=k,
            input="component",
            parameters=sum(p.numel() for p in m.parameters()),
            counted_flops=None,
        )
        for k, m in flow.named_children()
    ]
    params.append(
        dict(
            module="FlowNet",
            input="32x32 one condition",
            **measure(
                flow,
                lambda: flow(
                    torch.zeros(1, 3, 32, 32),
                    torch.zeros(1),
                    torch.zeros(1, dtype=torch.long),
                ),
            ),
        )
    )
    report(4, params)


def report(n, rows):
    target = ROOT / f"experiment{n}/outputs/report_replay"
    target.mkdir(parents=True, exist_ok=True)
    table(target / "complexity.csv", rows)


if __name__ == "__main__":
    main()
