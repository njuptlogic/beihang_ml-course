import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from src.training_monitor import monitor_run


def test_sampled_steps_and_epoch_streams(tmp_path, capsys):
    with monitor_run(tmp_path, "seed42/affine", interval=3, ema=True) as monitor:
        for epoch in (1, 2):
            for batch in range(1, 6):
                # 最后一批仅 2 张，准确率的分母须使用实际样本数。
                size = 2 if batch == 5 else 4
                monitor.batch(epoch, batch, 5, torch.tensor(0.3), torch.tensor(1), size, 0.001)
            assert capsys.readouterr().out == ""
            monitor.perturbation(epoch, "val", "shift_3px", 0.6, 0.75, 0.89)
            monitor.perturbation(epoch, "ema_val", "shift_3px", 0.25, 0.92, 0.91)
            monitor.epoch(dict(epoch=epoch, lr=0.001, seconds=1.2, train_loss=0.35,
                               train_accuracy=0.9, val_loss=0.4, val_accuracy=0.89,
                               ema_val_loss=0.3, ema_val_accuracy=0.91))
            lines = capsys.readouterr().out.splitlines()
            assert len(lines) == 1
            assert lines[0].startswith(f"seed42/affine epoch {epoch}  train loss=")
    directory = tmp_path / "tensorboard/seed42/affine"
    train = EventAccumulator(str(directory / "train")).Reload()
    accuracy = train.Scalars("batch/accuracy_pct")
    assert [p.step for p in accuracy] == [1, 3, 5, 6, 8, 10]
    assert [p.value for p in accuracy] == [25, 25, 50, 25, 25, 50]
    for split, expected in [("train", 90), ("val", 89), ("ema_val", 91)]:
        stream = EventAccumulator(str(directory / split)).Reload()
        points = stream.Scalars("epoch/accuracy_pct")
        assert [p.step for p in points] == [1, 2]
        assert [p.value for p in points] == pytest.approx([expected, expected])
    for split, (loss, acc_pct, drop_pp) in [
        ("val", (0.6, 75, 14)), ("ema_val", (0.25, 92, -1)),
    ]:
        stream = EventAccumulator(str(directory / split)).Reload()
        for tag, value in [("loss", loss), ("accuracy_pct", acc_pct), ("drop_pp", drop_pp)]:
            points = stream.Scalars(f"robustness/shift_3px/{tag}")
            assert [p.step for p in points] == [1, 2]
            assert [p.value for p in points] == pytest.approx([value, value])


def test_validation_views_use_fixed_geometry_and_training_background():
    from src.views import validation_views

    black = -0.42
    x = torch.full((2, 1, 15, 15), black)
    x[0, 0, 7, 7] = 1.
    x[0, 0, 10, 7] = .5
    initial_rng = torch.get_rng_state()
    views = {name: list(images) for name, images in validation_views(x, black)}
    assert torch.equal(initial_rng, torch.get_rng_state())
    for images in views.values():
        for transformed in images:
            torch.testing.assert_close(transformed[1], x[1])
    for shifted, (y, col) in zip(views["shift_3px"], [(7, 4), (7, 10), (4, 7), (10, 7)], strict=True):
        assert shifted[0, 0, y, col] == 1
    left, right = views["rotate_12deg"]
    torch.testing.assert_close(left.flip([-1]), right)
    assert not torch.allclose(left, right)
    large, small = views["grid_scale_10pct"]
    assert large[0, 0, 11, 7] > small[0, 0, 11, 7]
