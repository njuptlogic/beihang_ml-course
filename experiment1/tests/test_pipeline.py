import io
import zipfile

import numpy as np
from PIL import Image
import pytest
import torch

from src.data import read_archive
from src.evaluation import classification_metrics
from src.models import build_model


def make_zip(path, filenames, size=(28, 28)):
    with zipfile.ZipFile(path, "w") as z:
        for name in filenames:
            buf = io.BytesIO(); Image.new("L", size, 127).save(buf, format="BMP")
            z.writestr(name, buf.getvalue())


def test_label_is_filename_prefix_not_folder(tmp_path):
    path = tmp_path / "data.zip"
    make_zip(path, ["TrainingSet/9_20.bmp", "TrainingSet/2_1.bmp"])
    x, y, names = read_archive(path)
    assert x.shape == (2, 28, 28)
    assert y.tolist() == [2, 9]
    assert names.tolist() == ["TrainingSet/2_1.bmp", "TrainingSet/9_20.bmp"]


def test_reject_incorrect_image_count(tmp_path):
    path = tmp_path / "data.zip"; make_zip(path, ["1_0.bmp"])
    with pytest.raises(ValueError, match="期望"):
        read_archive(path, expected_count=60000)


@pytest.mark.parametrize("filename,size", [("unknown.bmp", (28, 28)), ("1_0.bmp", (32, 32))])
def test_reject_invalid_label_or_shape(tmp_path, filename, size):
    path = tmp_path / "data.zip"; make_zip(path, [filename], size)
    with pytest.raises(ValueError): read_archive(path)


@pytest.mark.parametrize("name", ["mlp", "cnn"])
def test_model_checkpoint_roundtrip(name, tmp_path):
    torch.set_num_threads(2)
    model = build_model(name).eval()
    x = torch.rand(3, 1, 28, 28)
    with torch.inference_mode(): expected = model(x)
    assert expected.shape == (3, 10)
    path = tmp_path / "model.pt"; torch.save(model.state_dict(), path)
    restored = build_model(name).eval()
    restored.load_state_dict(torch.load(path, weights_only=True))
    with torch.inference_mode(): actual = restored(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_metrics_against_known_predictions():
    y = np.array([0, 1, 2, 2])
    probabilities = np.full((4, 10), .01)
    probabilities[np.arange(4), [0, 1, 1, 2]] = .91
    metrics = classification_metrics(y, probabilities)
    assert metrics["accuracy"] == .75
    assert metrics["errors"] == 1
    assert metrics["confusion_matrix"][2][1] == 1
    assert metrics["classification_report"]["2"]["recall"] == .5


@pytest.mark.parametrize("name", ["cnn", "mlp"])
def test_transfer_shape(name):
    # 非方形、奇数边长、RGB 与非十分类同时覆盖。
    model = build_model(name, input_dim=(3, 31, 35), num_classes=5).eval()
    assert model(torch.zeros(2, 3, 31, 35)).shape == (2, 5)


def test_view_mix_matches_explicit_neighbors():
    from src.views import ShiftAverage

    # A position-sensitive readout exposes padding, radius and mixing errors.
    class ReadPixel(torch.nn.Module):
        def forward(self, images):
            return images[:, 0, 0, :1]

    x = torch.arange(9, dtype=torch.float32).reshape(1, 1, 3, 3)
    wrapper = ShiftAverage(ReadPixel(), radius=1, mix=.5, black=-1.)
    expected = .5 * 0 + .5 * (1 + 3 - 1 - 1) / 4
    torch.testing.assert_close(wrapper(x), torch.tensor([[expected]]))
