import io
import numpy as np
import pytest
from PIL import Image
from backend.prediction import ImageValidationError, classify, load_classes, prepare_image


def test_prepare_image_converts_rgb_and_resizes(image_file):
    batch = prepare_image(image_file)
    assert batch.shape == (1, 224, 224, 3)
    assert batch.dtype == np.float32
    assert batch.min() >= 0 and batch.max() <= 255


@pytest.mark.parametrize("payload", [b"", b"not an image"])
def test_invalid_image_rejected(payload):
    with pytest.raises(ImageValidationError):
        prepare_image(io.BytesIO(payload))


def test_image_size_limit():
    with pytest.raises(ImageValidationError, match="tamaño máximo"):
        prepare_image(io.BytesIO(b"x" * 12), max_bytes=10)


def test_class_indices_and_top_three():
    classes = [f"class-{i}" for i in range(12)]
    class Model:
        def predict(self, batch, verbose=0):
            values = np.zeros((1, 12)); values[0, 4] = .7; values[0, 1] = .2; values[0, 8] = .1
            return values
    result = classify(Model(), classes, np.zeros((1, 224, 224, 3)))
    assert result["category"] == "class-4"
    assert result["confidence"] == pytest.approx(.7)
    assert [x["category"] for x in result["top3"]] == ["class-4", "class-1", "class-8"]


def test_bad_prediction_shape_rejected():
    class Model:
        def predict(self, batch, verbose=0): return [[.5, .5]]
    with pytest.raises(Exception, match="salida inválida"):
        classify(Model(), [str(i) for i in range(12)], np.zeros((1, 224, 224, 3)))
