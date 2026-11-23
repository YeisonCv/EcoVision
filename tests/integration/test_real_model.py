"""Explicit, opt-in verification of the trained model artifact."""
import io
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from backend.prediction import classify, load_classes, load_model, model_paths, prepare_image


@pytest.mark.skipif(os.getenv("ECOVISION_REAL_MODEL") != "1", reason="Run explicitly with ECOVISION_REAL_MODEL=1; loading TensorFlow is resource intensive.")
def test_real_model_loads_and_predicts():
    model_path, classes_path = model_paths()
    if not model_path.is_file() or not classes_path.is_file():
        pytest.skip("Modelo o classes.txt no disponible.")
    classes = load_classes(classes_path)
    model = load_model(model_path)
    img = io.BytesIO()
    Image.new("RGB", (224, 224), (127, 127, 127)).save(img, format="PNG")
    img.seek(0)
    batch = prepare_image(img)
    raw = np.asarray(model.predict(batch, verbose=0)).reshape(-1)
    result = classify(model, classes, batch)
    assert raw.shape == (12,) and np.isfinite(raw).all()
    assert np.all((raw >= 0) & (raw <= 1))
    assert result["category"] == classes[int(np.argmax(raw))]
    assert 0 <= result["confidence"] <= 1
