"""Explicit, opt-in verification of the trained model artifact."""
import io
import os

import numpy as np
import pytest
from PIL import Image

from backend.prediction import classify, load_classes, load_model, model_paths, prepare_image


@pytest.mark.skipif(os.getenv("ECOVISION_REAL_MODEL") != "1", reason="Run explicitly with ECOVISION_REAL_MODEL=1; loading TensorFlow is resource intensive.")
def test_real_model_loads_and_predicts(app, client):
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

    # Simulate the browser dimensions and quality with a reproducible noisy source.
    pixels = np.random.default_rng(42).integers(0, 256, (1000, 1600, 3), dtype=np.uint8)
    original = Image.fromarray(pixels)
    source = io.BytesIO()
    original.save(source, format="PNG")
    assert source.tell() > 4 * 1024 * 1024
    source.seek(0)
    original_result = classify(model, classes, prepare_image(source, 25 * 1024 * 1024))
    assert original_result["category"] in classes

    prepared = io.BytesIO()
    original.resize((1024, 640), Image.Resampling.BILINEAR).save(prepared, format="JPEG", quality=85)
    assert prepared.tell() < int(4 * 1024 * 1024 * .95)
    prepared.seek(0)
    app.extensions["classifier"] = model
    response = client.post("/api/predict", data={"image": (prepared, "ecovision.jpg")})
    assert response.status_code == 201
    assert response.json["category"] in classes
    assert 0 <= response.json["confidence"] <= 1
    assert len(response.json["top3"]) == 3
