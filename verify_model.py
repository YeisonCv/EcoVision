"""Run the real model smoke check: python verify_model.py"""
import io
import numpy as np
from PIL import Image
from backend.prediction import classify, load_classes, load_model, model_paths, prepare_image

model_path, classes_path = model_paths()
classes = load_classes(classes_path)
model = load_model(model_path)
sample = io.BytesIO()
Image.new("RGB", (224, 224), (128, 128, 128)).save(sample, format="PNG")
sample.seek(0)
batch = prepare_image(sample)
raw = np.asarray(model.predict(batch, verbose=0)).reshape(-1)
result = classify(model, classes, batch)
assert len(classes) == 12
assert raw.shape == (12,) and np.isfinite(raw).all() and np.all((raw >= 0) & (raw <= 1))
assert result["category"] == classes[int(np.argmax(raw))]
assert 0 <= result["confidence"] <= 1
print(f"Modelo verificado: {len(classes)} clases, salida válida, categoría={result['category']}, confianza={result['confidence']:.4f}")
