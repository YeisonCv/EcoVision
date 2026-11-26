"""Image validation and inference for the trained EcoVision classifier."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile, UnidentifiedImageError

ImageFile.LOAD_TRUNCATED_IMAGES = False
ALLOWED_FORMATS = {"JPEG", "PNG"}
# Bound decoded RGB memory (roughly 60 MiB at the limit, excluding Pillow/model
# buffers) as well as compressed upload size, which is bounded by the caller.
MAX_IMAGE_PIXELS = 20_000_000


class ImageValidationError(ValueError):
    pass


class UnsupportedImageExtensionError(ValueError):
    pass


class ModelUnavailableError(RuntimeError):
    pass


def load_classes(path: str | Path) -> list[str]:
    try:
        classes = [line.strip() for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    except OSError as exc:
        raise ModelUnavailableError("No se pudo leer el archivo de clases.") from exc
    if len(classes) != 12 or len(set(classes)) != 12:
        raise ModelUnavailableError("classes.txt debe contener exactamente 12 clases únicas.")
    return classes


def validate_upload(filename: str, stream) -> None:
    """Check empty content before the extension, without decoding image bytes."""
    first_byte = stream.read(1)
    stream.seek(0)
    if not first_byte:
        raise ImageValidationError("El archivo está vacío.")
    if Path(filename).suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        raise UnsupportedImageExtensionError("Formato no permitido. Usa JPG, JPEG o PNG.")


def prepare_image(stream, max_bytes: int = 10 * 1024 * 1024) -> np.ndarray:
    data = stream.read(max_bytes + 1)
    if not data:
        raise ImageValidationError("El archivo está vacío.")
    if len(data) > max_bytes:
        raise ImageValidationError("La imagen supera el tamaño máximo permitido.")
    try:
        from io import BytesIO
        with Image.open(BytesIO(data)) as image:
            if image.format not in ALLOWED_FORMATS:
                raise ImageValidationError("Formato no permitido. Usa JPG, JPEG o PNG.")
            if image.width < 1 or image.height < 1 or image.width * image.height > MAX_IMAGE_PIXELS:
                raise ImageValidationError("Las dimensiones de la imagen no son válidas.")
            image.verify()
        with Image.open(BytesIO(data)) as image:
            image = image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
            return np.asarray(image, dtype=np.float32)[None, ...]
    except ImageValidationError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ImageValidationError("El archivo no contiene una imagen válida o está dañado.") from exc


def load_model(path: str | Path):
    if not Path(path).is_file():
        raise ModelUnavailableError("No se encontró el modelo entrenado.")
    try:
        import tensorflow as tf
        return tf.keras.models.load_model(path, compile=False)
    except Exception as exc:
        raise ModelUnavailableError("No se pudo cargar el modelo entrenado; revisa la versión de TensorFlow/Keras.") from exc


def classify(model, classes: list[str], batch: np.ndarray) -> dict:
    try:
        raw = model.predict(batch, verbose=0)
        probabilities = np.asarray(raw, dtype=np.float64).reshape(-1)
    except Exception as exc:
        raise ModelUnavailableError("Falló la inferencia del modelo.") from exc
    if (probabilities.size != len(classes) or not np.isfinite(probabilities).all()
            or (probabilities < 0).any() or (probabilities > 1).any()
            or not np.isclose(probabilities.sum(), 1.0, atol=1e-3)):
        raise ModelUnavailableError("El modelo devolvió una salida inválida.")
    index = int(np.argmax(probabilities))
    top = np.argsort(probabilities)[::-1][:3]
    return {"category": classes[index], "confidence": float(probabilities[index]),
            "top3": [{"category": classes[int(i)], "confidence": float(probabilities[i])} for i in top]}


def model_paths():
    root = Path(os.getenv("MODEL_DIR", Path(__file__).resolve().parents[1] / "modelo"))
    return root / os.getenv("MODEL_FILENAME", "waste_classifier.keras"), root / os.getenv("CLASSES_FILENAME", "classes.txt")
