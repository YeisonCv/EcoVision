"""One real model prediction against the ephemeral Docker PostgreSQL stack."""
from __future__ import annotations

import json
import os
from io import BytesIO
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from uuid import uuid4

import pytest
from PIL import Image
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url


def _isolated_database_url() -> str:
    value = os.getenv("ECOVISION_TEST_DATABASE_URL")
    if not value:
        pytest.skip("E2E requiere ECOVISION_TEST_DATABASE_URL del contenedor PostgreSQL aislado.")
    url = make_url(value)
    expected_port = int(os.getenv("TEST_POSTGRES_PORT", "55432"))
    if (url.drivername != "postgresql+psycopg" or url.host != "127.0.0.1"
            or url.port != expected_port or url.database != "ecovision_test"
            or url.username != "ecovision_test"):
        pytest.fail("E2E solo acepta el PostgreSQL local ecovision_test; no se muestra la URL.", pytrace=False)
    return value


def _request(base_url: str, path: str, *, method="GET", body=None, content_type=None):
    headers = {"Content-Type": content_type} if content_type else {}
    request = Request(base_url.rstrip("/") + path, data=body, headers=headers, method=method)
    try:
        response = urlopen(request, timeout=90)
    except HTTPError as error:
        response = error
    with response:
        return response.status, json.loads(response.read().decode("utf-8"))


@pytest.mark.e2e
def test_real_prediction_persists_to_isolated_postgresql():
    database_url = _isolated_database_url()
    base_url = os.getenv("ECOVISION_E2E_BASE_URL", "http://127.0.0.1:5002")
    if not base_url.startswith("http://127.0.0.1:"):
        pytest.fail("E2E solo admite el puerto de loopback local.", pytrace=False)

    # Verify identity, expected baseline, and model/DB availability before the
    # single POST that inserts the E2E record.
    engine = create_engine(database_url, pool_pre_ping=True)
    with engine.connect() as connection:
        identity = connection.execute(text("SELECT current_database(), current_user")).one()
        assert identity == ("ecovision_test", "ecovision_test")
        revisions = connection.execute(text("SELECT version_num FROM alembic_version")).scalars().all()
        assert revisions == ["85b59e312c4e"]
        count_before = connection.execute(text("SELECT count(*) FROM predictions")).scalar_one()

    status, health = _request(base_url, "/api/health")
    assert status == 200
    assert health == {"status": "ok", "model": "available", "database": "available"}

    image_buffer = BytesIO()
    Image.new("RGB", (96, 96), color=(70, 130, 90)).save(image_buffer, format="JPEG", quality=90)
    filename = f"ecovision-e2e-{uuid4().hex}.jpg"
    boundary = f"----EcoVisionBoundary{uuid4().hex}"
    payload = (f"--{boundary}\r\n"
               f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'
               "Content-Type: image/jpeg\r\n\r\n").encode("ascii") + image_buffer.getvalue()
    payload += f"\r\n--{boundary}--\r\n".encode("ascii")
    status, prediction = _request(base_url, "/api/predict", method="POST", body=payload,
                                  content_type=f"multipart/form-data; boundary={boundary}")
    assert status == 201
    assert prediction["category"] in {
        "battery", "biological", "brown-glass", "cardboard", "clothes", "green-glass",
        "metal", "paper", "plastic", "shoes", "trash", "white-glass",
    }
    assert 0 <= prediction["confidence"] <= 1
    record = prediction["prediction"]
    assert record["filename"] == filename
    assert record["category"] == prediction["category"]
    assert record["confidence"] == pytest.approx(prediction["confidence"])

    with engine.connect() as connection:
        identity = connection.execute(text("SELECT current_database(), current_user")).one()
        assert identity == ("ecovision_test", "ecovision_test")
        rows = connection.execute(text(
            "SELECT id, filename, category, confidence FROM predictions WHERE filename = :filename"
        ), {"filename": filename}).all()
        count_after = connection.execute(text("SELECT count(*) FROM predictions")).scalar_one()
    assert len(rows) == 1
    assert rows[0].id == record["id"]
    assert rows[0].category == prediction["category"]
    assert rows[0].confidence == pytest.approx(prediction["confidence"])
    assert count_after == count_before + 1

    status, history = _request(base_url, "/api/predictions?" + urlencode({"per_page": 100}))
    assert status == 200
    assert any(item["id"] == record["id"] and item["filename"] == filename for item in history["items"])
    status, statistics = _request(base_url, "/api/statistics")
    assert status == 200
    assert statistics["total"] == count_after
    assert any(row["category"] == prediction["category"] and row["count"] >= 1
               for row in statistics["by_category"])
    engine.dispose()
