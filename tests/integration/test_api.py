import io
from flask import request
from sqlalchemy.exc import SQLAlchemyError

from app import create_app


def test_health_reports_model_and_database(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json == {"status": "ok", "model": "available", "database": "available"}


def test_prediction_persists_and_appears_in_history(client, image_file):
    response = client.post("/api/predict", data={"image": (image_file, "test.jpg")}, content_type="multipart/form-data")
    assert response.status_code == 201
    assert response.json["category"] == "cardboard"
    assert response.json["confidence"] == .75
    assert response.json["prediction"]["filename"] == "test.jpg"
    listing = client.get("/api/predictions")
    assert listing.status_code == 200 and listing.json["total"] == 1
    detail = client.get(f"/api/predictions/{listing.json['items'][0]['id']}")
    assert detail.status_code == 200


def test_prediction_rejects_missing_empty_and_unsupported_files(client):
    received_file = {}

    @client.application.before_request
    def inspect_uploaded_file():
        uploaded = request.files.get("image")
        if uploaded is not None:
            content = uploaded.stream.read()
            uploaded.stream.seek(0)
            received_file.update(filename=uploaded.filename, content=content)

    assert client.post("/api/predict").status_code == 400
    empty_file = io.BytesIO(b"")
    response = client.post("/api/predict", data={"image": (empty_file, "empty.jpg")}, content_type="multipart/form-data")
    assert response.status_code == 400

    gif_file = io.BytesIO(b"abc")
    gif_response = client.post("/api/predict", data={"image": (gif_file, "x.gif")}, content_type="multipart/form-data")
    assert received_file == {"filename": "x.gif", "content": b"abc"}
    assert len(received_file["content"]) == 3
    assert gif_response.status_code == 415, gif_response.get_json()
    assert client.post("/api/predict", data={"image": (b"abc", "corrupt.jpg")}, content_type="multipart/form-data").status_code == 400


def test_statistics_and_not_found(client):
    response = client.get("/api/statistics")
    assert response.status_code == 200 and response.json["total"] == 0
    assert client.get("/api/predictions/999").status_code == 404
    missing_route = client.get("/api/predictions/not-an-id")
    assert missing_route.status_code == 404 and missing_route.is_json


def test_pagination_validation(client):
    assert client.get("/api/predictions?page=bad").status_code == 400
    for query in ("page=0", "page=10001", "per_page=0", "per_page=101",
                  "date_from=2026-02-30", "date_from=2026-02-02&date_to=2026-02-01",
                  "category=not-a-class", "date_from=", "page=1&page=2"):
        response = client.get(f"/api/predictions?{query}")
        assert response.status_code == 400, query


def test_upload_size_and_decoded_dimensions_are_bounded(client, app, image_file, monkeypatch):
    app.config.update(MAX_CONTENT_LENGTH=256, MAX_IMAGE_MB=1)
    oversized = client.post("/api/predict", data={"image": (io.BytesIO(b"x" * 1024), "large.jpg")},
                            content_type="multipart/form-data")
    assert oversized.status_code == 413
    app.config.update(MAX_CONTENT_LENGTH=10 * 1024 * 1024)

    from backend import prediction
    monkeypatch.setattr(prediction, "MAX_IMAGE_PIXELS", 100)
    from PIL import Image
    too_many_pixels = io.BytesIO()
    Image.new("RGB", (11, 10)).save(too_many_pixels, format="JPEG")
    too_many_pixels.seek(0)
    response = client.post("/api/predict", data={"image": (too_many_pixels, "large-dimensions.jpg")},
                           content_type="multipart/form-data")
    assert response.status_code == 400


def test_inference_failure_returns_safe_error_and_does_not_persist(client, app, image_file):
    class BrokenModel:
        def predict(self, *_args, **_kwargs):
            raise RuntimeError("internal model path or secret must not be exposed")
    app.extensions["classifier"] = BrokenModel()
    response = client.post("/api/predict", data={"image": (image_file, "test.jpg")},
                           content_type="multipart/form-data")
    assert response.status_code == 500
    assert response.json == {"error": "No fue posible analizar la imagen."}
    assert app.extensions["test_session"].records == []


def test_database_errors_return_safe_service_errors(client, app, monkeypatch):
    session = app.extensions["test_session"]
    def broken_query(*_args, **_kwargs):
        raise SQLAlchemyError("database details must not be returned")
    monkeypatch.setattr(session, "query", broken_query)
    from database.models import Prediction
    monkeypatch.setattr(type(Prediction.query), "paginate",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(SQLAlchemyError("query failed")))
    history = client.get("/api/predictions")
    stats = client.get("/api/statistics")
    assert history.status_code == 503 and "database details" not in history.get_data(as_text=True)
    assert stats.status_code == 503 and "database details" not in stats.get_data(as_text=True)


def test_health_reports_database_failure_without_details(client, app, monkeypatch):
    def broken_execute(*_args, **_kwargs):
        raise SQLAlchemyError("connection secret must not be returned")
    monkeypatch.setattr(app.extensions["test_session"], "execute", broken_execute)
    response = client.get("/api/health")
    assert response.status_code == 503
    assert response.json["database"] == "unavailable"
    assert "connection secret" not in response.get_data(as_text=True)


def test_cors_allows_only_configured_origins(client):
    allowed = client.get("/api/statistics", headers={"Origin": "https://frontend.example"})
    denied = client.get("/api/statistics", headers={"Origin": "https://attacker.example"})
    assert allowed.headers.get("Access-Control-Allow-Origin") == "https://frontend.example"
    assert denied.headers.get("Access-Control-Allow-Origin") is None
    assert "Access-Control-Allow-Credentials" not in allowed.headers


def test_prediction_rate_limit_uses_remote_address_without_trusting_forwarded_header():
    rate_app = create_app({"TESTING": True, "RATELIMIT_ENABLED": True,
                           "RATELIMIT_STORAGE_URI": "memory://",
                           "SQLALCHEMY_DATABASE_URI": "postgresql+psycopg://test:test@127.0.0.1:1/ecovision_test"})
    client = rate_app.test_client()
    statuses = [client.post("/api/predict", headers={"X-Forwarded-For": f"198.51.100.{n}"}).status_code
                for n in range(1, 7)]
    assert statuses[:5] == [400] * 5
    assert statuses[5] == 429


def test_proxy_headers_are_used_only_when_trusted_hops_are_explicit():
    proxy_app = create_app({"TESTING": True, "TRUSTED_PROXY_HOPS": 1,
                            "SQLALCHEMY_DATABASE_URI": "postgresql+psycopg://test:test@127.0.0.1:1/ecovision_test"})
    proxy_app.add_url_rule("/_test_remote_addr", view_func=lambda: request.remote_addr)
    response = proxy_app.test_client().get("/_test_remote_addr",
        headers={"X-Forwarded-For": "198.51.100.10, 198.51.100.20"},
        environ_overrides={"REMOTE_ADDR": "203.0.113.5"})
    assert proxy_app.config["TRUSTED_PROXY_HOPS"] == 1
    assert response.get_data(as_text=True) == "198.51.100.20"
