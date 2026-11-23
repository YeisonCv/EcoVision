import io
from flask import request


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


def test_pagination_validation(client):
    assert client.get("/api/predictions?page=bad").status_code == 400
