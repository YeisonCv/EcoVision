import io
import pytest
from PIL import Image
from app import create_app
from database import db


class FakeModel:
    def predict(self, batch, verbose=0):
        assert batch.shape == (1, 224, 224, 3)
        out = [0.0] * 12
        out[3] = .75
        out[8] = .15
        out[0] = .10
        return [out]


@pytest.fixture
def app():
    app = create_app({"TESTING": True, "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:"})
    app.extensions["classifier"] = FakeModel()
    app.extensions["classes"] = ["battery", "biological", "brown-glass", "cardboard", "clothes", "green-glass", "metal", "paper", "plastic", "shoes", "trash", "white-glass"]
    app.extensions["model_error"] = None
    with app.app_context():
        db.create_all()
        yield app
        db.session.remove()
        db.drop_all()


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def image_file():
    output = io.BytesIO()
    Image.new("RGB", (40, 30), (30, 120, 50)).save(output, format="JPEG")
    output.seek(0)
    return output
