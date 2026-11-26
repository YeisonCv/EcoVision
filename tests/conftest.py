import io
import os
from datetime import datetime, timezone

import pytest
from PIL import Image

# A syntactically valid, unreachable PostgreSQL URL. Tests replace persistence and
# never establish a connection to this address or to the user's Supabase project.
os.environ["DATABASE_URL"] = "postgresql+psycopg://test:test@127.0.0.1:1/ecovision_test"
os.environ["PYTHON_DOTENV_DISABLED"] = "1"

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


class FakeQuery:
    def __init__(self, session, *entities):
        self.session = session
        self.entities = entities

    def filter(self, *_args, **_kwargs):
        return self

    def order_by(self, *_args, **_kwargs):
        return self

    def group_by(self, *_args, **_kwargs):
        return self

    def limit(self, *_args, **_kwargs):
        return self

    def scalar(self):
        from sqlalchemy import func
        expression = self.entities[0]
        name = getattr(expression, "name", None)
        if name == "count":
            return len(self.session.records)
        if name == "avg":
            values = [record.confidence for record in self.session.records]
            return sum(values) / len(values) if values else None
        return None

    def all(self):
        if len(self.entities) != 2:
            return list(self.session.records)
        first = self.entities[0]
        if getattr(first, "name", None) == "date":
            counts = {}
            for record in self.session.records:
                day = record.created_at.date()
                counts[day] = counts.get(day, 0) + 1
            return sorted(counts.items())
        counts = {}
        for record in self.session.records:
            counts[record.category] = counts.get(record.category, 0) + 1
        return sorted(counts.items(), key=lambda pair: pair[1], reverse=True)

    def first(self):
        records = list(self.session.records)
        return records[-1] if records else None

    def paginate(self, page=1, per_page=20, error_out=False):
        records = list(self.session.records)
        start = (page - 1) * per_page
        return type("Pagination", (), {
            "items": records[start:start + per_page], "total": len(records),
            "pages": (len(records) + per_page - 1) // per_page,
        })()


class FakeSession:
    def __init__(self):
        self.records = []
        self.next_id = 1

    def __call__(self):
        return self

    def query(self, *entities):
        return FakeQuery(self, *entities)

    def execute(self, *_args, **_kwargs):
        return object()

    def add(self, record):
        record.id = self.next_id
        self.next_id += 1
        record.created_at = datetime.now(timezone.utc)
        self.records.append(record)

    def commit(self):
        pass

    def flush(self):
        pass

    def rollback(self):
        pass

    def remove(self):
        pass

    def get(self, model, record_id):
        return next((record for record in self.records if record.id == record_id), None)


@pytest.fixture
def app():
    from database import db
    from database.models import Prediction
    from unittest.mock import patch

    session = FakeSession()
    with patch.object(db, "session", session), patch.object(Prediction, "query", FakeQuery(session, Prediction)):
        app = create_app({"TESTING": True, "RATELIMIT_ENABLED": False,
                          "CORS_ALLOWED_ORIGINS": {"https://frontend.example"},
                          "SQLALCHEMY_DATABASE_URI": os.environ["DATABASE_URL"]})
        app.extensions["classifier"] = FakeModel()
        app.extensions["classes"] = ["battery", "biological", "brown-glass", "cardboard", "clothes", "green-glass", "metal", "paper", "plastic", "shoes", "trash", "white-glass"]
        app.extensions["model_error"] = None
        app.extensions["test_session"] = session
        yield app
        session.remove()


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def image_file():
    output = io.BytesIO()
    Image.new("RGB", (40, 30), (30, 120, 50)).save(output, format="JPEG")
    output.seek(0)
    return output
