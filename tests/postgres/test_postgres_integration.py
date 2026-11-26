from __future__ import annotations

import os
from datetime import datetime, timezone
from uuid import uuid4

import pytest
from alembic import command
from sqlalchemy import DateTime, Float, Integer, inspect, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import IntegrityError

from app import create_app, migrate
from database import db
from database.models import Prediction

pytestmark = pytest.mark.postgres_integration
INITIAL_REVISION = "85b59e312c4e"
EXPECTED_TABLES = {"alembic_version", "predictions"}
EXPECTED_PUBLIC_OBJECTS = {
    "alembic_version": "r",
    "predictions": "r",
    "predictions_id_seq": "S",
}


def _isolated_test_url() -> str:
    value = os.getenv("ECOVISION_TEST_DATABASE_URL")
    if not value:
        pytest.skip("Define ECOVISION_TEST_DATABASE_URL para habilitar PostgreSQL de pruebas.")
    try:
        url = make_url(value)
        expected_port = int(os.getenv("TEST_POSTGRES_PORT", "55432"))
    except (TypeError, ValueError):
        pytest.fail("La configuración PostgreSQL de pruebas es inválida; no se muestra la URL.", pytrace=False)
    if (url.drivername != "postgresql+psycopg" or url.host != "127.0.0.1"
            or url.port != expected_port or url.database != "ecovision_test"
            or url.username != "ecovision_test"):
        pytest.fail(
            "Se rechazó la conexión: las pruebas solo admiten 127.0.0.1, el puerto local configurado, "
            "la base ecovision_test y el usuario ecovision_test. La URL no se muestra.",
            pytrace=False,
        )
    return value


@pytest.fixture(scope="session")
def postgres_app():
    url = _isolated_test_url()
    test_app = create_app({
        "TESTING": True,
        "RATELIMIT_ENABLED": False,
        "SQLALCHEMY_DATABASE_URI": url,
        "SQLALCHEMY_ENGINE_OPTIONS": {
            "pool_pre_ping": True,
            "pool_size": 2,
            "max_overflow": 0,
            "connect_args": {"application_name": "ecovision_postgres_integration"},
        },
    })

    with test_app.app_context():
        try:
            with db.engine.connect() as connection:
                identity = connection.execute(
                    text("SELECT current_database(), current_user")
                ).one()
                if identity != ("ecovision_test", "ecovision_test"):
                    pytest.fail(
                        "PostgreSQL rechazado: current_database/current_user no corresponden al entorno aislado.",
                        pytrace=False,
                    )

                tables = set(inspect(connection).get_table_names(schema="public"))
                public_objects = dict(connection.execute(text(
                    "SELECT c.relname, c.relkind FROM pg_class AS c "
                    "JOIN pg_namespace AS n ON n.oid = c.relnamespace "
                    "WHERE n.nspname = 'public' AND c.relkind IN ('r', 'p', 'v', 'm', 'S', 'f')"
                )).all())
                if not public_objects:
                    # Apply the migration only after checking the loopback URL and
                    # the live PostgreSQL database/user identity above.
                    command.upgrade(migrate.get_config(), INITIAL_REVISION)
                elif tables == EXPECTED_TABLES and public_objects == EXPECTED_PUBLIC_OBJECTS:
                    revisions = connection.execute(text("SELECT version_num FROM alembic_version")).scalars().all()
                    if revisions != [INITIAL_REVISION]:
                        pytest.fail(
                            "La base PostgreSQL de pruebas ya tiene una revisión inesperada; no se modificó.",
                            pytrace=False,
                        )
                    existing_rows = connection.execute(text("SELECT count(*) FROM predictions")).scalar_one()
                    if existing_rows:
                        pytest.fail(
                            "La base PostgreSQL de pruebas contiene registros; recrea solo el contenedor efímero "
                            "antes de repetir la suite.",
                            pytrace=False,
                        )
                else:
                    pytest.fail(
                        "El esquema PostgreSQL de pruebas no está vacío ni coincide exactamente con el baseline; "
                        "no se modificó.",
                        pytrace=False,
                    )
        except pytest.fail.Exception:
            raise
        except Exception:
            pytest.fail(
                "No se pudo verificar o inicializar el PostgreSQL aislado. Revisa Docker y la configuración "
                "local; no se muestra la URL ni el error de conexión.",
                pytrace=False,
            )

        with db.engine.connect() as connection:
            revisions = connection.execute(text("SELECT version_num FROM alembic_version")).scalars().all()
            if revisions != [INITIAL_REVISION]:
                pytest.fail("La revisión aplicada no coincide con la migración inicial esperada.", pytrace=False)

    try:
        yield test_app
    finally:
        with test_app.app_context():
            db.session.remove()
            db.engine.dispose()


@pytest.fixture
def postgres_client(postgres_app):
    return postgres_app.test_client()


def _insert(app, *, filename: str, category: str = "plastic", confidence: float = 0.75,
            created_at: datetime | None = None) -> int:
    with app.app_context():
        record = Prediction(
            filename=filename,
            category=category,
            confidence=confidence,
            created_at=created_at or datetime.now(timezone.utc),
        )
        db.session.add(record)
        db.session.commit()
        return record.id


def test_initial_migration_creates_expected_postgresql_schema(postgres_app):
    with postgres_app.app_context():
        inspector = inspect(db.engine)
        assert set(inspector.get_table_names(schema="public")) == EXPECTED_TABLES

        columns = {column["name"]: column for column in inspector.get_columns("predictions", schema="public")}
        assert set(columns) == {"id", "filename", "category", "confidence", "created_at"}
        assert isinstance(columns["id"]["type"], Integer)
        assert columns["id"]["nullable"] is False
        assert "nextval" in (columns["id"]["default"] or "")
        assert columns["filename"]["type"].length == 255 and not columns["filename"]["nullable"]
        assert columns["category"]["type"].length == 80 and not columns["category"]["nullable"]
        assert isinstance(columns["confidence"]["type"], Float) and not columns["confidence"]["nullable"]
        assert isinstance(columns["created_at"]["type"], DateTime)
        assert columns["created_at"]["type"].timezone and not columns["created_at"]["nullable"]
        assert columns["created_at"]["default"] is None
        assert inspector.get_pk_constraint("predictions", schema="public")["constrained_columns"] == ["id"]
        assert {index["name"] for index in inspector.get_indexes("predictions", schema="public")} == {
            "ix_predictions_category", "ix_predictions_created_at",
        }
        assert inspector.get_unique_constraints("predictions", schema="public") == []
        assert inspector.get_foreign_keys("predictions", schema="public") == []
        assert inspector.get_check_constraints("predictions", schema="public") == []


def test_insert_list_detail_and_new_session_persist(postgres_app, postgres_client):
    filename = f"it-session-{uuid4().hex}.jpg"
    prediction_id = _insert(postgres_app, filename=filename, confidence=0.82)

    listing = postgres_client.get("/api/predictions?category=plastic&per_page=100")
    assert listing.status_code == 200
    assert any(item["id"] == prediction_id and item["filename"] == filename for item in listing.json["items"])

    detail = postgres_client.get(f"/api/predictions/{prediction_id}")
    assert detail.status_code == 200
    assert detail.json["filename"] == filename
    assert detail.json["confidence"] == pytest.approx(0.82)

    with postgres_app.app_context():
        db.session.remove()
        reopened = db.session.get(Prediction, prediction_id)
        assert reopened is not None and reopened.filename == filename


def test_pagination_category_and_date_filters_use_postgresql(postgres_app, postgres_client):
    prefix = f"it-filter-{uuid4().hex}"
    day = datetime(2024, 3, 14, 12, tzinfo=timezone.utc)
    first = _insert(postgres_app, filename=f"{prefix}-1.jpg", category="paper", created_at=day)
    second = _insert(postgres_app, filename=f"{prefix}-2.jpg", category="paper", created_at=day.replace(hour=13))
    _insert(postgres_app, filename=f"{prefix}-other.jpg", category="metal", created_at=day)

    page_one = postgres_client.get("/api/predictions?category=paper&date_from=2024-03-14&date_to=2024-03-14&per_page=1&page=1")
    page_two = postgres_client.get("/api/predictions?category=paper&date_from=2024-03-14&date_to=2024-03-14&per_page=1&page=2")
    assert page_one.status_code == page_two.status_code == 200
    assert page_one.json["total"] == 2 and page_one.json["pages"] == 2
    assert page_one.json["items"][0]["id"] != page_two.json["items"][0]["id"]
    assert {page_one.json["items"][0]["id"], page_two.json["items"][0]["id"]} == {first, second}


def test_statistics_are_calculated_from_postgresql_records(postgres_app, postgres_client):
    before = postgres_client.get("/api/statistics").json["total"]
    prefix = f"it-stats-{uuid4().hex}"
    _insert(postgres_app, filename=f"{prefix}-1.jpg", category="shoes", confidence=0.6)
    _insert(postgres_app, filename=f"{prefix}-2.jpg", category="shoes", confidence=0.8)

    response = postgres_client.get("/api/statistics")
    assert response.status_code == 200
    assert response.json["total"] == before + 2
    shoes = next(row for row in response.json["by_category"] if row["category"] == "shoes")
    assert shoes["count"] >= 2


def test_failed_commit_rolls_back_without_persisting_invalid_row(postgres_app):
    filename = f"it-rollback-{uuid4().hex}.jpg"
    with postgres_app.app_context():
        db.session.add(Prediction(filename=filename, category="plastic", confidence=0.7))
        db.session.add(Prediction(filename=None, category="plastic", confidence=0.9))
        with pytest.raises(IntegrityError):
            db.session.commit()
        db.session.rollback()
        assert db.session.query(Prediction).filter_by(filename=filename).first() is None


def test_missing_identifier_returns_not_found_from_postgresql(postgres_client):
    response = postgres_client.get("/api/predictions/2147483000")
    assert response.status_code == 404
