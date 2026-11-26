import os
import runpy
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy.engine import make_url

from app import create_app, load_environment, migrate, resolve_database_url
from database import db


VALID_TEST_URL = "postgresql+psycopg://test:test@127.0.0.1:1/ecovision_test"


@pytest.mark.parametrize("value", [None, "", "   "])
def test_database_url_is_required(value, monkeypatch):
    if value is None:
        monkeypatch.delenv("DATABASE_URL", raising=False)
    else:
        monkeypatch.setenv("DATABASE_URL", value)
    with pytest.raises(RuntimeError, match="DATABASE_URL es obligatoria"):
        resolve_database_url()


@pytest.mark.parametrize("value", [
    "sqlite:///local.db",
    "postgresql://user:secret@localhost/db",
    "postgresql+psycopg2://user:secret@localhost/db",
    "no-es-una-url",
])
def test_non_psycopg3_database_urls_are_rejected_without_leaking_credentials(value):
    with pytest.raises(RuntimeError) as error:
        resolve_database_url(value)
    assert "secret" not in str(error.value)
    assert "localhost" not in str(error.value)


def test_psycopg3_postgresql_url_is_accepted():
    assert resolve_database_url(VALID_TEST_URL) == VALID_TEST_URL


def test_remote_postgresql_requires_ca_and_verify_full(tmp_path, monkeypatch):
    monkeypatch.delenv("SUPABASE_CA_FILE", raising=False)
    remote_url = "postgresql+psycopg://user:secret@db.example.supabase.co:5432/postgres?sslmode=require"
    with pytest.raises(RuntimeError, match="SUPABASE_CA_FILE") as error:
        resolve_database_url(remote_url)
    assert "secret" not in str(error.value)

    ca_file = tmp_path / "supabase-ca.crt"
    ca_file.write_text("test-only certificate placeholder", encoding="ascii")
    resolved = resolve_database_url(remote_url, str(ca_file))
    resolved_url = make_url(resolved)
    assert resolved_url.query["sslmode"] == "verify-full"
    assert resolved_url.query["sslrootcert"] == str(ca_file.resolve())
    assert "secret" not in str(error.value)


def test_create_app_does_not_create_tables_automatically():
    with patch.object(db, "create_all") as create_all:
        create_app({"TESTING": True, "SQLALCHEMY_DATABASE_URI": VALID_TEST_URL})
    create_all.assert_not_called()


def test_flask_migrate_is_registered_with_the_existing_sqlalchemy_extension():
    app = create_app({"TESTING": True, "SQLALCHEMY_DATABASE_URI": VALID_TEST_URL})
    migrate_config = app.extensions["migrate"]
    assert migrate_config.db is db
    assert migrate.db is db
    assert Path(migrate.directory).resolve() == (Path(__file__).resolve().parents[2] / "migrations").resolve()


def test_migration_history_is_readable_without_a_database_connection():
    app = create_app({"TESTING": True, "SQLALCHEMY_DATABASE_URI": VALID_TEST_URL})
    with patch("sqlalchemy.engine.Engine.connect", side_effect=AssertionError("unexpected database connection")):
        result = app.test_cli_runner().invoke(args=["db", "history"])
    assert result.exit_code == 0
    with app.app_context():
        scripts = ScriptDirectory.from_config(migrate.get_config())
    revision = scripts.get_revision(scripts.get_current_head())
    assert revision.doc == "initial predictions schema"


def test_manual_migration_revision_can_be_generated_without_database_connection(tmp_path):
    source = Path(__file__).resolve().parents[2] / "migrations"
    target = tmp_path / "migrations"
    shutil.copytree(source, target)
    config = Config(str(target / "alembic.ini"))
    config.set_main_option("script_location", str(target).replace("%", "%%"))

    with patch("sqlalchemy.engine.Engine.connect", side_effect=AssertionError("unexpected database connection")):
        generated = command.revision(config, message="offline generation verification")
    assert generated is not None
    assert generated.doc == "offline generation verification"


def test_initial_migration_matches_the_prediction_model_without_executing_ddl():
    migration_path = Path(__file__).resolve().parents[2] / "migrations" / "versions" / "85b59e312c4e_initial_predictions_schema.py"
    migration = runpy.run_path(str(migration_path))
    statements = []
    with patch("alembic.op.create_table", side_effect=lambda *args, **kwargs: statements.append(("table", args, kwargs))), \
         patch("alembic.op.create_index", side_effect=lambda *args, **kwargs: statements.append(("index", args, kwargs))):
        migration["upgrade"]()

    table = statements[0]
    assert table[0] == "table" and table[1][0] == "predictions"
    columns = {column.name: column for column in table[1][1:] if hasattr(column, "type")}
    assert set(columns) == {"id", "filename", "category", "confidence", "created_at"}
    assert columns["id"].primary_key and not columns["id"].nullable
    assert columns["filename"].type.length == 255 and not columns["filename"].nullable
    assert columns["category"].type.length == 80 and not columns["category"].nullable
    assert not columns["confidence"].nullable and columns["confidence"].server_default is None
    assert columns["created_at"].type.timezone and not columns["created_at"].nullable
    assert columns["created_at"].server_default is None
    assert {item[1][0] for item in statements if item[0] == "index"} == {
        "ix_predictions_category", "ix_predictions_created_at",
    }
    with pytest.raises(RuntimeError, match="proteger los registros"):
        migration["downgrade"]()


def test_importing_app_does_not_call_create_all_or_connect_to_a_database():
    script = (
        "from database import db; calls=[]; db.create_all=lambda: calls.append('ddl'); "
        "import alembic.command; alembic.command.upgrade=lambda *a, **k: calls.append('migration'); "
        "import sqlalchemy.engine; sqlalchemy.engine.Engine.connect=lambda *a, **k: "
        "(_ for _ in ()).throw(AssertionError('unexpected database connection')); "
        "import app; print('startup_schema_calls=' + str(len(calls)))"
    )
    environment = os.environ.copy()
    environment["DATABASE_URL"] = VALID_TEST_URL
    result = subprocess.run([sys.executable, "-c", script], cwd=os.getcwd(), env=environment,
                            capture_output=True, text=True, check=True)
    assert result.stdout.strip().endswith("startup_schema_calls=0")


@pytest.mark.skipif(sys.platform == "win32", reason="El importador de Gunicorn requiere los módulos Unix fcntl y pwd.")
def test_gunicorn_wsgi_import_does_not_create_tables_or_connect():
    script = (
        "from database import db; calls=[]; db.create_all=lambda: calls.append('ddl'); "
        "import sqlalchemy.engine; sqlalchemy.engine.Engine.connect=lambda *a, **k: "
        "(_ for _ in ()).throw(AssertionError('unexpected database connection')); "
        "import backend.prediction as prediction; prediction.load_model=lambda path: object(); "
        "from gunicorn.app.wsgiapp import WSGIApplication; server=WSGIApplication(); "
        "server.app_uri='app:app'; server.load(); print('ddl_calls=' + str(len(calls)))"
    )
    environment = os.environ.copy()
    environment["DATABASE_URL"] = VALID_TEST_URL
    result = subprocess.run([sys.executable, "-c", script], cwd=os.getcwd(), env=environment,
                            capture_output=True, text=True, check=True)
    assert result.stdout.strip().endswith("ddl_calls=0")


def test_import_fails_clearly_when_database_url_is_missing():
    script = "import app"
    environment = os.environ.copy()
    environment.pop("DATABASE_URL", None)
    # Prevent a developer .env from supplying a URL in this isolated check.
    environment["PYTHON_DOTENV_DISABLED"] = "1"
    result = subprocess.run([sys.executable, "-c", script], cwd=os.getcwd(), env=environment,
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "DATABASE_URL es obligatoria" in result.stderr
    assert "password" not in result.stderr.lower()


def test_environment_loader_reads_file_and_keeps_exported_value(tmp_path, monkeypatch):
    dotenv_path = tmp_path / ".env.test"
    dotenv_path.write_text("ECOVISION_DOTENV_TEST_VALUE=from-file\n", encoding="utf-8")
    monkeypatch.delenv("ECOVISION_DOTENV_TEST_VALUE", raising=False)
    assert load_environment(dotenv_path)
    assert os.environ["ECOVISION_DOTENV_TEST_VALUE"] == "from-file"

    monkeypatch.setenv("ECOVISION_DOTENV_TEST_VALUE", "from-environment")
    load_environment(dotenv_path)
    assert os.environ["ECOVISION_DOTENV_TEST_VALUE"] == "from-environment"
