import os
import runpy
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy.engine import make_url
from sqlalchemy.pool import NullPool

from app import create_app, load_environment, migrate, resolve_database_url
from database import db


VALID_TEST_URL = "postgresql+psycopg://test:test@127.0.0.1:1/ecovision_test"
TEST_CA_PEM = """-----BEGIN CERTIFICATE-----
MIICzjCCAbagAwIBAgIJAKXKgmY4/1ljMA0GCSqGSIb3DQEBCwUAMBwxGjAYBgNV
BAMTEUVjb1Zpc2lvbiBUZXN0IENBMB4XDTI2MTEyNTA2MDk0NFoXDTI3MTEyNjA2
MDk0NFowHDEaMBgGA1UEAxMRRWNvVmlzaW9uIFRlc3QgQ0EwggEiMA0GCSqGSIb3
DQEBAQUAA4IBDwAwggEKAoIBAQC/xacHc1AlTyH1EjO8gUpuG/FrrmZGe8fHcpPM
f7zaj8KfGSYx8QXKf0kwqnpeyfc6PftulYouutRE4LAaMkO9baPUpv9LyoTO7gsz
Si2TG0gPFD0rOdwut6a6SnruRXaypJl6YMHqSgkXDs2HL3IJVzIsLl7J67J05UVv
fOoMpYkOVjo9cAKYsj+rBVfmN5L1wZDI0++ShwdqAmEhPRYepbYs9m7opzRycO5o
KbKqcwSSfMrqbIaC6AopkKMDDJcNxH71ENmGBB7Xa7LZWxQ+U9RpSRKJL6XZ4a0F
T9WEKvjWstkBWTaS2HJKzIMcKv5eOahLM92uqOOKVbIfBAotAgMBAAGjEzARMA8G
A1UdEwEB/wQFMAMBAf8wDQYJKoZIhvcNAQELBQADggEBAFgYLjX/YHQlGDTLdE2p
gKavYJNt1AYcv5ZbN5TOf+rEbXvnNgDMPMmgAdCeYgxv5E05KvxfDOKxLw8CA1Xd
ditkB3lcmRacPIOJX+dUunK2ZUxai1xm/COSqgzKPqvAEKqkLvqqBhr3FJhB+DhK
CpH3IwB0bBLe72RxWFr97mnjiQCoyHJolR92VVM07sq40lYNZdVMH9GUDwIb68Lh
C4YW5s132NY1DzTZjGV5Ys3NMIe/fEr7SJa2JAglwyYJtpFeI4UyDp1pLOrntlF5
mbvXz9oWg5SoL8Waz7Yz7dSvmtek//OmoIb2ou1APuWq4+tvgApWgIHSz0Hu++r1
KCA=
-----END CERTIFICATE-----"""


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
    import app

    app._ca_certificate_files.clear()
    monkeypatch.setattr(app, "_ca_temp_directory", lambda: str(tmp_path))
    monkeypatch.delenv("SUPABASE_CA_FILE", raising=False)
    monkeypatch.delenv("SUPABASE_CA_CERT", raising=False)
    remote_url = "postgresql+psycopg://user:secret@db.example.supabase.co:5432/postgres?sslmode=require"
    with pytest.raises(RuntimeError, match="SUPABASE_CA_FILE o SUPABASE_CA_CERT") as error:
        resolve_database_url(remote_url)
    assert "secret" not in str(error.value)

    resolved = resolve_database_url(remote_url, ca_cert=TEST_CA_PEM)
    resolved_url = make_url(resolved)
    assert resolved_url.query["sslmode"] == "verify-full"
    assert Path(resolved_url.query["sslrootcert"]).is_file()
    assert "secret" not in str(error.value)


def test_valid_ca_file_is_used_when_pem_is_not_configured(tmp_path, monkeypatch):
    ca_file = tmp_path / "supabase-ca.crt"
    ca_file.write_text(TEST_CA_PEM, encoding="ascii")
    monkeypatch.setenv("SUPABASE_CA_FILE", str(ca_file))
    monkeypatch.delenv("SUPABASE_CA_CERT", raising=False)

    resolved = make_url(resolve_database_url(
        "postgresql+psycopg://user:secret@db.example.supabase.co:5432/postgres",
    ))
    assert resolved.query["sslmode"] == "verify-full"
    assert resolved.query["sslrootcert"] == str(ca_file.resolve())


def test_valid_pem_takes_priority_over_ca_file(tmp_path, monkeypatch):
    import app

    app._ca_certificate_files.clear()
    ca_file = tmp_path / "supabase-ca.crt"
    ca_file.write_text(TEST_CA_PEM, encoding="ascii")
    monkeypatch.setattr(app, "_ca_temp_directory", lambda: str(tmp_path))
    monkeypatch.setenv("SUPABASE_CA_FILE", str(ca_file))
    monkeypatch.setenv("SUPABASE_CA_CERT", TEST_CA_PEM)

    resolved = make_url(resolve_database_url(
        "postgresql+psycopg://user:secret@db.example.supabase.co:5432/postgres?sslmode=disable&sslrootcert=wrong",
    ))
    pem_path = Path(resolved.query["sslrootcert"])
    assert pem_path != ca_file.resolve()
    assert pem_path.read_text(encoding="ascii").strip() == TEST_CA_PEM
    assert resolved.query["sslmode"] == "verify-full"


def test_invalid_pem_does_not_fall_back_to_valid_ca_file(tmp_path, monkeypatch):
    ca_file = tmp_path / "supabase-ca.crt"
    ca_file.write_text(TEST_CA_PEM, encoding="ascii")
    monkeypatch.setenv("SUPABASE_CA_FILE", str(ca_file))
    monkeypatch.setenv("SUPABASE_CA_CERT", "invalid-secret-pem")

    with pytest.raises(RuntimeError, match="SUPABASE_CA_CERT") as error:
        resolve_database_url("postgresql+psycopg://user:secret@db.example.supabase.co:5432/postgres")
    assert "invalid-secret-pem" not in str(error.value)
    assert "secret" not in str(error.value)


def test_ca_cert_environment_is_supported_without_ca_file(tmp_path, monkeypatch):
    import app

    app._ca_certificate_files.clear()
    monkeypatch.delenv("SUPABASE_CA_FILE", raising=False)
    monkeypatch.setenv("SUPABASE_CA_CERT", TEST_CA_PEM.replace("\n", "\\n"))
    monkeypatch.setattr(app, "_ca_temp_directory", lambda: str(tmp_path))

    resolved = make_url(resolve_database_url(
        "postgresql+psycopg://user:secret@db.example.supabase.co:5432/postgres",
    ))
    certificate_path = Path(resolved.query["sslrootcert"])
    assert certificate_path.parent == tmp_path
    assert certificate_path.read_text(encoding="ascii").strip() == TEST_CA_PEM
    assert resolved.query["sslmode"] == "verify-full"
    assert resolved.drivername == "postgresql+psycopg"

    from sqlalchemy import create_engine
    engine = create_engine(resolved)
    try:
        _args, connect_kwargs = engine.dialect.create_connect_args(engine.url)
        assert connect_kwargs["sslmode"] == "verify-full"
        assert connect_kwargs["sslrootcert"] == str(certificate_path)
    finally:
        engine.dispose()


def test_invalid_ca_file_falls_back_to_valid_pem(tmp_path, monkeypatch):
    import app

    app._ca_certificate_files.clear()
    invalid_file = tmp_path / "missing-or-invalid.crt"
    invalid_file.write_text("not a certificate", encoding="ascii")
    monkeypatch.setenv("SUPABASE_CA_FILE", str(invalid_file))
    monkeypatch.setenv("SUPABASE_CA_CERT", TEST_CA_PEM)
    monkeypatch.setattr(app, "_ca_temp_directory", lambda: str(tmp_path))

    resolved = make_url(resolve_database_url(
        "postgresql+psycopg://user:secret@db.example.supabase.co:5432/postgres",
    ))
    assert Path(resolved.query["sslrootcert"]).name.startswith("ecovision-supabase-ca-")


@pytest.mark.parametrize("certificate", [
    "", "not PEM", "-----BEGIN CERTIFICATE-----\\ninvalid\\n-----END CERTIFICATE-----",
    "-----BEGIN CERTIFICATE-----\\nAQID\\n-----END CERTIFICATE-----",
])
def test_empty_or_invalid_pem_fails_safely(certificate, monkeypatch, caplog):
    monkeypatch.delenv("SUPABASE_CA_FILE", raising=False)
    monkeypatch.setenv("SUPABASE_CA_CERT", certificate)
    with pytest.raises(RuntimeError, match="SUPABASE_CA_CERT") as error:
        resolve_database_url("postgresql+psycopg://user:secret@db.example.supabase.co:5432/postgres")
    if certificate:
        assert certificate not in str(error.value)
        assert certificate not in caplog.text
    assert "secret" not in str(error.value)


def test_nonexistent_ca_file_without_pem_fails_clearly(monkeypatch):
    monkeypatch.setenv("SUPABASE_CA_FILE", "Z:/path/that/does/not/exist/supabase-ca.crt")
    monkeypatch.delenv("SUPABASE_CA_CERT", raising=False)
    with pytest.raises(RuntimeError, match="SUPABASE_CA_FILE o SUPABASE_CA_CERT"):
        resolve_database_url("postgresql+psycopg://user:secret@db.example.supabase.co:5432/postgres")


def test_temp_ca_file_is_private_and_reused_under_concurrent_initialization(tmp_path, monkeypatch):
    import stat
    import app

    app._ca_certificate_files.clear()
    monkeypatch.delenv("SUPABASE_CA_FILE", raising=False)
    monkeypatch.delenv("SUPABASE_CA_CERT", raising=False)
    monkeypatch.setattr(app, "_ca_temp_directory", lambda: str(tmp_path))
    url = "postgresql+psycopg://user:secret@db.example.supabase.co:5432/postgres"

    with patch("app.os.chmod", wraps=os.chmod) as chmod, ThreadPoolExecutor(max_workers=8) as executor:
        paths = list(executor.map(lambda _: make_url(resolve_database_url(url, ca_cert=TEST_CA_PEM)).query["sslrootcert"], range(16)))

    assert len(set(paths)) == 1
    chmod.assert_called_once_with(paths[0], 0o600)
    ca_path = Path(paths[0])
    assert ca_path.parent == tmp_path
    if os.name == "posix":
        assert stat.S_IMODE(ca_path.stat().st_mode) == 0o600
    else:
        # Windows does not expose POSIX mode bits as its ACL; mkstemp still
        # creates the file under the current user's private temp directory.
        assert str(ca_path).startswith(str(tmp_path))
    assert ca_path.read_text(encoding="ascii").strip() == TEST_CA_PEM
    assert len(list(tmp_path.glob("ecovision-supabase-ca-*.crt"))) == 1


def test_ca_temp_directory_is_tmp_on_linux(monkeypatch):
    import app

    monkeypatch.setattr(app.os, "name", "posix")
    assert app._ca_temp_directory() == "/tmp"


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


@pytest.mark.parametrize("port", [6543, 5432])
def test_serverless_engine_passes_tls_and_transaction_pooler_options_to_psycopg(port, tmp_path, monkeypatch):
    import app

    monkeypatch.setattr(app, "_ca_temp_directory", lambda: str(tmp_path))
    monkeypatch.setattr(app, "_ca_certificate_files", {})
    monkeypatch.setenv("SUPABASE_CA_CERT", TEST_CA_PEM)
    monkeypatch.setenv("DATABASE_URL",
                       f"postgresql+psycopg://test:test@pooler.example.invalid:{port}/test?connect_timeout=5")
    test_app = create_app({"TESTING": True})
    with test_app.app_context():
        engine = db.engine
        assert isinstance(engine.pool, NullPool)
        assert engine.hide_parameters is True
        connect = Mock(side_effect=RuntimeError("DBAPI intercepted"))
        monkeypatch.setattr(engine.dialect.dbapi, "connect", connect)
        with pytest.raises(RuntimeError, match="DBAPI intercepted"):
            engine.connect()

    kwargs = connect.call_args.kwargs
    assert kwargs["host"] == "pooler.example.invalid"
    assert kwargs["port"] == port
    assert kwargs["sslmode"] == "verify-full"
    assert Path(kwargs["sslrootcert"]).parent == tmp_path
    assert kwargs["connect_timeout"] == "5"
    if port == 6543:
        assert "prepare_threshold" in kwargs and kwargs["prepare_threshold"] is None
    else:
        assert "prepare_threshold" not in kwargs


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
    monkeypatch.delenv("PYTHON_DOTENV_DISABLED", raising=False)
    dotenv_path = tmp_path / ".env.test"
    dotenv_path.write_text("ECOVISION_DOTENV_TEST_VALUE=from-file\n", encoding="utf-8")
    monkeypatch.delenv("ECOVISION_DOTENV_TEST_VALUE", raising=False)
    assert load_environment(dotenv_path)
    assert os.environ["ECOVISION_DOTENV_TEST_VALUE"] == "from-file"

    monkeypatch.setenv("ECOVISION_DOTENV_TEST_VALUE", "from-environment")
    load_environment(dotenv_path)
    assert os.environ["ECOVISION_DOTENV_TEST_VALUE"] == "from-environment"
