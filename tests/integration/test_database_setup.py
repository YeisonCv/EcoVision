import os
import subprocess
import sys

from sqlalchemy import inspect

from app import create_app, load_environment
from database import db


def test_import_does_not_create_tables():
    """Import the module against an isolated in-memory URL and inspect its global app."""
    script = (
        "import app; from sqlalchemy import inspect; "
        "ctx=app.app.app_context(); ctx.push(); "
        "print('created' if inspect(app.db.engine).has_table('predictions') else 'absent')"
    )
    environment = os.environ.copy()
    environment["DATABASE_URL"] = "sqlite:///:memory:"
    result = subprocess.run([sys.executable, "-c", script], cwd=os.getcwd(), env=environment,
                            capture_output=True, text=True, check=True)
    assert result.stdout.strip().endswith("absent")


def test_schema_is_created_only_when_explicitly_requested():
    app = create_app({"TESTING": True, "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:"})
    with app.app_context():
        assert not inspect(db.engine).has_table("predictions")

    result = app.test_cli_runner().invoke(args=["init-db"])
    assert result.exit_code == 0
    assert "Esquema de EcoVision inicializado" in result.output

    with app.app_context():
        assert inspect(db.engine).has_table("predictions")


def test_environment_loader_reads_file_and_keeps_exported_value(tmp_path, monkeypatch):
    dotenv_path = tmp_path / ".env.test"
    dotenv_path.write_text("ECOVISION_DOTENV_TEST_VALUE=from-file\n", encoding="utf-8")
    monkeypatch.delenv("ECOVISION_DOTENV_TEST_VALUE", raising=False)
    assert load_environment(dotenv_path)
    assert os.environ["ECOVISION_DOTENV_TEST_VALUE"] == "from-file"

    monkeypatch.setenv("ECOVISION_DOTENV_TEST_VALUE", "from-environment")
    load_environment(dotenv_path)
    assert os.environ["ECOVISION_DOTENV_TEST_VALUE"] == "from-environment"
