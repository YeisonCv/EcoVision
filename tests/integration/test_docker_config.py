from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_dockerfile_copies_the_complete_migrations_directory():
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert re.search(r"(?m)^COPY\s+migrations\s+\./migrations\s*$", dockerfile)
    for required_file in ("env.py", "alembic.ini", "script.py.mako"):
        assert (PROJECT_ROOT / "migrations" / required_file).is_file()
    assert any((PROJECT_ROOT / "migrations" / "versions").glob("*.py"))


def test_compose_requires_database_url_without_startup_migrations():
    compose = (PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    local_compose = (PROJECT_ROOT / "docker-compose.local.yml").read_text(encoding="utf-8")
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert re.search(r"(?m)^\s*DATABASE_URL:\s*\$\{DATABASE_URL:\?", compose)
    assert re.search(r"(?m)^\s*SUPABASE_CA_FILE:\s*/run/secrets/supabase-ca\.crt$", compose)
    assert re.search(r"(?m)^\s*file:\s*\$\{SUPABASE_CA_FILE:\?", compose)
    assert "ports:" not in compose
    assert '"127.0.0.1:${PORT:-5000}:5000"' in local_compose
    assert not re.search(r"(?m)^\s*(command|entrypoint):", compose)
    assert "flask db upgrade" not in compose.lower()
    assert re.search(r'(?m)^CMD\s+\["gunicorn"', dockerfile)
    assert re.search(r"(?m)^USER\s+10001:10001$", dockerfile)
    assert "flask db upgrade" not in dockerfile.lower()


def test_e2e_compose_isolated_from_production_configuration():
    e2e = (PROJECT_ROOT / "docker-compose.e2e.yml").read_text(encoding="utf-8")
    assert "postgres-test" in e2e
    assert "ecovision_test_only@postgres-test" in e2e
    assert '"127.0.0.1:${E2E_PORT:-5002}:5000"' in e2e
    assert "read_only: true" in e2e
    assert "flask db upgrade" not in e2e.lower()
