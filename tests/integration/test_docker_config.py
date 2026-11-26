from pathlib import Path
from fnmatch import fnmatchcase
import re


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def expand_braces(pattern):
    group = re.search(r"\{([^{}]+)\}", pattern)
    if group is None:
        return {pattern}
    return {
        expanded
        for option in group[1].split(",")
        for expanded in expand_braces(pattern[:group.start()] + option + pattern[group.end():])
    }


def test_dockerfile_copies_the_complete_migrations_directory():
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert re.search(r"(?m)^COPY\s+migrations\s+\./migrations\s*$", dockerfile)
    assert "COPY requirements-docker.txt ./" in dockerfile
    assert "pip install -r requirements-docker.txt" in dockerfile
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


def test_vercel_bundle_excludes_non_runtime_and_local_files_but_includes_app_assets():
    import json

    config = json.loads((PROJECT_ROOT / "vercel.json").read_text(encoding="utf-8"))
    function = config["functions"]["app.py"]
    assert config["fluid"] is True
    assert config["regions"] == ["pdx1"]
    assert function["maxDuration"] == 60
    assert "memory" not in function
    for field in ("includeFiles", "excludeFiles"):
        assert isinstance(function[field], str)
        assert 0 < len(function[field]) <= 256, field
    included = expand_braces(function["includeFiles"])
    excluded = expand_braces(function["excludeFiles"])
    assert {
        "backend/**", "database/**", "migrations/**",
        "visualizacion/templates/**", "visualizacion/static/**",
        "modelo/waste_classifier.keras", "modelo/classes.txt",
    } <= included
    assert {
        "tests/**", "docs/**", "data/**", "instance/**", "**/secrets/**",
        "**/__pycache__/**", "**/.pytest_cache/**", "**/.env*",
        "modelo/best_model.keras", "modelo/training_curves.png", "ecovision*.tar",
        "**/*.crt", "**/*.pem", "**/*.pyc", ".venv/**", "venv/**",
        "verify_model.py", "Dockerfile", "docker*.yml",
        "requirements-dev.txt", "requirements-docker.txt", "pytest.ini",
    } <= excluded
    required_paths = (
        "app.py", "requirements.txt", "backend/prediction.py", "database/models.py",
        "migrations/env.py", "visualizacion/templates/classifier.html",
        "visualizacion/static/js/app.js", "visualizacion/static/css/app.css",
        "modelo/waste_classifier.keras", "modelo/classes.txt",
    )
    for path in required_paths:
        assert (PROJECT_ROOT / path).is_file()
        assert not any(fnmatchcase(path, pattern) or (
            pattern.startswith("**/") and fnmatchcase(path, pattern[3:])
        ) for pattern in excluded), path
    assert (PROJECT_ROOT / ".python-version").read_text(encoding="utf-8").strip() == "3.12"
    assert "tensorflow-cpu==2.20.0" in (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")


def test_vercelignore_excludes_local_secrets_and_large_artifacts():
    patterns = set((PROJECT_ROOT / ".vercelignore").read_text(encoding="utf-8").splitlines())
    assert {".env*", "secrets/", "*.crt", "*.pem", "data/", "instance/", "tests/", "docs/", "ecovision*.tar"} <= patterns
    assert "modelo/waste_classifier.keras" not in patterns
    assert "modelo/classes.txt" not in patterns


def test_gunicorn_is_only_a_docker_and_development_dependency():
    runtime = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")
    docker = (PROJECT_ROOT / "requirements-docker.txt").read_text(encoding="utf-8")
    development = (PROJECT_ROOT / "requirements-dev.txt").read_text(encoding="utf-8")
    assert "gunicorn" not in runtime.lower()
    assert "gunicorn==23.0.0" in docker
    assert "gunicorn==23.0.0" in development
