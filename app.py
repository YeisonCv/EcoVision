from __future__ import annotations

import os
import threading
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from flask_migrate import Migrate
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.exceptions import HTTPException
from sqlalchemy import func, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError, SQLAlchemyError
from werkzeug.utils import secure_filename

from backend.prediction import (ImageValidationError, ModelUnavailableError,
                                UnsupportedImageExtensionError, classify, load_classes,
                                load_model, model_paths, prepare_image, validate_upload)
from database import db
from database.models import Prediction

PROJECT_ROOT = Path(__file__).resolve().parent
migrate = Migrate()
limiter = Limiter(key_func=get_remote_address, default_limits=["120 per minute"])


def load_environment(dotenv_path=PROJECT_ROOT / ".env"):
    """Load local defaults without replacing variables exported by the caller."""
    return load_dotenv(dotenv_path=dotenv_path, override=False)


load_environment()


def resolve_database_url(configured_url=None, ca_file=None):
    """Require psycopg 3 and enforce certificate validation for remote PostgreSQL."""
    if configured_url is None:
        configured_url = os.getenv("DATABASE_URL")
    if not configured_url or not configured_url.strip():
        raise RuntimeError("DATABASE_URL es obligatoria; configura la conexión PostgreSQL de EcoVision.")

    try:
        parsed_url = make_url(configured_url)
    except (ArgumentError, TypeError):
        raise RuntimeError("DATABASE_URL no es una URL PostgreSQL válida para psycopg 3.") from None
    if parsed_url.drivername != "postgresql+psycopg" or not parsed_url.host or not parsed_url.database:
        raise RuntimeError("DATABASE_URL debe usar PostgreSQL con el driver psycopg 3 y especificar host y base de datos.")

    # Local PostgreSQL is reserved for isolated development/integration tests.
    # The Docker E2E service has a fixed, test-only database and user identity.
    local_hosts = {"localhost", "127.0.0.1", "::1"}
    isolated_compose_test = (os.getenv("ECOVISION_E2E") == "1"
                             and parsed_url.host == "postgres-test"
                             and parsed_url.database == "ecovision_test"
                             and parsed_url.username == "ecovision_test")
    if parsed_url.host not in local_hosts and not isolated_compose_test:
        ca_file = ca_file or os.getenv("SUPABASE_CA_FILE")
        if not ca_file or not Path(ca_file).is_file():
            raise RuntimeError("La conexión PostgreSQL remota requiere SUPABASE_CA_FILE con el certificado CA oficial.")
        query = dict(parsed_url.query)
        query.update(sslmode="verify-full", sslrootcert=str(Path(ca_file).resolve()))
        parsed_url = parsed_url.set(query=query)
    return parsed_url.render_as_string(hide_password=False)


def create_app(test_config=None):
    app = Flask(__name__, template_folder="visualizacion/templates", static_folder="visualizacion/static")
    max_image_mb = int(os.getenv("MAX_IMAGE_MB", "10"))
    configured_url = (test_config or {}).get("SQLALCHEMY_DATABASE_URI")
    app.config.update(MAX_IMAGE_MB=max(1, min(25, max_image_mb)), SQLALCHEMY_TRACK_MODIFICATIONS=False,
                      SQLALCHEMY_DATABASE_URI=resolve_database_url(configured_url),
                      SQLALCHEMY_ENGINE_OPTIONS={"pool_pre_ping": True, "pool_recycle": 1800,
                                                 "pool_size": 5, "max_overflow": 0})
    app.config["MAX_CONTENT_LENGTH"] = app.config["MAX_IMAGE_MB"] * 1024 * 1024
    if test_config:
        app.config.update(test_config)
        # Test configuration cannot bypass the PostgreSQL-only runtime policy.
        app.config["SQLALCHEMY_DATABASE_URI"] = resolve_database_url(app.config["SQLALCHEMY_DATABASE_URI"])
    local_origins = "http://localhost:5000,http://127.0.0.1:5000,http://localhost:3000,http://127.0.0.1:3000"
    configured_origins = app.config.get("CORS_ALLOWED_ORIGINS", os.getenv("CORS_ALLOWED_ORIGINS", local_origins))
    app.config["CORS_ALLOWED_ORIGINS"] = {
        origin.strip().rstrip("/") for origin in configured_origins.split(",") if origin.strip()
    } if isinstance(configured_origins, str) else set(configured_origins)
    app.config.setdefault("RATELIMIT_STORAGE_URI", os.getenv("RATE_LIMIT_STORAGE_URI", "memory://"))
    app.config.setdefault("RATELIMIT_ENABLED", True)
    configured_proxy_hops = app.config.get("TRUSTED_PROXY_HOPS", os.getenv("TRUSTED_PROXY_HOPS", "0"))
    try:
        trusted_proxy_hops = int(configured_proxy_hops)
    except (TypeError, ValueError):
        raise RuntimeError("TRUSTED_PROXY_HOPS debe ser un entero entre 0 y 10.") from None
    if trusted_proxy_hops < 0 or trusted_proxy_hops > 10:
        raise RuntimeError("TRUSTED_PROXY_HOPS debe ser un entero entre 0 y 10.")
    app.config["TRUSTED_PROXY_HOPS"] = trusted_proxy_hops
    if trusted_proxy_hops:
        from werkzeug.middleware.proxy_fix import ProxyFix
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=trusted_proxy_hops)
    db.init_app(app)
    migrate.init_app(app, db)
    limiter.init_app(app)
    model_path, classes_path = model_paths()
    app.extensions["classifier"] = None
    app.extensions["classes"] = []
    app.extensions["model_error"] = None
    # Gunicorn is configured with several request threads. Serialize calls into
    # the single in-process model to bound peak memory and avoid concurrent
    # TensorFlow predict() calls sharing the same model instance.
    app.extensions["inference_lock"] = threading.Lock()
    try:
        app.extensions["classes"] = load_classes(classes_path)
        if not app.config.get("TESTING") or app.config.get("LOAD_MODEL_IN_TESTS", False):
            app.extensions["classifier"] = load_model(model_path)
    except ModelUnavailableError as exc:
        app.extensions["model_error"] = str(exc)
        app.logger.error("Model initialization failed: %s", exc)

    @app.get("/")
    def dashboard():
        return render_template("dashboard.html")

    @app.get("/clasificador")
    def classifier_page():
        return render_template("classifier.html")

    @app.get("/historial")
    def history_page():
        return render_template("history.html")

    @app.get("/estadisticas")
    def statistics_page():
        return render_template("statistics.html")

    @app.get("/api/health")
    @limiter.limit("30 per minute")
    def health():
        try:
            db.session.execute(text("SELECT 1"))
            database_status = "available"
        except SQLAlchemyError:
            db.session.rollback()
            database_status = "unavailable"
        model_status = "available" if app.extensions["classifier"] is not None and not app.extensions["model_error"] else "unavailable"
        status = 200 if database_status == "available" and model_status == "available" else 503
        return jsonify({"status": "ok" if status == 200 else "degraded", "model": model_status,
                        "database": database_status}), status

    @app.post("/api/predict")
    @limiter.limit("5 per minute")
    def predict():
        uploaded = request.files.get("image")
        if uploaded is None or not uploaded.filename:
            return jsonify(error="Debes seleccionar una imagen."), 400
        try:
            validate_upload(uploaded.filename, uploaded.stream)
        except UnsupportedImageExtensionError as exc:
            return jsonify(error=str(exc)), 415
        except ImageValidationError as exc:
            return jsonify(error=str(exc)), 400
        model = app.extensions["classifier"]
        if model is None:
            return jsonify(error="El modelo no está disponible. Revisa la configuración del servidor."), 503
        try:
            with app.extensions["inference_lock"]:
                batch = prepare_image(uploaded.stream, app.config["MAX_CONTENT_LENGTH"])
                result = classify(model, app.extensions["classes"], batch)
        except ImageValidationError as exc:
            return jsonify(error=str(exc)), 400
        except ModelUnavailableError:
            app.logger.exception("Prediction failed")
            return jsonify(error="No fue posible analizar la imagen."), 500
        try:
            record = Prediction(filename=secure_filename(uploaded.filename)[:255] or "imagen", category=result["category"], confidence=result["confidence"])
            db.session.add(record)
            db.session.commit()
        except SQLAlchemyError:
            db.session.rollback()
            app.logger.exception("Could not persist prediction")
            return jsonify(error="La imagen se analizó, pero no se pudo guardar el resultado."), 503
        return jsonify({**result, "prediction": record.as_dict(), "message": "Resultado del modelo; la confianza es una probabilidad, no una garantía."}), 201

    @app.get("/api/predictions")
    @limiter.limit("30 per minute")
    def predictions():
        for key in ("page", "per_page", "category", "date_from", "date_to"):
            values = request.args.getlist(key)
            if len(values) > 1 or (values and not values[0].strip()):
                return jsonify(error=f"El parámetro {key} está vacío o repetido."), 400
        try:
            page = int(request.args.get("page", "1"))
            per_page = int(request.args.get("per_page", "20"))
        except ValueError:
            return jsonify(error="Los parámetros de paginación deben ser números enteros."), 400
        if page < 1 or page > 10_000 or per_page < 1 or per_page > 100:
            return jsonify(error="La página debe estar entre 1 y 10000 y per_page entre 1 y 100."), 400
        query = Prediction.query
        category = request.args.get("category")
        if category:
            if len(category) > 80 or category not in app.extensions["classes"]:
                return jsonify(error="La categoría solicitada no es válida."), 400
            query = query.filter(Prediction.category == category)
        date_from, date_to = request.args.get("date_from"), request.args.get("date_to")
        try:
            if date_from:
                parsed_day = date.fromisoformat(date_from)
                if parsed_day.isoformat() != date_from:
                    raise ValueError
                parsed_from = datetime.combine(parsed_day, datetime.min.time(), tzinfo=timezone.utc)
                query = query.filter(Prediction.created_at >= parsed_from)
            if date_to:
                parsed_day = date.fromisoformat(date_to)
                if parsed_day.isoformat() != date_to:
                    raise ValueError
                parsed_to = datetime.combine(parsed_day, datetime.min.time(), tzinfo=timezone.utc)
                query = query.filter(Prediction.created_at < parsed_to + timedelta(days=1))
        except ValueError:
            return jsonify(error="El formato de fecha debe ser AAAA-MM-DD."), 400
        if date_from and date_to and parsed_from > parsed_to:
            return jsonify(error="date_from no puede ser posterior a date_to."), 400
        try:
            result = query.order_by(Prediction.created_at.desc(), Prediction.id.desc()).paginate(page=page, per_page=per_page, error_out=False)
        except SQLAlchemyError:
            db.session.rollback()
            app.logger.exception("Could not query prediction history")
            return jsonify(error="No fue posible consultar el historial."), 503
        return jsonify(items=[item.as_dict() for item in result.items], page=page, per_page=per_page, total=result.total, pages=result.pages)

    @app.get("/api/predictions/<int:prediction_id>")
    @limiter.limit("60 per minute")
    def prediction_detail(prediction_id):
        try:
            record = db.session.get(Prediction, prediction_id)
        except SQLAlchemyError:
            db.session.rollback()
            app.logger.exception("Could not query prediction detail")
            return jsonify(error="No fue posible consultar la predicción."), 503
        if record is None:
            return jsonify(error="No se encontró la predicción solicitada."), 404
        return jsonify(record.as_dict())

    @app.get("/api/statistics")
    @limiter.limit("20 per minute")
    def statistics():
        try:
            total = db.session.query(func.count(Prediction.id)).scalar() or 0
            average = db.session.query(func.avg(Prediction.confidence)).scalar()
            rows = (db.session.query(Prediction.category, func.count(Prediction.id))
                    .group_by(Prediction.category).order_by(func.count(Prediction.id).desc())
                    .limit(max(1, len(app.extensions["classes"]))).all())
            latest = Prediction.query.order_by(Prediction.created_at.desc()).first()
            cutoff = datetime.now(timezone.utc) - timedelta(days=89)
            daily = (db.session.query(func.date(Prediction.created_at), func.count(Prediction.id))
                     .filter(Prediction.created_at >= cutoff).group_by(func.date(Prediction.created_at))
                     .order_by(func.date(Prediction.created_at)).all())
        except SQLAlchemyError:
            db.session.rollback()
            app.logger.exception("Could not query prediction statistics")
            return jsonify(error="No fue posible consultar las estadísticas."), 503
        return jsonify(total=total, most_common=rows[0][0] if rows else None,
                       average_confidence=float(average) if average is not None else None,
                       by_category=[{"category": c, "count": n} for c, n in rows],
                       daily=[{"date": str(day), "count": n} for day, n in daily],
                       latest=latest.as_dict() if latest else None)

    @app.errorhandler(413)
    def too_large(_):
        return jsonify(error=f"El archivo supera el tamaño máximo de {app.config['MAX_IMAGE_MB']} MB."), 413

    @app.after_request
    def apply_api_cors(response):
        origin = request.headers.get("Origin")
        if origin:
            response.headers.add("Vary", "Origin")
            if origin.rstrip("/") in app.config["CORS_ALLOWED_ORIGINS"]:
                response.headers["Access-Control-Allow-Origin"] = origin
                response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
                response.headers["Access-Control-Allow-Headers"] = "Content-Type"
                response.headers["Access-Control-Max-Age"] = "600"
        return response

    @app.errorhandler(HTTPException)
    def handle_http_exception(error):
        if request.path.startswith("/api/"):
            return jsonify(error="La solicitud no pudo completarse."), error.code
        return error

    @app.errorhandler(Exception)
    def handle_unexpected_error(error):
        app.logger.exception("Unhandled application error")
        if request.path.startswith("/api/"):
            return jsonify(error="Ocurrió un error interno. Inténtalo de nuevo."), 500
        return jsonify(error="Ocurrió un error interno."), 500

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
