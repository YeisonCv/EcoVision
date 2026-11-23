from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from sqlalchemy import func, text
from sqlalchemy.exc import SQLAlchemyError
from werkzeug.utils import secure_filename

from backend.prediction import (ImageValidationError, ModelUnavailableError,
                                UnsupportedImageExtensionError, classify, load_classes,
                                load_model, model_paths, prepare_image, validate_upload)
from database import db
from database.models import Prediction

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATABASE_URL = "sqlite:///ecovision.db"


def load_environment(dotenv_path=PROJECT_ROOT / ".env"):
    """Load local defaults without replacing variables exported by the caller."""
    return load_dotenv(dotenv_path=dotenv_path, override=False)


load_environment()


def initialize_database(app):
    """Create missing tables only when explicitly called for this app."""
    with app.app_context():
        db.create_all()


def create_app(test_config=None):
    app = Flask(__name__, template_folder="visualizacion/templates", static_folder="visualizacion/static")
    max_image_mb = int(os.getenv("MAX_IMAGE_MB", "10"))
    app.config.update(MAX_IMAGE_MB=max(1, min(25, max_image_mb)), SQLALCHEMY_TRACK_MODIFICATIONS=False,
                      SQLALCHEMY_DATABASE_URI=os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL))
    app.config["MAX_CONTENT_LENGTH"] = app.config["MAX_IMAGE_MB"] * 1024 * 1024
    if test_config:
        app.config.update(test_config)
    db.init_app(app)
    model_path, classes_path = model_paths()
    app.extensions["classifier"] = None
    app.extensions["classes"] = []
    app.extensions["model_error"] = None
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
    def predict():
        uploaded = request.files.get("image")
        if uploaded is None or not uploaded.filename:
            return jsonify(error="Debes seleccionar una imagen."), 400
        try:
            validate_upload(uploaded.filename, uploaded.stream)
            batch = prepare_image(uploaded.stream, app.config["MAX_CONTENT_LENGTH"])
        except UnsupportedImageExtensionError as exc:
            return jsonify(error=str(exc)), 415
        except ImageValidationError as exc:
            return jsonify(error=str(exc)), 400
        model = app.extensions["classifier"]
        if model is None:
            return jsonify(error="El modelo no está disponible. Revisa la configuración del servidor."), 503
        try:
            result = classify(model, app.extensions["classes"], batch)
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
    def predictions():
        try:
            page = max(1, int(request.args.get("page", "1")))
            per_page = min(100, max(1, int(request.args.get("per_page", "20"))))
        except ValueError:
            return jsonify(error="Los parámetros de paginación deben ser números enteros."), 400
        query = Prediction.query
        category = request.args.get("category")
        if category:
            query = query.filter(Prediction.category == category)
        date_from, date_to = request.args.get("date_from"), request.args.get("date_to")
        try:
            if date_from:
                query = query.filter(Prediction.created_at >= datetime.fromisoformat(date_from).replace(tzinfo=timezone.utc))
            if date_to:
                query = query.filter(Prediction.created_at < datetime.fromisoformat(date_to).replace(tzinfo=timezone.utc) + timedelta(days=1))
        except ValueError:
            return jsonify(error="El formato de fecha debe ser AAAA-MM-DD."), 400
        result = query.order_by(Prediction.created_at.desc(), Prediction.id.desc()).paginate(page=page, per_page=per_page, error_out=False)
        return jsonify(items=[item.as_dict() for item in result.items], page=page, per_page=per_page, total=result.total, pages=result.pages)

    @app.get("/api/predictions/<int:prediction_id>")
    def prediction_detail(prediction_id):
        record = db.session.get(Prediction, prediction_id)
        if record is None:
            return jsonify(error="No se encontró la predicción solicitada."), 404
        return jsonify(record.as_dict())

    @app.get("/api/statistics")
    def statistics():
        total = db.session.query(func.count(Prediction.id)).scalar() or 0
        average = db.session.query(func.avg(Prediction.confidence)).scalar()
        rows = db.session.query(Prediction.category, func.count(Prediction.id)).group_by(Prediction.category).order_by(func.count(Prediction.id).desc()).all()
        latest = Prediction.query.order_by(Prediction.created_at.desc()).first()
        daily = db.session.query(func.date(Prediction.created_at), func.count(Prediction.id)).group_by(func.date(Prediction.created_at)).order_by(func.date(Prediction.created_at)).all()
        return jsonify(total=total, most_common=rows[0][0] if rows else None,
                       average_confidence=float(average) if average is not None else None,
                       by_category=[{"category": c, "count": n} for c, n in rows],
                       daily=[{"date": str(day), "count": n} for day, n in daily],
                       latest=latest.as_dict() if latest else None)

    @app.errorhandler(413)
    def too_large(_):
        return jsonify(error=f"El archivo supera el tamaño máximo de {app.config['MAX_IMAGE_MB']} MB."), 413

    @app.cli.command("init-db")
    def init_db_command():
        """Explicitly create missing tables for the configured database."""
        initialize_database(app)
        print("Esquema de EcoVision inicializado.")

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
