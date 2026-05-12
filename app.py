# -*- coding: utf-8 -*-
import os
from datetime import timedelta

from flask import Flask, send_from_directory

from config import Config
from database.db_handler import init_db
from ml_models.model_loader import init_models, start_models_loading_thread
from routes.admin_routes import admin_bp
from routes.auth_routes import auth_bp
from routes.dashboard_routes import dashboard_bp
from routes.history_routes import history_bp
from routes.image_routes import image_bp
from routes.main_routes import main_bp
from routes.text_routes import text_bp
from routes.video_routes import video_bp


def create_app():
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )
    app.config.from_object("config.Config")
    app.permanent_session_lifetime = timedelta(days=14)
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
    if os.environ.get("SESSION_COOKIE_SECURE", "").strip().lower() in ("1", "true", "yes", "on"):
        app.config["SESSION_COOKIE_SECURE"] = True

    @app.context_processor
    def _inject_current_user():
        from utils.auth import current_user as auth_current_user

        return {"current_user": auth_current_user()}

    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(os.path.join(Config.BASE_DIR, "ml_models", "saved_models"), exist_ok=True)

    with app.app_context():
        init_db()

    # Local dev on Windows with Flask reloader can restart repeatedly and interrupt the background
    # model-loading thread. Allow eager load when explicitly requested.
    eager = os.environ.get("TRUELENS_EAGER_LOAD_MODELS", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if eager:
        init_models()
    else:
        start_models_loading_thread()

    app.register_blueprint(main_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(admin_bp)
    app.register_blueprint(text_bp, url_prefix="/detect")
    app.register_blueprint(image_bp, url_prefix="/detect")
    app.register_blueprint(video_bp, url_prefix="/detect")
    app.register_blueprint(history_bp, url_prefix="/history")

    @app.route("/assets/<path:filename>")
    def assets_files(filename):
        return send_from_directory(os.path.join(Config.BASE_DIR, "assets"), filename)

    return app


if __name__ == "__main__":
    application = create_app()
    # Defaults are chosen to avoid watchdog/reloader loops on Windows.
    debug = os.environ.get("FLASK_DEBUG", "").strip() in ("1", "true", "True")
    use_reloader = os.environ.get("FLASK_USE_RELOADER", "").strip() in ("1", "true", "True")
    application.run(
        debug=debug,
        use_reloader=use_reloader,
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "5000")),
    )
