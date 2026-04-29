# -*- coding: utf-8 -*-
import os

from flask import Flask, send_from_directory

from config import Config
from database.db_handler import init_db
from ml_models.model_loader import init_models
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

    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(os.path.join(Config.BASE_DIR, "ml_models", "saved_models"), exist_ok=True)

    with app.app_context():
        init_db()

    init_models()

    app.register_blueprint(main_bp)
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
    application.run(debug=True, host="127.0.0.1", port=5000)
