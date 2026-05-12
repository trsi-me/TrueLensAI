# -*- coding: utf-8 -*-
import os

from flask import Blueprint, flash, jsonify, redirect, render_template, session, url_for

from database.db_handler import get_stats

main_bp = Blueprint("main", __name__)

# Placeholder numbers shown only when no scans have been recorded yet (total_scans = 0)
_PLACEHOLDER = {
    "total_scans": 1248,
    "fake_detected": 612,
    "real_detected": 636,
    "uncertain_detected": 42,
}


def _home_stats_for_template():
    raw = get_stats()
    total = int(raw.get("total_scans") or 0)
    if total > 0:
        return {
            "total_scans": total,
            "fake_detected": int(raw.get("fake_detected") or 0),
            "real_detected": int(raw.get("real_detected") or 0),
            "uncertain_detected": int(raw.get("uncertain_detected") or 0),
            "stats_estimate": False,
        }
    return {
        "total_scans": _PLACEHOLDER["total_scans"],
        "fake_detected": _PLACEHOLDER["fake_detected"],
        "real_detected": _PLACEHOLDER["real_detected"],
        "uncertain_detected": _PLACEHOLDER["uncertain_detected"],
        "stats_estimate": True,
    }


@main_bp.route("/")
def index():
    stats = _home_stats_for_template()
    return render_template("index.html", stats=stats)


@main_bp.route("/about")
def about():
    return render_template("about.html")


@main_bp.route("/report")
def analysis_report():
    data = session.pop("truelens_report", None)
    if not data:
        flash("No report data. Run an analysis first, then open the full report again.", "error")
        return redirect(url_for("main.index"))
    return render_template("report.html", report=data)


@main_bp.route("/_debug/models")
def debug_models():
    # TRUELENS_DEBUG=1 — model paths/sizes probe for deployment.
    if os.environ.get("TRUELENS_DEBUG", "").strip() != "1":
        return jsonify({"ok": False, "error": "disabled"}), 404
    from config import Config
    from ml_models.model_loader import (
        get_image_detector,
        get_text_detector,
        wait_for_models_init,
    )

    wait_for_models_init()

    def _sz(p: str):
        try:
            return os.path.getsize(p) if os.path.isfile(p) else None
        except OSError:
            return None

    img = get_image_detector()
    txt = get_text_detector()
    return jsonify(
        {
            "ok": True,
            "image_loaded": bool(img is not None and img.is_loaded()),
            "text_loaded": bool(txt is not None),
            "paths": {
                "image_h5": Config.IMAGE_MODEL_PATH,
                "image_tflite": Config.IMAGE_MODEL_TFLITE_PATH,
                "text_pkl": Config.TEXT_MODEL_PATH,
                "tfidf_pkl": Config.TFIDF_PATH,
            },
            "sizes": {
                "image_h5": _sz(Config.IMAGE_MODEL_PATH),
                "image_tflite": _sz(Config.IMAGE_MODEL_TFLITE_PATH),
                "text_pkl": _sz(Config.TEXT_MODEL_PATH),
                "tfidf_pkl": _sz(Config.TFIDF_PATH),
            },
        }
    )
