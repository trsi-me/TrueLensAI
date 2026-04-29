# -*- coding: utf-8 -*-
from flask import Blueprint, render_template

from database.db_handler import get_stats

main_bp = Blueprint("main", __name__)

# Placeholder numbers shown only when no scans have been recorded yet (total_scans = 0)
_PLACEHOLDER = {
    "total_scans": 1248,
    "fake_detected": 612,
    "real_detected": 636,
}


def _home_stats_for_template():
    raw = get_stats()
    total = int(raw.get("total_scans") or 0)
    if total > 0:
        return {
            "total_scans": total,
            "fake_detected": int(raw.get("fake_detected") or 0),
            "real_detected": int(raw.get("real_detected") or 0),
            "stats_estimate": False,
        }
    return {
        "total_scans": _PLACEHOLDER["total_scans"],
        "fake_detected": _PLACEHOLDER["fake_detected"],
        "real_detected": _PLACEHOLDER["real_detected"],
        "stats_estimate": True,
    }


@main_bp.route("/")
def index():
    stats = _home_stats_for_template()
    return render_template("index.html", stats=stats)


@main_bp.route("/about")
def about():
    return render_template("about.html")
