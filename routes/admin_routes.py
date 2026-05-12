# -*- coding: utf-8 -*-
from flask import Blueprint, render_template

from database.db_handler import get_all_history, get_stats, list_users_admin
from utils.access_control import admin_required

admin_bp = Blueprint("admin", __name__, url_prefix="/admin")


@admin_bp.route("/")
@admin_required
def admin_home():
    users = list_users_admin(300)
    stats = get_stats()
    return render_template("admin_home.html", users=users, stats=stats)


@admin_bp.route("/detections")
@admin_required
def admin_detections():
    rows = get_all_history(400)
    return render_template("admin_detections.html", records=rows)
