# -*- coding: utf-8 -*-
from flask import Blueprint, render_template

from database.db_handler import get_user_dashboard_counts
from utils.access_control import login_required
from utils.auth import current_user_id

dashboard_bp = Blueprint("dashboard", __name__)


@dashboard_bp.route("/dashboard")
@login_required
def dashboard():
    uid = current_user_id()
    assert uid is not None
    counts = get_user_dashboard_counts(uid)
    return render_template("dashboard.html", counts=counts)
