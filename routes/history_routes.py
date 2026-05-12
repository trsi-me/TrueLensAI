# -*- coding: utf-8 -*-
from flask import Blueprint, render_template

from database.db_handler import clear_all_history, delete_history_record, get_history_for_user
from utils.auth import current_user_id

history_bp = Blueprint("history", __name__)


def _json(success: bool, data=None, error=None):
    from flask import jsonify

    return jsonify(
        {"success": success, "data": data if data is not None else {}, "error": error}
    )


@history_bp.route("/")
def history_page():
    return render_template("history.html")


@history_bp.route("/data")
def history_data():
    uid = current_user_id()
    if uid is None:
        return _json(True, {"records": [], "need_login": True}, None)
    rows = get_history_for_user(uid, 200)
    return _json(True, {"records": rows, "need_login": False}, None)


@history_bp.route("/delete/<int:record_id>", methods=["DELETE"])
def delete_record(record_id: int):
    uid = current_user_id()
    if uid is None:
        return _json(False, None, "Please sign in to delete history."), 401
    if delete_history_record(record_id, uid):
        return _json(True, {"deleted": record_id}, None)
    return _json(False, None, "Record not found.")


@history_bp.route("/clear", methods=["DELETE"])
def clear_history():
    uid = current_user_id()
    if uid is None:
        return _json(False, None, "Please sign in to clear history."), 401
    clear_all_history(uid)
    return _json(True, {"cleared": True}, None)
