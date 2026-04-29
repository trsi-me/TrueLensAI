# -*- coding: utf-8 -*-
from flask import Blueprint, render_template

from database.db_handler import clear_all_history, delete_history_record, get_all_history

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
    rows = get_all_history(200)
    return _json(True, {"records": rows}, None)


@history_bp.route("/delete/<int:record_id>", methods=["DELETE"])
def delete_record(record_id: int):
    if delete_history_record(record_id):
        return _json(True, {"deleted": record_id}, None)
    return _json(False, None, "Record not found.")


@history_bp.route("/clear", methods=["DELETE"])
def clear_history():
    clear_all_history()
    return _json(True, {"cleared": True}, None)
