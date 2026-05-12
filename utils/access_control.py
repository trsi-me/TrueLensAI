# -*- coding: utf-8 -*-
from functools import wraps
from typing import Any, Callable, TypeVar, cast

from flask import flash, redirect, request, url_for

from database.db_handler import get_user_public
from utils.auth import current_user_id

F = TypeVar("F", bound=Callable[..., Any])


def login_required(view: F) -> F:
    @wraps(view)
    def wrapped(*args, **kwargs):
        if current_user_id() is None:
            flash("Please sign in to access this page.", "error")
            return redirect(url_for("auth.login", next=request.path))
        return view(*args, **kwargs)

    return cast(F, wrapped)


def admin_required(view: F) -> F:
    @wraps(view)
    def wrapped(*args, **kwargs):
        uid = current_user_id()
        if uid is None:
            flash("Please sign in.", "error")
            return redirect(url_for("auth.login", next=request.path))
        u = get_user_public(uid)
        if not u or not int(u.get("is_admin") or 0):
            flash("You do not have permission to access the admin area.", "error")
            return redirect(url_for("main.index"))
        return view(*args, **kwargs)

    return cast(F, wrapped)
