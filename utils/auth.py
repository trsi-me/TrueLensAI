# -*- coding: utf-8 -*-
from typing import Optional

from flask import session

from database.db_handler import get_user_public

SESSION_USER_ID = "user_id"


def login_user_id(user_id: int) -> None:
    session[SESSION_USER_ID] = int(user_id)
    session.permanent = True


def logout_user() -> None:
    session.pop(SESSION_USER_ID, None)


def current_user_id() -> Optional[int]:
    raw = session.get(SESSION_USER_ID)
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def current_user() -> Optional[dict]:
    uid = current_user_id()
    if uid is None:
        return None
    u = get_user_public(uid)
    if u is None:
        logout_user()
        return None
    return u
