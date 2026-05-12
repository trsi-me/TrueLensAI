# -*- coding: utf-8 -*-
import os
import sqlite3

from flask import Blueprint, flash, redirect, render_template, request, url_for
from werkzeug.security import check_password_hash, generate_password_hash

from database.db_handler import (
    count_users,
    create_user,
    get_user_by_email,
    get_user_public,
    update_user_display_name,
    update_user_password_hash,
)
from utils.auth import current_user_id, login_user_id, logout_user
from utils.validators import (
    validate_display_name,
    validate_email,
    validate_password_for_register,
)

auth_bp = Blueprint("auth", __name__, url_prefix="/auth")


def _safe_next(target: str) -> str:
    if not target or not isinstance(target, str):
        return url_for("main.index")
    t = target.strip()
    if not t.startswith("/") or t.startswith("//"):
        return url_for("main.index")
    return t


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user_id() is not None:
        return redirect(url_for("main.index"))
    if request.method == "POST":
        email = request.form.get("email") or ""
        password = request.form.get("password") or ""
        nxt = _safe_next(request.form.get("next") or "")
        ok, err = validate_email(email)
        if not ok:
            flash(err, "error")
            return render_template("auth_login.html", next_url=nxt)
        if not password:
            flash("Password is required.", "error")
            return render_template("auth_login.html", next_url=nxt)
        row = get_user_by_email(email)
        if row is None or not check_password_hash(row["password_hash"], password):
            flash("Incorrect email or password.", "error")
            return render_template("auth_login.html", next_url=nxt)
        login_user_id(int(row["id"]))
        flash("You are signed in.", "success")
        return redirect(nxt)
    nxt = _safe_next(request.args.get("next") or "")
    return render_template("auth_login.html", next_url=nxt)


@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    if current_user_id() is not None:
        return redirect(url_for("main.index"))
    if request.method == "POST":
        email = request.form.get("email") or ""
        password = request.form.get("password") or ""
        confirm = request.form.get("confirm_password") or ""
        display_name = request.form.get("display_name") or ""
        ok, err = validate_email(email)
        if not ok:
            flash(err, "error")
            return render_template("auth_register.html")
        ok, err = validate_password_for_register(password)
        if not ok:
            flash(err, "error")
            return render_template("auth_register.html")
        if password != confirm:
            flash("Passwords do not match.", "error")
            return render_template("auth_register.html")
        ok, err = validate_display_name(display_name)
        if not ok:
            flash(err, "error")
            return render_template("auth_register.html")
        h = generate_password_hash(password)
        admin_email = os.environ.get("TRUELENS_ADMIN_EMAIL", "").strip().lower()
        is_admin = 1 if admin_email and email.strip().lower() == admin_email else 0
        if is_admin == 0 and os.environ.get("TRUELENS_BOOTSTRAP_ADMIN", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        ):
            if count_users() == 0:
                is_admin = 1
        try:
            uid = create_user(email, h, display_name or None, is_admin=is_admin)
        except sqlite3.IntegrityError:
            flash("An account with this email already exists.", "error")
            return render_template("auth_register.html")
        login_user_id(uid)
        flash("Account created. Welcome!", "success")
        return redirect(url_for("main.index"))
    return render_template("auth_register.html")


@auth_bp.route("/logout", methods=["POST"])
def logout():
    logout_user()
    flash("You have been signed out.", "success")
    return redirect(url_for("main.index"))


@auth_bp.route("/account", methods=["GET", "POST"])
def account():
    uid = current_user_id()
    if uid is None:
        flash("Please sign in to manage your account.", "error")
        return redirect(url_for("auth.login", next=request.path))
    user = get_user_public(uid)
    if user is None:
        logout_user()
        return redirect(url_for("auth.login"))
    if request.method == "POST":
        action = (request.form.get("action") or "").strip().lower()
        if action == "profile":
            display_name = request.form.get("display_name") or ""
            ok, err = validate_display_name(display_name)
            if not ok:
                flash(err, "error")
                return render_template("auth_account.html", user=user)
            update_user_display_name(uid, display_name or None)
            user = get_user_public(uid)
            flash("Profile updated.", "success")
            return render_template("auth_account.html", user=user)
        if action == "password":
            current_pw = request.form.get("current_password") or ""
            new_pw = request.form.get("new_password") or ""
            confirm = request.form.get("confirm_new_password") or ""
            full = get_user_by_email(user["email"])
            if full is None:
                logout_user()
                return redirect(url_for("auth.login"))
            if not check_password_hash(full["password_hash"], current_pw):
                flash("Current password is incorrect.", "error")
                return render_template("auth_account.html", user=user)
            ok, err = validate_password_for_register(new_pw)
            if not ok:
                flash(err, "error")
                return render_template("auth_account.html", user=user)
            if new_pw != confirm:
                flash("New passwords do not match.", "error")
                return render_template("auth_account.html", user=user)
            update_user_password_hash(uid, generate_password_hash(new_pw))
            flash("Password changed.", "success")
            return render_template("auth_account.html", user=user)
        flash("Unknown action.", "error")
    return render_template("auth_account.html", user=user)
