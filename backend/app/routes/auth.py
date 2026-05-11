"""Authentication endpoints.

The auth surface ships in Phase 0:

    POST /api/auth/signup              public
    POST /api/auth/login               public
    POST /api/auth/logout              login-required
    GET  /api/auth/me                  login-required
    POST /api/auth/change-password     login-required

All return JSON. Errors come back with an ``error`` (human) and ``code``
(machine) field so the Phase-5 frontend can branch without parsing
prose.
"""

from __future__ import annotations

import re

from flask import Blueprint, jsonify, request, session

from app.auth import get_current_user, login_required
from app.models import change_password, create_user, get_user_by_id, verify_user

auth_bp = Blueprint("auth", __name__)


_USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_MIN_PASSWORD_LEN = 8


def _validation_error(message: str, *, code: str = "VALIDATION_ERROR", status: int = 400):
    return jsonify({"error": message, "code": code}), status


def _start_session(user: dict) -> None:
    session.clear()
    session["user_id"] = user["id"]
    session["username"] = user["username"]
    session["display_name"] = user["display_name"]
    session["user_role"] = user["role"]
    session.permanent = True


def _public_user(user: dict) -> dict:
    """Strip the row down to what's safe to ship to the client."""
    return {
        "id": user["id"],
        "username": user["username"],
        "display_name": user["display_name"],
        "role": user["role"],
    }


@auth_bp.route("/signup", methods=["POST"])
def signup():
    payload = request.get_json(silent=True) or {}
    username = (payload.get("username") or "").strip()
    password = payload.get("password") or ""
    email = (payload.get("email") or "").strip() or None
    display_name = (payload.get("display_name") or "").strip() or None

    if not _USERNAME_RE.match(username):
        return _validation_error(
            "username must be 3-32 chars from [A-Za-z0-9_.-]",
            code="USERNAME_INVALID",
        )
    if len(password) < _MIN_PASSWORD_LEN:
        return _validation_error(
            f"password must be at least {_MIN_PASSWORD_LEN} characters",
            code="PASSWORD_TOO_SHORT",
        )
    if email is not None and not _EMAIL_RE.match(email):
        return _validation_error("email is malformed", code="EMAIL_INVALID")

    try:
        user = create_user(
            username=username,
            password=password,
            email=email,
            display_name=display_name,
        )
    except ValueError:
        return _validation_error(
            "username or email already taken",
            code="ALREADY_EXISTS",
            status=409,
        )

    _start_session(user)
    return jsonify({"success": True, "user": _public_user(user)})


@auth_bp.route("/login", methods=["POST"])
def login():
    payload = request.get_json(silent=True) or {}
    username = (payload.get("username") or "").strip()
    password = payload.get("password") or ""

    if not username or not password:
        return _validation_error(
            "username and password are required",
            code="MISSING_CREDENTIALS",
        )

    user = verify_user(username, password)
    if user is None:
        return jsonify({"error": "invalid credentials", "code": "INVALID_CREDENTIALS"}), 401

    _start_session(user)
    return jsonify({"success": True, "user": _public_user(user)})


@auth_bp.route("/logout", methods=["POST"])
@login_required
def logout():
    session.clear()
    return jsonify({"success": True})


@auth_bp.route("/me", methods=["GET"])
@login_required
def me():
    current = get_current_user()
    # Re-fetch from DB so a stale session that survived a user-row delete
    # surfaces as a 401 instead of cached lies.
    if current is None:
        return jsonify({"error": "Authentication required", "code": "AUTH_REQUIRED"}), 401
    user = get_user_by_id(current["id"])
    if user is None or not user.get("is_active"):
        session.clear()
        return jsonify({"error": "Authentication required", "code": "AUTH_REQUIRED"}), 401
    return jsonify({"user": _public_user(user)})


@auth_bp.route("/change-password", methods=["POST"])
@login_required
def change_password_route():
    payload = request.get_json(silent=True) or {}
    current = payload.get("current_password") or ""
    new = payload.get("new_password") or ""

    if len(new) < _MIN_PASSWORD_LEN:
        return _validation_error(
            f"new password must be at least {_MIN_PASSWORD_LEN} characters",
            code="PASSWORD_TOO_SHORT",
        )

    user = get_current_user()
    if user is None:
        return jsonify({"error": "Authentication required", "code": "AUTH_REQUIRED"}), 401

    if not change_password(user["id"], current, new):
        return jsonify(
            {"error": "current password is incorrect", "code": "WRONG_CURRENT_PASSWORD"}
        ), 400
    return jsonify({"success": True})
