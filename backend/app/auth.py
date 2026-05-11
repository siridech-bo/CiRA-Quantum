"""Authentication decorators + current-user helper.

Session-based auth (server-side session, secure cookie). The route
handlers populate ``session['user_id']`` etc. on a successful login;
``@login_required`` checks for that on every protected endpoint, and
``@admin_required`` additionally checks the role.

Identical pattern to CiRA Oculus / v1 spec — no novelty here.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any

from flask import jsonify, session

from app.config import ROLE_ADMIN


def login_required(f: Callable) -> Callable:
    @wraps(f)
    def wrapper(*args: Any, **kwargs: Any):
        if "user_id" not in session:
            return jsonify({"error": "Authentication required", "code": "AUTH_REQUIRED"}), 401
        return f(*args, **kwargs)

    return wrapper


def admin_required(f: Callable) -> Callable:
    @wraps(f)
    def wrapper(*args: Any, **kwargs: Any):
        if "user_id" not in session:
            return jsonify({"error": "Authentication required", "code": "AUTH_REQUIRED"}), 401
        if session.get("user_role") != ROLE_ADMIN:
            return jsonify({"error": "Admin access required", "code": "ADMIN_REQUIRED"}), 403
        return f(*args, **kwargs)

    return wrapper


def get_current_user() -> dict[str, Any] | None:
    """Return a small dict snapshot of the logged-in user, or ``None``.

    Phase 4's route handlers consume this to enforce job ownership; Phase 5's
    frontend hits ``/api/auth/me`` to populate the Pinia auth store on boot.
    """
    if "user_id" not in session:
        return None
    return {
        "id": session.get("user_id"),
        "username": session.get("username"),
        "display_name": session.get("display_name"),
        "role": session.get("user_role"),
    }
