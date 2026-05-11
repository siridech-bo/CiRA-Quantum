"""BYOK key-management endpoints.

    GET    /api/keys              list user's stored providers (no values)
    PUT    /api/keys/<provider>   store/update a key for a provider
    DELETE /api/keys/<provider>   remove a stored key

All routes require an authenticated session; the stored key is encrypted
with :func:`app.crypto.encrypt_api_key` under
``KEY_ENCRYPTION_SECRET``. The plaintext is *never* echoed back — not
on PUT, not on GET, not in errors. The list endpoint omits both
plaintext and ciphertext for the same reason: the dashboard wants
provider names + creation dates, not opaque blobs.
"""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from app.auth import get_current_user, login_required
from app.config import KEY_ENCRYPTION_SECRET
from app.crypto import encrypt_api_key
from app.formulation import list_providers
from app.models import delete_api_key, list_api_keys, put_api_key

keys_bp = Blueprint("keys", __name__)


def _is_known_provider(name: str) -> bool:
    try:
        return name in list_providers()
    except Exception:
        # If the registry hasn't bootstrapped yet (e.g. during testing
        # with a fresh app), be permissive — the route doesn't dispatch
        # to the provider, only stores its key.
        return name in {"claude", "openai", "local"}


@keys_bp.route("", methods=["GET"])
@login_required
def list_keys():
    user = get_current_user()
    rows = list_api_keys(user["id"])
    return jsonify({"keys": rows})


@keys_bp.route("/<provider>", methods=["PUT"])
@login_required
def put_key(provider: str):
    if not _is_known_provider(provider):
        return jsonify({
            "error": f"unknown provider {provider!r}",
            "code": "UNKNOWN_PROVIDER",
        }), 400

    payload = request.get_json(silent=True) or {}
    key = (payload.get("key") or "").strip()
    if not key:
        return jsonify({
            "error": "key must be a non-empty string",
            "code": "EMPTY_KEY",
        }), 400

    user = get_current_user()
    cipher = encrypt_api_key(key, KEY_ENCRYPTION_SECRET)
    put_api_key(user["id"], provider, cipher)
    return jsonify({"success": True, "provider": provider})


@keys_bp.route("/<provider>", methods=["DELETE"])
@login_required
def remove_key(provider: str):
    user = get_current_user()
    removed = delete_api_key(user["id"], provider)
    return jsonify({"success": True, "removed": removed})
