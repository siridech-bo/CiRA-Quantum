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

import time

from flask import Blueprint, jsonify, request

from app.auth import get_current_user, login_required
from app.config import KEY_ENCRYPTION_SECRET
from app.crypto import decrypt_api_key, encrypt_api_key
from app.formulation import list_providers
from app.models import (
    delete_api_key,
    get_api_key_ciphertext,
    list_api_keys,
    put_api_key,
)

keys_bp = Blueprint("keys", __name__)


# Quantum-hardware BYOK providers. Distinct from LLM formulation
# providers — these authenticate against quantum-cloud APIs (Origin
# Quantum, future IBM Q, etc.) rather than LLM endpoints. The keys
# table is provider-agnostic, so we just expand the allow-list.
_QUANTUM_PROVIDERS = frozenset({"originqc"})


def _is_known_provider(name: str) -> bool:
    if name in _QUANTUM_PROVIDERS:
        return True
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


# ---- POST /<provider>/test --------------------------------------------------


def _test_originqc(api_key: str) -> dict:
    """Submit a 1-qubit Hadamard+measure to the cheap cloud simulator
    (``full_amplitude``) and wait for the result. Returns a structured
    dict — never raises. Total wall time is typically 5-10 seconds.
    Confirms (a) the key authenticates with Origin's cloud, (b) the
    pyqpanda3 submission pipeline still works, and (c) the cloud
    simulator returns parseable probabilities.
    """
    try:
        import pyqpanda3.core as pq3
        import pyqpanda3.qcloud as qcloud_mod
    except ImportError:
        return {
            "ok": False,
            "error": "pyqpanda3 not installed on the server",
            "code": "PYQPANDA_MISSING",
        }

    t0 = time.perf_counter()
    try:
        service = qcloud_mod.QCloudService(api_key=api_key)
        backend = service.backend("full_amplitude")
    except Exception as e:  # noqa: BLE001
        return {
            "ok": False,
            "error": f"Cloud auth setup failed: {type(e).__name__}: {e}",
            "code": "AUTH_SETUP_FAILED",
            "elapsed_ms": int((time.perf_counter() - t0) * 1000),
        }

    prog = pq3.QProg()
    prog << pq3.H(0)
    prog << pq3.measure(0, 0)

    try:
        # Simulator path: NO QCloudOptions (raises in pyqpanda3 v0.3.5).
        job = backend.run(prog, 100)
        job_id = job.job_id()
        result = job.result()
        probs = result.get_probs()
    except Exception as e:  # noqa: BLE001
        return {
            "ok": False,
            "error": f"Cloud submission failed: {type(e).__name__}: {e}",
            "code": "SUBMIT_FAILED",
            "elapsed_ms": int((time.perf_counter() - t0) * 1000),
        }

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    return {
        "ok": True,
        "message": (
            f"Authenticated; submitted a 1-qubit H+measure job to "
            f"full_amplitude and got back {len(probs)} outcome(s)."
        ),
        "job_id": job_id,
        "probabilities": dict(probs),
        "elapsed_ms": elapsed_ms,
    }


_PROVIDER_TESTERS = {
    "originqc": _test_originqc,
}


@keys_bp.route("/<provider>/test", methods=["POST"])
@login_required
def test_key(provider: str):
    """Liveness-check a stored BYOK key by running a minimal cheap
    operation against the provider's API. The test never burns paid
    resources (the originqc test hits the free cloud simulator, not
    the QPU).
    """
    user = get_current_user()
    tester = _PROVIDER_TESTERS.get(provider)
    if tester is None:
        return jsonify({
            "ok": False,
            "error": f"no liveness test wired for provider {provider!r}",
            "code": "UNSUPPORTED",
        }), 400

    cipher = get_api_key_ciphertext(user["id"], provider)
    if cipher is None:
        return jsonify({
            "ok": False,
            "error": f"no stored {provider!r} key on file",
            "code": "NO_KEY",
        }), 404

    try:
        api_key = decrypt_api_key(cipher, KEY_ENCRYPTION_SECRET)
    except ValueError as e:
        return jsonify({
            "ok": False,
            "error": f"stored key is unreadable: {e}",
            "code": "DECRYPT_FAILED",
        }), 500

    result = tester(api_key)
    return jsonify(result), (200 if result["ok"] else 502)
