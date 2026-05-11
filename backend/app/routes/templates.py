"""Problem-template endpoints.

    GET  /api/templates                    summary list + category counts
    GET  /api/templates/categories         counts per category (cached chips)
    GET  /api/templates/modules            grouped lessons (v2 Modules view)
    GET  /api/templates/<id>               full template detail
    POST /api/solve/from-template/<id>     submit a job pre-filled from a template

All routes require an authenticated session. The from-template POST is
parked under ``/api/solve/...`` rather than ``/api/templates/...`` so
the frontend's solve store can route both code paths through the same
backend module on submit.
"""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from app.auth import get_current_user, login_required
from app.config import KEY_ENCRYPTION_SECRET
from app.crypto import decrypt_api_key
from app.formulation import list_providers
from app.models import create_job, get_api_key_ciphertext, get_job
from app.templates import (
    aggregate_categories,
    get_template,
    list_modules,
    list_templates,
)

templates_bp = Blueprint("templates", __name__)


# Fields the gallery card needs — kept lean so the list endpoint stays
# under a few KB even with hundreds of templates.
_SUMMARY_FIELDS = (
    "id", "title", "category", "difficulty", "summary",
    "tags", "estimated_solve_time_seconds",
)


def _summary(t: dict) -> dict:
    out = {field: t.get(field) for field in _SUMMARY_FIELDS}
    # Templates that are part of a Module carry a lightweight pointer so
    # the gallery can show "lesson 1 of 5" without re-fetching detail.
    if t.get("module"):
        out["module_id"] = t["module"].get("module_id")
        out["module_order"] = t["module"].get("order")
    return out


@templates_bp.route("", methods=["GET"])
@login_required
def index():
    return jsonify({
        "templates": [_summary(t) for t in list_templates()],
        "categories": aggregate_categories(),
    })


@templates_bp.route("/categories", methods=["GET"])
@login_required
def categories():
    return jsonify({"categories": aggregate_categories()})


@templates_bp.route("/modules", methods=["GET"])
@login_required
def modules():
    return jsonify({"modules": list_modules()})


@templates_bp.route("/<template_id>", methods=["GET"])
@login_required
def detail(template_id: str):
    t = get_template(template_id)
    if t is None:
        return jsonify({"error": f"unknown template {template_id!r}",
                        "code": "NOT_FOUND"}), 404
    return jsonify({"template": t})


# ---- POST /api/solve/from-template/<id> -----------------------------------
#
# Registered on its own blueprint with the ``/api`` prefix so the path is
# ``/api/solve/from-template/<id>``. Reuses the orchestrator launcher from
# the solve route module — exactly the same code path; only the
# ``problem_statement`` source differs.

solve_from_template_bp = Blueprint("solve_from_template", __name__)


@solve_from_template_bp.route("/solve/from-template/<template_id>", methods=["POST"])
@login_required
def solve_from_template(template_id: str):
    user = get_current_user()
    template = get_template(template_id)
    if template is None:
        return jsonify({
            "error": f"unknown template {template_id!r}",
            "code": "NOT_FOUND",
        }), 404

    payload = request.get_json(silent=True) or {}
    provider = (payload.get("provider") or "").strip()

    if provider not in (list_providers() or {"claude", "openai", "local"}):
        return jsonify({
            "error": f"unknown provider {provider!r}",
            "code": "UNKNOWN_PROVIDER",
        }), 400

    # Resolve the API key — same precedence as the plain /api/solve route.
    inline = payload.get("api_key")
    use_stored = bool(payload.get("use_stored_key", False))
    if provider == "local":
        api_key = ""
    elif inline and not use_stored:
        api_key = inline
    else:
        cipher = get_api_key_ciphertext(user["id"], provider)
        if cipher is None:
            return jsonify({
                "error": f"no stored key for {provider!r}",
                "code": "NO_API_KEY",
            }), 402
        try:
            api_key = decrypt_api_key(cipher, KEY_ENCRYPTION_SECRET)
        except ValueError as e:
            return jsonify({
                "error": f"stored key for {provider!r} is unreadable: {e}",
                "code": "STORED_KEY_BROKEN",
            }), 402

    expected = template.get("expected_optimum")
    job_id = create_job(
        user["id"],
        template["problem_statement"],
        provider,
        template_id=template_id,
        expected_optimum=expected,
    )

    # Late import so we share the route module's monkeypatch-friendly
    # ``_launch_pipeline_in_background`` (test fixtures replace it with a
    # synchronous stub).
    from app.routes import solve as solve_module
    solve_module._launch_pipeline_in_background(
        job_id=job_id,
        user_id=user["id"],
        problem_statement=template["problem_statement"],
        provider_name=provider,
        api_key=api_key,
    )

    job = get_job(job_id, user_id=user["id"])
    return jsonify({"success": True, "job": job})
