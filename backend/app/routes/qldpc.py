"""qLDPC — Quantum Low-Density Parity-Check codes blueprint.

Sprint 0 shipped the public code-family registry surface (gallery,
single-family detail, capability health).

Sprint 1 adds the Phase 1 math surface:

* ``GET /api/qldpc/code-families/<id>/matrix``        H_X / H_Z + CSS check
* ``GET /api/qldpc/code-families/<id>/distance``      code distance (?exact=)
* ``GET /api/qldpc/code-families/<id>/tanner-graph``  Tanner graph for Sprint 2

When the ``qldpc`` PyPI package isn't installed the Sprint 1 endpoints
return 503 with ``code: QLDPC_STACK_MISSING`` and point at the install
command. The Sprint 0 endpoints keep working in both states; when
``qldpc`` is installed they overlay live ``n, k, d`` from the live
factories so the gallery cards stay accurate.
"""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from app.qldpc import (
    get_code_family,
    get_code_family_live,
    list_code_families,
    list_code_families_live,
)

qldpc_bp = Blueprint("qldpc", __name__)


# ---- Health ----------------------------------------------------------------


@qldpc_bp.route("/health", methods=["GET"])
def qldpc_health():
    """Tells the frontend which qLDPC capabilities are available.

    Sprint 1 lights up ``qldpc_lib`` (matrix gen + distance + CSS check)
    once the ``[qldpc]`` extra is installed. ``stim`` and ``qiskit_qec``
    light up in Sprints 3 and 4 respectively.
    """
    return jsonify({
        "status": "ok",
        "phase": "qLDPC Sprint 1 — Phase 1 math",
        "capabilities": {
            "qldpc_lib": _can_import("qldpc"),
            "stim": _can_import("stim"),
            "qiskit_qec": _can_import("qiskit_qec"),
            "networkx": _can_import("networkx"),
        },
    })


def _can_import(module: str) -> bool:
    try:
        __import__(module)
        return True
    except Exception:
        return False


# ---- Code families (public) -----------------------------------------------


@qldpc_bp.route("/code-families", methods=["GET"])
def qldpc_code_families():
    """Gallery list. When ``qldpc`` is installed each entry includes
    live ``n, k, d`` computed via the factory + distance bound."""
    if _can_import("qldpc"):
        items = list_code_families_live()
    else:
        items = list_code_families()
    return jsonify({"code_families": items, "total": len(items)})


@qldpc_bp.route("/code-families/<family_id>", methods=["GET"])
def qldpc_code_family_detail(family_id: str):
    """Single-family detail. Same canned-vs-live policy as the list endpoint."""
    if _can_import("qldpc"):
        c = get_code_family_live(family_id)
    else:
        c = get_code_family(family_id)
    if c is None:
        return jsonify({"error": "Code family not found"}), 404
    return jsonify(c)


# ---- Sprint 1: parity-check matrices + CSS verification --------------------


@qldpc_bp.route("/code-families/<family_id>/matrix", methods=["GET"])
def qldpc_code_family_matrix(family_id: str):
    """Return H_X and H_Z plus the CSS commutativity verification.

    The CSS check confirms ``H_X · H_Zᵀ ≡ 0 (mod 2)``. A well-formed
    CSS code always passes; a residual of N≠0 means N pairs of X/Z
    stabilizers anti-commute, which would break the syndrome calculus
    (this would be a bug in the generator, not a user error).
    """
    if get_code_family(family_id) is None:
        return jsonify({"error": "Code family not found"}), 404

    missing = _require_qldpc_stack()
    if missing is not None:
        return missing

    from app.qldpc.generators import get_factory
    from app.qldpc.serializers import code_to_matrix_payload, verify_css_commutativity

    factory = get_factory(family_id)
    if factory is None:
        return jsonify({
            "error": "No live factory wired up for this code family yet.",
            "code": "FACTORY_MISSING",
        }), 501

    code = factory()
    matrix = code_to_matrix_payload(code)
    css = verify_css_commutativity(code)
    return jsonify({
        "family_id": family_id,
        "matrix": matrix,
        "css_check": css,
    })


@qldpc_bp.route("/code-families/<family_id>/distance", methods=["GET"])
def qldpc_code_family_distance(family_id: str):
    """Compute the code distance. ``?exact=true`` for the slow exact path.

    Default is the fast upper bound from ``get_distance(bound=True)`` —
    returns within a few hundred milliseconds for all four families.
    Exact distance is feasible for Surface / Toric at our default
    parameters; for HGP / BB it can take minutes and isn't recommended
    in an interactive request.
    """
    if get_code_family(family_id) is None:
        return jsonify({"error": "Code family not found"}), 404

    missing = _require_qldpc_stack()
    if missing is not None:
        return missing

    from app.qldpc.generators import get_factory
    from app.qldpc.serializers import compute_distance

    factory = get_factory(family_id)
    if factory is None:
        return jsonify({
            "error": "No live factory wired up for this code family yet.",
            "code": "FACTORY_MISSING",
        }), 501

    exact_arg = (request.args.get("exact") or "").lower()
    exact = exact_arg in {"1", "true", "yes", "on"}

    code = factory()
    payload = compute_distance(code, exact=exact)
    payload["family_id"] = family_id
    return jsonify(payload)


@qldpc_bp.route("/code-families/<family_id>/tanner-graph", methods=["GET"])
def qldpc_code_family_tanner(family_id: str):
    """Return the Tanner subgraphs (X and Z) as JSON node/edge lists.

    Sprint 1 shape: ``{graph_x, graph_z, node_count_x, node_count_z}``.

    Sprint 2 query param: ``?strategy=<bipartite|kamada_kawai|spring|circular>``
    layers 2D positions + routing metrics on top, returning additionally
    ``{positions_x, positions_z, metrics_x, metrics_z, strategy,
      available_strategies}``. When ``strategy`` is omitted the response
    matches Sprint 1 exactly (backward compatible).
    """
    if get_code_family(family_id) is None:
        return jsonify({"error": "Code family not found"}), 404

    missing = _require_qldpc_stack()
    if missing is not None:
        return missing

    from app.qldpc.generators import get_factory
    from app.qldpc.layouts import (
        SUPPORTED_STRATEGIES,
        build_nx_graph,
        layout_payload,
    )
    from app.qldpc.serializers import code_to_tanner_payload

    factory = get_factory(family_id)
    if factory is None:
        return jsonify({
            "error": "No live factory wired up for this code family yet.",
            "code": "FACTORY_MISSING",
        }), 501

    code = factory()
    payload = code_to_tanner_payload(code)
    payload["family_id"] = family_id
    payload["available_strategies"] = list(SUPPORTED_STRATEGIES)

    strategy = (request.args.get("strategy") or "").strip()
    if strategy:
        if strategy not in SUPPORTED_STRATEGIES:
            return jsonify({
                "error": f"Unknown layout strategy '{strategy}'. "
                         f"Supported: {', '.join(SUPPORTED_STRATEGIES)}",
                "code": "UNKNOWN_STRATEGY",
                "available_strategies": list(SUPPORTED_STRATEGIES),
            }), 400
        gx = build_nx_graph(payload["graph_x"])
        gz = build_nx_graph(payload["graph_z"])
        lx = layout_payload(gx, strategy)
        lz = layout_payload(gz, strategy)
        payload["strategy"] = strategy
        payload["positions_x"] = lx["positions"]
        payload["positions_z"] = lz["positions"]
        payload["metrics_x"] = lx["metrics"]
        payload["metrics_z"] = lz["metrics"]

    return jsonify(payload)


# ---- Helpers ---------------------------------------------------------------


def _require_qldpc_stack():
    """Return a 503 response if ``qldpc`` isn't installed, else ``None``."""
    if not _can_import("qldpc"):
        return jsonify({
            "error": "The qldpc Python package is not installed on the server. "
                     "Run `pip install \".[qldpc]\"`.",
            "code": "QLDPC_STACK_MISSING",
        }), 503
    return None
