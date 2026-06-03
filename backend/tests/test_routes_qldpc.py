"""qLDPC route tests — Sprint 0 (smoke) + Sprint 1 (math endpoints).

Mirrors the style of ``test_routes_admin.py``: a tmp-DB fixture so the
test suite can run in parallel, then HTTP calls against the blueprint.
All qLDPC endpoints are public — no auth setup needed.

Sprint 1 endpoints (matrix / distance / tanner-graph) are gated on the
``qldpc`` PyPI lib being installed. Tests that exercise the live math
paths use ``pytest.importorskip("qldpc")`` so the file passes on
contributor machines without the ``[qldpc]`` extra.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def isolated_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db_path = tmp_path / "qldpc.db"
    monkeypatch.setenv("SECRET_KEY", "qldpc-test-secret")
    from app import config as config_module
    monkeypatch.setattr(config_module, "DATABASE_PATH", str(db_path))
    from app import models as models_module
    importlib.reload(models_module)
    monkeypatch.setattr(models_module, "DATABASE_PATH", str(db_path))
    models_module.init_db()

    from app import create_app
    app = create_app({"TESTING": True})
    return app


def test_qldpc_health_returns_ok_with_capabilities(isolated_app):
    client = isolated_app.test_client()
    r = client.get("/api/qldpc/health")
    assert r.status_code == 200
    payload = r.get_json()
    assert payload["status"] == "ok"
    assert "Sprint" in payload["phase"]
    caps = payload["capabilities"]
    # All four capability flags must be present (even if False).
    assert set(caps.keys()) == {"qldpc_lib", "stim", "qiskit_qec", "networkx"}
    for v in caps.values():
        assert isinstance(v, bool)


def test_qldpc_code_families_lists_all_four(isolated_app):
    client = isolated_app.test_client()
    r = client.get("/api/qldpc/code-families")
    assert r.status_code == 200
    payload = r.get_json()
    assert payload["total"] >= 4
    ids = {f["id"] for f in payload["code_families"]}
    assert {"bicycle", "surface", "hypergraph_product", "toric"} <= ids


def test_qldpc_code_family_detail_returns_known(isolated_app):
    client = isolated_app.test_client()
    r = client.get("/api/qldpc/code-families/surface")
    assert r.status_code == 200
    fam = r.get_json()
    assert fam["id"] == "surface"
    assert fam["category"] == "topological"
    # Distance must not exceed physical qubits.
    assert fam["d"] <= fam["n"]


def test_qldpc_code_family_detail_404s_on_unknown(isolated_app):
    client = isolated_app.test_client()
    r = client.get("/api/qldpc/code-families/does-not-exist")
    assert r.status_code == 404
    assert "error" in r.get_json()


def test_qldpc_endpoints_are_public(isolated_app):
    """No login required for any of the Sprint 0 endpoints."""
    client = isolated_app.test_client()
    # Confirm anonymous access works (no auth cookie set).
    for path in [
        "/api/qldpc/health",
        "/api/qldpc/code-families",
        "/api/qldpc/code-families/bicycle",
    ]:
        r = client.get(path)
        assert r.status_code == 200, f"{path} should be public, got {r.status_code}"


# ---- Sprint 1: matrix / distance / tanner-graph endpoints ------------------

# These exercise the live math path; require the qldpc PyPI lib.
qldpc_lib = pytest.importorskip("qldpc")


@pytest.fixture(autouse=True)
def _reset_live_cache():
    from app.qldpc.code_families import invalidate_live_cache
    invalidate_live_cache()
    yield
    invalidate_live_cache()


def test_matrix_endpoint_returns_well_formed_css_payload(isolated_app):
    client = isolated_app.test_client()
    r = client.get("/api/qldpc/code-families/surface/matrix")
    assert r.status_code == 200
    body = r.get_json()
    assert body["family_id"] == "surface"

    mx = body["matrix"]
    assert mx["n"] == 169
    assert mx["k"] == 1
    assert mx["num_checks_x"] == len(mx["matrix_x"])
    assert mx["num_checks_z"] == len(mx["matrix_z"])
    # All entries 0 or 1
    assert {v for row in mx["matrix_x"] for v in row} <= {0, 1}

    css = body["css_check"]
    assert css["commutes"] is True
    assert css["residual_nonzero_count"] == 0


def test_matrix_endpoint_404s_for_unknown_family(isolated_app):
    client = isolated_app.test_client()
    r = client.get("/api/qldpc/code-families/does-not-exist/matrix")
    assert r.status_code == 404


def test_distance_endpoint_default_returns_upper_bound(isolated_app):
    client = isolated_app.test_client()
    r = client.get("/api/qldpc/code-families/surface/distance")
    assert r.status_code == 200
    body = r.get_json()
    assert body["mode"] == "upper_bound"
    assert body["distance"] == 13
    assert body["time_ms"] >= 0


def test_distance_endpoint_exact_for_surface(isolated_app):
    client = isolated_app.test_client()
    r = client.get("/api/qldpc/code-families/surface/distance?exact=true")
    assert r.status_code == 200
    body = r.get_json()
    assert body["mode"] == "exact"
    assert body["distance"] == 13


def test_tanner_graph_endpoint_returns_subgraphs(isolated_app):
    client = isolated_app.test_client()
    r = client.get("/api/qldpc/code-families/surface/tanner-graph")
    assert r.status_code == 200
    body = r.get_json()
    assert body["family_id"] == "surface"
    for key in ("graph_x", "graph_z"):
        graph = body[key]
        assert isinstance(graph["nodes"], list) and len(graph["nodes"]) > 0
        assert isinstance(graph["edges"], list)
        for node in graph["nodes"][:5]:
            assert node["type"] in {"data", "check"}
    # Sprint 1 baseline: no strategy → no layout fields.
    assert "positions_x" not in body
    assert "metrics_x" not in body
    # Sprint 2: always advertise the available strategies even on the
    # no-strategy path so the frontend can populate its selector.
    assert set(body["available_strategies"]) == {
        "bipartite", "kamada_kawai", "spring", "circular",
    }


# Sprint 2: layout strategies ------------------------------------------------


def test_tanner_graph_with_bipartite_strategy_returns_positions_and_metrics(isolated_app):
    client = isolated_app.test_client()
    r = client.get(
        "/api/qldpc/code-families/hypergraph_product/tanner-graph?strategy=bipartite"
    )
    assert r.status_code == 200
    body = r.get_json()
    assert body["strategy"] == "bipartite"
    # Every node in graph_x has a position.
    node_ids = {n["id"] for n in body["graph_x"]["nodes"]}
    assert set(body["positions_x"].keys()) == node_ids
    # Positions are 2-element lists of floats.
    for coord in list(body["positions_x"].values())[:5]:
        assert isinstance(coord, list) and len(coord) == 2
    # Metrics shape is consistent.
    mx = body["metrics_x"]
    assert mx["num_edges"] >= 1
    assert mx["min_edge_length"] <= mx["avg_edge_length"] <= mx["max_edge_length"]
    assert mx["edge_crossings"] >= 0
    # Same for graph_z.
    assert "positions_z" in body
    assert "metrics_z" in body


@pytest.mark.parametrize("strategy", ["bipartite", "kamada_kawai", "spring", "circular"])
def test_all_strategies_work_for_a_small_family(isolated_app, strategy: str):
    client = isolated_app.test_client()
    r = client.get(
        f"/api/qldpc/code-families/hypergraph_product/tanner-graph?strategy={strategy}"
    )
    assert r.status_code == 200, r.get_data(as_text=True)
    body = r.get_json()
    assert body["strategy"] == strategy
    assert len(body["positions_x"]) > 0


def test_unknown_strategy_returns_400(isolated_app):
    client = isolated_app.test_client()
    r = client.get(
        "/api/qldpc/code-families/surface/tanner-graph?strategy=does-not-exist"
    )
    assert r.status_code == 400
    body = r.get_json()
    assert body["code"] == "UNKNOWN_STRATEGY"
    assert set(body["available_strategies"]) == {
        "bipartite", "kamada_kawai", "spring", "circular",
    }


def test_code_family_detail_includes_live_flag_when_qldpc_installed(isolated_app):
    client = isolated_app.test_client()
    r = client.get("/api/qldpc/code-families/surface")
    assert r.status_code == 200
    fam = r.get_json()
    # When qldpc is installed the live overlay marks the entry.
    assert fam.get("live") is True
    assert fam["n"] == 169 and fam["k"] == 1 and fam["d"] == 13


def test_stack_missing_returns_503(isolated_app, monkeypatch):
    """If qldpc can't be imported, the live endpoints return 503.

    We monkey-patch the route module's ``_can_import`` helper rather
    than blocking the actual qldpc package — that's surgical and
    doesn't leak state into other tests.
    """
    from app.routes import qldpc as routes_qldpc
    monkeypatch.setattr(
        routes_qldpc, "_can_import",
        lambda module: False if module == "qldpc" else True,
    )

    client = isolated_app.test_client()
    r = client.get("/api/qldpc/code-families/surface/matrix")
    assert r.status_code == 503
    body = r.get_json()
    assert body["code"] == "QLDPC_STACK_MISSING"

    # Same gating on the other Sprint 1 endpoints.
    r2 = client.get("/api/qldpc/code-families/surface/distance")
    assert r2.status_code == 503
    r3 = client.get("/api/qldpc/code-families/surface/tanner-graph")
    assert r3.status_code == 503
