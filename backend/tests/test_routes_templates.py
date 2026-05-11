"""Phase 5B — template routes + solve-from-template route tests."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db_path = tmp_path / "templates.db"
    monkeypatch.setenv("SECRET_KEY", "phase5b-test-secret")

    from app import config as config_module
    monkeypatch.setattr(config_module, "DATABASE_PATH", str(db_path))

    from app import models as models_module
    importlib.reload(models_module)
    monkeypatch.setattr(models_module, "DATABASE_PATH", str(db_path))

    # Skip the real pipeline — we only verify the route surface here.
    from app.routes import solve as solve_module

    def _sync_stub_runner(*, job_id, **kwargs):
        models_module.update_job(
            job_id,
            status="complete",
            num_variables=5,
            num_constraints=1,
            interpreted_solution=f"stub for {job_id}",
        )

    monkeypatch.setattr(
        solve_module, "_launch_pipeline_in_background", _sync_stub_runner
    )

    from app import create_app
    return create_app(test_config={"TESTING": True})


@pytest.fixture
def client(app):
    return app.test_client()


def _signup(client, username: str = "alice"):
    r = client.post("/api/auth/signup", json={
        "username": username, "password": "verysecret123",
    })
    assert r.status_code == 200, r.json
    return r.json["user"]


# ---- GET /api/templates ----


def test_get_templates_requires_auth(client):
    r = client.get("/api/templates")
    assert r.status_code == 401


def test_get_templates_returns_summary_list(client):
    _signup(client)
    r = client.get("/api/templates")
    assert r.status_code == 200
    body = r.json
    assert "templates" in body
    assert "categories" in body
    assert len(body["templates"]) >= 10
    # Each summary row has just the fields the gallery card needs.
    summary = body["templates"][0]
    for field in (
        "id", "title", "category", "difficulty", "summary", "tags",
        "estimated_solve_time_seconds",
    ):
        assert field in summary
    # Full template fields should NOT appear in the summary (kept lean).
    assert "problem_statement" not in summary


# ---- GET /api/templates/<id> ----


def test_get_template_by_id_returns_full(client):
    _signup(client)
    r = client.get("/api/templates/knapsack_classic")
    assert r.status_code == 200
    t = r.json["template"]
    assert t["id"] == "knapsack_classic"
    # The full object includes everything the detail modal needs.
    for field in (
        "problem_statement", "real_world_example", "expected_pattern",
        "expected_optimum", "expected_solution_summary", "learning_notes",
    ):
        assert field in t


def test_get_template_by_id_404(client):
    _signup(client)
    r = client.get("/api/templates/does_not_exist")
    assert r.status_code == 404
    assert r.json["code"] == "NOT_FOUND"


# ---- GET /api/templates/categories ----


def test_categories_endpoint_returns_counts(client):
    _signup(client)
    r = client.get("/api/templates/categories")
    assert r.status_code == 200
    cats = r.json["categories"]
    names = {c["name"] for c in cats}
    for required in ("allocation", "scheduling", "routing", "graph", "finance", "logic"):
        assert required in names
    total = sum(c["count"] for c in cats)
    assert total >= 10


# ---- POST /api/solve/from-template/<id> ----


def test_solve_from_template_creates_job_with_template_id(client):
    _signup(client)
    client.put("/api/keys/claude", json={"key": "sk-test"})
    r = client.post(
        "/api/solve/from-template/knapsack_classic",
        json={"provider": "claude", "use_stored_key": True},
    )
    assert r.status_code == 200, r.json
    job = r.json["job"]
    assert job["template_id"] == "knapsack_classic"
    assert job["expected_optimum"] == 26


def test_solve_from_template_unknown_id_returns_404(client):
    _signup(client)
    r = client.post(
        "/api/solve/from-template/does_not_exist",
        json={"provider": "claude", "api_key": "sk-test"},
    )
    assert r.status_code == 404


def test_solve_from_template_sets_expected_optimum(client):
    _signup(client)
    r = client.post(
        "/api/solve/from-template/max_cut_6node",
        json={"provider": "claude", "api_key": "sk-test"},
    )
    assert r.status_code == 200, r.json
    job_id = r.json["job"]["id"]
    detail = client.get(f"/api/jobs/{job_id}").json["job"]
    assert detail["template_id"] == "max_cut_6node"
    assert detail["expected_optimum"] == 7


# ---- Modules view (v2 addition) -------------------------------------------


def test_modules_endpoint_groups_by_module_id(client):
    _signup(client)
    r = client.get("/api/templates/modules")
    assert r.status_code == 200
    modules = r.json["modules"]
    assert "qubo_foundations" in modules
    lessons = modules["qubo_foundations"]
    assert len(lessons) >= 5
    # Lessons are pre-sorted by `module.order` for the frontend.
    orders = [lesson["module"]["order"] for lesson in lessons]
    assert orders == sorted(orders)
