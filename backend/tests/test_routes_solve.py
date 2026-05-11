"""Phase 4 — solve/jobs route tests.

The solve endpoint launches the pipeline on a background thread. Tests
inject a mock orchestrator that runs synchronously and writes a fixed
``complete`` status so route assertions don't depend on real provider /
solver work. The orchestrator's own correctness is covered in
``test_pipeline.py``.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db_path = tmp_path / "solve.db"
    monkeypatch.setenv("SECRET_KEY", "phase4-solve-secret")

    from app import config as config_module
    monkeypatch.setattr(config_module, "DATABASE_PATH", str(db_path))

    from app import models as models_module
    importlib.reload(models_module)
    monkeypatch.setattr(models_module, "DATABASE_PATH", str(db_path))

    # Patch the solve route's background-thread runner to a synchronous
    # in-test stub. The orchestrator's behavior is its own tests' problem;
    # here we only care about the HTTP surface.
    from app.routes import solve as solve_module

    def _sync_stub_runner(*, job_id, user_id, problem_statement,
                          provider_name, api_key, **kwargs):
        # Fast-forward: write a complete status AND emit the terminal
        # event onto the shared bus so any SSE subscriber unblocks
        # cleanly. Tests that don't open the SSE stream simply ignore
        # the emitted events.
        models_module.update_job(
            job_id,
            status="complete",
            num_variables=5,
            num_constraints=1,
            interpreted_solution=f"stub solution for job {job_id}",
            solve_time_ms=12,
            completed_at="2026-05-11T00:00:00",
        )
        from app.pipeline import get_event_bus
        bus = get_event_bus()
        for status in ("formulating", "compiling", "validating", "solving"):
            bus.emit(job_id, status)
        bus.emit(job_id, "complete", solve_time_ms=12)

    monkeypatch.setattr(solve_module, "_launch_pipeline_in_background", _sync_stub_runner)

    from app import create_app
    return create_app(test_config={"TESTING": True})


@pytest.fixture
def client(app):
    return app.test_client()


def _signup(client, username: str):
    r = client.post("/api/auth/signup", json={
        "username": username, "password": "verysecret123",
    })
    assert r.status_code == 200, r.json
    return r.json["user"]


# ---- spec'd tests ------------------------------------------------------------


def test_solve_requires_auth(client):
    r = client.post("/api/solve", json={
        "problem_statement": "pack a knapsack",
        "provider": "claude",
        "api_key": "x",
    })
    assert r.status_code == 401
    assert r.json["code"] == "AUTH_REQUIRED"


def test_solve_with_invalid_provider_returns_400(client):
    _signup(client, "alice")
    r = client.post("/api/solve", json={
        "problem_statement": "knapsack",
        "provider": "skynet",
        "api_key": "x",
    })
    assert r.status_code == 400
    assert r.json["code"] == "UNKNOWN_PROVIDER"


def test_solve_creates_job_in_db(client):
    _signup(client, "alice")
    r = client.post("/api/solve", json={
        "problem_statement": "knapsack",
        "provider": "claude",
        "api_key": "sk-test",
    })
    assert r.status_code == 200, r.json
    job_id = r.json["job"]["id"]
    assert isinstance(job_id, str) and len(job_id) > 0

    # The stub orchestrator marked the job complete synchronously.
    fetched = client.get(f"/api/jobs/{job_id}")
    assert fetched.status_code == 200
    assert fetched.json["job"]["status"] == "complete"


def test_get_job_owner_only(client):
    """User A cannot read user B's job — must return 404 (not 403, so we
    don't leak existence)."""
    _signup(client, "alice")
    r = client.post("/api/solve", json={
        "problem_statement": "k", "provider": "claude", "api_key": "x",
    })
    alice_job_id = r.json["job"]["id"]
    client.post("/api/auth/logout")

    _signup(client, "bob")
    r2 = client.get(f"/api/jobs/{alice_job_id}")
    assert r2.status_code == 404, r2.json


def test_get_job_admin_can_see_any(client):
    """The seeded admin reads any user's job."""
    _signup(client, "alice")
    r = client.post("/api/solve", json={
        "problem_statement": "k", "provider": "claude", "api_key": "x",
    })
    alice_job_id = r.json["job"]["id"]
    client.post("/api/auth/logout")

    login = client.post("/api/auth/login", json={
        "username": "admin", "password": "admin123",
    })
    assert login.status_code == 200
    r2 = client.get(f"/api/jobs/{alice_job_id}")
    assert r2.status_code == 200
    assert r2.json["job"]["id"] == alice_job_id


def test_delete_job_removes_from_db(client):
    _signup(client, "alice")
    r = client.post("/api/solve", json={
        "problem_statement": "k", "provider": "claude", "api_key": "x",
    })
    job_id = r.json["job"]["id"]

    delete = client.delete(f"/api/jobs/{job_id}")
    assert delete.status_code == 200
    assert delete.json["success"] is True

    follow_up = client.get(f"/api/jobs/{job_id}")
    assert follow_up.status_code == 404


def test_jobs_list_paginated_and_filtered_by_user(client):
    """A user sees only their own jobs; the response is paginated."""
    _signup(client, "alice")
    for _ in range(3):
        client.post("/api/solve", json={
            "problem_statement": "k", "provider": "claude", "api_key": "x",
        })
    client.post("/api/auth/logout")

    _signup(client, "bob")
    client.post("/api/solve", json={
        "problem_statement": "k", "provider": "claude", "api_key": "x",
    })

    r = client.get("/api/jobs?page=1&page_size=10")
    assert r.status_code == 200
    payload = r.json
    assert payload["total"] == 1   # bob's own
    assert len(payload["jobs"]) == 1


# ---- additional sanity tests -------------------------------------------------


def test_solve_rejects_oversized_problem_statement(client):
    _signup(client, "alice")
    huge = "x" * 9000
    r = client.post("/api/solve", json={
        "problem_statement": huge,
        "provider": "claude",
        "api_key": "x",
    })
    assert r.status_code == 400
    assert r.json["code"] == "PROBLEM_TOO_LONG"


def test_solve_requires_api_key_or_stored_key(client):
    """If the user has no stored key for the provider AND doesn't send
    one inline, the response is 402 (BYOK gate)."""
    _signup(client, "alice")
    r = client.post("/api/solve", json={
        "problem_statement": "k",
        "provider": "claude",
        "use_stored_key": True,
    })
    assert r.status_code == 402
    assert r.json["code"] == "NO_API_KEY"


def test_solve_uses_stored_key_when_requested(client):
    """Storing a key + asking for `use_stored_key: true` works without an
    inline api_key."""
    _signup(client, "alice")
    client.put("/api/keys/claude", json={"key": "sk-ant-stored-key"})
    r = client.post("/api/solve", json={
        "problem_statement": "k",
        "provider": "claude",
        "use_stored_key": True,
    })
    assert r.status_code == 200, r.json
    assert r.json["job"]["status"] in ("queued", "complete")


def test_get_nonexistent_job_returns_404(client):
    _signup(client, "alice")
    r = client.get("/api/jobs/does-not-exist")
    assert r.status_code == 404


def test_jobs_stream_returns_sse_content_type(client):
    """The SSE route's content-type signals event-stream so the
    browser's EventSource picks it up, and the body actually carries the
    expected event sequence — the stub orchestrator emits all five
    stages onto the bus during the synchronous POST handler."""
    _signup(client, "alice")
    r = client.post("/api/solve", json={
        "problem_statement": "k", "provider": "claude", "api_key": "x",
    })
    job_id = r.json["job"]["id"]
    response = client.get(f"/api/jobs/{job_id}/stream")
    assert response.headers["Content-Type"].startswith("text/event-stream")
    # The stub already pushed terminal "complete" before we subscribed, so
    # the bus replays history and closes — no blocking.
    body = response.get_data(as_text=True)
    assert "event: status" in body
    assert "complete" in body
