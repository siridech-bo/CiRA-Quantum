"""Tests for the Phase 7 admin endpoints.

Pattern matches the other ``test_routes_*.py`` files: monkeypatch the
SQLite path to a tmp DB, seed a couple of users (one admin, one regular),
hit the endpoints with each session, and verify the gating + payload
structure.
"""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def isolated_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db_path = tmp_path / "admin.db"
    monkeypatch.setenv("SECRET_KEY", "admin-test-secret")
    from app import config as config_module
    monkeypatch.setattr(config_module, "DATABASE_PATH", str(db_path))
    from app import models as models_module
    importlib.reload(models_module)
    monkeypatch.setattr(models_module, "DATABASE_PATH", str(db_path))
    models_module.init_db()
    # Seed: admin user, regular user, one job, one stored key.
    admin = models_module.create_user("admin_alice", "p4sswordpass", role="admin")
    regular = models_module.create_user("regular_bob", "p4sswordpass")
    job_id = models_module.create_job(
        regular["id"], "knapsack thing", "claude",
    )
    models_module.put_api_key(regular["id"], "claude", b"ciphertext-blob")

    from app import create_app
    app = create_app({"TESTING": True})
    return app, models_module, admin, regular, job_id


def _login_as(client, username: str, password: str = "p4sswordpass"):
    r = client.post(
        "/api/auth/login",
        json={"username": username, "password": password},
    )
    assert r.status_code == 200, r.get_data(as_text=True)


def test_admin_endpoints_require_admin(isolated_app):
    app, _, _admin, _regular, _job_id = isolated_app
    client = app.test_client()
    # Anonymous: 401
    assert client.get("/api/admin/users").status_code == 401
    assert client.get("/api/admin/jobs").status_code == 401
    assert client.get("/api/admin/overview").status_code == 401
    # Logged in as regular user: 403
    _login_as(client, "regular_bob")
    assert client.get("/api/admin/users").status_code == 403
    assert client.get("/api/admin/jobs").status_code == 403
    assert client.get("/api/admin/overview").status_code == 403


def test_admin_users_lists_all_users_with_providers(isolated_app):
    app, _, _admin, regular, _job_id = isolated_app
    client = app.test_client()
    _login_as(client, "admin_alice")
    r = client.get("/api/admin/users")
    assert r.status_code == 200
    payload = r.get_json()
    assert payload["total"] >= 2  # default admin + alice + bob
    users_by_name = {u["username"]: u for u in payload["users"]}
    assert "admin_alice" in users_by_name
    assert "regular_bob" in users_by_name
    bob = users_by_name["regular_bob"]
    assert bob["role"] == "user"
    assert bob["is_active"] is True
    assert bob["providers_on_file"] == ["claude"]
    assert bob["total_jobs"] == 1
    # No ciphertext / plaintext key material in the response anywhere.
    body_str = r.get_data(as_text=True)
    assert "ciphertext-blob" not in body_str


def test_admin_jobs_paginated_and_filterable(isolated_app):
    app, models, _admin, regular, _job_id = isolated_app
    # Seed a few more jobs in various statuses.
    for i in range(3):
        models.create_job(regular["id"], f"problem {i}", "claude")
    # Mark one job as 'complete'.
    completed_id = models.create_job(regular["id"], "done already", "claude")
    models.update_job(completed_id, status="complete")

    client = app.test_client()
    _login_as(client, "admin_alice")
    r = client.get("/api/admin/jobs?page=1&page_size=10")
    assert r.status_code == 200
    payload = r.get_json()
    assert payload["total"] >= 5
    assert "username" in payload["jobs"][0]
    # Filter by status.
    r2 = client.get("/api/admin/jobs?status=complete")
    assert r2.status_code == 200
    payload2 = r2.get_json()
    assert all(j["status"] == "complete" for j in payload2["jobs"])
    assert payload2["status_filter"] == "complete"


def test_admin_overview_structure(isolated_app):
    app, _, _admin, _regular, _job_id = isolated_app
    client = app.test_client()
    _login_as(client, "admin_alice")
    r = client.get("/api/admin/overview")
    assert r.status_code == 200
    payload = r.get_json()
    assert "users" in payload
    assert "jobs" in payload
    assert payload["users"]["total"] >= 2
    assert payload["users"]["admins"] >= 1
    assert payload["jobs"]["total"] >= 1
    assert isinstance(payload["jobs"]["by_status"], dict)
    assert isinstance(payload["jobs"]["top_providers"], list)
    assert "pending_cloud_jobs" in payload
