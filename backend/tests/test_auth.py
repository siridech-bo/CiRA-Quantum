"""Phase 0 — authentication tests.

The 8 tests below match the v2 spec's Phase-0 test list exactly. Each
test uses an isolated SQLite database under a pytest ``tmp_path`` so
the suite never touches the dev ``backend/data/app.db`` file.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Build a fresh Flask app with an isolated SQLite DB. Reloads the
    config and models modules so the `DATABASE_PATH` module constant
    sees the monkeypatched env var."""
    db_path = tmp_path / "test_app.db"
    monkeypatch.setenv("CIRA_DB_PATH", str(db_path))
    monkeypatch.setenv("SECRET_KEY", "test-only-secret-not-for-prod-use-x")

    # `app.config.DATABASE_PATH` is computed at import time, so we override
    # it directly on the module after import.
    from app import config as config_module
    monkeypatch.setattr(config_module, "DATABASE_PATH", str(db_path))

    # Force a reload of `app.models` so it re-resolves DATABASE_PATH.
    from app import models as models_module
    importlib.reload(models_module)
    monkeypatch.setattr(models_module, "DATABASE_PATH", str(db_path))

    # Now build the Flask app (which calls init_db() against the new path).
    from app import create_app
    flask_app = create_app(test_config={"TESTING": True})
    yield flask_app


@pytest.fixture
def client(app):
    return app.test_client()


# ---- 1 ----

def test_signup_creates_user(client):
    r = client.post("/api/auth/signup", json={
        "username": "alice",
        "password": "verysecret123",
        "display_name": "Alice",
        "email": "alice@example.com",
    })
    assert r.status_code == 200, r.json
    body = r.json
    assert body["success"] is True
    assert body["user"]["username"] == "alice"
    assert body["user"]["display_name"] == "Alice"
    assert body["user"]["role"] == "user"
    # Session cookie should now be set.
    me = client.get("/api/auth/me")
    assert me.status_code == 200
    assert me.json["user"]["username"] == "alice"


# ---- 2 ----

def test_signup_rejects_duplicate_username(client):
    client.post("/api/auth/signup", json={
        "username": "bob", "password": "verysecret123",
    })
    r = client.post("/api/auth/signup", json={
        "username": "bob", "password": "anotherone456",
    })
    assert r.status_code == 409
    assert r.json["code"] == "ALREADY_EXISTS"


# ---- 3 ----

def test_login_with_valid_credentials_succeeds(client):
    client.post("/api/auth/signup", json={
        "username": "carol", "password": "verysecret123",
    })
    # Drop the session cookie via logout so we're truly logging in fresh.
    client.post("/api/auth/logout")
    r = client.post("/api/auth/login", json={
        "username": "carol", "password": "verysecret123",
    })
    assert r.status_code == 200
    assert r.json["user"]["username"] == "carol"


# ---- 4 ----

def test_login_with_invalid_credentials_fails(client):
    r = client.post("/api/auth/login", json={
        "username": "nobody", "password": "whatever",
    })
    assert r.status_code == 401
    assert r.json["code"] == "INVALID_CREDENTIALS"


# ---- 5 ----

def test_me_returns_user_when_authenticated(client):
    client.post("/api/auth/signup", json={
        "username": "dave", "password": "verysecret123",
    })
    r = client.get("/api/auth/me")
    assert r.status_code == 200
    assert r.json["user"]["username"] == "dave"


# ---- 6 ----

def test_me_returns_401_when_not_authenticated(client):
    r = client.get("/api/auth/me")
    assert r.status_code == 401
    assert r.json["code"] == "AUTH_REQUIRED"


# ---- 7 ----

def test_logout_clears_session(client):
    client.post("/api/auth/signup", json={
        "username": "eve", "password": "verysecret123",
    })
    # Logged in immediately after signup
    assert client.get("/api/auth/me").status_code == 200
    logout = client.post("/api/auth/logout")
    assert logout.status_code == 200
    assert logout.json["success"] is True
    # Subsequent /me should be unauthenticated
    me = client.get("/api/auth/me")
    assert me.status_code == 401


# ---- 8 ----

def test_default_admin_created_on_init(client):
    """The schema-bootstrap path seeds `admin / admin123` on first run."""
    r = client.post("/api/auth/login", json={
        "username": "admin", "password": "admin123",
    })
    assert r.status_code == 200, r.json
    assert r.json["user"]["username"] == "admin"
    assert r.json["user"]["role"] == "admin"


# ---- Bonus: health check (separate route, same blueprint set) ----

def test_health_endpoint_is_public(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json["status"] == "ok"
