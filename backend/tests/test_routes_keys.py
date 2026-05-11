"""Phase 4 — BYOK key-management routes."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db_path = tmp_path / "keys.db"
    monkeypatch.setenv("SECRET_KEY", "phase4-keys-secret")

    from app import config as config_module
    monkeypatch.setattr(config_module, "DATABASE_PATH", str(db_path))

    from app import models as models_module
    importlib.reload(models_module)
    monkeypatch.setattr(models_module, "DATABASE_PATH", str(db_path))

    from app import create_app
    return create_app(test_config={"TESTING": True})


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def alice_session(client):
    """Sign up a user and return the client with the session cookie attached."""
    r = client.post("/api/auth/signup", json={
        "username": "alice", "password": "verysecret123", "display_name": "Alice",
    })
    assert r.status_code == 200, r.json
    return client


# ---- 1 ----

def test_put_key_stores_encrypted(alice_session, tmp_path, monkeypatch):
    me = alice_session.get("/api/auth/me").json["user"]
    r = alice_session.put("/api/keys/claude", json={"key": "sk-ant-test-12345"})
    assert r.status_code == 200, r.json
    assert r.json["success"] is True
    assert r.json["provider"] == "claude"

    # The stored ciphertext must NOT equal the plaintext — confirmation that
    # encryption is actually happening at the storage layer.
    from app import models as m
    cipher = m.get_api_key_ciphertext(user_id=me["id"], provider="claude")
    assert cipher is not None
    assert b"sk-ant-test-12345" not in cipher


# ---- 2 ----

def test_get_keys_lists_providers_no_values(alice_session):
    alice_session.put("/api/keys/claude", json={"key": "sk-ant-test"})
    alice_session.put("/api/keys/openai", json={"key": "sk-oa-test"})
    r = alice_session.get("/api/keys")
    assert r.status_code == 200
    keys = r.json["keys"]
    providers = {k["provider"] for k in keys}
    assert providers == {"claude", "openai"}
    # Ciphertext or plaintext must NEVER appear in the response.
    body_text = r.get_data(as_text=True)
    assert "sk-ant-test" not in body_text
    assert "sk-oa-test" not in body_text
    assert "encrypted_key" not in body_text


# ---- 3 ----

def test_delete_key_removes(alice_session):
    alice_session.put("/api/keys/claude", json={"key": "sk-ant-test"})
    r = alice_session.delete("/api/keys/claude")
    assert r.status_code == 200
    assert r.json["success"] is True
    # GET should now report no keys
    list_r = alice_session.get("/api/keys")
    assert list_r.json["keys"] == []


# ---- Bonus: ownership + authz coverage ----

def test_keys_require_auth(client):
    """Unauthenticated requests get 401 across the whole /api/keys surface."""
    for method, path in [
        ("get", "/api/keys"),
        ("put", "/api/keys/claude"),
        ("delete", "/api/keys/claude"),
    ]:
        method_fn = getattr(client, method)
        kwargs = {"json": {"key": "x"}} if method == "put" else {}
        r = method_fn(path, **kwargs)
        assert r.status_code == 401, f"{method} {path} returned {r.status_code}"


def test_one_user_cannot_see_anothers_keys(client):
    """Each session's GET /api/keys returns only that user's providers."""
    # Bob signs up + stores a key
    client.post("/api/auth/signup", json={
        "username": "bob", "password": "verysecret123",
    })
    client.put("/api/keys/claude", json={"key": "sk-bob-key"})
    client.post("/api/auth/logout")

    # Carol signs up + stores a DIFFERENT key
    client.post("/api/auth/signup", json={
        "username": "carol", "password": "verysecret123",
    })
    r = client.get("/api/keys")
    assert r.status_code == 200
    assert r.json["keys"] == []  # Carol's key list is empty, NOT bob's


def test_unknown_provider_is_rejected(alice_session):
    r = alice_session.put("/api/keys/skynet", json={"key": "x"})
    assert r.status_code == 400
    assert r.json["code"] == "UNKNOWN_PROVIDER"


def test_put_key_rejects_empty_value(alice_session):
    r = alice_session.put("/api/keys/claude", json={"key": ""})
    assert r.status_code == 400


def test_put_key_round_trip_decrypts_correctly():
    """The stored ciphertext must decrypt back to the original plaintext
    under the platform's KEY_ENCRYPTION_SECRET — that's the BYOK contract."""
    from app.config import KEY_ENCRYPTION_SECRET
    from app.crypto import decrypt_api_key, encrypt_api_key
    pt = "sk-ant-test-round-trip"
    cipher = encrypt_api_key(pt, KEY_ENCRYPTION_SECRET)
    assert decrypt_api_key(cipher, KEY_ENCRYPTION_SECRET) == pt
