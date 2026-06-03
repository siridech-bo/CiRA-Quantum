"""Tests for the QML-1 blueprint shell.

Same isolation pattern as ``test_routes_admin.py``: monkeypatch the
SQLite path to a tmp DB, seed a user, exercise the public + auth-gated
endpoints, and verify the payload structure.
"""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def isolated_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db_path = tmp_path / "qml.db"
    monkeypatch.setenv("SECRET_KEY", "qml-test-secret")
    from app import config as config_module
    monkeypatch.setattr(config_module, "DATABASE_PATH", str(db_path))
    from app import models as models_module
    importlib.reload(models_module)
    monkeypatch.setattr(models_module, "DATABASE_PATH", str(db_path))
    models_module.init_db()
    regular = models_module.create_user("qml_alice", "p4sswordpass")

    from app import create_app
    app = create_app({"TESTING": True})
    return app, models_module, regular


def _login(client, username: str, password: str = "p4sswordpass"):
    r = client.post(
        "/api/auth/login",
        json={"username": username, "password": password},
    )
    assert r.status_code == 200, r.get_data(as_text=True)


def test_qml_health_reports_capabilities(isolated_app):
    app, *_ = isolated_app
    client = app.test_client()
    r = client.get("/api/qml/health")
    assert r.status_code == 200
    payload = r.get_json()
    assert payload["status"] == "ok"
    assert "QML-1" in payload["phase"]
    assert set(payload["capabilities"]) == {"pennylane", "sklearn", "qiskit_ibm_runtime"}
    # Every capability flag is a boolean, even if the import failed.
    for flag in payload["capabilities"].values():
        assert isinstance(flag, bool)


def test_qml_datasets_listing_is_public(isolated_app):
    app, *_ = isolated_app
    client = app.test_client()
    # No login — dataset gallery must be reachable like the benchmark dashboard.
    r = client.get("/api/qml/datasets")
    assert r.status_code == 200
    payload = r.get_json()
    assert payload["total"] == 6  # planning decision: 6 datasets
    ids = {d["id"] for d in payload["datasets"]}
    assert ids == {"moons", "circles", "iris", "wine", "mnist_0v1", "breast_cancer"}
    # Every entry carries the gallery-card fields.
    for d in payload["datasets"]:
        for field in ("id", "title", "category", "difficulty",
                      "n_features", "n_classes", "n_samples", "summary"):
            assert field in d, f"missing {field} in dataset {d.get('id')}"


def test_qml_dataset_detail_and_404(isolated_app):
    app, *_ = isolated_app
    client = app.test_client()
    r = client.get("/api/qml/datasets/moons")
    assert r.status_code == 200
    d = r.get_json()
    assert d["id"] == "moons"
    assert d["n_classes"] == 2

    r2 = client.get("/api/qml/datasets/does-not-exist")
    assert r2.status_code == 404


def test_qml_dataset_preview_2d_returns_raw_features(isolated_app):
    """Moons is natively 2D — preview returns standard-scaled (x, y)
    pairs with no PCA applied."""
    app, *_ = isolated_app
    client = app.test_client()
    r = client.get("/api/qml/datasets/moons/preview")
    assert r.status_code == 200
    payload = r.get_json()
    assert payload["id"] == "moons"
    assert payload["pca_applied"] is False
    # Class names + feature names come back in a shape the scatter
    # plot can use directly.
    assert payload["classes"] == ["inner", "outer"]
    assert payload["feature_names"] == ["x", "y"]
    # 200 points capped, every one has x/y/label.
    assert payload["n_points"] == 200
    assert len(payload["points"]) == 200
    p0 = payload["points"][0]
    assert {"x", "y", "label"} == set(p0.keys())
    assert p0["label"] in (0, 1)


def test_qml_dataset_preview_high_dim_applies_pca(isolated_app):
    """Breast-cancer is 30D — preview PCA-projects to 2D so the gallery
    scatter plot still works."""
    app, *_ = isolated_app
    client = app.test_client()
    r = client.get("/api/qml/datasets/breast_cancer/preview")
    assert r.status_code == 200
    payload = r.get_json()
    assert payload["pca_applied"] is True
    assert payload["feature_names"] == ["PC1", "PC2"]
    # PCA + 200-point cap → exactly 200 points.
    assert payload["n_points"] == 200


def test_qml_dataset_preview_404_on_unknown(isolated_app):
    app, *_ = isolated_app
    client = app.test_client()
    r = client.get("/api/qml/datasets/does-not-exist/preview")
    assert r.status_code == 404


def test_qml_jobs_requires_auth(isolated_app):
    app, *_ = isolated_app
    client = app.test_client()
    assert client.get("/api/qml/jobs").status_code == 401
    assert client.get("/api/qml/jobs/anything").status_code == 401
    assert client.delete("/api/qml/jobs/anything").status_code == 401


def test_qml_jobs_crud_roundtrip(isolated_app):
    app, models, regular = isolated_app
    # Seed a couple of QML training jobs directly via the helper.
    job_a = models.create_qml_job(regular["id"], "moons", "vqc")
    job_b = models.create_qml_job(regular["id"], "iris", "vqc")

    client = app.test_client()
    _login(client, "qml_alice")

    # List
    r = client.get("/api/qml/jobs")
    assert r.status_code == 200
    payload = r.get_json()
    assert payload["total"] == 2
    job_ids = {j["id"] for j in payload["jobs"]}
    assert job_ids == {job_a, job_b}

    # Detail
    r = client.get(f"/api/qml/jobs/{job_a}")
    assert r.status_code == 200
    assert r.get_json()["dataset_id"] == "moons"
    assert r.get_json()["model"] == "vqc"
    assert r.get_json()["status"] == "queued"

    # Other user's job 404s (security boundary).
    other = models.create_user("qml_bob", "p4sswordpass")
    other_job = models.create_qml_job(other["id"], "moons", "vqc")
    r = client.get(f"/api/qml/jobs/{other_job}")
    assert r.status_code == 404

    # Delete
    r = client.delete(f"/api/qml/jobs/{job_a}")
    assert r.status_code == 200
    assert r.get_json()["deleted"] is True
    # Second delete returns 404 because the row is gone.
    assert client.delete(f"/api/qml/jobs/{job_a}").status_code == 404
