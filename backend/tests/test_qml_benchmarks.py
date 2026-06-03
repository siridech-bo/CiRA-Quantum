"""Tests for the QML-7 TrainRecord archive + route surface."""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def isolated_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db_path = tmp_path / "qml_bench.db"
    archive_path = tmp_path / "qml_archive"
    archive_path.mkdir()
    monkeypatch.setenv("SECRET_KEY", "qml-bench-test")
    monkeypatch.setenv("CIRA_QML_BENCH_ARCHIVE", str(archive_path))

    from app import config as config_module
    monkeypatch.setattr(config_module, "DATABASE_PATH", str(db_path))
    from app import models as models_module
    importlib.reload(models_module)
    monkeypatch.setattr(models_module, "DATABASE_PATH", str(db_path))
    models_module.init_db()

    admin = models_module.create_user("admin_qb", "p4sswordpass", role="admin")
    regular = models_module.create_user("regular_qb", "p4sswordpass")

    # Seed a completed VQC training job for the archiver to capture.
    metrics = {
        "weights": [[0.1, 0.2], [0.3, 0.4]],  # gets stripped on archive
        "bias": 0.0,
        "final_train_accuracy": 0.92,
        "final_test_accuracy": 0.86,
        "final_loss": 0.42,
        "confusion_matrix": [[20, 3], [4, 23]],
        "train_time_ms": 9421,
        "n_qubits": 2,
        "pca_applied": False,
        "classes": ["inner", "outer"],
        "feature_names": ["x", "y"],
        "baselines": [
            {"name": "logreg", "title": "Logistic Regression",
             "library": "scikit-learn", "version": "1.4.0", "family": "linear",
             "train_accuracy": 0.78, "test_accuracy": 0.76, "train_time_ms": 12,
             "confusion_matrix": [[18, 5], [7, 20]], "notes": ""},
        ],
        "scatter_points": [{"x": 0.1, "y": 0.0, "label": 0, "split": "test"}],
    }
    parent_id = models_module.create_qml_job(regular["id"], "moons", "vqc")
    models_module.update_qml_job(
        parent_id,
        status="complete",
        hyperparameters=json.dumps({"n_qubits": 2, "n_layers": 2, "n_epochs": 30}),
        training_history=json.dumps([
            {"epoch": 1, "loss": 0.7, "train_accuracy": 0.6, "test_accuracy": 0.55},
            {"epoch": 2, "loss": 0.5, "train_accuracy": 0.78, "test_accuracy": 0.74},
        ]),
        metrics=json.dumps(metrics),
        train_time_ms=9421,
        completed_at="2026-05-21T10:00:00",
    )

    from app import create_app
    # Reload records module so it picks up the CIRA_QML_BENCH_ARCHIVE
    # env var the test just set.
    from app.qml import records as records_module
    importlib.reload(records_module)
    app = create_app({"TESTING": True})
    return app, models_module, admin, regular, parent_id, archive_path


def _login(client, username, password="p4sswordpass"):
    r = client.post("/api/auth/login",
                    json={"username": username, "password": password})
    assert r.status_code == 200, r.get_data(as_text=True)


# ---- Pure-records unit tests ----------------------------------------------


def test_build_record_from_job_strips_weights(isolated_app):
    """build_record_from_job should drop weights/scatter_points/decision_grid."""
    _, models, _, regular, parent_id, _archive = isolated_app
    parent = models.get_qml_job(parent_id, is_admin=True)
    from app.qml import records as rec_mod
    record = rec_mod.build_record_from_job(
        job_row=parent,
        contributor_display_name="regular_qb",
        qpu_run_rows=[],
    )
    assert "weights" not in record.metrics
    assert "scatter_points" not in record.metrics
    assert record.metrics["final_test_accuracy"] == 0.86
    assert record.metrics["final_train_accuracy"] == 0.92
    assert record.dataset_id == "moons"
    assert record.model == "vqc"
    assert record.contributor_display_name == "regular_qb"
    # Baselines persist intact.
    assert len(record.baselines) == 1
    assert record.baselines[0]["name"] == "logreg"
    # Training history persists in full.
    assert len(record.training_history) == 2
    # repro_hash is 16 hex chars.
    assert len(record.repro_hash) == 16
    int(record.repro_hash, 16)  # parses as hex


def test_write_and_load_record_roundtrip(isolated_app):
    _, models, _, regular, parent_id, archive = isolated_app
    parent = models.get_qml_job(parent_id, is_admin=True)
    from app.qml import records as rec_mod
    record = rec_mod.build_record_from_job(
        job_row=parent, contributor_display_name="regular_qb",
    )
    path = rec_mod.write_record(record)
    assert path.exists()
    loaded = rec_mod.load_record(record.record_id)
    assert loaded is not None
    assert loaded.record_id == record.record_id
    assert loaded.dataset_id == record.dataset_id
    assert loaded.metrics["final_test_accuracy"] == 0.86


def test_summarize_lean_projection(isolated_app):
    _, models, _, regular, parent_id, _archive = isolated_app
    parent = models.get_qml_job(parent_id, is_admin=True)
    from app.qml import records as rec_mod
    record = rec_mod.build_record_from_job(
        job_row=parent, contributor_display_name="regular_qb",
    )
    summary = rec_mod.summarize(record)
    # Heavy fields aren't there.
    assert "training_history" not in summary
    assert "qpu_runs" not in summary
    # Headline numbers are.
    assert summary["final_test_accuracy"] == 0.86
    assert summary["dataset_id"] == "moons"
    assert summary["model"] == "vqc"
    assert summary["n_baselines"] == 1
    assert summary["has_real_qpu_run"] is False


def test_bibtex_entry_has_required_fields(isolated_app):
    _, models, _, regular, parent_id, _archive = isolated_app
    parent = models.get_qml_job(parent_id, is_admin=True)
    from app.qml import records as rec_mod
    record = rec_mod.build_record_from_job(
        job_row=parent, contributor_display_name="regular_qb",
    )
    bib = rec_mod.bibtex_entry(record)
    assert bib.startswith("@misc{")
    assert "title = " in bib
    assert "author = " in bib
    assert "year = " in bib
    assert record.repro_hash in bib


# ---- Route-layer tests ----------------------------------------------------


def test_route_list_empty_archive(isolated_app):
    app, *_ = isolated_app
    client = app.test_client()
    r = client.get("/api/qml/benchmarks")
    assert r.status_code == 200
    payload = r.get_json()
    assert payload["total"] == 0
    assert payload["records"] == []
    assert payload["facets"] == {"datasets": {}, "models": {}}


def test_route_list_is_public_no_auth_needed(isolated_app):
    app, *_ = isolated_app
    client = app.test_client()
    # No login.
    r = client.get("/api/qml/benchmarks")
    assert r.status_code == 200


def test_route_archive_requires_admin(isolated_app):
    app, _, _admin, _regular, parent_id, _archive = isolated_app
    client = app.test_client()
    # Anonymous: 401.
    assert client.post(f"/api/qml/benchmarks/archive/{parent_id}").status_code == 401
    # Regular user: 403.
    _login(client, "regular_qb")
    r = client.post(f"/api/qml/benchmarks/archive/{parent_id}")
    assert r.status_code == 403


def test_route_archive_rejects_incomplete_parent(isolated_app):
    app, models, _admin, _regular, _parent_id, _archive = isolated_app
    # Seed a NOT-yet-complete job so the gate fires.
    other_parent = models.create_qml_job(_regular["id"], "moons", "vqc")
    # Status stays "queued" by default.
    client = app.test_client()
    _login(client, "admin_qb")
    r = client.post(f"/api/qml/benchmarks/archive/{other_parent}")
    assert r.status_code == 400
    assert r.get_json()["code"] == "PARENT_NOT_COMPLETE"


def test_route_archive_writes_record_and_credits_contributor(isolated_app):
    app, _, _admin, _regular, parent_id, archive = isolated_app
    client = app.test_client()
    _login(client, "admin_qb")
    r = client.post(
        f"/api/qml/benchmarks/archive/{parent_id}",
        json={"notes": "reference run for the moons primer"},
    )
    assert r.status_code == 201, r.get_data(as_text=True)
    payload = r.get_json()
    record_id = payload["record_id"]
    # Record exists on disk + lists correctly.
    files = list(archive.glob("*.json"))
    assert len(files) == 1
    assert files[0].name.startswith(record_id)
    # The contributor credited is the original RUNNER, not the
    # archiving admin (so users get academic credit, gatekeeper doesn't).
    summary = payload["summary"]
    assert summary["contributor_display_name"] == "regular_qb"
    assert summary["dataset_id"] == "moons"
    assert summary["final_test_accuracy"] == 0.86


def test_route_list_after_archive_returns_record(isolated_app):
    app, _, _admin, _regular, parent_id, _archive = isolated_app
    client = app.test_client()
    _login(client, "admin_qb")
    archived = client.post(
        f"/api/qml/benchmarks/archive/{parent_id}", json={},
    ).get_json()
    # Public list (no auth).
    client_anon = app.test_client()
    r = client_anon.get("/api/qml/benchmarks")
    assert r.status_code == 200
    payload = r.get_json()
    assert payload["total"] == 1
    record_ids = [r["record_id"] for r in payload["records"]]
    assert archived["record_id"] in record_ids
    # Facets populated.
    assert payload["facets"]["datasets"]["moons"] == 1
    assert payload["facets"]["models"]["vqc"] == 1


def test_route_list_supports_dataset_filter(isolated_app):
    app, _, _admin, _regular, parent_id, _archive = isolated_app
    client = app.test_client()
    _login(client, "admin_qb")
    client.post(f"/api/qml/benchmarks/archive/{parent_id}", json={})

    # Match.
    r = client.get("/api/qml/benchmarks?dataset_id=moons")
    assert r.get_json()["total"] == 1
    # Miss.
    r = client.get("/api/qml/benchmarks?dataset_id=iris")
    assert r.get_json()["total"] == 0


def test_route_detail_and_cite(isolated_app):
    app, _, _admin, _regular, parent_id, _archive = isolated_app
    client = app.test_client()
    _login(client, "admin_qb")
    archived = client.post(
        f"/api/qml/benchmarks/archive/{parent_id}", json={},
    ).get_json()
    record_id = archived["record_id"]

    # Detail.
    r = client.get(f"/api/qml/benchmarks/{record_id}")
    assert r.status_code == 200
    record = r.get_json()
    assert record["dataset_id"] == "moons"
    assert "training_history" in record
    assert len(record["training_history"]) == 2

    # Cite — text/plain BibTeX.
    r = client.get(f"/api/qml/benchmarks/{record_id}/cite")
    assert r.status_code == 200
    assert r.mimetype == "text/plain"
    body = r.get_data(as_text=True)
    assert body.startswith("@misc{")


def test_route_delete_requires_admin(isolated_app):
    app, _, _admin, _regular, parent_id, _archive = isolated_app
    client = app.test_client()
    _login(client, "admin_qb")
    archived = client.post(
        f"/api/qml/benchmarks/archive/{parent_id}", json={},
    ).get_json()
    record_id = archived["record_id"]
    # Regular user: 403.
    other_client = app.test_client()
    _login(other_client, "regular_qb")
    assert other_client.delete(f"/api/qml/benchmarks/{record_id}").status_code == 403
    # Admin: 200 + idempotent 404 on the re-delete.
    assert client.delete(f"/api/qml/benchmarks/{record_id}").status_code == 200
    assert client.delete(f"/api/qml/benchmarks/{record_id}").status_code == 404
