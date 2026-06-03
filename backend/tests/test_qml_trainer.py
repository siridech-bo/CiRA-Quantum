"""Tests for the QML-2 trainer + VQC.

The trainer is the heaviest test in the suite because it actually runs
PennyLane gradient steps. We keep it cheap by using Moons with 4 epochs;
that's enough to verify the loss decreases, history is emitted, and
the result struct populates correctly.
"""
from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest

# Skip module if either PennyLane or sklearn is missing.
pytest.importorskip("pennylane")
pytest.importorskip("sklearn")

from app.qml import data_loader
from app.qml.vqc import VQCConfig, train_vqc


def test_train_vqc_loss_decreases_on_moons():
    """Sanity: a tiny VQC reduces BCE on a separable dataset."""
    split = data_loader.load("moons", max_qubits=2)
    cfg = VQCConfig(
        n_qubits=2, n_layers=2, n_epochs=4,
        batch_size=32, learning_rate=0.2, seed=42,
    )
    emitted = []
    result = train_vqc(
        split.X_train, split.y_train,
        split.X_test, split.y_test,
        config=cfg,
        on_epoch=lambda i, m: emitted.append(m),
    )
    # Callback fires once per epoch.
    assert len(emitted) == cfg.n_epochs
    # Loss trend should be downward (allow the last epoch to wobble a bit).
    assert result.history[-1]["loss"] < result.history[0]["loss"] + 1e-2
    # Accuracy on the test set is at least chance + something.
    assert result.final_test_accuracy >= 0.5
    # Confusion matrix has the right shape and totals to the test-set size.
    cm = np.array(result.confusion_matrix)
    assert cm.shape == (2, 2)
    assert cm.sum() == split.X_test.shape[0]
    # Weights serialized as nested lists, bias as a plain float.
    assert isinstance(result.weights, list)
    assert all(isinstance(row, list) for row in result.weights)
    assert isinstance(result.bias, float)


def test_train_vqc_history_fields():
    split = data_loader.load("moons", max_qubits=2)
    cfg = VQCConfig(n_qubits=2, n_layers=1, n_epochs=2, batch_size=32, seed=1)
    result = train_vqc(
        split.X_train, split.y_train,
        split.X_test, split.y_test,
        config=cfg,
    )
    assert len(result.history) == 2
    for h in result.history:
        assert {"epoch", "loss", "train_accuracy", "test_accuracy"}.issubset(h)


@pytest.fixture
def isolated_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Same isolation pattern as the other route test files."""
    db_path = tmp_path / "qml_trainer.db"
    monkeypatch.setenv("SECRET_KEY", "qml-trainer-test")
    from app import config as config_module
    monkeypatch.setattr(config_module, "DATABASE_PATH", str(db_path))
    from app import models as models_module
    importlib.reload(models_module)
    monkeypatch.setattr(models_module, "DATABASE_PATH", str(db_path))
    models_module.init_db()
    user = models_module.create_user("qml_trainer_user", "p4sswordpass")

    from app.pipeline import events as events_module
    events_module.reset_event_bus_for_tests()

    from app import create_app
    app = create_app({"TESTING": True})
    return app, models_module, user


def _login(client, username, password="p4sswordpass"):
    r = client.post("/api/auth/login", json={"username": username, "password": password})
    assert r.status_code == 200, r.get_data(as_text=True)


def test_run_training_job_persists_metrics(isolated_app):
    """End-to-end through the orchestrator: row goes queued → training → complete."""
    app, models, user = isolated_app
    from app.pipeline.events import get_event_bus
    from app.qml.trainer import run_training_job

    job_id = models.create_qml_job(user["id"], "moons", "vqc")
    run_training_job(
        job_id=job_id,
        dataset_id="moons",
        hyperparameters={
            "n_qubits": 2, "n_layers": 1, "n_epochs": 2,
            "batch_size": 32, "learning_rate": 0.2, "seed": 0,
        },
        event_bus=get_event_bus(),
    )
    row = models.get_qml_job(job_id, user_id=user["id"])
    assert row["status"] == "complete"
    assert row["train_time_ms"] is not None and row["train_time_ms"] > 0
    import json as _json
    metrics = _json.loads(row["metrics"])
    assert "final_test_accuracy" in metrics
    assert "confusion_matrix" in metrics
    # Educational transparency: circuit_info + feature_names + classes
    # must be present so the detail page can render the explainer.
    ci = metrics["circuit_info"]
    assert ci["backend_kind"] == "statevector"
    assert ci["is_real_hardware"] is False
    assert ci["n_qubits"] == 2
    assert ci["n_layers"] == 1
    assert ci["n_trainable_params"] == 2 * 1 + 1
    assert "AngleEmbedding" in ci["encoding"]
    assert "BasicEntanglerLayers" in ci["entangler"]
    assert metrics["feature_names"] == ["x", "y"]
    assert metrics["classes"] == ["inner", "outer"]
    history = _json.loads(row["training_history"])
    assert len(history) == 2
    # QML-3 — classical baselines run on the same split and land in the
    # final metrics blob alongside the VQC result.
    baselines = metrics["baselines"]
    assert len(baselines) == 4
    assert {b["name"] for b in baselines} == {
        "logreg", "svm_rbf", "random_forest", "mlp",
    }
    for b in baselines:
        assert 0.0 <= b["test_accuracy"] <= 1.0
        assert b["library"] == "scikit-learn"
    # QML-4 — final decision grid + scatter points for the boundary plot.
    grid = metrics["decision_grid"]
    assert grid["resolution"] == 20
    assert len(grid["probabilities"]) == 20 * 20
    assert grid["x_min"] < grid["x_max"]
    assert grid["y_min"] < grid["y_max"]
    scatter = metrics["scatter_points"]
    assert isinstance(scatter, list) and len(scatter) > 0
    s0 = scatter[0]
    assert {"x", "y", "label", "split"} == set(s0.keys())


def test_decision_grid_shape_and_bounds():
    """compute_decision_grid is the engine behind the live boundary."""
    from app.qml.vqc import _build_circuit, compute_decision_grid
    from pennylane import numpy as pnp
    circuit = _build_circuit(2, 1)
    weights = pnp.array([[0.1, 0.2]], requires_grad=True)
    bias = 0.0
    X = np.array([[0.0, 0.0], [1.0, 1.0], [-1.0, -1.0]], dtype=np.float32)
    grid = compute_decision_grid(circuit, weights, bias, X, resolution=10, pad=0.25)
    assert grid["resolution"] == 10
    assert len(grid["probabilities"]) == 100
    # Every probability is in [0, 1].
    assert all(0.0 <= p <= 1.0 for p in grid["probabilities"])
    # Bounds extend past the data by exactly `pad`.
    assert grid["x_min"] == pytest.approx(-1.25)
    assert grid["x_max"] == pytest.approx(1.25)


def test_trainer_emits_decision_grid_events(isolated_app):
    """The trainer pushes ``decision_grid`` SSE events for 2-qubit jobs."""
    app, models, user = isolated_app
    from app.pipeline.events import EventBus
    from app.qml.trainer import run_training_job

    job_id = models.create_qml_job(user["id"], "moons", "vqc")
    bus = EventBus()
    events: list[dict] = []
    import threading
    done = threading.Event()
    def _drain():
        for ev in bus.subscribe(job_id):
            events.append(ev)
            if ev["status"] in ("complete", "error"):
                done.set()
                return
    t = threading.Thread(target=_drain, daemon=True)
    t.start()

    # 6 epochs at every-3 cadence → at least 2 boundary events
    # (one at epoch 3, one at epoch 6).
    run_training_job(
        job_id=job_id,
        dataset_id="moons",
        hyperparameters={"n_qubits": 2, "n_layers": 1, "n_epochs": 6, "seed": 0},
        event_bus=bus,
    )
    done.wait(timeout=60)

    boundary_events = [e for e in events if e["status"] == "decision_grid"]
    assert len(boundary_events) >= 2
    g = boundary_events[0]["grid"]
    assert g["resolution"] == 20
    assert len(g["probabilities"]) == 400


def test_training_event_includes_circuit_info(isolated_app):
    """The 'training' event must carry circuit_info so the frontend can
    render the gate diagram BEFORE the first epoch lands."""
    app, models, user = isolated_app
    from app.pipeline.events import EventBus
    from app.qml.trainer import run_training_job

    job_id = models.create_qml_job(user["id"], "moons", "vqc")
    bus = EventBus()
    events: list[dict] = []
    # Subscribe in a thread so we can drain while the trainer runs.
    import threading
    done = threading.Event()
    def _drain():
        for ev in bus.subscribe(job_id):
            events.append(ev)
            if ev["status"] in ("complete", "error"):
                done.set()
                return
    t = threading.Thread(target=_drain, daemon=True)
    t.start()

    run_training_job(
        job_id=job_id,
        dataset_id="moons",
        hyperparameters={"n_qubits": 2, "n_layers": 1, "n_epochs": 1, "seed": 0},
        event_bus=bus,
    )
    done.wait(timeout=30)

    training_event = next((e for e in events if e["status"] == "training"), None)
    assert training_event is not None, f"no training event in {[e['status'] for e in events]}"
    ci = training_event["circuit_info"]
    assert ci["backend_name"] == "PennyLane default.qubit"
    assert ci["is_real_hardware"] is False
    assert ci["n_qubits"] == 2


def test_train_route_rejects_unknown_dataset(isolated_app):
    app, *_ = isolated_app
    client = app.test_client()
    _login(client, "qml_trainer_user")
    r = client.post("/api/qml/train", json={"dataset_id": "nope", "model": "vqc"})
    assert r.status_code == 400
    assert r.get_json()["code"] == "UNKNOWN_DATASET"


def test_train_route_rejects_unsupported_model(isolated_app):
    app, *_ = isolated_app
    client = app.test_client()
    _login(client, "qml_trainer_user")
    r = client.post("/api/qml/train", json={"dataset_id": "moons", "model": "qcnn"})
    assert r.status_code == 400
    assert r.get_json()["code"] == "UNSUPPORTED_MODEL"


def test_train_route_requires_auth(isolated_app):
    app, *_ = isolated_app
    client = app.test_client()
    r = client.post("/api/qml/train", json={"dataset_id": "moons"})
    assert r.status_code == 401


def test_train_route_creates_job_row(isolated_app, monkeypatch):
    """POST /api/qml/train returns 201 with a queued job and persists the row.

    We monkeypatch ``launch_training_thread`` to a no-op so the test
    doesn't actually run PennyLane in the request handler.
    """
    app, models, user = isolated_app

    from app.qml import trainer as trainer_module
    called: list[dict] = []

    def _fake_launch(**kwargs):
        called.append(kwargs)

    monkeypatch.setattr(trainer_module, "launch_training_thread", _fake_launch)
    # The route does ``from app.qml.trainer import launch_training_thread``
    # inside the function body, so the monkeypatch above on the module is
    # what gets picked up at call time.

    client = app.test_client()
    _login(client, "qml_trainer_user")
    r = client.post("/api/qml/train", json={
        "dataset_id": "moons", "model": "vqc",
        "hyperparameters": {"n_epochs": 3},
    })
    assert r.status_code == 201, r.get_data(as_text=True)
    job = r.get_json()["job"]
    assert job["status"] == "queued"
    assert job["dataset_id"] == "moons"
    # The trainer was scheduled exactly once with our job_id.
    assert len(called) == 1
    assert called[0]["job_id"] == job["id"]
    assert called[0]["dataset_id"] == "moons"
