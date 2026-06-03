"""Tests for the QML-6 Origin-Quantum cloud inference path.

Strategy mirrors ``test_qml_qpu_ibmq``: mock pyqpanda3 so tests don't
touch the real Origin cloud. We exercise:

* ``submit_inference`` returns a clean envelope and picks the right
  sample index.
* Real-QPU backends are gated behind ``ENABLE_ORIGIN_REAL_HARDWARE``.
* ``try_materialize`` for DONE / QUEUED / ERROR states.
* The route-layer submit + refresh full circle, with a fake pyqpanda3
  module installed by the fixture.
"""
from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest


# ---- Fake pyqpanda3 --------------------------------------------------------


class _FakeQuery:
    """Stand-in for the object returned by ``QCloudJob.query()``."""
    def __init__(self, *, probs=None, status="DONE", err=""):
        self._probs = probs or {}
        self._status = status
        self._err = err

    def job_status(self):
        return self._status

    def error_message(self):
        return self._err

    def get_probs(self):
        return dict(self._probs)


class _FakeCloudJob:
    """Class-level registry of canned responses by job_id, so the
    materializer's lookup-by-id behaviour is testable."""
    _registry: dict[str, _FakeQuery] = {}

    def __init__(self, cloud_job_id):
        self._id = cloud_job_id

    def query(self):
        return _FakeCloudJob._registry.get(
            self._id,
            _FakeQuery(probs={}, status="QUEUED", err=""),
        )


class _FakeBackend:
    def __init__(self, name):
        self.name = name
        self.last_run = None

    def run(self, prog, shots, *opts):
        self.last_run = (prog, shots, opts)
        # Backend.run returns a "Job" with a job_id() method that maps
        # to a registered _FakeQuery (set up by the test).
        return _FakeBackendJob(f"origin-fake-{self.name}-001")


class _FakeBackendJob:
    def __init__(self, job_id):
        self._id = job_id

    def job_id(self):
        return self._id


class _FakeService:
    def __init__(self, *, api_key, url):
        self.api_key = api_key
        self.url = url

    def backend(self, name):
        return _FakeBackend(name)


class _FakeQCloudOptions:
    def __init__(self):
        self.flags = {}

    def set_optimization(self, v): self.flags["optimization"] = v
    def set_mapping(self, v): self.flags["mapping"] = v
    def set_amend(self, v): self.flags["amend"] = v
    def set_is_prob_counts(self, v): self.flags["prob_counts"] = v


class _FakeQProg:
    """Records the gates that get << into it so we can assert the
    AngleEmbedding + entangler structure was built right."""
    def __init__(self):
        self.gates = []

    def __lshift__(self, gate):
        self.gates.append(gate)
        return self


def _gate(name, *args):
    return ("gate", name, args)


@pytest.fixture(autouse=True)
def _install_fake_pyqpanda3(monkeypatch):
    """Install a fake ``pyqpanda3`` + submodules in ``sys.modules`` so
    every test in this file picks them up via lazy import. The fixture
    is autouse so individual tests can just configure ``_FakeCloudJob._registry``
    and the inference module's imports resolve to our fakes."""
    _FakeCloudJob._registry = {}

    fake_pyqpanda3 = types.ModuleType("pyqpanda3")
    fake_core = types.ModuleType("pyqpanda3.core")
    fake_qcloud = types.ModuleType("pyqpanda3.qcloud")

    # core: gate factories + QProg.
    fake_core.QProg = _FakeQProg
    fake_core.RX = lambda q, a: _gate("RX", q, a)
    fake_core.RY = lambda q, a: _gate("RY", q, a)
    fake_core.CNOT = lambda c, t: _gate("CNOT", c, t)
    fake_core.measure = lambda q, c: _gate("MEASURE", q, c)

    # qcloud: service + job + options.
    fake_qcloud.QCloudService = _FakeService
    fake_qcloud.QCloudJob = _FakeCloudJob
    fake_qcloud.QCloudOptions = _FakeQCloudOptions

    monkeypatch.setitem(sys.modules, "pyqpanda3", fake_pyqpanda3)
    monkeypatch.setitem(sys.modules, "pyqpanda3.core", fake_core)
    monkeypatch.setitem(sys.modules, "pyqpanda3.qcloud", fake_qcloud)

    # Ensure the qpu_originqc module reloads against the fakes.
    if "app.qml.qpu_originqc" in sys.modules:
        importlib.reload(sys.modules["app.qml.qpu_originqc"])
    yield
    # Clean up the registry between tests.
    _FakeCloudJob._registry = {}


# ---- submit_inference unit tests -------------------------------------------


def test_submit_inference_returns_envelope():
    from app.qml import qpu_originqc as mod
    X = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, -0.5]], dtype=np.float32)
    y = np.array([1, 0, 1], dtype=np.int64)
    env = mod.submit_inference(
        api_key="fake-origin-key",
        weights=[[0.1, 0.2], [0.3, 0.4]],
        bias=0.0,
        X_test=X,
        y_test=y,
        sample_index=1,
        shots=512,
    )
    assert env["cloud_job_id"].startswith("origin-fake-full_amplitude")
    assert env["backend_name"] == "full_amplitude"
    assert env["sample_index"] == 1
    assert env["sample_true_label"] == 0
    assert env["sample_x"] == [1.0, 0.0]
    assert env["is_real_hardware"] is False
    assert env["n_qubits"] == 2


def test_submit_inference_validates_inputs():
    from app.qml import qpu_originqc as mod
    X = np.zeros((2, 2), dtype=np.float32)
    y = np.zeros(2, dtype=np.int64)
    with pytest.raises(ValueError, match="api_key is required"):
        mod.submit_inference(api_key="", weights=[[0.1, 0.2]], bias=0.0,
                             X_test=X, y_test=y, shots=128)
    with pytest.raises(ValueError, match="shots"):
        mod.submit_inference(api_key="k", weights=[[0.1, 0.2]], bias=0.0,
                             X_test=X, y_test=y, shots=99_999)
    with pytest.raises(ValueError, match="sample_index out of range"):
        mod.submit_inference(api_key="k", weights=[[0.1, 0.2]], bias=0.0,
                             X_test=X, y_test=y, sample_index=5, shots=128)


def test_submit_inference_blocks_real_hw_without_flag(monkeypatch):
    from app.qml import qpu_originqc as mod
    monkeypatch.delenv("ENABLE_ORIGIN_REAL_HARDWARE", raising=False)
    X = np.zeros((1, 2), dtype=np.float32)
    y = np.zeros(1, dtype=np.int64)
    with pytest.raises(RuntimeError, match="ENABLE_ORIGIN_REAL_HARDWARE"):
        mod.submit_inference(
            api_key="k", weights=[[0.1, 0.2]], bias=0.0,
            X_test=X, y_test=y, sample_index=0, shots=128,
            backend_name="WK_C180",
        )


def test_submit_inference_passes_options_for_real_qpu(monkeypatch):
    """When the env flag is set, real-QPU runs go through with QCloudOptions."""
    from app.qml import qpu_originqc as mod
    monkeypatch.setenv("ENABLE_ORIGIN_REAL_HARDWARE", "1")
    X = np.array([[0.0, 0.0]], dtype=np.float32)
    y = np.array([0], dtype=np.int64)
    env = mod.submit_inference(
        api_key="k", weights=[[0.1, 0.2]], bias=0.0,
        X_test=X, y_test=y, sample_index=0, shots=2048,
        backend_name="WK_C180",
    )
    assert env["is_real_hardware"] is True
    assert env["backend_name"] == "WK_C180"


# ---- try_materialize unit tests --------------------------------------------


def test_try_materialize_done_correct_prediction():
    """High P(|0⟩) → ⟨Z⟩ > 0 → σ > 0.5 → predicts class 1.
    Set true_label = 1 → ``correct`` should be True."""
    from app.qml import qpu_originqc as mod
    _FakeCloudJob._registry["origin-job-A"] = _FakeQuery(
        probs={"0": 0.92, "1": 0.08}, status="FINISHED",
    )
    out = mod.try_materialize(
        api_key="k",
        cloud_job_id="origin-job-A",
        weights=[[0.1, 0.2]],
        bias=0.0,
        sample_index=3,
        sample_true_label=1,
    )
    assert out["terminal"] is True
    assert out["status"] == "complete"
    m = out["metrics"]
    assert m["sample_index"] == 3
    assert m["predicted_label"] == 1
    assert m["correct"] is True
    assert m["p_zero_on_q0"] == pytest.approx(0.92, abs=1e-3)


def test_try_materialize_done_incorrect_prediction():
    from app.qml import qpu_originqc as mod
    _FakeCloudJob._registry["origin-job-B"] = _FakeQuery(
        probs={"0": 0.10, "1": 0.90}, status="FINISHED",
    )
    out = mod.try_materialize(
        api_key="k",
        cloud_job_id="origin-job-B",
        weights=[[0.1, 0.2]],
        bias=0.0,
        sample_index=0,
        sample_true_label=1,  # mismatch: low P(|0⟩) → predicts class 0
    )
    assert out["terminal"] is True
    assert out["status"] == "complete"
    m = out["metrics"]
    assert m["predicted_label"] == 0
    assert m["correct"] is False


def test_try_materialize_queued():
    from app.qml import qpu_originqc as mod
    _FakeCloudJob._registry["origin-job-C"] = _FakeQuery(
        probs={}, status="QUEUED",
    )
    out = mod.try_materialize(
        api_key="k", cloud_job_id="origin-job-C",
        weights=[[0.1, 0.2]], bias=0.0,
        sample_index=0, sample_true_label=0,
    )
    assert out["terminal"] is False
    assert out["status"] == "queued"
    assert out["live_status"] == "QUEUED"


def test_try_materialize_error():
    from app.qml import qpu_originqc as mod
    _FakeCloudJob._registry["origin-job-D"] = _FakeQuery(
        probs={}, status="FAILED", err="synthetic cloud failure",
    )
    out = mod.try_materialize(
        api_key="k", cloud_job_id="origin-job-D",
        weights=[[0.1, 0.2]], bias=0.0,
        sample_index=0, sample_true_label=0,
    )
    assert out["terminal"] is True
    assert out["status"] == "error"
    assert "synthetic cloud failure" in out["error"]


# ---- Route-layer integration tests -----------------------------------------


@pytest.fixture
def isolated_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db_path = tmp_path / "qml_origin.db"
    monkeypatch.setenv("SECRET_KEY", "qml-origin-test")
    monkeypatch.setenv("KEY_ENCRYPTION_SECRET", "test" * 8)
    from app import config as config_module
    monkeypatch.setattr(config_module, "DATABASE_PATH", str(db_path))
    from app import models as models_module
    importlib.reload(models_module)
    monkeypatch.setattr(models_module, "DATABASE_PATH", str(db_path))
    models_module.init_db()

    user = models_module.create_user("origin_alice", "p4sswordpass")

    metrics = {
        "weights": [[0.1, 0.2], [0.3, 0.4]],
        "bias": 0.0,
        "scatter_points": [
            {"x": 0.0, "y": 0.0, "label": 0, "split": "train"},
            {"x": 0.1, "y": 0.0, "label": 1, "split": "test"},
            {"x": 0.9, "y": 1.0, "label": 0, "split": "test"},
        ],
    }
    parent_id = models_module.create_qml_job(user["id"], "moons", "vqc")
    models_module.update_qml_job(parent_id, status="complete",
                                 metrics=json.dumps(metrics))

    from app.crypto import encrypt_api_key
    from app.config import KEY_ENCRYPTION_SECRET
    encrypted = encrypt_api_key("fake-origin-key", KEY_ENCRYPTION_SECRET)
    models_module.put_api_key(user["id"], "originqc", encrypted)

    from app import create_app
    app = create_app({"TESTING": True})
    return app, models_module, user, parent_id


def _login(client, username, password="p4sswordpass"):
    r = client.post("/api/auth/login",
                    json={"username": username, "password": password})
    assert r.status_code == 200, r.get_data(as_text=True)


def test_route_origin_submit_requires_auth(isolated_app):
    app, *_ = isolated_app
    client = app.test_client()
    r = client.post("/api/qml/jobs/whatever/qpu/originqc", json={})
    assert r.status_code == 401


def test_route_origin_submit_rejects_byok_missing(isolated_app):
    app, models, user, parent_id = isolated_app
    models.delete_api_key(user["id"], "originqc")
    client = app.test_client()
    _login(client, "origin_alice")
    r = client.post(f"/api/qml/jobs/{parent_id}/qpu/originqc",
                    json={"shots": 256})
    assert r.status_code == 400
    assert r.get_json()["code"] == "BYOK_MISSING"


def test_route_origin_submit_creates_run(isolated_app):
    app, models, user, parent_id = isolated_app
    client = app.test_client()
    _login(client, "origin_alice")
    r = client.post(
        f"/api/qml/jobs/{parent_id}/qpu/originqc",
        json={"shots": 512, "backend_name": "full_amplitude", "sample_index": 0},
    )
    assert r.status_code == 201, r.get_data(as_text=True)
    run = r.get_json()["qpu_run"]
    assert run["provider"] == "originqc"
    assert run["status"] == "submitted"
    assert run["cloud_job_id"].startswith("origin-fake-full_amplitude")
    ctx = json.loads(run["submission_context"])
    assert ctx["sample_index"] == 0
    assert ctx["sample_true_label"] == 1  # from the seeded scatter


def test_route_refresh_materializes_origin_run(isolated_app):
    """Submit on Origin → poll → expect single-point metrics persisted."""
    app, models, user, parent_id = isolated_app
    client = app.test_client()
    _login(client, "origin_alice")

    submit = client.post(
        f"/api/qml/jobs/{parent_id}/qpu/originqc",
        json={"shots": 512, "sample_index": 0},
    ).get_json()["qpu_run"]

    # The submission above already registered cloud-job-id
    # "origin-fake-full_amplitude-001" in _FakeBackend.run; wire a
    # DONE result against that id with counts matching the seeded
    # test point's true label (sample_index=0 → label=1, so we want
    # the prediction to be 1 → high P(|0⟩)).
    _FakeCloudJob._registry[submit["cloud_job_id"]] = _FakeQuery(
        probs={"0": 0.88, "1": 0.12}, status="FINISHED",
    )

    r = client.post(f"/api/qml/qpu-runs/{submit['id']}/refresh")
    assert r.status_code == 200
    run = r.get_json()["qpu_run"]
    assert run["status"] == "complete"
    m = json.loads(run["metrics"])
    assert m["mode"] == "single_point"
    assert m["sample_index"] == 0
    assert m["predicted_label"] == 1
    assert m["correct"] is True
    assert m["is_real_hardware"] is False  # simulator backend
