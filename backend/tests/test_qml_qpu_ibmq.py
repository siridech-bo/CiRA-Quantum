"""Tests for the QML-5 IBM-Quantum cloud inference path.

Same strategy as ``test_qaoa_ibmq_sampler``: mock the qiskit-ibm-runtime
classes so tests don't need real cloud access. We exercise:

* ``submit_inference`` end-to-end with a mocked service + sampler
* ``try_materialize`` for DONE / ERROR / QUEUED states
* ``_parse_inference_result`` arithmetic on a fixed counts dict
* The route surface: gating on auth, BYOK, parent-job state, and the
  refresh polling endpoint flipping a row from ``submitted`` → ``complete``.
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---- Fakes -----------------------------------------------------------------


class _FakeCounts:
    """Stand-in for a SamplerV2 PubResult's classical register record.

    Exposes ``get_counts()`` returning the per-pub shot distribution.
    """
    def __init__(self, counts):
        self._counts = counts

    def get_counts(self):
        return dict(self._counts)


class _FakeData:
    """``pub.data.c`` is the common location for a single classical
    register in modern qiskit. We give it the ``get_counts`` interface."""
    def __init__(self, counts):
        self.c = _FakeCounts(counts)


class _FakePub:
    def __init__(self, counts):
        self.data = _FakeData(counts)


class _FakeJob:
    def __init__(self, pubs, job_id="ibm-fake-qml-job-001", status="DONE"):
        self._pubs = pubs
        self._id = job_id
        self._status = status

    def job_id(self):
        return self._id

    def status(self):
        return self._status

    def queue_position(self):
        return 7 if self._status == "QUEUED" else None

    def error_message(self):
        return "synthetic failure" if self._status in ("ERROR", "FAILED") else None

    def result(self):
        return self._pubs  # iterable of pubs is enough


class _FakeSamplerV2:
    """Records the call signature + returns a configured _FakeJob."""
    def __init__(self, mode):
        self.mode = mode
        self.last_call = None

    def run(self, circuits, *, shots):
        # Build a synthetic counts dict per circuit. For determinism we
        # alternate: even idx → mostly |0⟩, odd idx → mostly |1⟩. That
        # gives us a non-trivial predicted-label vector to assert against.
        pubs = []
        for i, _ in enumerate(circuits):
            if i % 2 == 0:
                pubs.append(_FakePub({"0": int(shots * 0.85), "1": int(shots * 0.15)}))
            else:
                pubs.append(_FakePub({"0": int(shots * 0.15), "1": int(shots * 0.85)}))
        self.last_call = {"shots": shots, "n_circuits": len(circuits)}
        return _FakeJob(pubs)


class _FakeBackend:
    name = "ibm_fake_brisbane"

    def configuration(self):  # pragma: no cover — defensive
        cfg = MagicMock()
        cfg.simulator = False
        return cfg


class _FakeService:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def least_busy(self, **_kwargs):
        return _FakeBackend()

    def backend(self, name):
        b = _FakeBackend()
        b.name = name
        return b

    def job(self, cloud_job_id):
        # The default test fixture below patches this to return whatever
        # state the test wants.
        return _FakeJob([], job_id=cloud_job_id, status="DONE")


# Lightweight stand-in for the real transpiler pipeline. We don't have a
# real backend in tests, so we wire ``generate_preset_pass_manager`` to
# return an object whose ``.run(circuit)`` is a passthrough. Mirrors
# what the production code does on a Heron backend without actually
# needing one.
class _PassThroughPM:
    def run(self, circuit):
        return circuit


def _fake_generate_preset_pass_manager(**_kwargs):
    return _PassThroughPM()


# ---- submit_inference unit tests -------------------------------------------


pytest.importorskip("qiskit")
pytest.importorskip("qiskit_ibm_runtime")


def _patch_runtime(monkeypatch, sampler_class=_FakeSamplerV2,
                   service_class=_FakeService):
    """Wire the fake Runtime classes + transpile into the module
    under test. We patch at the *use site* so other tests that import
    qiskit normally still work."""
    from app.qml import qpu_ibmq as mod
    monkeypatch.setattr(
        "qiskit_ibm_runtime.QiskitRuntimeService", service_class,
        raising=True,
    )
    monkeypatch.setattr(
        "qiskit_ibm_runtime.SamplerV2", sampler_class,
        raising=True,
    )
    monkeypatch.setattr(
        "qiskit.transpiler.generate_preset_pass_manager",
        _fake_generate_preset_pass_manager,
        raising=True,
    )
    return mod


def test_submit_inference_returns_envelope(monkeypatch):
    mod = _patch_runtime(monkeypatch)
    weights = [[0.1, 0.2], [0.3, 0.4]]
    X_test = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, -0.5]], dtype=np.float32)

    env = mod.submit_inference(
        api_key="fake-key",
        weights=weights,
        bias=0.0,
        X_test=X_test,
        shots=512,
    )
    assert env["cloud_job_id"] == "ibm-fake-qml-job-001"
    assert env["backend_name"] == "ibm_fake_brisbane"
    assert env["shots"] == 512
    assert env["n_test_points"] == 3
    assert env["n_qubits"] == 2
    assert env["n_layers"] == 2


def test_submit_inference_validates_inputs(monkeypatch):
    mod = _patch_runtime(monkeypatch)
    with pytest.raises(ValueError, match="api_key is required"):
        mod.submit_inference(
            api_key="", weights=[[0.1, 0.2]], bias=0.0,
            X_test=np.zeros((1, 2)), shots=128,
        )
    with pytest.raises(ValueError, match="shots must be in"):
        mod.submit_inference(
            api_key="k", weights=[[0.1, 0.2]], bias=0.0,
            X_test=np.zeros((1, 2)), shots=99_999,
        )
    with pytest.raises(ValueError, match="X_test"):
        mod.submit_inference(
            api_key="k", weights=[[0.1, 0.2]], bias=0.0,
            X_test=np.zeros(1), shots=128,
        )
    with pytest.raises(ValueError, match="weights shape mismatch"):
        mod.submit_inference(
            api_key="k", weights=[[0.1]], bias=0.0,
            X_test=np.zeros((1, 2)), shots=128,
        )


def test_submit_inference_caps_test_points(monkeypatch):
    mod = _patch_runtime(monkeypatch)
    with pytest.raises(ValueError, match="cap is"):
        mod.submit_inference(
            api_key="k", weights=[[0.1, 0.2]], bias=0.0,
            X_test=np.zeros((100, 2), dtype=np.float32),
            shots=128,
        )


# ---- try_materialize unit tests --------------------------------------------


def _service_returning(job: _FakeJob):
    class _S(_FakeService):
        def job(self, cloud_job_id):  # noqa: ARG002
            return job
    return _S


def test_try_materialize_done(monkeypatch):
    # Convention: ⟨Z⟩ = 2·P(|0⟩) − 1, then σ(⟨Z⟩ + b). High P(|0⟩) →
    # σ > 0.5 → predicts class 1. (Same convention as the local trainer
    # in app.qml.vqc — the QPU inference must mirror it for honest
    # comparison.)
    pubs = [
        _FakePub({"0": 850, "1": 150}),  # mostly |0⟩ → predicts class 1
        _FakePub({"0": 150, "1": 850}),  # mostly |1⟩ → predicts class 0
    ]
    mod = _patch_runtime(
        monkeypatch,
        service_class=_service_returning(_FakeJob(pubs, status="DONE")),
    )
    X_test = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    y_test = np.array([1, 0], dtype=np.int64)

    out = mod.try_materialize(
        api_key="k", cloud_job_id="ibm-fake-job",
        weights=[[0.1, 0.2]], bias=0.0,
        X_test=X_test, y_test=y_test, shots=1000,
    )
    assert out["terminal"] is True
    assert out["status"] == "complete"
    m = out["metrics"]
    # Both predictions match labels → 100% accuracy.
    assert m["test_accuracy"] == 1.0
    # cm = [[TN, FP], [FN, TP]]: y=0 + pred=0 → TN, y=1 + pred=1 → TP.
    assert m["confusion_matrix"] == [[1, 0], [0, 1]]
    assert len(m["probabilities"]) == 2


def test_try_materialize_queued(monkeypatch):
    mod = _patch_runtime(
        monkeypatch,
        service_class=_service_returning(_FakeJob([], status="QUEUED")),
    )
    out = mod.try_materialize(
        api_key="k", cloud_job_id="ibm-fake-job",
        weights=[[0.1, 0.2]], bias=0.0,
        X_test=np.zeros((1, 2), dtype=np.float32),
        y_test=np.zeros(1, dtype=np.int64),
        shots=128,
    )
    assert out["terminal"] is False
    assert out["status"] == "queued"
    assert out["live_status"] == "QUEUED"
    assert out["queue_position"] == 7


def test_try_materialize_error(monkeypatch):
    mod = _patch_runtime(
        monkeypatch,
        service_class=_service_returning(_FakeJob([], status="ERROR")),
    )
    out = mod.try_materialize(
        api_key="k", cloud_job_id="ibm-fake-job",
        weights=[[0.1, 0.2]], bias=0.0,
        X_test=np.zeros((1, 2), dtype=np.float32),
        y_test=np.zeros(1, dtype=np.int64),
        shots=128,
    )
    assert out["terminal"] is True
    assert out["status"] == "error"
    assert "ERROR" in out["error"]


# ---- Route-layer integration tests -----------------------------------------


@pytest.fixture
def isolated_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db_path = tmp_path / "qml_qpu.db"
    monkeypatch.setenv("SECRET_KEY", "qml-qpu-test")
    monkeypatch.setenv("KEY_ENCRYPTION_SECRET", "test" * 8)  # 32 chars
    from app import config as config_module
    monkeypatch.setattr(config_module, "DATABASE_PATH", str(db_path))
    from app import models as models_module
    importlib.reload(models_module)
    monkeypatch.setattr(models_module, "DATABASE_PATH", str(db_path))
    models_module.init_db()

    user = models_module.create_user("qpu_alice", "p4sswordpass")

    # Seed a "complete" parent VQC job with the metrics shape the
    # route needs (scatter_points + weights + bias).
    # Test labels chosen so the synthetic pubs in
    # test_route_refresh_materializes_complete_run produce 100% accuracy
    # against the same convention used by the local VQC trainer
    # (high P(|0⟩) → predicts class 1).
    metrics = {
        "weights": [[0.1, 0.2], [0.3, 0.4]],
        "bias": 0.0,
        "scatter_points": [
            {"x": 0.0, "y": 0.0, "label": 0, "split": "train"},
            {"x": 1.0, "y": 1.0, "label": 1, "split": "train"},
            {"x": 0.1, "y": 0.0, "label": 1, "split": "test"},
            {"x": 0.9, "y": 1.0, "label": 0, "split": "test"},
        ],
    }
    parent_id = models_module.create_qml_job(user["id"], "moons", "vqc")
    models_module.update_qml_job(parent_id, status="complete",
                                 metrics=json.dumps(metrics))

    # Encrypt a fake IBM Quantum key and store it under the user.
    from app.crypto import encrypt_api_key
    from app.config import KEY_ENCRYPTION_SECRET
    encrypted = encrypt_api_key("fake-ibm-key", KEY_ENCRYPTION_SECRET)
    models_module.put_api_key(user["id"], "ibm_quantum", encrypted)

    from app import create_app
    app = create_app({"TESTING": True})
    return app, models_module, user, parent_id


def _login(client, username, password="p4sswordpass"):
    r = client.post("/api/auth/login",
                    json={"username": username, "password": password})
    assert r.status_code == 200, r.get_data(as_text=True)


def test_route_submit_requires_auth(isolated_app):
    app, *_ = isolated_app
    client = app.test_client()
    r = client.post("/api/qml/jobs/whatever/qpu/ibmq", json={})
    assert r.status_code == 401


def test_route_submit_404_on_unknown_parent(isolated_app):
    app, *_ = isolated_app
    client = app.test_client()
    _login(client, "qpu_alice")
    r = client.post("/api/qml/jobs/no-such-parent/qpu/ibmq", json={})
    assert r.status_code == 404


def test_route_submit_rejects_byok_missing(isolated_app, monkeypatch):
    """Drop the BYOK key from the seeded user, then expect a 400."""
    app, models, user, parent_id = isolated_app
    # Stub qiskit_ibm_runtime so the BYOK check is the only gate that
    # can fail in this test.
    monkeypatch.setattr(
        "qiskit_ibm_runtime.QiskitRuntimeService", _FakeService, raising=True,
    )
    monkeypatch.setattr(
        "qiskit_ibm_runtime.SamplerV2", _FakeSamplerV2, raising=True,
    )
    models.delete_api_key(user["id"], "ibm_quantum")

    client = app.test_client()
    _login(client, "qpu_alice")
    r = client.post(f"/api/qml/jobs/{parent_id}/qpu/ibmq", json={"shots": 128})
    assert r.status_code == 400
    assert r.get_json()["code"] == "BYOK_MISSING"


def test_route_submit_creates_qpu_run(isolated_app, monkeypatch):
    """Full POST → expect a row in qml_qpu_runs with cloud_job_id."""
    app, models, user, parent_id = isolated_app
    monkeypatch.setattr(
        "qiskit_ibm_runtime.QiskitRuntimeService", _FakeService, raising=True,
    )
    monkeypatch.setattr(
        "qiskit_ibm_runtime.SamplerV2", _FakeSamplerV2, raising=True,
    )
    monkeypatch.setattr(
        "qiskit.transpiler.generate_preset_pass_manager",
        _fake_generate_preset_pass_manager,
        raising=True,
    )

    client = app.test_client()
    _login(client, "qpu_alice")
    r = client.post(
        f"/api/qml/jobs/{parent_id}/qpu/ibmq",
        json={"shots": 256, "backend_name": "ibm_fake_brisbane"},
    )
    assert r.status_code == 201, r.get_data(as_text=True)
    run = r.get_json()["qpu_run"]
    assert run["status"] == "submitted"
    assert run["cloud_job_id"] == "ibm-fake-qml-job-001"
    assert run["backend_name"] == "ibm_fake_brisbane"
    assert run["shots"] == 256


def test_route_refresh_materializes_complete_run(isolated_app, monkeypatch):
    """Submit → refresh → expect status=complete + metrics populated."""
    app, models, user, parent_id = isolated_app
    monkeypatch.setattr(
        "qiskit_ibm_runtime.QiskitRuntimeService", _FakeService, raising=True,
    )
    monkeypatch.setattr(
        "qiskit_ibm_runtime.SamplerV2", _FakeSamplerV2, raising=True,
    )
    monkeypatch.setattr(
        "qiskit.transpiler.generate_preset_pass_manager",
        _fake_generate_preset_pass_manager,
        raising=True,
    )

    client = app.test_client()
    _login(client, "qpu_alice")
    submit = client.post(
        f"/api/qml/jobs/{parent_id}/qpu/ibmq",
        json={"shots": 1024},
    ).get_json()["qpu_run"]

    # Now stub the service so try_materialize returns DONE with synthetic
    # counts that yield 100% on this 2-point test split. Convention:
    # high P(|0⟩) → ⟨Z⟩ > 0 → σ > 0.5 → predicts class 1.
    pubs = [
        _FakePub({"0": 900, "1": 100}),  # mostly |0⟩ → preds class 1 (label = 1)
        _FakePub({"0": 100, "1": 900}),  # mostly |1⟩ → preds class 0 (label = 0)
    ]
    monkeypatch.setattr(
        "qiskit_ibm_runtime.QiskitRuntimeService",
        _service_returning(_FakeJob(pubs, status="DONE")),
        raising=True,
    )

    r = client.post(f"/api/qml/qpu-runs/{submit['id']}/refresh")
    assert r.status_code == 200
    run = r.get_json()["qpu_run"]
    assert run["status"] == "complete"
    m = json.loads(run["metrics"])
    assert m["test_accuracy"] == 1.0
    assert m["is_real_hardware"] is True
    assert m["backend_name"] == "ibm_fake_brisbane"


def test_route_lists_qpu_runs_for_parent(isolated_app, monkeypatch):
    app, models, user, parent_id = isolated_app
    monkeypatch.setattr(
        "qiskit_ibm_runtime.QiskitRuntimeService", _FakeService, raising=True,
    )
    monkeypatch.setattr(
        "qiskit_ibm_runtime.SamplerV2", _FakeSamplerV2, raising=True,
    )
    monkeypatch.setattr(
        "qiskit.transpiler.generate_preset_pass_manager",
        _fake_generate_preset_pass_manager,
        raising=True,
    )

    client = app.test_client()
    _login(client, "qpu_alice")
    for _ in range(3):
        client.post(f"/api/qml/jobs/{parent_id}/qpu/ibmq", json={"shots": 128})

    r = client.get(f"/api/qml/jobs/{parent_id}/qpu")
    assert r.status_code == 200
    payload = r.get_json()
    assert payload["total"] == 3
    for run in payload["qpu_runs"]:
        assert run["qml_job_id"] == parent_id
