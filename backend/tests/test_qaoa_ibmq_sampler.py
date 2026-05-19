"""Tests for the QAOAIBMQSampler (Phase 11).

Strategy: mock ``qiskit_ibm_runtime.QiskitRuntimeService`` + ``SamplerV2``
+ the returned PrimitiveResult, so tests never touch the real IBM
Quantum cloud. The Qiskit-side classes we mock are well-documented and
stable, so a thin mock is enough to exercise our sampler's bitstring →
SampleSet conversion, qaoa_extras population, async submit / materialize
paths, and BYOK guards.

Tests:
* construction guards (api_key required, validated kwargs)
* sample() end-to-end on a tiny knapsack-style BQM, with a mocked
  cloud result whose counts include the optimum
* submit_async returns the expected envelope (cloud_job_id, trained
  angles, etc.)
* try_materialize handles DONE / ERROR / RUNNING states correctly
* Empty BQM short-circuits cleanly on both sync and async paths
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import dimod
import pytest


@pytest.fixture
def tiny_bqm() -> dimod.BinaryQuadraticModel:
    """A trivial 2-qubit BQM whose minimum at (0,1) has energy -6.
    Mimics a knapsack 'pick item B' shape."""
    linear = {"item_A": -5.0, "item_B": -6.0}
    quadratic = {("item_A", "item_B"): 2.0}
    return dimod.BinaryQuadraticModel(linear, quadratic, 0.0, dimod.BINARY)


@pytest.fixture
def empty_bqm() -> dimod.BinaryQuadraticModel:
    return dimod.BinaryQuadraticModel({}, {}, 0.0, dimod.BINARY)


# ---- Mocks ------------------------------------------------------------------


class _FakeBitArray:
    """Stand-in for qiskit's BitArray. We only need ``get_counts``."""

    def __init__(self, counts: dict[str, int]):
        self._counts = counts

    def get_counts(self) -> dict[str, int]:
        return dict(self._counts)


class _FakeData:
    def __init__(self, counts: dict[str, int]):
        self.meas = _FakeBitArray(counts)


class _FakePubResult:
    def __init__(self, counts: dict[str, int]):
        self.data = _FakeData(counts)


class _FakePrimitiveResult(list):
    """qiskit's PrimitiveResult is iterable / indexable; we return a
    list of PubResults. Subclassing list is the simplest mock."""

    def __init__(self, pub_results: list[_FakePubResult]):
        super().__init__(pub_results)


class _FakeJob:
    """Stand-in for qiskit_ibm_runtime's RuntimeJobV2."""

    def __init__(self, counts: dict[str, int], job_id: str = "ibm-fake-job-12345"):
        self._counts = counts
        self._id = job_id
        self._status = "DONE"

    def job_id(self) -> str:
        return self._id

    def status(self) -> str:
        return self._status

    def result(self) -> _FakePrimitiveResult:
        return _FakePrimitiveResult([_FakePubResult(self._counts)])

    def queue_position(self):
        return None

    def error_message(self) -> str:
        return ""


class _FakeBackend:
    def __init__(self, name: str = "ibm_fake_heron"):
        self.name = name
        self.target = MagicMock()


class _FakeSamplerV2:
    """Stand-in for qiskit_ibm_runtime.SamplerV2. ``run`` returns a job
    pre-seeded with measurement counts."""

    last_constructed_with = None

    def __init__(self, mode=None, options=None):
        type(self).last_constructed_with = {"mode": mode}
        # Inject counts via class-level injection point. Tests set
        # ``_FakeSamplerV2.fixed_counts`` before calling sample().
        self._counts = getattr(type(self), "fixed_counts", {"01": 80, "10": 15, "11": 3, "00": 2})

    def run(self, pubs, *, shots=None):
        return _FakeJob(self._counts)


class _FakeService:
    """Stand-in for QiskitRuntimeService."""

    last_constructed_with = None

    def __init__(self, *, channel: str, token: str, instance: str | None = None):
        type(self).last_constructed_with = {
            "channel": channel,
            "token_len": len(token),
            "instance": instance,
        }

    def backend(self, name: str) -> _FakeBackend:
        return _FakeBackend(name=name)

    def least_busy(self, *, operational=True, simulator=False, min_num_qubits=1):
        return _FakeBackend()

    def backends(self, *, operational=True, simulator=False):
        return [_FakeBackend("ibm_fake_heron"), _FakeBackend("ibm_fake_eagle")]

    def job(self, job_id: str) -> _FakeJob:
        return _FakeJob({"01": 100}, job_id=job_id)


class _FakePassManager:
    def run(self, qc):
        # No-op: return the same circuit. We're not testing transpiler
        # behaviour here.
        return qc


@pytest.fixture
def mocked_qiskit(monkeypatch):
    """Patch all qiskit-ibm-runtime entry points the sampler touches.
    Tests can override ``_FakeSamplerV2.fixed_counts`` to seed
    different measurement distributions."""
    import qiskit_ibm_runtime
    monkeypatch.setattr(qiskit_ibm_runtime, "QiskitRuntimeService", _FakeService)
    monkeypatch.setattr(qiskit_ibm_runtime, "SamplerV2", _FakeSamplerV2)
    # The sampler imports these at function-scope, so patch the module too.
    monkeypatch.setattr(
        "qiskit.transpiler.generate_preset_pass_manager",
        lambda **kwargs: _FakePassManager(),
    )


# ---- Tests ------------------------------------------------------------------


def test_init_requires_api_key():
    from app.optimization.qaoa_ibmq_sampler import QAOAIBMQSampler
    with pytest.raises(ValueError, match="api_key is required"):
        QAOAIBMQSampler(api_key="")


def test_init_validates_kwargs():
    from app.optimization.qaoa_ibmq_sampler import QAOAIBMQSampler
    with pytest.raises(ValueError, match="layer must be"):
        QAOAIBMQSampler(api_key="dummy", layer=0)
    with pytest.raises(ValueError, match="shots must be"):
        QAOAIBMQSampler(api_key="dummy", shots=0)


def test_empty_bqm_short_circuits_sample(empty_bqm, mocked_qiskit):
    from app.optimization.qaoa_ibmq_sampler import QAOAIBMQSampler
    sampler = QAOAIBMQSampler(api_key="dummy")
    ss = sampler.sample(empty_bqm)
    # An empty BQM yields a single zero-variable sample.
    assert len(ss) == 1
    assert ss.first.energy == 0.0


def test_empty_bqm_short_circuits_submit_async(empty_bqm, mocked_qiskit):
    from app.optimization.qaoa_ibmq_sampler import QAOAIBMQSampler
    sampler = QAOAIBMQSampler(api_key="dummy")
    result = sampler.submit_async(empty_bqm)
    assert result["empty"] is True
    assert "cloud_job_id" in result


def test_sample_end_to_end_with_mocked_cloud(tiny_bqm, mocked_qiskit):
    from app.optimization.qaoa_ibmq_sampler import QAOAIBMQSampler
    # Mocked cloud returns a 4-bitstring distribution. The sampler
    # evaluates each against the BQM and reports the lowest-energy
    # sample as ``ss.first``. For our unconstrained 2-var BQM
    # (linear=-5,-6; quadratic=+2) the four bit-pairs give energies:
    #   (0,0)=0  (0,1)=-6  (1,0)=-5  (1,1)=-5+-6+2=-9
    # Picking both items is the true minimum at -9.
    _FakeSamplerV2.fixed_counts = {"10": 80, "01": 15, "11": 3, "00": 2}
    sampler = QAOAIBMQSampler(api_key="dummy", layer=1, shots=100, top_k=4)
    ss = sampler.sample(tiny_bqm)
    assert ss.first.energy == pytest.approx(-9.0)
    # Qiskit info hooks populated
    assert "qaoa_top_bitstrings" in ss.info
    assert "qaoa_trained_gammas" in ss.info
    assert "cloud_job_id" in ss.info
    assert ss.info["cloud_is_real_hardware"] is True
    # Reset class-level state for other tests
    _FakeSamplerV2.fixed_counts = {"01": 80, "10": 15, "11": 3, "00": 2}


def test_submit_async_returns_envelope(tiny_bqm, mocked_qiskit):
    from app.optimization.qaoa_ibmq_sampler import QAOAIBMQSampler
    sampler = QAOAIBMQSampler(api_key="dummy", layer=1, shots=50)
    envelope = sampler.submit_async(tiny_bqm)
    assert envelope["cloud_job_id"] == "ibm-fake-job-12345"
    assert envelope["backend_name"] == "ibm_fake_heron"  # least_busy mock
    assert envelope["layer"] == 1
    assert envelope["shots"] == 50
    assert isinstance(envelope["trained_gammas"], list)
    assert isinstance(envelope["trained_betas"], list)
    assert len(envelope["trained_gammas"]) == 1  # 1 layer


def test_try_materialize_done(mocked_qiskit):
    """A job in DONE state returns a primitive_result + status='complete'."""
    from app.optimization.qaoa_ibmq_sampler import QAOAIBMQSampler
    sampler = QAOAIBMQSampler(api_key="dummy")
    r = sampler.try_materialize("ibm-fake-job-12345")
    assert r is not None
    assert r["terminal"] is True
    assert r["status"] == "complete"
    assert "primitive_result" in r


def test_try_materialize_in_progress(monkeypatch, mocked_qiskit):
    """A queued/running job returns terminal=False with the live status."""
    from app.optimization.qaoa_ibmq_sampler import QAOAIBMQSampler

    class _RunningJob(_FakeJob):
        def status(self):
            return "QUEUED"

    class _RunningService(_FakeService):
        def job(self, job_id):
            return _RunningJob({}, job_id=job_id)

    monkeypatch.setattr(
        "qiskit_ibm_runtime.QiskitRuntimeService", _RunningService,
    )
    sampler = QAOAIBMQSampler(api_key="dummy")
    r = sampler.try_materialize("ibm-fake-job-12345")
    assert r is not None
    assert r["terminal"] is False
    assert r["status"] == "queued"
    assert r["live_status"] == "QUEUED"


def test_try_materialize_cloud_error(monkeypatch, mocked_qiskit):
    """A job ending in CANCELLED / ERROR surfaces as status='error'."""
    from app.optimization.qaoa_ibmq_sampler import QAOAIBMQSampler

    class _ErroredJob(_FakeJob):
        def status(self):
            return "ERROR"

        def error_message(self):
            return "simulated cloud-side failure"

    class _ErroredService(_FakeService):
        def job(self, job_id):
            return _ErroredJob({}, job_id=job_id)

    monkeypatch.setattr(
        "qiskit_ibm_runtime.QiskitRuntimeService", _ErroredService,
    )
    sampler = QAOAIBMQSampler(api_key="dummy")
    r = sampler.try_materialize("ibm-fake-job-12345")
    assert r is not None
    assert r["terminal"] is True
    assert r["status"] == "error"
    assert "simulated cloud-side failure" in r["error"]
