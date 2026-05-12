"""Phase 9B — QAOACloudSampler tests.

Two layers:

1. **Pure unit tests** — construct, configure, validate. No cloud
   calls, no pyqpanda imports beyond what's strictly local.
2. **Mocked-cloud tests** — monkeypatch ``pyqpanda3.qcloud.QCloudService``
   so ``sample()`` runs end-to-end with a fake cloud round-trip. Exercises
   the local-training path and the bitstring-to-SampleSet conversion
   without ever touching the network.

Real-hardware integration tests live in
``test_qaoa_cloud_integration.py`` (gated behind
``ENABLE_ORIGIN_REAL_HARDWARE=1`` env var). Those cost real QPU
minutes and are not run by default in this suite.
"""

from __future__ import annotations

import dimod
import pytest

from app.optimization.qaoa_cloud_sampler import (
    QAOACloudSampler,
    _real_hardware_enabled,
)

# ----- Pure unit tests (no cloud) ----------------------------------------


def test_requires_api_key():
    with pytest.raises(ValueError, match="api_key is required"):
        QAOACloudSampler(api_key="")
    with pytest.raises(ValueError, match="api_key is required"):
        QAOACloudSampler(api_key="   ")


def test_constructor_validation():
    with pytest.raises(ValueError, match="layer must be"):
        QAOACloudSampler(api_key="dummy", layer=0)
    with pytest.raises(ValueError, match="shots must be"):
        QAOACloudSampler(api_key="dummy", shots=0)
    with pytest.raises(ValueError, match="max_qubits must be"):
        QAOACloudSampler(api_key="dummy", max_qubits=0)
    with pytest.raises(ValueError, match="max_submissions must be"):
        QAOACloudSampler(api_key="dummy", max_submissions=0)
    with pytest.raises(ValueError, match="top_k must be"):
        QAOACloudSampler(api_key="dummy", top_k=0)


def test_real_hardware_backend_requires_feature_flag(monkeypatch):
    """Without ENABLE_ORIGIN_REAL_HARDWARE=1, asking for WK_C180 must
    raise during construction — never accidentally submit to the QPU."""
    monkeypatch.delenv("ENABLE_ORIGIN_REAL_HARDWARE", raising=False)
    with pytest.raises(RuntimeError, match="real superconducting QPU"):
        QAOACloudSampler(api_key="dummy", backend_name="WK_C180")


def test_real_hardware_backend_with_flag_constructs(monkeypatch):
    """With the flag set, the QPU backend constructs cleanly (the
    actual cloud call is deferred until sample() is invoked)."""
    monkeypatch.setenv("ENABLE_ORIGIN_REAL_HARDWARE", "1")
    s = QAOACloudSampler(api_key="dummy", backend_name="WK_C180")
    assert s.properties["is_real_hardware"] is True


def test_simulator_backend_does_not_require_flag(monkeypatch):
    """Cloud simulators are not gated — full_amplitude / partial_amplitude
    / single_amplitude / PQPUMESH8 should always be constructible."""
    monkeypatch.delenv("ENABLE_ORIGIN_REAL_HARDWARE", raising=False)
    for backend in ("full_amplitude", "partial_amplitude", "single_amplitude"):
        s = QAOACloudSampler(api_key="dummy", backend_name=backend)
        assert s.properties["is_real_hardware"] is False


def test_stochastic_marker_set():
    """Forward-looking hook for Phase-2 replay tolerance — same as
    Phase 9A's local QAOASampler."""
    assert QAOACloudSampler._STOCHASTIC is True


def test_not_cqm_native():
    """Same QUBO-class dispatch as local QAOASampler — record_run
    routes through sample(bqm)."""
    assert not getattr(QAOACloudSampler, "_CQM_NATIVE", False)


# ----- Mocked-cloud tests -------------------------------------------------


class _FakeQCloudResult:
    """Stand-in for pyqpanda3.qcloud.QCloudResult. Returns a fixed
    probability distribution."""

    def __init__(self, probs: dict[str, float], job_id: str = "FAKE_JOB"):
        self._probs = dict(probs)
        self._job_id = job_id

    def get_probs(self) -> dict[str, float]:
        return dict(self._probs)

    def job_status(self):
        return "DONE"

    def job_id(self) -> str:
        return self._job_id


class _FakeQCloudJob:
    def __init__(self, probs: dict[str, float]):
        self._probs = probs
        self._id = "FAKE_JOB_12345"

    def job_id(self) -> str:
        return self._id

    def result(self):
        return _FakeQCloudResult(self._probs, job_id=self._id)


class _FakeBackend:
    def __init__(self, fixed_probs: dict[str, float]):
        self._probs = fixed_probs

    def run(self, _prog, _shots):
        return _FakeQCloudJob(self._probs)


class _FakeQCloudService:
    """Pretends to be a QCloudService — returns a backend that emits a
    fixed measurement distribution."""

    last_constructed_with: dict | None = None

    def __init__(self, *, api_key: str, url: str = ""):
        # Sanity: surface the api_key into a class var so tests can
        # confirm it was passed through. NEVER print or store the
        # actual production credential — tests always use "dummy".
        type(self).last_constructed_with = {"api_key_len": len(api_key), "url": url}
        self._fixed_probs = {
            "010": 0.55,
            "011": 0.20,
            "110": 0.15,
            "000": 0.10,
        }

    def backend(self, _name):
        return _FakeBackend(self._fixed_probs)


@pytest.fixture
def mocked_cloud(monkeypatch):
    """Replace pyqpanda3.qcloud.QCloudService inside the sampler's
    module-local import path with a fake that returns a fixed
    distribution. No network."""
    import pyqpanda3.qcloud as qcloud_mod

    monkeypatch.setattr(qcloud_mod, "QCloudService", _FakeQCloudService)
    yield


def test_sample_end_to_end_with_mocked_cloud(mocked_cloud):
    """Build a 3-var BQM, run sample(); the sampler should locally
    train QAOA params, build the circuit, and 'submit' it (to the
    fake), then return a SampleSet from the fake probabilities. The
    cloud_job_id and cloud_backend fields land on the SampleSet info."""
    linear = {"x0": 1.3, "x1": -1.0, "x2": -0.5}
    quadratic = {("x0", "x1"): -1.2, ("x1", "x2"): 0.9}
    bqm = dimod.BinaryQuadraticModel(linear, quadratic, 0.0, vartype=dimod.BINARY)

    s = QAOACloudSampler(api_key="dummy", backend_name="full_amplitude", layer=2, top_k=3)
    result = s.sample(bqm, seed=42)

    assert len(result) <= 3
    # The fake distribution puts most mass on "010" — energy(010) is
    # the user-facing minimum for this BQM, so the first row's energy
    # should be the smallest one we computed.
    energies = list(result.record.energy)
    assert energies[0] == min(energies)
    # Verify cloud metadata is recorded.
    assert result.info["cloud_backend"] == "full_amplitude"
    assert result.info["cloud_job_id"] == "FAKE_JOB_12345"
    assert result.info["cloud_is_real_hardware"] is False
    # The trained params should be present (4 floats for layer=2: 2 gammas + 2 betas).
    assert len(result.info["qaoa_trained_gammas"]) == 2
    assert len(result.info["qaoa_trained_betas"]) == 2


def test_qubit_budget_rejection(mocked_cloud):
    """A BQM exceeding the cloud sampler's max_qubits must raise
    ValueError with a clear pointer to a bigger backend."""
    big_n = 70
    linear = {f"x{i}": 1.0 for i in range(big_n)}
    bqm = dimod.BinaryQuadraticModel(linear, {}, 0.0, dimod.BINARY)

    s = QAOACloudSampler(api_key="dummy", backend_name="full_amplitude", max_qubits=64)
    with pytest.raises(ValueError, match="exceeds backend"):
        s.sample(bqm)


def test_submission_cap_enforces(mocked_cloud):
    """After max_submissions successful submissions, further sample()
    calls must raise rather than burning more cloud credits."""
    linear = {"x0": 1.0}
    bqm = dimod.BinaryQuadraticModel(linear, {}, 0.0, dimod.BINARY)

    s = QAOACloudSampler(api_key="dummy", backend_name="full_amplitude", max_submissions=2, top_k=2)
    s.sample(bqm)  # ok, count = 1
    s.sample(bqm)  # ok, count = 2
    with pytest.raises(RuntimeError, match="submission cap"):
        s.sample(bqm)


def test_empty_bqm_short_circuit():
    """Same degenerate-input handling as Phase 9A — no cloud call
    happens for an empty BQM."""
    bqm = dimod.BinaryQuadraticModel({}, {}, 7.5, dimod.BINARY)
    # No mocked_cloud fixture — empty BQM should never reach the cloud.
    s = QAOACloudSampler(api_key="dummy", backend_name="full_amplitude")
    result = s.sample(bqm)
    assert result.first.energy == pytest.approx(7.5, abs=1e-9)


# ----- Real-hardware enabled check ----------------------------------------


def test_real_hardware_enabled_helper(monkeypatch):
    """The ``_real_hardware_enabled`` helper reads the env flag and
    interprets it case-insensitively. Mirrors patterns in other
    feature-flag helpers in the codebase."""
    monkeypatch.delenv("ENABLE_ORIGIN_REAL_HARDWARE", raising=False)
    assert _real_hardware_enabled() is False

    for truthy in ("1", "true", "yes", "TRUE", "Yes"):
        monkeypatch.setenv("ENABLE_ORIGIN_REAL_HARDWARE", truthy)
        assert _real_hardware_enabled() is True

    for falsy in ("0", "false", "no", "off", ""):
        monkeypatch.setenv("ENABLE_ORIGIN_REAL_HARDWARE", falsy)
        assert _real_hardware_enabled() is False
