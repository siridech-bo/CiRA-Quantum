"""QML-6 — Origin Quantum cloud VQC inference.

Mirror of :mod:`app.qml.qpu_ibmq` for Origin's Wukong-class hardware,
with two important asymmetries the student needs to understand:

* **One circuit per submission.** Origin's QCloud API doesn't batch
  PUBs the way IBM Runtime's ``SamplerV2`` does — each ``backend.run()``
  call submits a single program. Batching the whole test set would
  mean N queue waits, taking hours on Wukong. So this path evaluates
  *one representative test point per run*. The user can launch multiple
  runs to build up an accuracy distribution.
* **Cloud simulator vs real QPU.** Origin exposes hosted simulators
  (``full_amplitude``, ``partial_amplitude``, …) and real QPUs
  (``WK_C180`` Wukong). The simulator path is free / fast and useful
  for debugging; the real QPU is queued, billable, and gated behind
  the ``ENABLE_ORIGIN_REAL_HARDWARE`` env flag (same convention as the
  optimization-side ``QAOACloudSampler``).

Pipeline
--------

1. Pick test point ``i`` (caller-supplied or random).
2. Build a pyqpanda3 ``QProg`` with the trained weights baked in:
   ``RX(x_i)`` per qubit → per-layer ``RY(θ_{l,q}) + ring CNOT`` →
   ``measure q0``.
3. Submit to the chosen Origin backend, get a cloud job_id.
4. Poll via ``QCloudJob.query()`` until terminal.
5. Recover P(|0⟩) on qubit 0 → ⟨Z⟩ → ``σ(⟨Z⟩ + b)`` → predicted label.

Public surface
--------------

* ``submit_inference(...) -> {cloud_job_id, backend_name, sample_index, ...}``
* ``try_materialize(cloud_job_id, run_context) -> dict | None``
"""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np

_DEFAULT_BACKEND = "full_amplitude"       # cheap cloud simulator
_DEFAULT_CLOUD_URL = "http://pyqanda-admin.qpanda.cn"
_DEFAULT_SHOTS = 2048
_MAX_SHOTS = 8192
_REAL_HARDWARE_BACKENDS = frozenset({"WK_C180", "HanYuan_01"})


def _real_hardware_enabled() -> bool:
    """Read the ENABLE_ORIGIN_REAL_HARDWARE feature flag from env.
    Matches the optimization-side convention so a single env var gates
    both apps' real-QPU paths."""
    return os.environ.get("ENABLE_ORIGIN_REAL_HARDWARE", "").strip().lower() in (
        "1", "true", "yes",
    )


# ---- Public submit / materialize ------------------------------------------


def submit_inference(
    *,
    api_key: str,
    weights: list[list[float]],
    bias: float,
    X_test: np.ndarray,
    y_test: np.ndarray,
    sample_index: int | None = None,
    backend_name: str = _DEFAULT_BACKEND,
    shots: int = _DEFAULT_SHOTS,
    url: str = _DEFAULT_CLOUD_URL,
    seed: int | None = None,
) -> dict[str, Any]:
    """Submit one cloud job evaluating a single test point on Origin.

    Returns the cloud ``job_id`` + the resolved backend name + the
    sample index we evaluated, so the materializer can reproduce the
    prediction.
    """
    if not api_key or not api_key.strip():
        raise ValueError("api_key is required for Origin Quantum cloud inference")
    if shots < 1 or shots > _MAX_SHOTS:
        raise ValueError(f"shots must be in [1, {_MAX_SHOTS}]")
    if X_test.ndim != 2:
        raise ValueError("X_test must be 2D (n_samples, n_features)")
    if X_test.shape[0] == 0:
        raise ValueError("X_test is empty")

    if (
        backend_name in _REAL_HARDWARE_BACKENDS
        and not _real_hardware_enabled()
    ):
        raise RuntimeError(
            f"Backend {backend_name!r} is a real superconducting QPU; "
            "set ENABLE_ORIGIN_REAL_HARDWARE=1 to enable submissions."
        )

    n_qubits = X_test.shape[1]
    n_layers = len(weights)
    if n_layers == 0 or any(len(row) != n_qubits for row in weights):
        raise ValueError(f"weights shape mismatch: expected (n_layers, {n_qubits})")

    if sample_index is None:
        rng = random.Random(seed)
        sample_index = rng.randrange(X_test.shape[0])
    if sample_index < 0 or sample_index >= X_test.shape[0]:
        raise ValueError(f"sample_index out of range: {sample_index}")

    # Lazy import — pyqpanda3 is the heavy dep on this path, only loaded
    # when the user actually submits.
    from pyqpanda3.qcloud import QCloudOptions, QCloudService

    service = QCloudService(api_key=api_key, url=url)
    try:
        backend = service.backend(backend_name)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to acquire Origin backend {backend_name!r}: "
            f"{type(e).__name__}: {e}"
        ) from e

    prog = _build_inference_prog(
        x_features=X_test[sample_index].tolist(),
        weights=weights,
        n_qubits=n_qubits,
        n_layers=n_layers,
    )

    is_real_hw = backend_name in _REAL_HARDWARE_BACKENDS
    if is_real_hw:
        options = QCloudOptions()
        options.set_optimization(True)
        options.set_mapping(True)
        options.set_amend(True)
        options.set_is_prob_counts(True)
        job = backend.run(prog, int(shots), options)
    else:
        # Simulator path: no QCloudOptions (pyqpanda3 v0.3.5 enforces).
        job = backend.run(prog, int(shots))
    cloud_job_id = job.job_id()

    return {
        "cloud_job_id": cloud_job_id,
        "backend_name": backend_name,
        "shots": int(shots),
        "n_qubits": int(n_qubits),
        "n_layers": int(n_layers),
        "sample_index": int(sample_index),
        "sample_x": [float(v) for v in X_test[sample_index].tolist()],
        "sample_true_label": int(y_test[sample_index]),
        "is_real_hardware": bool(is_real_hw),
    }


def try_materialize(
    *,
    api_key: str,
    cloud_job_id: str,
    weights: list[list[float]],
    bias: float,
    sample_index: int,
    sample_true_label: int,
    url: str = _DEFAULT_CLOUD_URL,
) -> dict[str, Any] | None:
    """Poll Origin once. Returns ``None`` only when the optimization-side
    convention "no terminal state yet" is desired; here we always return
    a dict with ``terminal: bool`` so the caller can mirror the IBM
    materializer's flow.
    """
    from pyqpanda3.qcloud import QCloudJob, QCloudService

    try:
        QCloudService(api_key=api_key, url=url)
    except Exception as e:  # noqa: BLE001
        return {
            "terminal": True, "status": "error",
            "error": f"QCloudService init failed: {type(e).__name__}: {e}",
            "cloud_job_id": cloud_job_id,
        }
    try:
        job = QCloudJob(cloud_job_id)
        q = job.query()
    except Exception as e:  # noqa: BLE001
        return {
            "terminal": True, "status": "error",
            "error": f"job.query() failed: {type(e).__name__}: {e}",
            "cloud_job_id": cloud_job_id,
        }

    try:
        status = q.job_status()
        status_str = (
            str(status).rsplit(".", 1)[-1] if hasattr(status, "name") else str(status)
        ).upper()
    except Exception:  # pragma: no cover — defensive
        status_str = "UNKNOWN"

    try:
        err = q.error_message() or ""
    except Exception:  # pragma: no cover
        err = ""
    try:
        probs = q.get_probs()
    except Exception:  # pragma: no cover
        probs = {}

    if probs:
        metrics = _parse_inference_result(
            dict(probs), bias=bias,
            sample_index=sample_index,
            sample_true_label=sample_true_label,
        )
        return {
            "terminal": True,
            "status": "complete",
            "cloud_job_id": cloud_job_id,
            "metrics": metrics,
        }
    if err or status_str in {"FAILED", "ERROR", "CANCELLED", "CANCELED"}:
        return {
            "terminal": True,
            "status": "error",
            "error": err or f"cloud status {status_str}",
            "cloud_job_id": cloud_job_id,
        }
    return {
        "terminal": False,
        "status": "running" if status_str == "RUNNING" else "queued",
        "live_status": status_str,
        "cloud_job_id": cloud_job_id,
    }


# ---- Circuit construction --------------------------------------------------


def _build_inference_prog(
    *,
    x_features: list[float],
    weights: list[list[float]],
    n_qubits: int,
    n_layers: int,
):
    """Build the pyqpanda3 QProg matching the local PennyLane circuit.

    Mirrors :func:`app.qml.vqc._build_circuit` step-for-step:
    - AngleEmbedding (RX) for each feature
    - Per layer: RY(θ_{l,q}) on every qubit, then ring CNOT
    - Measure q0 in the Z basis
    """
    import pyqpanda3.core as pq3

    prog = pq3.QProg()
    # AngleEmbedding: RX(x_i) on qubit i.
    for q in range(n_qubits):
        prog << pq3.RX(q, float(x_features[q]))
    # BasicEntanglerLayers: RY + ring CNOT per layer.
    for layer in range(n_layers):
        for q in range(n_qubits):
            prog << pq3.RY(q, float(weights[layer][q]))
        if n_qubits >= 2:
            for q in range(n_qubits):
                prog << pq3.CNOT(q, (q + 1) % n_qubits)
    # Measure qubit 0 into classical bit 0.
    prog << pq3.measure(0, 0)
    return prog


# ---- Result parsing --------------------------------------------------------


def _parse_inference_result(
    probs: dict,
    *,
    bias: float,
    sample_index: int,
    sample_true_label: int,
) -> dict[str, Any]:
    """Recover ``⟨Z⟩`` on qubit 0 from Origin's probability dict, then
    map through ``σ(·+b)`` and the 0.5 threshold to a predicted label.

    Origin returns probabilities keyed by bitstring (or sometimes by an
    integer, depending on backend / version). Our circuit only measures
    qubit 0, so the keys are "0" / "1" — but for portability we also
    handle multi-bit strings by summing over states where the rightmost
    bit (qubit 0 in little-endian) is 0.
    """
    p0 = 0.0
    for key, p in probs.items():
        s = str(key)
        # If the key is multi-bit, qubit 0 is the rightmost char in
        # little-endian (the convention pyqpanda3 uses for QPU runs).
        if s and s[-1] == "0":
            p0 += float(p)
        elif s == "0":
            p0 += float(p)
    p0 = max(0.0, min(1.0, p0))
    expect_z = 2.0 * p0 - 1.0
    prob_class1 = 1.0 / (1.0 + np.exp(-(expect_z + bias)))
    pred = 1 if prob_class1 >= 0.5 else 0

    return {
        "sample_index": int(sample_index),
        "sample_true_label": int(sample_true_label),
        "predicted_label": int(pred),
        "prob_class1": round(float(prob_class1), 4),
        "p_zero_on_q0": round(float(p0), 4),
        "expect_z": round(float(expect_z), 4),
        "correct": bool(pred == sample_true_label),
    }
