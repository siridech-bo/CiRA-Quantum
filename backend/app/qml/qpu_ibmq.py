"""QML-5 — IBM Quantum cloud VQC inference.

We deliberately **do not train on the QPU**. Parameter-shift on a small
VQC at typical shot counts costs tens of thousands of cloud submissions
per epoch — useless for an educational portal. Instead, the student
trains locally (the QML-2 path on a statevector simulator), and this
module re-evaluates the *trained* circuit on a real superconducting
QPU so they can see what shot noise + device error actually look like.

Pipeline
--------

1. Reconstruct the trained VQC for each test point: ``AngleEmbedding(x_i)``
   followed by ``BasicEntanglerLayers(weights)``, then a Z-basis
   measurement on qubit 0.
2. Transpile once for the chosen backend at ``optimization_level=1``.
3. Submit all test-point circuits as a single batch PUB to
   ``SamplerV2.run(...)`` — one cloud submission per QPU run.
4. Poll the cloud job via ``service.job(<id>).status()`` until terminal.
5. Parse the per-circuit shot counts → probability of measuring
   ``|0⟩`` on qubit 0 → ``⟨Z⟩`` → ``σ(⟨Z⟩+b)`` → predicted label.

Safety
------

* **BYOK** — IBM Quantum API token supplied per-call. Never logged.
* **Soft caps**: max 64 test points per submission (the simulator
  already covers larger sets exactly), max 4096 shots per circuit.
* **Read-only** with respect to the trained parameters; this module
  never mutates ``qml_jobs`` rows.

Public surface
--------------

* ``submit_inference(...) -> {cloud_job_id, backend_name, ...}``
* ``try_materialize(cloud_job_id, run_context) -> dict | None``
"""

from __future__ import annotations

from typing import Any

import numpy as np

_DEFAULT_SHOTS = 1024
_DEFAULT_OPT_LEVEL = 1
_MAX_TEST_POINTS = 64
_MAX_SHOTS = 4096
_REAL_HARDWARE_HINT = "ibm-quantum:heron-or-eagle"


# ---- Public submit / materialize ------------------------------------------


def submit_inference(
    *,
    api_key: str,
    weights: list[list[float]],
    bias: float,
    X_test: np.ndarray,
    backend_name: str | None = None,
    shots: int = _DEFAULT_SHOTS,
    channel: str = "ibm_quantum_platform",
    instance: str | None = None,
    optimization_level: int = _DEFAULT_OPT_LEVEL,
) -> dict[str, Any]:
    """Submit one cloud job that batches every test-point circuit.

    Returns the cloud ``job_id`` + the resolved backend name +
    everything the materializer needs to rebuild the predicted labels
    once the job finishes.
    """
    if not api_key or not api_key.strip():
        raise ValueError("api_key is required for IBM Quantum cloud inference")
    if shots < 1 or shots > _MAX_SHOTS:
        raise ValueError(f"shots must be in [1, {_MAX_SHOTS}]")
    if X_test.ndim != 2:
        raise ValueError("X_test must be 2D (n_samples, n_features)")
    if X_test.shape[0] > _MAX_TEST_POINTS:
        raise ValueError(
            f"X_test has {X_test.shape[0]} points; cap is "
            f"{_MAX_TEST_POINTS} per QPU submission (subsample first)"
        )
    n_qubits = X_test.shape[1]
    n_layers = len(weights)
    if n_layers == 0 or any(len(row) != n_qubits for row in weights):
        raise ValueError(
            f"weights shape mismatch: expected (n_layers, {n_qubits})"
        )

    # Lazy imports — keeps the route layer fast for clients that never
    # touch the IBM path.
    from qiskit.circuit import Parameter
    from qiskit.transpiler import generate_preset_pass_manager
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

    service = QiskitRuntimeService(
        channel=channel, token=api_key,
        **({"instance": instance} if instance else {}),
    )
    backend = (
        service.backend(backend_name)
        if backend_name
        else service.least_busy(
            operational=True, simulator=False, min_num_qubits=n_qubits,
        )
    )
    resolved_backend_name = backend.name

    # Build one parametric "template" circuit, transpile once for the
    # backend, then bind the per-point input angles. The trainable
    # rotation angles + bias are baked in directly because they're
    # already known at submission time.
    x_params = [Parameter(f"x{i}") for i in range(n_qubits)]
    circuit = _build_inference_circuit(x_params, weights, n_qubits, n_layers)

    # Newer IBM Heron backends (ibm_kingston, ibm_fez, …) advertise a
    # preferred translation plugin ``ibm_dynamic_circuits`` that ships
    # in a separate package. Without it installed, ``transpile(..., backend=...)``
    # raises ``TranspilerError: Invalid plugin name ibm_dynamic_circuits``.
    # Same workaround as ``qaoa_ibmq_sampler``: override the translation
    # stage to the built-in ``translator`` method which has no plugin
    # dependency.
    pm = generate_preset_pass_manager(
        backend=backend,
        optimization_level=optimization_level,
        translation_method="translator",
    )
    transpiled = pm.run(circuit)

    bound = [
        transpiled.assign_parameters(dict(zip(x_params, X_test[i].tolist())))
        for i in range(X_test.shape[0])
    ]

    sampler = SamplerV2(mode=backend)
    job = sampler.run(bound, shots=shots)
    cloud_job_id = job.job_id() if callable(getattr(job, "job_id", None)) else None

    return {
        "cloud_job_id": cloud_job_id,
        "backend_name": resolved_backend_name,
        "shots": int(shots),
        "n_test_points": int(X_test.shape[0]),
        "n_qubits": int(n_qubits),
        "n_layers": int(n_layers),
        "is_real_hardware": not bool(getattr(backend, "configuration", lambda: None)().simulator
                                     if hasattr(backend, "configuration") else False),
    }


def try_materialize(
    *,
    api_key: str,
    cloud_job_id: str,
    weights: list[list[float]],
    bias: float,
    X_test: np.ndarray,
    y_test: np.ndarray,
    shots: int,
    channel: str = "ibm_quantum_platform",
    instance: str | None = None,
) -> dict[str, Any] | None:
    """Poll once. Returns ``None`` if the IBM job isn't terminal yet, or
    a structured result dict if it is (success or cloud-side error).

    The materializer needs the same ``weights`` + ``X_test`` + ``y_test``
    the submitter saw, so the route stashes them in the
    ``qml_qpu_runs`` row's submission_context — the poller passes them
    back here when polling.
    """
    from qiskit_ibm_runtime import QiskitRuntimeService

    service = QiskitRuntimeService(
        channel=channel, token=api_key,
        **({"instance": instance} if instance else {}),
    )
    try:
        job = service.job(cloud_job_id)
    except Exception as e:  # noqa: BLE001
        return {"terminal": True, "status": "error",
                "error": f"job lookup failed: {type(e).__name__}: {e}"}

    try:
        status = job.status()
    except Exception as e:  # noqa: BLE001
        return {"terminal": True, "status": "error",
                "error": f"status() failed: {type(e).__name__}: {e}"}

    status_str = (status.name if hasattr(status, "name") else str(status)).upper()

    if status_str in {"DONE", "COMPLETED", "FINISHED"}:
        try:
            primitive_result = job.result()
        except Exception as e:  # noqa: BLE001
            return {"terminal": True, "status": "error",
                    "error": f"result() failed: {type(e).__name__}: {e}",
                    "cloud_job_id": cloud_job_id}
        metrics = _parse_inference_result(
            primitive_result, X_test, y_test, bias=bias,
        )
        return {
            "terminal": True,
            "status": "complete",
            "cloud_job_id": cloud_job_id,
            "metrics": metrics,
        }

    if status_str in {"ERROR", "FAILED", "CANCELLED", "CANCELED"}:
        err = ""
        try:
            err = str(job.error_message() or "")
        except Exception:  # pragma: no cover
            pass
        return {
            "terminal": True,
            "status": "error",
            "error": f"IBM cloud reported {status_str}: {err}",
            "cloud_job_id": cloud_job_id,
        }

    # Still queued / running. Return queue info for the live UI.
    queue_pos = None
    try:
        queue_pos = job.queue_position()
    except Exception:  # pragma: no cover — older runtime versions
        pass
    return {
        "terminal": False,
        "status": "running" if status_str == "RUNNING" else "queued",
        "live_status": status_str,
        "queue_position": queue_pos,
        "cloud_job_id": cloud_job_id,
    }


# ---- Circuit construction --------------------------------------------------


def _build_inference_circuit(x_params, weights: list[list[float]], n_qubits: int, n_layers: int):
    """Build the parametric VQC matching the local PennyLane circuit.

    Mirrors :func:`app.qml.vqc._build_circuit` step-for-step:
    - AngleEmbedding (RX) for each feature
    - For each layer: RY(θ_{l,q}) on every qubit, then ring CNOT.
    """
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(n_qubits, 1)  # one classical bit (qubit 0 readout)
    # AngleEmbedding: RX(x_i) on qubit i.
    for q in range(n_qubits):
        qc.rx(x_params[q], q)
    # BasicEntanglerLayers: RY + ring CNOT per layer.
    for layer in range(n_layers):
        for q in range(n_qubits):
            qc.ry(float(weights[layer][q]), q)
        if n_qubits >= 2:
            for q in range(n_qubits):
                qc.cx(q, (q + 1) % n_qubits)
    # Measure qubit 0 in the Z basis. We only need P(|0⟩) on q0 to get ⟨Z⟩.
    qc.measure(0, 0)
    return qc


# ---- Result parsing --------------------------------------------------------


def _parse_inference_result(
    primitive_result,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    bias: float,
) -> dict[str, Any]:
    """Walk every PubResult in the batch, recover per-point ``⟨Z⟩``,
    apply ``σ(·+b)``, threshold at 0.5, compute accuracy."""
    n = X_test.shape[0]
    probs_class1: list[float] = []
    for i in range(n):
        pub = primitive_result[i]
        counts = _counts_from_pub(pub)
        total = sum(counts.values()) or 1
        # qubit 0 was measured into classical bit 0. The single-bit
        # outcome is "0" or "1" (Qiskit prints little-endian for
        # multi-bit registers, but we only have one bit, so this is
        # unambiguous).
        p0 = counts.get("0", 0) / total
        p1 = 1.0 - p0
        # ⟨Z⟩ = P(|0⟩) − P(|1⟩) = 2·p0 − 1.
        expect_z = 2.0 * p0 - 1.0
        probs_class1.append(1.0 / (1.0 + np.exp(-(expect_z + bias))))

    preds = np.array([1 if p >= 0.5 else 0 for p in probs_class1], dtype=np.int64)
    accuracy = float(np.mean(preds == y_test))
    cm = _confusion(preds, y_test)
    return {
        "test_accuracy": accuracy,
        "confusion_matrix": cm,
        "probabilities": [round(float(p), 4) for p in probs_class1],
        "predictions": preds.tolist(),
    }


def _counts_from_pub(pub) -> dict[str, int]:
    """Best-effort extraction of a counts dict from a SamplerV2 PubResult.

    The Qiskit Runtime API exposes counts via
    ``pub.data.<creg_name>.get_counts()``. Our circuit has a single
    nameless classical register; depending on the qiskit version that
    becomes ``pub.data.c``, ``pub.data.meas``, or a single-key dict.
    Walk the candidates until one yields a counts dict.
    """
    data = getattr(pub, "data", pub)
    for attr in ("c", "meas", "creg", "register"):
        rec = getattr(data, attr, None)
        if rec is not None and hasattr(rec, "get_counts"):
            return rec.get_counts()
    # Fallback: BitArray accessed by index / iteration. Reconstruct
    # counts from raw shot strings if necessary.
    try:
        # In newer qiskit, pub.data is a DataBin whose only field is the
        # classical register; iterate its attributes.
        for name in dir(data):
            if name.startswith("_"):
                continue
            rec = getattr(data, name)
            if hasattr(rec, "get_counts"):
                return rec.get_counts()
    except Exception:  # pragma: no cover — defensive
        pass
    return {}


def _confusion(preds: np.ndarray, y: np.ndarray) -> list[list[int]]:
    tn = int(np.sum((preds == 0) & (y == 0)))
    fp = int(np.sum((preds == 1) & (y == 0)))
    fn = int(np.sum((preds == 0) & (y == 1)))
    tp = int(np.sum((preds == 1) & (y == 1)))
    return [[tn, fp], [fn, tp]]
