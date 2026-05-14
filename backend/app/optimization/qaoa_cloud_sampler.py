"""OriginQC cloud QAOA adapter — Phase 9B.

Submits trained QAOA circuits to Origin Quantum's cloud, targeting
either their hosted statevector simulators (``full_amplitude``,
``partial_amplitude``, ``single_amplitude``) or the real
superconducting QPUs (``WK_C180`` Wukong, etc.).

Architecture — hybrid local-train + cloud-execute
-------------------------------------------------

Cloud QPU submissions cost real queue time (typically 30 s – several
minutes per call) and, for real hardware, billable QPU minutes. The
naive approach — running the full variational optimizer loop against
the cloud — would mean 30-50 cloud round-trips per record, taking
hours and burning credits.

Instead, this sampler:

1. **Trains QAOA parameters locally** using the Phase-9A path
   (``pyqpanda_alg.QAOA.qaoa.QAOA`` against the CPU statevector
   simulator). This is the same algorithm and produces the same
   optimized ``(gamma, beta)`` parameters the cloud submission would
   eventually converge to.
2. **Rebuilds the trained circuit by hand** using pyqpanda3
   primitives — Hadamard initial state, alternating problem-unitary +
   mixer-unitary layers parameterized by the trained ``gamma``,
   ``beta``.
3. **Submits the trained circuit once** to the chosen cloud backend,
   waits for the queue + execution, and parses the measurement
   probabilities into a ``dimod.SampleSet``.

Result: one cloud submission per archived record. Real-QPU records
carry real-QPU noise characteristics in their measurement
distributions; cloud-simulator records exercise the same dispatch
path without QPU cost. The local-training step is identical to
``QAOASampler`` (Phase 9A), so the algorithm trace is reproducible.

Safety
------

* **BYOK.** The Origin API key is supplied at construction (or pulled
  from the BYOK store at sample time by callers like the Flask route).
  The key is never logged or echoed by this class.
* **Soft submission cap.** Each ``QAOACloudSampler`` instance enforces
  a per-instance ``max_submissions`` cap (default 5). Once exceeded,
  ``sample()`` raises rather than burning more cloud credits. The
  cap is per-Python-process, not per-user — the Phase-7 hardening
  work adds proper per-user rate limits.
* **Real-hardware feature flag.** Real QPU backends (``WK_C180`` and
  any other non-simulator) are gated behind the environment variable
  ``ENABLE_ORIGIN_REAL_HARDWARE=1``. By default, attempting to
  construct against a real-QPU backend raises. This prevents
  accidental burns during dev.

Qubit budget
------------

Real QPUs (``WK_C180``: 169 working qubits as of probe time) lift the
20-qubit CPU-simulator cap from Phase 9A. The cloud-simulator
backends scale further (statevector simulation cost is the cloud
operator's problem). Each backend declares its own ``max_qubits``
from the chip metadata; this sampler defaults to 64 for cloud paths,
which fits every Phase-2 instance after Lagrange-lowering.
"""

from __future__ import annotations

import os
import time
from typing import Any

import dimod
import numpy as np

# Backends that count as "real superconducting hardware" — accessing
# these requires the ENABLE_ORIGIN_REAL_HARDWARE feature flag.
_REAL_HARDWARE_BACKENDS = frozenset({"WK_C180", "HanYuan_01"})
_DEFAULT_CLOUD_URL = "http://pyqanda-admin.qpanda.cn"
_DEFAULT_BACKEND = "full_amplitude"  # cheap cloud simulator

# Qubit caps per backend type, set from the step-1.6/1.7 empirical
# probes:
#   - Real QPU (WK_C180): n=7 dense QAOA completes in ~10s; n=8 hangs
#     past the cloud's 3-minute processing budget. So the wall is
#     between 7 and 8 regardless of coefficient magnitude.
#   - Cloud simulator (full_amplitude): no measured wall under 30
#     qubits; statevector cost is the cloud's problem, not ours.
#     64 is a generous default that fits every Phase-2 instance
#     after Lagrange-lowering.
_DEFAULT_MAX_QUBITS_SIMULATOR = 64
_DEFAULT_MAX_QUBITS_REAL_QPU = 7
_DEFAULT_LAYER = 3
_DEFAULT_SHOTS = 2048
_DEFAULT_MAX_SUBMISSIONS = 5
_DEFAULT_TOP_K = 10


def _real_hardware_enabled() -> bool:
    """Read the ENABLE_ORIGIN_REAL_HARDWARE feature flag from env.
    Accepts (case-insensitive) ``1``, ``true``, ``yes``."""
    return os.environ.get("ENABLE_ORIGIN_REAL_HARDWARE", "").strip().lower() in (
        "1", "true", "yes",
    )


# Origin Quantum's pyqpanda3 v0.3.5 ``QCloudJob.result()`` blocks
# until the cloud reports a status it considers terminal. In practice
# we've observed it polling forever when the cloud-side job is
# ``Failed`` (e.g. transient "Submit pilot task error" rejections),
# hanging the orchestrator's daemon thread indefinitely. This helper
# wraps the wait with a wall-clock timeout AND surfaces Failed /
# Error states immediately instead of waiting for them to "resolve".
def _wait_for_cloud_job(job, timeout_s: float):
    """Poll a ``QCloudJob`` until it completes or fails; never block
    forever. Returns the ``QCloudResult`` on success, raises
    ``RuntimeError`` on failure or timeout."""
    import time
    poll_interval = 1.0
    deadline = time.time() + timeout_s

    while time.time() < deadline:
        try:
            status = job.status()
        except Exception as e:  # noqa: BLE001 — defensive against SDK-level errors
            raise RuntimeError(f"job.status() raised: {type(e).__name__}: {e}") from e

        # pyqpanda3 returns ``JobStatus`` enum values; strip the prefix
        # for case-insensitive name matching. Treat any unfamiliar
        # status as "in progress" — never as terminal-success.
        name = (
            status.name if hasattr(status, "name")
            else str(status).rsplit(".", 1)[-1]
        ).upper()

        if name in ("COMPLETED", "FINISHED", "SUCCESS"):
            return job.query()

        if name in ("FAILED", "ERROR", "CANCELLED", "CANCELED", "ABORTED"):
            # Try to pull a useful error message off the result before bailing.
            err_msg = ""
            try:
                r = job.query()
                err_msg = r.error_message() or ""
            except Exception:  # pragma: no cover — defensive
                pass
            raise RuntimeError(
                f"Cloud-side job reported status {name!r}"
                + (f": {err_msg}" if err_msg else ""),
            )

        # "QUEUED", "RUNNING", "COMPUTING", "PENDING", "UNKNOWN", … →
        # keep waiting.
        time.sleep(poll_interval)

    raise RuntimeError(
        f"Timed out waiting for cloud job after {timeout_s:.0f} s "
        f"(last status: {name!r})"
    )


class QAOACloudSampler(dimod.Sampler):
    """Hybrid local-train + cloud-execute QAOA on OriginQC's cloud."""

    _STOCHASTIC = True

    def __init__(
        self,
        *,
        api_key: str,
        backend_name: str = _DEFAULT_BACKEND,
        url: str = _DEFAULT_CLOUD_URL,
        layer: int = _DEFAULT_LAYER,
        shots: int = _DEFAULT_SHOTS,
        max_qubits: int | None = None,
        max_submissions: int = _DEFAULT_MAX_SUBMISSIONS,
        top_k: int = _DEFAULT_TOP_K,
        train_optimizer: str = "SLSQP",
    ):
        if not api_key or not api_key.strip():
            raise ValueError("api_key is required for cloud QAOA")
        if layer < 1:
            raise ValueError("layer must be >= 1")
        if shots < 1:
            raise ValueError("shots must be >= 1")
        if max_submissions < 1:
            raise ValueError("max_submissions must be >= 1")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        # If max_qubits wasn't specified explicitly, pick a sensible
        # default based on the backend type. Real QPUs have a hard
        # empirical wall (see step-1.6/1.7 diagnostics); simulators do
        # not.
        is_real_hw = backend_name in _REAL_HARDWARE_BACKENDS
        if max_qubits is None:
            max_qubits = (
                _DEFAULT_MAX_QUBITS_REAL_QPU if is_real_hw
                else _DEFAULT_MAX_QUBITS_SIMULATOR
            )
        if max_qubits < 1:
            raise ValueError("max_qubits must be >= 1")

        # Real-hardware gating: refuse to construct against a QPU
        # backend unless the env flag is explicitly on.
        if backend_name in _REAL_HARDWARE_BACKENDS and not _real_hardware_enabled():
            raise RuntimeError(
                f"Backend {backend_name!r} is a real superconducting QPU. "
                "Set ENABLE_ORIGIN_REAL_HARDWARE=1 in the environment to "
                "submit jobs against it. Simulator backends "
                f"({sorted(_simulator_backend_hints())}) work without the "
                "feature flag."
            )

        self._api_key = api_key
        self._backend_name = backend_name
        self._url = url
        self._layer = int(layer)
        self._shots = int(shots)
        self._max_qubits = int(max_qubits)
        self._max_submissions = int(max_submissions)
        self._top_k = int(top_k)
        self._train_optimizer = str(train_optimizer)
        self._submissions_so_far = 0

    # ----- dimod.Sampler interface -----

    @property
    def parameters(self) -> dict:
        return {
            "layer": [],
            "shots": [],
            "top_k": [],
            "max_qubits": [],
            "seed": [],
        }

    @property
    def properties(self) -> dict:
        return {
            "backend_name": self._backend_name,
            "is_real_hardware": self._backend_name in _REAL_HARDWARE_BACKENDS,
            "url": self._url,
            "max_qubits": self._max_qubits,
            "stochastic": True,
        }

    def sample(
        self,
        bqm: dimod.BinaryQuadraticModel,
        seed: int | None = None,
    ) -> dimod.SampleSet:
        # ---- Empty-BQM short-circuit (matches Phase-9A QAOASampler) ----
        if bqm.num_variables == 0:
            samples_array = np.empty((1, 0), dtype=np.int8)
            return dimod.SampleSet.from_samples(
                (samples_array, []),
                vartype=bqm.vartype,
                energy=[float(bqm.offset)],
            )

        # ---- Qubit budget guard ----
        n = bqm.num_variables
        if n > self._max_qubits:
            raise ValueError(
                f"QAOACloudSampler: this CQM lowered to {n} variables, which "
                f"exceeds backend {self._backend_name!r}'s qubit budget of "
                f"{self._max_qubits}. Pick a backend with more qubits or "
                "route to a classical / quantum-inspired tier."
            )

        # ---- Submission cap guard ----
        if self._submissions_so_far >= self._max_submissions:
            raise RuntimeError(
                f"QAOACloudSampler: per-instance submission cap of "
                f"{self._max_submissions} reached. Construct a new sampler "
                "instance if you need more submissions in this session — the "
                "cap is a Phase-9B safety guard against accidental QPU "
                "credit burns during development."
            )

        bqm_bin = bqm.change_vartype(dimod.BINARY, inplace=False)
        variables = list(bqm_bin.variables)
        idx_for_var = {v: i for i, v in enumerate(variables)}

        # ---- Step 1: train QAOA parameters locally ----
        # Build sympy expression from BQM, same as Phase 9A.
        import sympy as sp
        from pyqpanda_alg.QAOA.qaoa import QAOA as QAOA_local

        sym_for_var = {v: sp.Symbol(f"x{i}") for i, v in enumerate(variables)}
        expr: Any = sp.Float(float(bqm_bin.offset))
        for v, coeff in bqm_bin.linear.items():
            if coeff == 0:
                continue
            expr = expr + sp.Float(float(coeff)) * sym_for_var[v]
        for (u, w), coeff in bqm_bin.quadratic.items():
            if coeff == 0:
                continue
            expr = expr + sp.Float(float(coeff)) * sym_for_var[u] * sym_for_var[w]

        local_qaoa = QAOA_local(expr)
        train_started = time.perf_counter()
        _local_probs, trained_params, train_loss = local_qaoa.run(
            layer=self._layer,
            optimizer=self._train_optimizer,
        )
        train_elapsed = time.perf_counter() - train_started

        # The optimized params are interleaved [g_0, b_0, g_1, b_1, ...].
        gammas = [float(trained_params[2 * p]) for p in range(self._layer)]
        betas = [float(trained_params[2 * p + 1]) for p in range(self._layer)]

        # ---- Step 2: rebuild the trained circuit by hand ----
        import pyqpanda3.core as pq3
        from pyqpanda_alg.QAOA.qaoa import (
            pauli_z_operator_to_circuit,
            problem_to_z_operator,
        )

        problem_op = problem_to_z_operator(expr, norm=False)
        qubits = list(range(n))

        prog = pq3.QProg()
        # Initial state: |+>^n via Hadamards
        for q in qubits:
            prog << pq3.H(q)
        # P alternating QAOA layers. ``pauli_z_operator_to_circuit``
        # returns a (QCircuit, scalar_offset) tuple — the offset is the
        # identity-coefficient term that contributes a global phase
        # we can drop for sampling-only use.
        for p in range(self._layer):
            sub_circuit, _identity_offset = pauli_z_operator_to_circuit(
                problem_op, qubits, gamma=gammas[p]
            )
            prog << sub_circuit
            for q in qubits:
                prog << pq3.RX(q, 2.0 * betas[p])
        # Measure all qubits into classical registers 0..n-1
        for i, q in enumerate(qubits):
            prog << pq3.measure(q, i)

        # ---- Step 3: submit to cloud and wait ----
        # ``QCloudBackend.run()`` is overloaded but pyqpanda3 v0.3.5
        # enforces a strict signature split at the C++ layer:
        #   * Simulator backends (``full_amplitude``, etc.):
        #     ``run(prog, shots)`` — passing ``QCloudOptions`` raises
        #     ``RuntimeError: Run with QCloudOptions is only for QPU``.
        #   * QPU backends (``WK_C180``, ``HanYuan_01``):
        #     ``run(prog, shots, QCloudOptions)`` — the options enable
        #     the mapping / optimization / amend passes the web UI
        #     submits by default.
        # This used to be flexible (v0.3.x earlier allowed options on
        # simulators too); the Phase 9D investigation traced our
        # "instant-fail" symptom to this tightening.
        from pyqpanda3.qcloud import QCloudOptions, QCloudService

        service = QCloudService(api_key=self._api_key, url=self._url)
        try:
            backend = service.backend(self._backend_name)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"Failed to acquire backend {self._backend_name!r} from "
                f"{self._url}: {type(e).__name__}: {e}"
            ) from e

        is_real_hw = self._backend_name in _REAL_HARDWARE_BACKENDS
        submit_started = time.perf_counter()
        # Wall-clock timeout for each cloud wait. Real-QPU runs need
        # more headroom (queue + ~minute of execution); simulator runs
        # should complete in seconds.
        timeout_s = 300.0 if is_real_hw else 60.0

        # Retry-on-transient. Origin's QPanda-source jobs hit a
        # different cloud-side scheduler queue than the web Composer
        # uses, and that queue intermittently rejects pilot tasks with
        # "Submit pilot task error, please try again later" — the
        # screenshots from Phase 9D show Failed jobs sandwiched between
        # successful ones, so a retry within a minute has a high hit
        # rate. Backoff is 2 / 5 / 10 seconds — short enough to stay
        # under the orchestrator's perceived "still working" budget.
        retry_delays = [2.0, 5.0, 10.0]
        max_attempts = len(retry_delays) + 1  # 1 initial + 3 retries
        last_error: Exception | None = None
        result = None
        job_id = ""

        for attempt in range(max_attempts):
            try:
                if is_real_hw:
                    options = QCloudOptions()
                    # Match the OriginQC web UI defaults on the QPU path.
                    options.set_optimization(True)
                    options.set_mapping(True)
                    options.set_amend(True)
                    options.set_is_prob_counts(True)
                    job = backend.run(prog, self._shots, options)
                else:
                    # Simulator path: NO QCloudOptions (raises in v0.3.5).
                    job = backend.run(prog, self._shots)
                job_id = job.job_id()
                result = _wait_for_cloud_job(job, timeout_s)
                break  # success
            except Exception as e:  # noqa: BLE001
                last_error = e
                msg = str(e).lower()
                # Only retry transient pilot-task errors. Other errors
                # (auth, malformed circuit, qubit budget) won't get
                # better by waiting.
                is_transient = (
                    "pilot task" in msg
                    or "please try again" in msg
                    or "timed out waiting" in msg
                )
                if not is_transient or attempt == max_attempts - 1:
                    break
                time.sleep(retry_delays[attempt])

        if result is None:
            raise RuntimeError(
                f"Cloud submission to {self._backend_name} failed after "
                f"{max_attempts} attempts: {type(last_error).__name__}: {last_error}"
            ) from last_error
        submit_elapsed = time.perf_counter() - submit_started

        # Count this against the per-instance cap *after* a successful
        # submission. A failure to submit doesn't burn the cap (the
        # job likely never reached the QPU).
        self._submissions_so_far += 1

        # ---- Step 4: parse measurement probabilities ----
        # ``get_probs()`` returns {bitstring: probability} in the same
        # shape as the local QAOASampler. From here, the path mirrors
        # Phase 9A: take top-K by probability, recompute energies via
        # the BQM, sort ascending, return.
        cloud_probs = result.get_probs()
        if not cloud_probs:
            raise RuntimeError(
                f"Cloud backend {self._backend_name!r} returned an empty "
                "probability distribution. Job ID: {job_id}"
            )

        top_by_prob = sorted(cloud_probs.items(), key=lambda kv: kv[1], reverse=True)[
            : self._top_k
        ]

        rows: list[tuple[float, np.ndarray, str, float]] = []
        for bitstring, prob in top_by_prob:
            # Pad to n bits in case the cloud emitted a shorter string
            # (e.g. when leading qubits measured 0 and the backend
            # trims them). The bitstring positions correspond to qubits
            # 0..n-1 in order.
            padded = bitstring.zfill(n)[-n:]
            row = np.zeros(n, dtype=np.int8)
            for i_var, bit_char in enumerate(padded):
                row[idx_for_var[variables[i_var]]] = int(bit_char)
            energy = float(bqm_bin.energy(dict(zip(variables, row, strict=True))))
            rows.append((energy, row, bitstring, float(prob)))

        rows.sort(key=lambda t: t[0])
        samples_array = np.stack([r[1] for r in rows]) if rows else np.zeros((0, n), dtype=np.int8)

        ss = dimod.SampleSet.from_samples(
            (samples_array, variables),
            vartype=dimod.BINARY,
            energy=[r[0] for r in rows],
        )
        ss.info["qaoa_top_bitstrings"] = [r[2] for r in rows]
        ss.info["qaoa_top_probabilities"] = [r[3] for r in rows]
        ss.info["qaoa_layer"] = self._layer
        ss.info["qaoa_optimizer"] = self._train_optimizer
        ss.info["qaoa_trained_gammas"] = gammas
        ss.info["qaoa_trained_betas"] = betas
        ss.info["qaoa_train_loss"] = float(train_loss)
        ss.info["qaoa_train_elapsed_s"] = train_elapsed
        ss.info["cloud_backend"] = self._backend_name
        ss.info["cloud_url"] = self._url
        ss.info["cloud_job_id"] = job_id
        ss.info["cloud_submit_elapsed_s"] = submit_elapsed
        ss.info["cloud_shots"] = self._shots
        ss.info["cloud_is_real_hardware"] = self._backend_name in _REAL_HARDWARE_BACKENDS
        return ss


def _simulator_backend_hints() -> list[str]:
    """Names that are known cloud simulators (not real QPUs). Used in
    error messages and the registry's backend probe."""
    return ["full_amplitude", "partial_amplitude", "single_amplitude", "PQPUMESH8"]
