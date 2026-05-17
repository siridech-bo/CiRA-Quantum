"""IBM Quantum cloud QAOA adapter — Phase 11.

Submits trained QAOA circuits to IBM's superconducting QPUs via the
Qiskit Runtime SamplerV2 API. Parallel design to ``QAOACloudSampler``
(Origin), but Qiskit-pure end-to-end — no pyqpanda3 dependency on
this path.

Architecture — hybrid local-train + cloud-execute
-------------------------------------------------

Same pattern as the Origin sampler:

1. **Lower BQM → cost Hamiltonian ``H_C``** as a Qiskit ``SparsePauliOp``.
   Each linear bias ``h_i x_i`` becomes ``(h_i/2)(I - Z_i)``; each
   quadratic coupling ``J_ij x_i x_j`` becomes
   ``(J_ij/4)(I - Z_i - Z_j + Z_i Z_j)``. Constants are dropped — they
   don't affect the optimum and would only inflate the Hamiltonian.

2. **Train ``(γ, β)`` locally** using Qiskit's ``StatevectorEstimator``
   + ``scipy.optimize.minimize(method='COBYLA')``. The training loop
   evaluates ``⟨ψ(γ,β)|H_C|ψ(γ,β)⟩`` against an exact statevector
   simulator — no cloud round-trips during the variational phase.
   ``layer=1`` is the default; deeper layers need more parameters and
   the COBYLA optimizer scales fine through p ~ 6.

3. **Build the trained circuit** with the optimized angles bound in.

4. **Transpile for the target backend** at optimization_level=1 so the
   submission honors the chip's coupling map and basis-gate set.

5. **Submit via ``SamplerV2.run([(circuit,)], shots=N)``** and parse the
   resulting ``BitArray`` into a ``dimod.SampleSet``.

Safety
------

* **BYOK** — the IBM Quantum API token is supplied per-call at
  construction (live solve route) or read from ``IBM_QUANTUM_TOKEN``
  env var (benchmark scripts). Never logged.
* **Soft submission cap.** Same ``max_submissions`` guard as the
  Origin sampler — prevents accidental quota burns during dev. Open
  Plan gives 10 min/month plus a 180-min promotional bonus; the cap
  defaults to 5 submissions per sampler instance.
* **Dynamic backend** by default. Calls
  ``service.least_busy(operational=True, simulator=False, min_num_qubits=N)``
  so a retired chip name doesn't break the path. Passing an explicit
  ``backend_name`` overrides the auto-selection.

Public interface
----------------

* ``sample(bqm, seed=None) -> dimod.SampleSet`` — sync, blocks until
  the IBM job completes. Used by archival benchmark scripts.
* ``submit_async(bqm, seed=None) -> dict`` — non-blocking, returns
  the cloud job_id immediately. Used by the live multi-solver fan-out
  so the orchestrator never blocks on the IBM queue.
* ``try_materialize(job_id) -> dict | None`` — poll once; returns
  None if not yet terminal, or a fully-built result dict otherwise.
  The pending-jobs poller calls this periodically.
"""

from __future__ import annotations

import os
import time
from typing import Any

import dimod
import numpy as np

# A modest set of IBM superconducting QPUs known to host QAOA-shaped
# workloads. We use this purely as a presentation hint (e.g. for the
# ``hardware`` field of the registered ``SolverIdentity``); the real
# backend selection happens at runtime via ``least_busy()`` or the
# caller's ``backend_name`` override.
_REAL_HARDWARE_HINT = "ibm-quantum:heron-or-eagle"

_DEFAULT_LAYER = 1
_DEFAULT_SHOTS = 200
_DEFAULT_MAX_QUBITS = 20  # conservative — Open Plan jobs queue longer for bigger circuits
_DEFAULT_MAX_SUBMISSIONS = 5
_DEFAULT_TOP_K = 10
_DEFAULT_OPT_LEVEL = 1  # 0=no transpile, 3=heavy. Level 1 is the sweet spot for QAOA.


class QAOAIBMQSampler(dimod.Sampler):
    """Hybrid local-train + IBM-cloud-execute QAOA.

    All Qiskit-side state (training optimizer, Hamiltonian, circuit) is
    constructed once per ``sample()`` / ``submit_async()`` call so the
    instance can safely be reused across problems.
    """

    _STOCHASTIC = True

    def __init__(
        self,
        *,
        api_key: str,
        backend_name: str | None = None,
        instance: str | None = None,
        # qiskit-ibm-runtime v0.30+ renamed the legacy "ibm_quantum"
        # channel to "ibm_quantum_platform" when IBM merged their
        # consoles into the unified IBM Quantum Platform. The old name
        # raises ValueError on init in current SDK versions.
        channel: str = "ibm_quantum_platform",
        layer: int = _DEFAULT_LAYER,
        shots: int = _DEFAULT_SHOTS,
        max_qubits: int = _DEFAULT_MAX_QUBITS,
        max_submissions: int = _DEFAULT_MAX_SUBMISSIONS,
        top_k: int = _DEFAULT_TOP_K,
        train_optimizer: str = "COBYLA",
        optimization_level: int = _DEFAULT_OPT_LEVEL,
    ):
        if not api_key or not api_key.strip():
            raise ValueError("api_key is required for IBM Quantum cloud QAOA")
        if layer < 1:
            raise ValueError("layer must be >= 1")
        if shots < 1:
            raise ValueError("shots must be >= 1")
        if max_qubits < 1:
            raise ValueError("max_qubits must be >= 1")
        if max_submissions < 1:
            raise ValueError("max_submissions must be >= 1")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        self._api_key = api_key
        self._backend_name = backend_name  # None → least_busy at runtime
        self._instance = instance
        self._channel = channel
        self._layer = int(layer)
        self._shots = int(shots)
        self._max_qubits = int(max_qubits)
        self._max_submissions = int(max_submissions)
        self._top_k = int(top_k)
        self._train_optimizer = str(train_optimizer)
        self._optimization_level = int(optimization_level)
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
            "backend_name": self._backend_name or "(dynamic via least_busy)",
            "channel": self._channel,
            "is_real_hardware": True,
            "max_qubits": self._max_qubits,
            "stochastic": True,
        }

    def sample(
        self,
        bqm: dimod.BinaryQuadraticModel,
        seed: int | None = None,
    ) -> dimod.SampleSet:
        """Sync entry point: submit and wait. Used by benchmark scripts."""
        prepared = self._prepare(bqm, seed=seed)
        if prepared is None:
            # Empty-BQM short-circuit.
            return _empty_sampleset(bqm)
        submitted = self._submit_one(prepared)
        result = self._wait_for_terminal(submitted["job"], timeout_s=900.0)
        return self._sampleset_from_result(prepared, submitted, result)

    # ----- Async interface (used by the live multi-solver fan-out) -----

    def submit_async(
        self,
        bqm: dimod.BinaryQuadraticModel,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Submit and return immediately. Caller persists the returned
        ``cloud_job_id`` and polls ``try_materialize()`` later."""
        prepared = self._prepare(bqm, seed=seed)
        if prepared is None:
            return {"empty": True, **_empty_result_envelope()}
        submitted = self._submit_one(prepared)
        return {
            "cloud_job_id": submitted["job_id"],
            "backend_name": submitted["backend_name"],
            "trained_gammas": prepared["gammas"],
            "trained_betas": prepared["betas"],
            "layer": self._layer,
            "shots": self._shots,
            # Stash everything the poller needs to rebuild the SampleSet
            # without re-submitting. The "prepared" payload is JSON-able
            # (BQM is reconstructed from linear/quadratic dicts) but
            # large — we leave that for the poller to load from the BQM
            # blob stored alongside the pending entry.
        }

    def try_materialize(
        self,
        cloud_job_id: str,
    ) -> dict[str, Any] | None:
        """Poll once. Returns None if the IBM job isn't terminal yet,
        a structured result dict if it is (success or cloud-side error).
        """
        service = self._service()
        try:
            job = service.job(cloud_job_id)
        except Exception as e:  # noqa: BLE001
            return {"terminal": True, "error": f"job lookup failed: {type(e).__name__}: {e}"}

        try:
            status = job.status()
        except Exception as e:  # noqa: BLE001
            return {"terminal": True, "error": f"status() failed: {type(e).__name__}: {e}"}

        # qiskit-ibm-runtime returns status as a string in modern
        # versions (e.g. "DONE", "QUEUED", "RUNNING", "CANCELLED",
        # "ERROR"). Older versions returned a JobStatus enum.
        status_str = (
            status.name if hasattr(status, "name") else str(status)
        ).upper()

        if status_str in {"DONE", "COMPLETED", "FINISHED"}:
            try:
                result = job.result()
            except Exception as e:  # noqa: BLE001
                return {"terminal": True, "error": f"result() failed: {type(e).__name__}: {e}"}
            return {
                "terminal": True,
                "status": "complete",
                "primitive_result": result,
                "cloud_job_id": cloud_job_id,
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

        # QUEUED / RUNNING / VALIDATING / INITIALIZING → keep waiting
        queue_pos = None
        try:
            queue_pos = job.queue_position()
        except Exception:  # pragma: no cover — older runtime versions
            pass
        return {
            "terminal": False,
            "status": "queued",
            "live_status": status_str,
            "queue_position": queue_pos,
            "cloud_job_id": cloud_job_id,
        }

    # ----- Internal helpers -----

    def _service(self):
        """Construct a ``QiskitRuntimeService`` from the stored token.
        Done lazily so importing this module is cheap on machines that
        don't actually use the IBM path."""
        from qiskit_ibm_runtime import QiskitRuntimeService
        kwargs = {"channel": self._channel, "token": self._api_key}
        if self._instance is not None:
            kwargs["instance"] = self._instance
        return QiskitRuntimeService(**kwargs)

    def _pick_backend(self, service, num_qubits: int):
        """Resolve an actual backend. Honor an explicit name first;
        otherwise call ``least_busy``."""
        if self._backend_name:
            return service.backend(self._backend_name)
        return service.least_busy(
            operational=True,
            simulator=False,
            min_num_qubits=num_qubits,
        )

    def _prepare(
        self,
        bqm: dimod.BinaryQuadraticModel,
        *,
        seed: int | None,
    ) -> dict[str, Any] | None:
        """All the local work — Hamiltonian build, circuit construct,
        local QAOA training. Returns a dict the submit step consumes,
        or None for the empty-BQM short-circuit."""
        if bqm.num_variables == 0:
            return None
        if bqm.num_variables > self._max_qubits:
            raise ValueError(
                f"QAOAIBMQSampler: this CQM lowered to {bqm.num_variables} variables, "
                f"which exceeds the configured cap of {self._max_qubits}. Pick a "
                "smaller problem or route to a classical / quantum-inspired tier."
            )
        if self._submissions_so_far >= self._max_submissions:
            raise RuntimeError(
                f"QAOAIBMQSampler: per-instance submission cap of "
                f"{self._max_submissions} reached. Construct a new sampler "
                "instance if you need more submissions in this session."
            )

        bqm_bin = bqm.change_vartype(dimod.BINARY, inplace=False)
        variables = list(bqm_bin.variables)
        n = len(variables)

        # Build the cost Hamiltonian as a SparsePauliOp.
        from qiskit.quantum_info import SparsePauliOp

        # Convert x_i (∈ {0,1}) → (1 - Z_i)/2 substitution. The constant
        # offset doesn't affect the optimum so we drop it.
        pauli_terms: list[tuple[str, complex]] = []
        # Track which qubits have local Z biases / which pairs have ZZ
        # couplings for later structural reporting (qaoa_extras).
        linear_terms: list[tuple[int, float]] = []
        quadratic_terms: list[tuple[int, int, float]] = []

        for v, h in bqm_bin.linear.items():
            hv = float(h)
            if hv == 0.0:
                continue
            i = variables.index(v)
            # h * x = (h/2)(I - Z) → contributes -h/2 * Z_i (drop constant)
            label = ["I"] * n
            label[i] = "Z"
            pauli_terms.append(("".join(reversed(label)), -hv / 2.0))
            linear_terms.append((i, hv))

        for (u, v), J in bqm_bin.quadratic.items():
            Jv = float(J)
            if Jv == 0.0:
                continue
            i, j = variables.index(u), variables.index(v)
            i, j = sorted([i, j])
            # J * x_i * x_j = (J/4)(I - Z_i - Z_j + Z_i Z_j); contributes
            # -J/4 * Z_i, -J/4 * Z_j, and J/4 * Z_i Z_j (drop constant).
            for q in (i, j):
                label = ["I"] * n
                label[q] = "Z"
                pauli_terms.append(("".join(reversed(label)), -Jv / 4.0))
            label = ["I"] * n
            label[i] = "Z"
            label[j] = "Z"
            pauli_terms.append(("".join(reversed(label)), Jv / 4.0))
            quadratic_terms.append((i, j, Jv))

        if not pauli_terms:
            # Trivial BQM — every assignment has the same energy. Skip
            # training and just emit uniform samples.
            return {
                "trivial": True,
                "n": n,
                "variables": variables,
                "linear_terms": linear_terms,
                "quadratic_terms": quadratic_terms,
                "gammas": [0.0] * self._layer,
                "betas": [0.0] * self._layer,
            }

        # Coalesce duplicate Pauli strings.
        coeff_by_pauli: dict[str, complex] = {}
        for p, c in pauli_terms:
            coeff_by_pauli[p] = coeff_by_pauli.get(p, 0.0) + c
        labels, coeffs = zip(*coeff_by_pauli.items(), strict=True)
        H_C = SparsePauliOp(list(labels), coeffs=list(coeffs))

        # Build parameterized circuit.
        from qiskit import QuantumCircuit
        from qiskit.circuit import Parameter

        gammas_param = [Parameter(f"gamma_{p}") for p in range(self._layer)]
        betas_param = [Parameter(f"beta_{p}") for p in range(self._layer)]
        qc = QuantumCircuit(n)
        for q in range(n):
            qc.h(q)
        for p in range(self._layer):
            # Problem unitary e^{-i gamma H_C}.
            for (i, h) in linear_terms:
                qc.rz(-gammas_param[p] * h, i)
            for (i, j, J) in quadratic_terms:
                qc.cx(i, j)
                qc.rz(gammas_param[p] * J, j)
                qc.cx(i, j)
            # Mixer e^{-i beta sum_i X_i}.
            for q in range(n):
                qc.rx(2.0 * betas_param[p], q)
        qc.measure_all()

        # Local training.
        gammas_trained, betas_trained, train_loss = _train_qaoa_locally(
            qc, gammas_param, betas_param, H_C, seed=seed,
            optimizer=self._train_optimizer,
        )

        return {
            "trivial": False,
            "n": n,
            "variables": variables,
            "linear_terms": linear_terms,
            "quadratic_terms": quadratic_terms,
            "H_C": H_C,
            "qc": qc,
            "gammas_param": gammas_param,
            "betas_param": betas_param,
            "gammas": gammas_trained,
            "betas": betas_trained,
            "train_loss": train_loss,
            "bqm_bin": bqm_bin,
        }

    def _submit_one(self, prepared: dict[str, Any]) -> dict[str, Any]:
        """Transpile + submit, return the live job handle + metadata.
        Caller is responsible for waiting (or stashing for async)."""
        if prepared.get("trivial"):
            # No real submission needed — return a sentinel the wait
            # path detects and turns into uniform samples.
            return {
                "trivial": True,
                "job": None,
                "job_id": "trivial-no-cloud-roundtrip",
                "backend_name": "n/a",
            }

        from qiskit.transpiler import generate_preset_pass_manager
        from qiskit_ibm_runtime import SamplerV2

        service = self._service()
        backend = self._pick_backend(service, prepared["n"])
        backend_name = backend.name

        pm = generate_preset_pass_manager(
            backend=backend,
            optimization_level=self._optimization_level,
        )
        qc_isa = pm.run(prepared["qc"])
        # Bind the trained parameters.
        param_bindings = {
            **dict(zip(prepared["gammas_param"], prepared["gammas"], strict=True)),
            **dict(zip(prepared["betas_param"], prepared["betas"], strict=True)),
        }
        qc_bound = qc_isa.assign_parameters(param_bindings)

        sampler = SamplerV2(mode=backend)
        job = sampler.run([(qc_bound,)], shots=self._shots)
        self._submissions_so_far += 1
        return {
            "trivial": False,
            "job": job,
            "job_id": job.job_id(),
            "backend_name": backend_name,
        }

    def _wait_for_terminal(self, job, *, timeout_s: float):
        """Block until the IBM job completes or fails. Raises on timeout."""
        if job is None:
            return None
        deadline = time.time() + timeout_s
        poll = 2.0
        while time.time() < deadline:
            try:
                status = job.status()
            except Exception as e:  # noqa: BLE001
                raise RuntimeError(f"IBM job.status() failed: {type(e).__name__}: {e}")
            s = (status.name if hasattr(status, "name") else str(status)).upper()
            if s in {"DONE", "COMPLETED", "FINISHED"}:
                return job.result()
            if s in {"ERROR", "FAILED", "CANCELLED", "CANCELED"}:
                raise RuntimeError(f"IBM cloud job ended in status {s!r}")
            time.sleep(poll)
        raise RuntimeError(f"IBM cloud job did not complete within {timeout_s:.0f} s")

    def _sampleset_from_result(
        self,
        prepared: dict[str, Any],
        submitted: dict[str, Any],
        result,
    ) -> dimod.SampleSet:
        """Common ``PrimitiveResult`` → ``dimod.SampleSet`` mapping used
        by both the sync ``sample()`` and the poller's materialize step."""
        n = prepared["n"]
        variables = prepared["variables"]
        bqm_bin = prepared["bqm_bin"]

        if prepared.get("trivial") or result is None:
            samples_array = np.zeros((1, n), dtype=np.int8)
            return dimod.SampleSet.from_samples(
                (samples_array, variables),
                vartype=dimod.BINARY,
                energy=[float(bqm_bin.offset)] if hasattr(bqm_bin, "offset") else [0.0],
            )

        # PrimitiveResult contains one PubResult per submitted PUB; we
        # only sent one. Pull the BitArray and count bitstrings.
        pub_result = result[0]
        bit_array = pub_result.data.meas
        # bit_array.get_counts() returns {bitstring: count}.
        counts = bit_array.get_counts()
        total = sum(counts.values())
        probs = {bs: c / total for bs, c in counts.items()}
        top = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[: self._top_k]

        rows: list[tuple[float, np.ndarray, str, float]] = []
        for bitstring, p in top:
            padded = bitstring.zfill(n)[-n:]
            # Qiskit bitstrings are little-endian (rightmost char is qubit 0).
            # Convert to a left-aligned (q0 first) order for the dimod sample.
            le = padded[::-1]
            row = np.zeros(n, dtype=np.int8)
            for i, ch in enumerate(le):
                row[i] = int(ch)
            energy = float(bqm_bin.energy(dict(zip(variables, row, strict=True))))
            rows.append((energy, row, bitstring, float(p)))

        rows.sort(key=lambda t: t[0])
        samples_array = (
            np.stack([r[1] for r in rows]) if rows else np.zeros((0, n), dtype=np.int8)
        )
        ss = dimod.SampleSet.from_samples(
            (samples_array, variables),
            vartype=dimod.BINARY,
            energy=[r[0] for r in rows],
        )
        ss.info["qaoa_top_bitstrings"] = [r[2] for r in rows]
        ss.info["qaoa_top_probabilities"] = [r[3] for r in rows]
        ss.info["qaoa_layer"] = self._layer
        ss.info["qaoa_optimizer"] = self._train_optimizer
        ss.info["qaoa_trained_gammas"] = list(prepared["gammas"])
        ss.info["qaoa_trained_betas"] = list(prepared["betas"])
        ss.info["qaoa_train_loss"] = float(prepared.get("train_loss", 0.0))
        ss.info["cloud_backend"] = submitted["backend_name"]
        ss.info["cloud_job_id"] = submitted["job_id"]
        ss.info["cloud_shots"] = self._shots
        ss.info["cloud_is_real_hardware"] = True
        return ss


# ===== Module-level helpers =================================================


def _empty_sampleset(bqm: dimod.BinaryQuadraticModel) -> dimod.SampleSet:
    samples_array = np.empty((1, 0), dtype=np.int8)
    return dimod.SampleSet.from_samples(
        (samples_array, []),
        vartype=bqm.vartype,
        energy=[float(bqm.offset)],
    )


def _empty_result_envelope() -> dict[str, Any]:
    return {
        "cloud_job_id": "empty-no-cloud-roundtrip",
        "backend_name": "n/a",
        "trained_gammas": [],
        "trained_betas": [],
        "layer": 0,
        "shots": 0,
    }


def _train_qaoa_locally(
    qc, gammas_param, betas_param, H_C,
    *,
    seed: int | None,
    optimizer: str = "COBYLA",
) -> tuple[list[float], list[float], float]:
    """Run scipy.optimize against a local statevector simulator to find
    the best (γ, β) for the given cost Hamiltonian. Returns the trained
    angles and the final loss."""
    from qiskit.primitives import StatevectorEstimator
    from scipy.optimize import minimize

    estimator = StatevectorEstimator(seed=seed if seed is not None else 42)
    layer = len(gammas_param)

    def objective(x: np.ndarray) -> float:
        """x = [gamma_0, beta_0, gamma_1, beta_1, ...]"""
        bindings = {}
        for p in range(layer):
            bindings[gammas_param[p]] = float(x[2 * p])
            bindings[betas_param[p]] = float(x[2 * p + 1])
        qc_bound = qc.assign_parameters(bindings)
        # Strip the measure_all() — the estimator wants an unmeasured
        # circuit so it can compute the analytical expectation value.
        qc_unmeas = qc_bound.remove_final_measurements(inplace=False)
        pub_result = estimator.run([(qc_unmeas, H_C)]).result()[0]
        return float(pub_result.data.evs)

    # Modest starting point — γ in [0, π], β in [0, π/2]. Random seed
    # respects the user's ``seed`` argument so smoke tests reproduce.
    rng = np.random.default_rng(seed if seed is not None else 42)
    x0 = rng.uniform(low=0.1, high=1.0, size=2 * layer)

    res = minimize(
        objective,
        x0,
        method=optimizer,
        options={"maxiter": 50, "rhobeg": 0.3},
    )
    x_best = np.asarray(res.x)
    gammas = [float(x_best[2 * p]) for p in range(layer)]
    betas = [float(x_best[2 * p + 1]) for p in range(layer)]
    return gammas, betas, float(res.fun)
