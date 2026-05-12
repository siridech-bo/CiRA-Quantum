"""OriginQC pyqpanda QAOA adapter exposed as a ``dimod``-compatible sampler.

Phase 9A — Quantum tier, local simulator path.

QAOA is the Quantum Approximate Optimization Algorithm: a variational
quantum algorithm that prepares a parametric circuit whose measurement
distribution concentrates on low-energy bitstrings of a problem
Hamiltonian. We use OriginQC's pyqpanda3 stack via the high-level
``pyqpanda_alg.QUBO.QUBO_QAOA`` wrapper.

Key design notes:

* **NOT CQM-native** — unlike CP-SAT and HiGHS (Phase 8), QAOA is a
  QUBO solver. Constraints get baked into the objective as Lagrange
  penalty terms via ``dimod.cqm_to_bqm`` before the variational loop.
  Same pattern as ``gpu_sa``. The ``records.record_run`` dispatcher
  picks the BQM path automatically because ``_CQM_NATIVE`` is unset.

* **Local CPU simulator only** in 9A. The same wrapper transparently
  supports OriginQC's cloud backend (real superconducting QPU at
  ``qcloud.originqc.com.cn``) via a ``backend`` switch — that arm
  lands in Phase 9B alongside a fourth BYOK provider entry. The
  simulator path requires no external credentials, no network, no
  shot-noise from physics.

* **Qubit budget cap** — CPU statevector simulation memory is
  ``O(2**n_qubits)`` float64 complex. Beyond ~20 qubits the cost
  becomes painful (1M+ states), so we cap the adapter at
  ``MAX_QUBITS_CPU_SIM = 20`` by default. A CQM that would lower to
  more variables raises ``ValueError`` with a clear message pointing
  the user at Phase 9B for larger instances.

* **Stochastic results** — SPSA / SLSQP-trained QAOA is not bit-exact
  reproducible across runs. ``_STOCHASTIC = True`` is a forward-looking
  marker for Phase-2 ``replay_record``: a future change to replay
  tolerance will key off this attribute. In 9A nothing reads it yet.

* **Empty / trivial BQM short-circuit** mirrors ``gpu_sa.py``.
"""

from __future__ import annotations

from typing import Any

import dimod
import numpy as np

# Conservative cap for local CPU statevector simulation. 20 qubits is
# ~16M complex amplitudes (~256 MB at float64); beyond that, run-time
# climbs steeply. Phase 9B raises this for the cloud-real-hardware arm
# (Wukong runs at 72 qubits but with shot noise instead of statevector).
MAX_QUBITS_CPU_SIM = 20

# QAOA hyperparameter defaults. Layer depth p=3 is the sweet spot for
# small Phase-2 instances: layer 1 underfits, layer 5+ pays too much
# classical-optimizer wall time for marginal probability gain.
_DEFAULT_LAYER = 3
_DEFAULT_OPTIMIZER = "SLSQP"
_DEFAULT_TOP_K = 10  # Number of top-probability bitstrings to surface as samples.


class QAOASampler(dimod.Sampler):
    """pyqpanda QAOA wrapped as a ``dimod.Sampler``.

    QUBO-class solver. Reads a ``dimod.BinaryQuadraticModel``, converts
    it to a sympy expression, runs the variational quantum loop, and
    returns the top-K bitstrings by probability as a ``dimod.SampleSet``
    with each sample's energy recomputed against the BQM.

    The class deliberately does *not* set ``_CQM_NATIVE`` — constraints
    are handled by the orchestrator's BQM lowering with Lagrange penalty
    (Phase 2), exactly as for ``gpu_sa``.
    """

    _STOCHASTIC = True

    def __init__(
        self,
        layer: int = _DEFAULT_LAYER,
        optimizer: str = _DEFAULT_OPTIMIZER,
        top_k: int = _DEFAULT_TOP_K,
        max_qubits: int = MAX_QUBITS_CPU_SIM,
    ):
        if layer < 1:
            raise ValueError("layer must be >= 1")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if max_qubits < 1:
            raise ValueError("max_qubits must be >= 1")
        self._layer = int(layer)
        self._optimizer = str(optimizer)
        self._top_k = int(top_k)
        self._max_qubits = int(max_qubits)

    # ----- dimod.Sampler interface -----

    @property
    def parameters(self) -> dict:
        return {
            "layer": [],
            "optimizer": [],
            "top_k": [],
            "max_qubits": [],
            "seed": [],
        }

    @property
    def properties(self) -> dict:
        try:
            import pyqpanda3

            qpanda_version = getattr(pyqpanda3, "__version__", "unknown")
        except ImportError:  # pragma: no cover — Phase 9A gates the registry
            qpanda_version = "missing"
        return {
            "backend": "cpu_simulator",
            "pyqpanda3_version": qpanda_version,
            "max_qubits": self._max_qubits,
            "stochastic": True,
        }

    def sample(
        self,
        bqm: dimod.BinaryQuadraticModel,
        seed: int | None = None,
    ) -> dimod.SampleSet:
        """Run QAOA on the BQM and return a SampleSet of the top-K
        bitstrings by measurement probability."""
        # ---- Empty-BQM short-circuit ----
        # A BQM with no variables has exactly one assignment — the empty
        # one — whose energy is the offset. Pass a single 0-column row
        # so the SampleSet is well-formed.
        if bqm.num_variables == 0:
            samples_array = np.empty((1, 0), dtype=np.int8)
            return dimod.SampleSet.from_samples(
                (samples_array, []),
                vartype=bqm.vartype,
                energy=[float(bqm.offset)],
            )

        # ---- Qubit-budget guard ----
        n = bqm.num_variables
        if n > self._max_qubits:
            raise ValueError(
                f"QAOASampler: this CQM lowered to {n} variables, which "
                f"exceeds the local-simulator qubit cap of {self._max_qubits}. "
                "Phase 9B (Origin Quantum cloud + real Wukong QPU) lifts "
                "this cap; for now, route this instance through one of the "
                "classical or quantum-inspired tiers (cpsat, highs, gpu_sa, "
                "cpu_sa_neal)."
            )

        # ---- QUBO encoding: convert ±1/SPIN to 0/1/BINARY if needed ----
        # pyqpanda's QUBO wrapper expects a binary-form polynomial.
        # ``change_vartype`` is a no-op when already BINARY.
        bqm_bin = bqm.change_vartype(dimod.BINARY, inplace=False)

        # ---- Build sympy expression: f(x) = sum_i a_i*x_i + sum_{i<j} b_ij*x_i*x_j + c ----
        import sympy as sp
        from pyqpanda_alg.QUBO import QUBO_QAOA

        variables = list(bqm_bin.variables)
        # Fixed ordering — drives bitstring index ↔ variable name mapping.
        idx_for_var = {v: i for i, v in enumerate(variables)}
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

        # ---- Run QAOA ----
        # The pyqpanda_alg wrapper takes its hyperparameters via run().
        # scipy's gradient-based optimizers (SLSQP / BFGS / ...) don't
        # accept a `seed` option; pyqpanda's SPSA does but via a
        # different key. We honor `seed` only for SPSA in 9A and warn
        # softly for others (the QAOA initial parameters still vary
        # slightly across runs, so the result is reproducible-in-
        # distribution rather than bit-exact — see DECISIONS.md).
        optimizer_option: dict | None = None
        if seed is not None and self._optimizer.upper() == "SPSA":
            optimizer_option = {"options": {"seed": int(seed)}}

        runner = QUBO_QAOA(expr)
        result_dict = runner.run(
            layer=self._layer,
            optimizer=self._optimizer,
            optimizer_option=optimizer_option,
        )
        # ``result_dict`` is ``{bitstring: probability, ...}``. Bitstring
        # index 0 maps to the first sympy symbol (x0), which is
        # ``variables[0]`` per our ``sym_for_var`` construction.

        # ---- Pick top-K bitstrings, compute their energies, sort by energy ----
        # Sorting by energy (not by probability) makes ``ss.first.energy``
        # the lowest-energy bitstring QAOA produced — which is what the
        # Phase-5C dashboard reads as ``best_energy``. The probability
        # ranking is preserved in ``ss.info["qaoa_top_probabilities"]``
        # paired by bitstring for downstream inspection.
        top_by_prob = sorted(result_dict.items(), key=lambda kv: kv[1], reverse=True)[: self._top_k]

        rows: list[tuple[float, np.ndarray, str, float]] = []
        for bitstring, prob in top_by_prob:
            row = np.zeros(n, dtype=np.int8)
            for i_var, bit_char in enumerate(bitstring):
                row[idx_for_var[variables[i_var]]] = int(bit_char)
            energy = float(bqm_bin.energy(dict(zip(variables, row, strict=True))))
            rows.append((energy, row, bitstring, float(prob)))

        rows.sort(key=lambda t: t[0])  # ascending energy → .first is the minimum
        samples_array = np.stack([r[1] for r in rows]) if rows else np.zeros((0, n), dtype=np.int8)

        ss = dimod.SampleSet.from_samples(
            (samples_array, variables),
            vartype=dimod.BINARY,
            energy=[r[0] for r in rows],
        )

        # Record metadata so the dashboard can show probabilities and the
        # variational loop's loss alongside the energies.
        ss.info["qaoa_top_bitstrings"] = [r[2] for r in rows]
        ss.info["qaoa_top_probabilities"] = [r[3] for r in rows]
        ss.info["qaoa_layer"] = self._layer
        ss.info["qaoa_optimizer"] = self._optimizer
        return ss
