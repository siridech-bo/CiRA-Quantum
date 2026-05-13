"""Simulated Bifurcation sampler (Phase 9C).

Simulated Bifurcation (Goto et al. 2019) is a quantum-inspired
classical algorithm that simulates a network of nonlinear oscillators
whose continuous-variable dynamics, when bifurcated, settle into a
binary configuration corresponding to a low-energy Ising state.

This is the open algorithm sitting behind Fujitsu's proprietary
SQBM+ brand. We wrap the open ``simulated-bifurcation`` PyPI package
(Romain Demangeon et al.) as a ``dimod.Sampler`` so it slots into the
Phase-2 benchmark registry alongside our other QUBO heuristics.

Two algorithmic variants are exposed via the ``mode`` parameter:

* ``ballistic`` (bSB) — faster, lower solution quality. The
  ``simulated-bifurcation`` package's default.
* ``discrete`` (dSB) — slower per-step but with a discrete signum
  applied to oscillator positions, yielding tighter convergence to
  Ising-valid configurations. Empirically dominates bSB on harder
  instances at the cost of ~2x wall time.

The adapter is QUBO-only — no constraint awareness. Upstream
Lagrange-lowering handles CQM constraints the same way it does for
GPU SA and Parallel Tempering.

References:
    Goto, Tatsumura, & Dixon, "Combinatorial optimization by
    simulating adiabatic bifurcations in nonlinear Hamiltonian
    systems", Science Advances 5 (2019), eaav2372.
    Goto, Endo, & Tatsumura, "High-performance combinatorial
    optimization based on classical mechanics", Science Advances 7
    (2021), eabe7953.
"""

from __future__ import annotations

from typing import Literal

import dimod
import numpy as np

_DEFAULT_AGENTS = 32
_DEFAULT_MAX_STEPS = 1000
_DEFAULT_MODE: Literal["ballistic", "discrete"] = "ballistic"
_DEFAULT_NUM_READS = 10


class SimulatedBifurcationSampler(dimod.Sampler):
    """Wraps the ``simulated-bifurcation`` PyPI package's
    ``minimize`` function as a dimod.Sampler.

    Each call runs ``agents`` parallel SB trajectories from random
    initial conditions, then returns the top ``num_reads`` lowest-
    energy outcomes (with energies recomputed against the original
    BQM so the user-facing units are right).
    """

    _STOCHASTIC = True

    def __init__(
        self,
        mode: Literal["ballistic", "discrete"] = _DEFAULT_MODE,
    ):
        if mode not in ("ballistic", "discrete"):
            raise ValueError(
                f"mode must be 'ballistic' or 'discrete', got {mode!r}"
            )
        self._mode = mode

    @property
    def parameters(self) -> dict:
        return {
            "agents": [],
            "max_steps": [],
            "num_reads": [],
            "seed": [],
        }

    @property
    def properties(self) -> dict:
        return {
            "mode": self._mode,
            "stochastic": True,
            "supports_binary": True,
            "supports_spin": True,
        }

    def sample(
        self,
        bqm: dimod.BinaryQuadraticModel,
        agents: int = _DEFAULT_AGENTS,
        max_steps: int = _DEFAULT_MAX_STEPS,
        num_reads: int = _DEFAULT_NUM_READS,
        seed: int | None = None,
    ) -> dimod.SampleSet:
        if agents < 1:
            raise ValueError("agents must be >= 1")
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if num_reads < 1:
            raise ValueError("num_reads must be >= 1")

        # Empty-BQM short-circuit
        if bqm.num_variables == 0:
            samples = np.empty((1, 0), dtype=np.int8)
            return dimod.SampleSet.from_samples(
                (samples, []), vartype=bqm.vartype, energy=[float(bqm.offset)],
            )

        bqm_bin = bqm.change_vartype(dimod.BINARY, inplace=False)
        variables = list(bqm_bin.variables)
        n = len(variables)
        idx_for_var = {v: i for i, v in enumerate(variables)}

        # Build the (n, n) symmetric Q matrix and the (n,) linear vector b
        # that ``simulated-bifurcation`` expects for binary-domain
        # polynomial form. Energy = x^T Q x + b^T x + c.
        # Note: the package symmetrizes internally; passing an
        # upper-triangular Q + duplicating to lower would double-count.
        # We keep Q symmetric and split quadratic terms half-and-half.
        Q = np.zeros((n, n), dtype=np.float64)
        b = np.zeros(n, dtype=np.float64)
        for v, c in bqm_bin.linear.items():
            b[idx_for_var[v]] += float(c)
        for (u, w), c in bqm_bin.quadratic.items():
            i, j = idx_for_var[u], idx_for_var[w]
            if i == j:
                b[i] += float(c)  # x_i^2 == x_i for binary
            else:
                Q[i, j] += float(c) / 2.0
                Q[j, i] += float(c) / 2.0
        # offset is on bqm_bin; bqm_bin.energy() pulls it in automatically
        # when we recompute energies later, so no need to track it here.

        # ``sb.minimize`` ignores Python-level RNG seeding except via
        # torch. We respect the seed by seeding torch's RNG before
        # the call when one is provided.
        import torch
        if seed is not None:
            torch.manual_seed(int(seed))

        # Run SB. Returns (best_vector, best_value) by default — we
        # want multiple agents' results, so set best_only=False to
        # get one row per agent.
        import simulated_bifurcation as sb
        try:
            spins, _energies_unused = sb.minimize(
                Q, b,
                domain="binary",
                mode=self._mode,
                agents=int(agents),
                max_steps=int(max_steps),
                best_only=False,
                verbose=False,
                early_stopping=True,
            )
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"simulated_bifurcation.minimize failed: {type(e).__name__}: {e}"
            ) from e

        # ``spins`` is a torch tensor of shape (n,) when best_only=True
        # or (agents, n) when best_only=False — agents are rows. Convert
        # to numpy and round to nearest binary value (sb may return
        # values near 0 / 1 due to floating-point).
        spins_np = spins.detach().cpu().numpy()
        if spins_np.ndim == 1:
            spins_np = spins_np.reshape(1, -1)
        states = np.rint(spins_np).astype(np.int8)

        # Deduplicate (different agents often converge to identical
        # configurations) and recompute energies against the original
        # BQM for user-facing units.
        unique_states, _idx = np.unique(states, axis=0, return_index=True)
        energies = np.array(
            [
                bqm_bin.energy(dict(zip(variables, row, strict=True)))
                for row in unique_states
            ],
            dtype=np.float64,
        )

        order = np.argsort(energies)[:num_reads]
        sorted_states = unique_states[order]
        sorted_energies = energies[order].tolist()

        ss = dimod.SampleSet.from_samples(
            (sorted_states, variables),
            vartype=dimod.BINARY,
            energy=sorted_energies,
        )
        ss.info["sb_mode"] = self._mode
        ss.info["sb_agents"] = int(agents)
        ss.info["sb_max_steps"] = int(max_steps)
        ss.info["sb_unique_solutions"] = int(unique_states.shape[0])
        return ss
