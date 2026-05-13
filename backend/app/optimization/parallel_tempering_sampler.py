"""Parallel Tempering sampler (Phase 9C).

Parallel Tempering — also called Replica Exchange Monte Carlo — runs
N independent Metropolis chains at different inverse temperatures
``β_1 < β_2 < ... < β_N`` and periodically attempts to *swap*
configurations between adjacent replicas. The swap acceptance rule

    P(swap_{i, i+1}) = min(1, exp((β_i - β_{i+1}) * (E_{i+1} - E_i)))

preserves the detailed-balance condition on the joint distribution.
Hot replicas explore freely (rough landscapes have shallow barriers
at low β); cold replicas exploit the basins they fall into. The
swap chain lets a configuration discovered by a hot replica
percolate down to the cold replica where it can be polished.

On rough energy landscapes — Edwards-Anderson spin glasses, dense
random QUBOs, penalty-lifted CQMs with frustrated couplings — PT is
known to outperform vanilla SA by orders of magnitude in
time-to-solution. It's the standard "better SA" in the optimization
literature.

This implementation is deliberately simple: N replicas in a
contiguous Python loop, swap attempts every ``swap_interval`` sweeps,
return the lowest-energy chain at the end. No multi-threading — for
the Phase-2 instance sizes (≤50 variables after BQM lowering) a
single-threaded numpy implementation is faster than process pools
would be.

References:
    Hukushima & Nemoto, "Exchange Monte Carlo method and application
    to spin glass simulations", J. Phys. Soc. Jpn. 65 (1996), 1604.
"""

from __future__ import annotations

import dimod
import numpy as np

_DEFAULT_NUM_REPLICAS = 8
_DEFAULT_NUM_SWEEPS = 1000
_DEFAULT_SWAP_INTERVAL = 10
_DEFAULT_BETA_RANGE = (0.1, 5.0)
_DEFAULT_NUM_READS = 100  # how many low-T-replica snapshots to retain


class ParallelTemperingSampler(dimod.Sampler):
    """Custom Hukushima-Nemoto Parallel Tempering for binary QUBOs.

    Reads a ``dimod.BinaryQuadraticModel``. Constraint handling is
    upstream (the orchestrator's CQM→BQM lowering); this sampler only
    cares about the QUBO matrix.

    The result is a ``dimod.SampleSet`` containing the ``num_reads``
    lowest-energy configurations observed across the cold replica's
    history. The first sample is the global minimum encountered.
    """

    _STOCHASTIC = True

    def __init__(
        self,
        num_replicas: int = _DEFAULT_NUM_REPLICAS,
        beta_range: tuple[float, float] = _DEFAULT_BETA_RANGE,
    ):
        if num_replicas < 2:
            raise ValueError("num_replicas must be >= 2 (PT needs >=2 chains)")
        if beta_range[0] <= 0 or beta_range[1] <= beta_range[0]:
            raise ValueError(
                "beta_range must be (β_hot, β_cold) with 0 < β_hot < β_cold"
            )
        self._num_replicas = int(num_replicas)
        self._beta_range = (float(beta_range[0]), float(beta_range[1]))

    @property
    def parameters(self) -> dict:
        return {
            "num_sweeps": [],
            "num_reads": [],
            "swap_interval": [],
            "seed": [],
        }

    @property
    def properties(self) -> dict:
        return {
            "num_replicas": self._num_replicas,
            "beta_range": self._beta_range,
            "stochastic": True,
        }

    def sample(
        self,
        bqm: dimod.BinaryQuadraticModel,
        num_sweeps: int = _DEFAULT_NUM_SWEEPS,
        num_reads: int = _DEFAULT_NUM_READS,
        swap_interval: int = _DEFAULT_SWAP_INTERVAL,
        seed: int | None = None,
    ) -> dimod.SampleSet:
        if num_sweeps < 1:
            raise ValueError("num_sweeps must be >= 1")
        if num_reads < 1:
            raise ValueError("num_reads must be >= 1")
        if swap_interval < 1:
            raise ValueError("swap_interval must be >= 1")

        # Empty-BQM short-circuit (matches sibling samplers).
        if bqm.num_variables == 0:
            samples = np.empty((1, 0), dtype=np.int8)
            return dimod.SampleSet.from_samples(
                (samples, []), vartype=bqm.vartype, energy=[float(bqm.offset)],
            )

        bqm_bin = bqm.change_vartype(dimod.BINARY, inplace=False)
        variables = list(bqm_bin.variables)
        n = len(variables)
        idx_for_var = {v: i for i, v in enumerate(variables)}

        # Build a dense (n, n) QUBO matrix Q where energy(x) = x^T Q x + offset.
        # Diagonal terms are the linear coefficients; off-diagonal are
        # the quadratic ones (kept upper-triangular for accounting).
        Q = np.zeros((n, n), dtype=np.float64)
        for v, c in bqm_bin.linear.items():
            Q[idx_for_var[v], idx_for_var[v]] += float(c)
        for (u, w), c in bqm_bin.quadratic.items():
            i, j = idx_for_var[u], idx_for_var[w]
            if i == j:
                Q[i, i] += float(c)
            else:
                lo, hi = (i, j) if i < j else (j, i)
                Q[lo, hi] += float(c)
        offset = float(bqm_bin.offset)

        # Geometric schedule from hot β to cold β.
        beta_hot, beta_cold = self._beta_range
        betas = np.geomspace(beta_hot, beta_cold, self._num_replicas)

        rng = np.random.default_rng(seed)

        # Initialize each replica with a random binary configuration.
        states = rng.integers(0, 2, size=(self._num_replicas, n), dtype=np.int8)
        energies = _vector_energies(states, Q, offset)

        # Track the lowest-energy snapshots observed across the simulation.
        # We keep up to (num_reads * num_replicas) candidates and pick the
        # best at the end, so we get genuine "best of trajectory" rather
        # than just the final state.
        best_states_pool: list[np.ndarray] = []
        best_energies_pool: list[float] = []

        # Pre-compute the per-replica acceptance threshold workspace.
        for sweep in range(num_sweeps):
            # One Metropolis sweep per replica.
            for r in range(self._num_replicas):
                _metropolis_sweep_inplace(states[r], Q, betas[r], rng)
                energies[r] = _energy(states[r], Q, offset)

            # Snapshot the cold replica's state every sweep — it's the
            # one most likely to land in the global minimum.
            best_states_pool.append(states[-1].copy())
            best_energies_pool.append(energies[-1])

            # Swap attempt block — try every adjacent pair at intervals.
            if (sweep + 1) % swap_interval == 0:
                for i in range(self._num_replicas - 1):
                    delta = (betas[i] - betas[i + 1]) * (energies[i + 1] - energies[i])
                    if delta >= 0 or rng.random() < np.exp(delta):
                        # Swap configurations
                        tmp = states[i].copy()
                        states[i] = states[i + 1]
                        states[i + 1] = tmp
                        energies[i], energies[i + 1] = energies[i + 1], energies[i]

        # Build the final SampleSet from the best ``num_reads`` of the
        # cold replica's snapshots (with replacement allowed). Sort by
        # energy ascending.
        pool_energies = np.array(best_energies_pool)
        pool_states = np.array(best_states_pool, dtype=np.int8)
        order = np.argsort(pool_energies)[:num_reads]
        sorted_states = pool_states[order]
        sorted_energies = pool_energies[order].tolist()

        ss = dimod.SampleSet.from_samples(
            (sorted_states, variables),
            vartype=dimod.BINARY,
            energy=sorted_energies,
        )
        ss.info["pt_num_replicas"] = self._num_replicas
        ss.info["pt_beta_range"] = list(self._beta_range)
        ss.info["pt_num_sweeps"] = num_sweeps
        ss.info["pt_swap_interval"] = swap_interval
        return ss


# ---- numpy hot-path helpers ----------------------------------------


def _energy(x: np.ndarray, Q: np.ndarray, offset: float) -> float:
    """Compute energy of one configuration. Q is upper-triangular plus
    the diagonal (linear terms)."""
    return float(x @ Q @ x + offset)


def _vector_energies(X: np.ndarray, Q: np.ndarray, offset: float) -> np.ndarray:
    """Compute energies of multiple configurations at once."""
    # X shape: (num_replicas, n). Compute one (Qx) per row.
    Qx = X @ Q                                        # (num_replicas, n)
    energies = np.einsum("ij,ij->i", X, Qx)           # (num_replicas,)
    return energies + offset


def _metropolis_sweep_inplace(
    x: np.ndarray, Q: np.ndarray, beta: float, rng: np.random.Generator,
) -> None:
    """One Metropolis sweep — try flipping each bit in turn, accept
    with Metropolis criterion. Edits ``x`` in place."""
    n = x.shape[0]
    # Precompute h[i] = 2 * sum_j Q_full[i,j] * x[j] - Q_full[i,i]
    # where Q_full is the symmetric form. ΔE on flipping bit i is
    # (1 - 2*x[i]) * h[i] when Q is upper-triangular with diagonal.
    Q_full = Q + Q.T
    np.fill_diagonal(Q_full, np.diag(Q))  # symmetric off-diag, untouched diag
    for i in rng.permutation(n):
        # ΔE if we flip x[i]:
        # E(flipped) - E(current) = (1 - 2*x[i]) * (sum_j Q_full[i, j] * x[j])
        h_i = float(Q_full[i] @ x)
        delta_E = (1.0 - 2.0 * x[i]) * h_i
        if delta_E <= 0 or rng.random() < np.exp(-beta * delta_E):
            x[i] = 1 - x[i]
