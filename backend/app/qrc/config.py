"""QRC configuration — system presets, simulation, encoding, training.

Everything the QRC pipeline needs is described by four frozen-ish
dataclasses that are cheap to serialize (for the reproducibility hash)
and easy to override from the CLI or a YAML/JSON file.

Design note — **N-qubit generic from the start**. The plan
(``docs/QRC_Simulation_Plan.md``) is written around the 3-qubit SPINQ
Gemini Lab, but a headline research goal is to *prove that more qubits
buy more computational power* (memory capacity, NARMA accuracy) so the
lab has a quantitative case for acquiring a larger machine. So the
system is never hardcoded to 3 spins: a :class:`SystemConfig` carries
per-spin arrays (chemical shifts, T1, T2) plus a J-coupling matrix of
any size, and :func:`make_system_config` can *generate* a physically
plausible ``N``-spin NMR molecule for the scaling sweep.

Units convention (kept consistent everywhere downstream):

* frequencies / chemical shifts / J-couplings: **Hz** (angular
  conversion ``2*pi`` happens inside :mod:`app.qrc.system`)
* times (T1, T2, evolution ``tau``, sampling ``dt``): **seconds**
* input values fed to the reservoir: normalized to ``[0, 1]``

References: Hou et al. 2026 (PRL 136, 120602); Nakajima et al. 2018
(arXiv:1803.04574); Negoro et al. 2018 (arXiv:1806.10910).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np

# ---------------------------------------------------------------------------
# System (the physical spin network)
# ---------------------------------------------------------------------------


@dataclass
class SystemConfig:
    """A weak-coupling (Ising-J) NMR spin network of ``n_qubits`` spins.

    ``chemical_shifts`` are rotating-frame offsets in Hz. ``j_coupling``
    is a symmetric ``n×n`` matrix in Hz (diagonal ignored). ``t1`` /
    ``t2`` are per-spin relaxation / dephasing times in seconds. The
    optional ``labels`` name the nuclei for readable plots/logs.
    """

    n_qubits: int
    chemical_shifts: list[float]          # Hz, length n_qubits
    j_coupling: list[list[float]]         # Hz, n×n symmetric
    t1: list[float]                       # seconds, length n_qubits
    t2: list[float]                       # seconds, length n_qubits
    labels: list[str] = field(default_factory=list)
    # Larmor frequencies (MHz) are only needed to reason about
    # frequency-selective addressing; they don't enter the rotating-frame
    # dynamics, so they're metadata.
    larmor_mhz: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        n = self.n_qubits
        if len(self.chemical_shifts) != n:
            raise ValueError(f"chemical_shifts must have length {n}")
        if len(self.t1) != n or len(self.t2) != n:
            raise ValueError(f"t1/t2 must have length {n}")
        j = np.asarray(self.j_coupling, dtype=float)
        if j.shape != (n, n):
            raise ValueError(f"j_coupling must be {n}×{n}")
        if not np.allclose(j, j.T):
            raise ValueError("j_coupling must be symmetric")
        if not self.labels:
            self.labels = [f"q{i}" for i in range(n)]


@dataclass
class SimConfig:
    """Simulation / reservoir knobs.

    ``tau`` is the free-evolution time per input step; ``n_virtual``
    (V) is the number of temporal-multiplexing sample points within each
    ``tau`` window (Nakajima's virtual nodes). ``backend`` selects the
    QuTiP data layer — ``"numpy"`` (CPU, best for ≤ ~9 qubits) or
    ``"jax"`` (GPU via ``qutip-jax``; only pays off for larger systems
    or big batched parameter sweeps — see the plan §2.3).
    """

    tau: float = 0.03                     # s (30 ms — Paper-4-ish)
    n_virtual: int = 25                   # V virtual nodes (Paper 1)
    backend: Literal["numpy", "jax"] = "numpy"
    # Time-evolution strategy per reservoir step:
    #   "propagator" — precompute a dense Liouvillian superoperator once
    #     and reuse it every step. Fast for small N, but the superoperator
    #     is 4^N-dimensional (dense ~4.3 GB at N=7, ~550 GB at N=9), so it
    #     walls around 6–7 qubits.
    #   "action"     — apply exp(t·L) to the vectorized state via Krylov
    #     (scipy ``expm_multiply``) on the *sparse* Liouvillian. Exact and
    #     memory-lean (the Liouvillian is ~0.001% dense at N=9), and immune
    #     to the stiffness that cripples an adaptive ODE. Reaches N=9.
    #   "mesolve"    — integrate the 2^N×2^N density matrix with QuTiP's
    #     adaptive ODE. Correct, but the fast-precession/slow-relaxation
    #     stiffness makes it impractically slow past ~7 qubits.
    #   "gpu"        — same exact-exponential idea as ``action`` but the
    #     sparse Liouvillian matvecs run on the GPU (torch CUDA, complex64)
    #     via a Taylor + sub-stepping exp(t·L). ~15-25x faster than the CPU
    #     ``action`` path at N=9. Requires a CUDA torch build; raises if
    #     unavailable. Not chosen by ``auto`` (GPU presence varies) —
    #     request it explicitly.
    #   "auto"       — propagator for N ≤ 5, action for N ≥ 6 (CPU only).
    evolution_mode: Literal[
        "auto", "propagator", "action", "mesolve", "gpu"
    ] = "auto"
    # Pure-dephasing model. "physical": gamma_phi = 1/T2 - 1/(2 T1),
    # collapse op sqrt(gamma_phi/2) σz (correct Bloch-Redfield-consistent
    # form). "simple": sqrt(1/(2 T2)) σz, the shortcut written in the
    # plan §12 implementation notes. Physical is the default.
    dephasing_model: Literal["physical", "simple"] = "physical"
    # Small initial longitudinal polarization ε for the thermal-like
    # product initial state ρ0 = ⊗ (I + ε σz)/2. Kept small (high-temp
    # NMR limit). Dynamics are linear in ρ so the exact value only
    # scales signal amplitude, not the qualitative memory behavior.
    init_polarization: float = 0.05
    seed: int = 42


@dataclass
class EncodingConfig:
    """How a scalar input ``s ∈ [0,1]`` becomes an RF rotation.

    ``fn`` names the encoding function (see :mod:`app.qrc.encoding`).
    ``target_qubits`` lists which spins the input pulse addresses; an
    empty list means "all". ``phase_amplitude`` turns on the combined
    R_z(2πs)·R_x(θ) scheme that packs a second degree of freedom into
    each input.
    """

    fn: Literal[
        "arcsin_sqrt",   # Paper 4:  θ = arcsin(√s)
        "arccos",        # Paper 3:  θ = arccos(2s-1)
        "linear",        # θ = s·π
        "sinusoidal",    # θ = π·sin²(s)
        "logarithmic",   # θ = π·log(1+s)
        "polynomial",    # θ = π·s³
        "exponential",   # θ = π·(1-exp(-s))
    ] = "arcsin_sqrt"
    axis: Literal["x", "y"] = "x"
    target_qubits: list[int] = field(default_factory=list)
    phase_amplitude: bool = False


@dataclass
class TrainingConfig:
    """Readout training (ridge regression) + benchmark splits."""

    ridge_lambda: float = 1e-6
    cv_folds: int = 10
    # Ridge λ grid searched by cross-validation (log-spaced).
    lambda_grid: list[float] = field(
        default_factory=lambda: list(np.logspace(-9, 1, 11))
    )
    washout: int = 100                    # steps discarded before fitting
    n_train: int = 400
    n_test: int = 100
    device: Literal["cpu", "cuda"] = "cuda"
    seed: int = 42


@dataclass
class QRCConfig:
    """Top-level bundle passed around the pipeline."""

    system: SystemConfig
    sim: SimConfig = field(default_factory=SimConfig)
    encoding: EncodingConfig = field(default_factory=EncodingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    code_version: str = "qrc-0.1.0"

    def repro_hash(self) -> str:
        """SHA-256 (truncated) over the full config — same idea as the
        QML/optimization archives. Two runs with the same hash should
        reproduce up to library nondeterminism."""
        blob = json.dumps(asdict(self), sort_keys=True, default=float)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------


def spinq_3qubit() -> SystemConfig:
    """The SPINQ Gemini Lab ¹H/³¹P/¹⁹F system (plan §6.1.1).

    Chemical shifts are not given in the plan (only Larmor frequencies,
    which are absorbed by the rotating frame), so we use small distinct
    rotating-frame offsets that keep the three spins spectrally
    resolved — the qualitative memory/NARMA behavior is insensitive to
    their exact values, the J-couplings drive the entangling dynamics.
    """
    j = [
        [0.0, 42.0, 220.0],   # H–H(self,ignored), H–P, H–F
        [42.0, 0.0, 430.0],   # P–H, P–P, P–F
        [220.0, 430.0, 0.0],  # F–H, F–P, F–F
    ]
    return SystemConfig(
        n_qubits=3,
        chemical_shifts=[120.0, -80.0, 200.0],
        j_coupling=j,
        t1=[5.0, 4.5, 6.0],
        t2=[0.2, 0.15, 0.25],
        labels=["1H", "31P", "19F"],
        larmor_mhz=[27.3, 11.0, 25.5],
    )


def generic_nqubit(
    n: int,
    *,
    seed: int = 0,
    t1: float = 5.0,
    t2: float = 0.2,
    shift_spread_hz: float = 120.0,
    j_nearest_hz: float = 200.0,
    j_decay: float = 0.5,
) -> SystemConfig:
    """A physically plausible ``n``-spin NMR molecule for scaling studies.

    Chemical shifts are spread evenly across ``±shift_spread_hz`` (so the
    spins stay spectrally distinct). J-couplings follow a chain-like
    decay: nearest neighbours couple at ``j_nearest_hz`` and the strength
    falls off as ``j_decay**(|i-j|-1)`` with a little seeded jitter, which
    mimics how through-bond scalar couplings shrink with topological
    distance in a real molecule. Homogeneous ``t1``/``t2`` by default
    (override per-spin on the returned object if desired).

    This is deliberately generic — for the scaling sweep we care about
    *how* memory capacity grows with ``n``, not about matching one exact
    molecule. Use :func:`crotonic_acid_like` for a 9-spin Paper-4 stand-in.

    Why ``shift_spread_hz`` defaults to 120 Hz (not larger): transverse
    magnetization precesses at the chemical-shift frequency, and the
    temporal-multiplexing readout samples it at ``V / tau``. If a shift
    exceeds the sampling Nyquist (``V / (2·tau)`` — e.g. 133 Hz at V=8,
    τ=30 ms) the precession *aliases* and the reservoir's recall of the
    current input collapses (corr²(0) falls below the fading-memory gate).
    120 Hz keeps every spin readable across the V settings we use while
    staying spectrally distinct given the ~200 Hz J-couplings. Raise it
    only if you also raise ``SimConfig.n_virtual`` accordingly.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    rng = np.random.default_rng(seed)
    if n == 1:
        shifts = [0.0]
    else:
        shifts = list(np.linspace(-shift_spread_hz, shift_spread_hz, n))
    j = np.zeros((n, n))
    for i in range(n):
        for k in range(i + 1, n):
            base = j_nearest_hz * (j_decay ** (k - i - 1))
            jitter = 1.0 + 0.1 * rng.standard_normal()
            val = float(max(0.0, base * jitter))
            j[i, k] = j[k, i] = val
    return SystemConfig(
        n_qubits=n,
        chemical_shifts=[float(x) for x in shifts],
        j_coupling=j.tolist(),
        t1=[float(t1)] * n,
        t2=[float(t2)] * n,
        labels=[f"s{i}" for i in range(n)],
    )


def crotonic_acid_like() -> SystemConfig:
    """A 9-spin stand-in for the Paper 4 (Hou et al. 2026) system.

    Paper 4 uses ¹³C-labelled crotonic acid (a well-known 7-carbon +
    proton NMR benchmark). We don't reproduce its exact coupling table
    here; instead we build a 9-spin generic molecule with realistic
    scales so the 9-qubit end of the scaling sweep is meaningful. Swap
    in the published coupling constants when replicating Paper 4 exactly.
    """
    cfg = generic_nqubit(9, seed=4, t1=5.5, t2=0.2, j_nearest_hz=250.0)
    cfg.labels = [f"C{i}" if i < 4 else f"H{i-4}" for i in range(9)]
    return cfg


SYSTEM_PRESETS = {
    "spinq3": spinq_3qubit,
    "crotonic9": crotonic_acid_like,
}


def make_system_config(name_or_n: str | int, **kw) -> SystemConfig:
    """Resolve a preset name (``"spinq3"``/``"crotonic9"``) or an integer
    ``n`` (→ :func:`generic_nqubit`) to a :class:`SystemConfig`."""
    if isinstance(name_or_n, int):
        return generic_nqubit(name_or_n, **kw)
    if name_or_n in SYSTEM_PRESETS:
        return SYSTEM_PRESETS[name_or_n]()
    raise KeyError(
        f"unknown system preset {name_or_n!r}; "
        f"known: {sorted(SYSTEM_PRESETS)} or an int qubit count"
    )
