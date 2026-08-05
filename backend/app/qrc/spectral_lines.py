"""Physics-informed spectral-line selection for the FID readout (SOP §3.2).

The reservoir Hamiltonian is all-``σz`` (weak-coupling NMR Ising):

    H/h = Σ_i ν_i I_z^i + Σ_{i<j} J_ij I_z^i I_z^j        (I_z = σz/2)

so it is diagonal in the computational basis and the FID (transverse
magnetization ``Σ_k σx_k``) oscillates at a *finite, analytic* set of
**single-quantum transition frequencies**. Flipping readout spin ``k`` against
a fixed configuration of the other spins gives, to first order in weak coupling,

    f_k(z) = ν_k + ½ Σ_{j≠k} J_kj z_j ,   z_j ∈ {+1, −1}                 (Hz)

i.e. each resonance ``ν_k`` split into a ``2^{n-1}`` J-multiplet by its
spectators. That set is the *entire* spectral content of the FID — every one of
the standard 653 FFT peaks is one of these lines (or an FFT skirt of one).

**Physics-informed selection** (this module): enumerate the analytic lines,
then **merge lines closer than the decoherence linewidth** ``Δf = 1/(π T₂)``
(unresolvable under the reservoir's own dephasing). The surviving centres are
``D_eff`` features at *known* frequencies. Read them by a direct DFT projection
``X = fid_samples · Φ`` with ``Φ[m,k] = exp(i·2π f_k·t_m)`` — a fixed matrix,
fully differentiable (``torch.matmul``), so it is cheap enough to backprop
through for learnable encoding (regime B, §3.1) unlike the dense 653-FID.

Measured floors (``report_deff``): crotonic-9 (protons) ``D_eff ≈ 100`` (from
1280 raw transitions — the 653 readout oversamples ~6.5×); 6-spin generic
``D_eff ≈ 137``. The encoding sets only the line *amplitudes*; the frequencies
are fixed by ``H``, so ``Φ`` is a constant of the reservoir and the selection
does not depend on the input or a reference run.

Pure NumPy (+ optional torch for ``Phi``); no reservoir evolution, no GPU.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from app.qrc.config import SystemConfig


@dataclass
class LineSet:
    """The physics-informed reduced-FID feature basis for one reservoir."""

    freqs_hz: np.ndarray          # (D_eff,) resolvable line centres, Hz, sorted
    readout_qubits: tuple[int, ...]
    linewidth_hz: float           # merge threshold used (1/π·min T2 over readout)
    n_raw: int                    # raw single-quantum transitions before merge
    f_max_hz: float               # largest |line|, sets the Nyquist dwell

    @property
    def d_eff(self) -> int:
        return int(self.freqs_hz.size)

    def suggest_dwell(self, safety: float = 2.5) -> float:
        """Nyquist-safe FID dwell (s): ``dwell < 1/(2·f_max)`` with headroom."""
        if self.f_max_hz <= 0:
            return 3e-4
        return 1.0 / (2.0 * safety * self.f_max_hz)

    def suggest_samples(self, oversample: float = 2.0) -> int:
        """Minimum FID sample count M to read D_eff lines (``M ≳ oversample·D``)."""
        return int(np.ceil(oversample * self.d_eff))

    def projection(self, n_samples: int, dwell_s: float, backend: str = "numpy"):
        """Direct-DFT projection ``Φ`` (n_samples × D_eff) at the line freqs.

        ``X_k = Σ_m fid(t_m)·exp(i 2π f_k t_m)`` = ``fid_samples @ Φ``. Returned
        as a NumPy complex array, or a torch complex128 tensor for the
        differentiable path (``backend="torch"``).
        """
        t = np.arange(n_samples) * dwell_s                    # (M,)
        phi = np.exp(1j * 2 * np.pi * np.outer(t, self.freqs_hz))  # (M, D_eff)
        if backend == "torch":
            import torch

            return torch.as_tensor(phi, dtype=torch.complex128)
        return phi


def _lines_for_spin(nu: np.ndarray, J: np.ndarray, k: int,
                    spectators: list[int]) -> np.ndarray:
    """All ``2^len(spectators)`` first-order transition freqs for flipping ``k``."""
    js = J[k, spectators]
    signs = np.array(list(itertools.product((1.0, -1.0), repeat=len(spectators))))
    return nu[k] + 0.5 * signs @ js if spectators else np.array([nu[k]])


def _merge(freqs: np.ndarray, linewidth: float) -> np.ndarray:
    """Greedy 1-D clustering: a gap > linewidth starts a new resolvable line."""
    f = np.sort(np.asarray(freqs, dtype=float))
    if f.size == 0:
        return f
    centres = [f[0]]
    for x in f[1:]:
        if x - centres[-1] > linewidth:
            centres.append(x)
    return np.array(centres)


def physics_informed_lines(
    system: SystemConfig,
    readout_qubits: tuple[int, ...] | None = None,
    *,
    coupling_threshold_hz: float | None = None,
) -> LineSet:
    """Enumerate the FID's single-quantum lines and merge within the T₂ linewidth.

    ``readout_qubits`` defaults to all spins (crotonic → pass the protons, 4–8).
    ``coupling_threshold_hz`` optionally drops sub-threshold couplings as
    unresolvable splits before enumeration (a speed knob; the linewidth merge
    already removes their effect, so it does not change ``D_eff``).
    """
    nu = np.asarray(system.chemical_shifts, dtype=float)
    J = np.asarray(system.j_coupling, dtype=float)
    t2 = np.asarray(system.t2, dtype=float)
    n = system.n_qubits
    if readout_qubits is None:
        readout_qubits = tuple(range(n))

    all_freqs: list[np.ndarray] = []
    n_raw = 0
    for k in readout_qubits:
        spec = [j for j in range(n) if j != k]
        if coupling_threshold_hz is not None:
            spec = [j for j in spec if abs(J[k, j]) >= coupling_threshold_hz]
        fk = _lines_for_spin(nu, J, k, spec)
        n_raw += fk.size
        all_freqs.append(fk)

    freqs = np.concatenate(all_freqs)
    linewidth = 1.0 / (np.pi * float(t2[list(readout_qubits)].min()))
    centres = _merge(freqs, linewidth)
    return LineSet(
        freqs_hz=centres,
        readout_qubits=tuple(readout_qubits),
        linewidth_hz=linewidth,
        n_raw=int(n_raw),
        f_max_hz=float(np.abs(centres).max()) if centres.size else 0.0,
    )


def report_deff() -> None:
    """Print D_eff for the reference systems (crotonic-9 protons; 6-spin)."""
    from app.qrc.config import crotonic_acid_paper4, generic_nqubit

    for name, sysc, ro in [
        ("crotonic9_paper4 (protons 4-8)", crotonic_acid_paper4(), tuple(range(4, 9))),
        ("generic 6-spin (all)", generic_nqubit(6, seed=4), tuple(range(6))),
    ]:
        ls = physics_informed_lines(sysc, ro)
        print(f"{name:34s}  raw={ls.n_raw:5d}  linewidth={ls.linewidth_hz:5.2f}Hz  "
              f"D_eff={ls.d_eff:4d}  f_max={ls.f_max_hz:8.1f}Hz  "
              f"dwell~{ls.suggest_dwell()*1e3:.3f}ms  M~{ls.suggest_samples()}")


if __name__ == "__main__":
    report_deff()
