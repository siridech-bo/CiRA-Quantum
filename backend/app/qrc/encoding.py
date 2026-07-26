"""QRC input encoding — scalar → RF pulse → quantum operation (plan §8).

An encoding maps a normalized input ``s ∈ [0, 1]`` to a rotation angle
``θ`` and applies the corresponding RF pulse to the reservoir state. The
plan makes encoding a first-class research axis (§8.8: "encoding as an
optimization problem"), so this module exposes:

* seven **encoding functions** (Papers 1/3/4 + five alternatives),
* **multi-nucleus parallel** encoding via frequency selectivity — the
  SPINQ advantage of distinct ¹H/³¹P/¹⁹F channels (§8.4),
* **phase-amplitude** combined encoding R_z(2πs)·R_x(θ) (§8.3.3),
* pulse application as a unitary conjugation ρ → U ρ U† (RF pulses are
  fast compared to the free-evolution τ, so treated as instantaneous).

The pulse is a *unitary* applied outside the master equation; relaxation
acts during the subsequent free evolution in :mod:`app.qrc.system`.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

# ---------------------------------------------------------------------------
# Encoding functions:  s ∈ [0,1]  →  θ  (radians)
# ---------------------------------------------------------------------------


def _arcsin_sqrt(s):      # Paper 4 (Hou 2026): θ = arcsin(√s) ∈ [0, π/2]
    return np.arcsin(np.sqrt(np.clip(s, 0.0, 1.0)))


def _arccos(s):           # Paper 3 (Negoro 2018): θ = arccos(2s−1) ∈ [0, π]
    return np.arccos(np.clip(2.0 * s - 1.0, -1.0, 1.0))


def _linear(s):           # θ = s·π
    return np.asarray(s) * np.pi


def _sinusoidal(s):       # θ = π·sin²(s)  — emphasizes mid-range
    return np.pi * np.sin(np.asarray(s)) ** 2


def _logarithmic(s):      # θ = π·log(1+s)  — compresses dynamic range
    return np.pi * np.log1p(np.asarray(s))


def _polynomial(s):       # θ = π·s³  — emphasizes extremes
    return np.pi * np.asarray(s) ** 3


def _exponential(s):      # θ = π·(1−e^{−s})  — smooth saturation
    return np.pi * (1.0 - np.exp(-np.asarray(s)))


ENCODING_FNS: dict[str, Callable] = {
    "arcsin_sqrt": _arcsin_sqrt,
    "arccos": _arccos,
    "linear": _linear,
    "sinusoidal": _sinusoidal,
    "logarithmic": _logarithmic,
    "polynomial": _polynomial,
    "exponential": _exponential,
}


def angle_for(fn: str, s: float) -> float:
    """Rotation angle θ (rad) for input ``s`` under encoding ``fn``."""
    try:
        f = ENCODING_FNS[fn]
    except KeyError as exc:
        raise KeyError(
            f"unknown encoding {fn!r}; known: {sorted(ENCODING_FNS)}"
        ) from exc
    return float(f(s))


# ---------------------------------------------------------------------------
# Pulse construction / application
# ---------------------------------------------------------------------------


class Encoder:
    """Applies input pulses to reservoir states for one :class:`QRCSystem`.

    Bound to a system so it can build rotation operators once and reuse
    them. Supports single-value (broadcast to target qubits) and
    multi-value parallel encoding (one value per target qubit / nucleus).
    """

    def __init__(self, qrc_system, encoding_cfg) -> None:
        self.sys = qrc_system
        self.qt = qrc_system.qt
        self.cfg = encoding_cfg
        n = qrc_system.n
        self.targets = (
            list(encoding_cfg.target_qubits)
            if encoding_cfg.target_qubits
            else list(range(n))
        )

    def _rot2(self, axis: str, theta: float):
        """2×2 single-qubit rotation R_axis(θ) = exp(-i θ/2 σ_axis).

        Built in closed form — exponentiating the *full* 2^n operator per
        qubit per step is the dominant cost otherwise (≈1.3 s/step at N=8).
        Single-qubit pulses on distinct qubits commute, so the composite
        pulse is just the tensor product of these 2×2 gates.
        """
        c, si = np.cos(theta / 2.0), np.sin(theta / 2.0)
        if axis == "x":
            m = [[c, -1j * si], [-1j * si, c]]
        elif axis == "y":
            m = [[c, -si], [si, c]]
        else:  # z
            m = [[np.exp(-1j * theta / 2.0), 0], [0, np.exp(1j * theta / 2.0)]]
        return self.qt.Qobj(np.array(m, dtype=complex))

    def pulse_unitary(self, values):
        """Build the composite pulse unitary U for this input step.

        ``values`` is either a scalar (same input broadcast to all target
        qubits — global pulse, Paper 4 style) or a sequence with one entry
        per target qubit (frequency-selective parallel encoding, §8.4).
        With ``phase_amplitude`` on, each target also gets a preceding
        R_z(2πs) phase pulse (§8.3.3).

        Assembled as a tensor product of per-qubit 2×2 gates (identity on
        non-target qubits), which is exact because the single-qubit pulses
        act on distinct qubits and therefore commute.
        """
        vals = np.atleast_1d(np.asarray(values, dtype=float))
        if vals.size == 1:
            vals = np.repeat(vals, len(self.targets))
        if vals.size != len(self.targets):
            raise ValueError(
                f"got {vals.size} values for {len(self.targets)} target "
                f"qubits {self.targets}"
            )
        eye = self.qt.qeye(2)
        gates = [eye for _ in range(self.sys.n)]
        axis = self.cfg.axis
        for q, s in zip(self.targets, vals, strict=True):
            theta = angle_for(self.cfg.fn, float(s))
            u = self._rot2(axis, theta)
            if self.cfg.phase_amplitude:
                phi = 2.0 * np.pi * float(s)
                u = u * self._rot2("z", phi)
            gates[q] = u
        return self.qt.tensor(gates)

    def apply(self, rho, values):
        """Encode ``values`` into ``rho`` → U ρ U†."""
        U = self.pulse_unitary(values)
        return U * rho * U.dag()
