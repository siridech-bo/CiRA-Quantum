"""Pulse sequences as an engine primitive (one definition for simulation and hardware).

A ``Sequence`` is an ordered tuple of hard pulses, free delays and one acquisition.
Simulators execute it through a small backend protocol (``init / pulse / free /
acquire``); a hardware driver consumes the same object through ``to_dict()`` (event list
with absolute start times and RF phases). Builders reproduce the probe banks used in the
noise-spectroscopy milestones exactly (same element order and timings).

Conventions: hard (instantaneous) rotations about x (phase 0 deg) or y (phase 90 deg);
times in seconds; ``Acquire(n, dwell)`` samples the FID at m * dwell, m = 0..n-1,
starting at the end of the last element before it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

_PHASE = {"x": 0.0, "y": 90.0, "-x": 180.0, "-y": 270.0}


@dataclass(frozen=True)
class Pulse:
    axis: str           # "x" | "y" | "-x" | "-y"
    angle: float        # rad

    def __post_init__(self):
        if self.axis not in _PHASE:
            raise ValueError(f"unknown pulse axis {self.axis!r}")


@dataclass(frozen=True)
class Delay:
    t: float            # s

    def __post_init__(self):
        if self.t < 0:
            raise ValueError("negative delay")


@dataclass(frozen=True)
class Acquire:
    n: int
    dwell: float        # s


@dataclass(frozen=True)
class Sequence:
    name: str
    elements: tuple

    def __post_init__(self):
        acq = [e for e in self.elements if isinstance(e, Acquire)]
        if len(acq) != 1 or not isinstance(self.elements[-1], Acquire):
            raise ValueError("a sequence needs exactly one Acquire, as its last element")

    @property
    def acquisition(self) -> Acquire:
        return self.elements[-1]

    @property
    def prep_duration(self) -> float:
        """Time from the first element to the start of acquisition (s)."""
        return float(sum(e.t for e in self.elements if isinstance(e, Delay)))

    def timeline(self):
        """Yield (t_start, element); pulses are instantaneous."""
        t = 0.0
        for e in self.elements:
            yield t, e
            if isinstance(e, Delay):
                t += e.t

    def to_dict(self) -> dict[str, Any]:
        ev = []
        for t, e in self.timeline():
            if isinstance(e, Pulse):
                ev.append({"type": "pulse", "t": t, "phase_deg": _PHASE[e.axis],
                           "angle_rad": e.angle})
            elif isinstance(e, Delay):
                ev.append({"type": "delay", "t": t, "duration": e.t})
            else:
                ev.append({"type": "acquire", "t": t, "n": e.n, "dwell": e.dwell})
        return {"name": self.name, "events": ev}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Sequence":
        inv = {v: k for k, v in _PHASE.items()}
        el = []
        for e in d["events"]:
            if e["type"] == "pulse":
                el.append(Pulse(inv[e["phase_deg"]], e["angle_rad"]))
            elif e["type"] == "delay":
                el.append(Delay(e["duration"]))
            else:
                el.append(Acquire(e["n"], e["dwell"]))
        return cls(d["name"], tuple(el))


# --- builders ----------------------------------------------------------------
def cpmg(n_pi: int, T: float, n_acq: int, dwell: float) -> Sequence:
    """pi/2_x, then n_pi pi_y pulses evenly over T (tau/2, tau, ..., tau, tau/2), acquire.
    n_pi = 0 is a free delay T (FID after delay). The Milestone-0 probe."""
    el = [Pulse("x", np.pi / 2)]
    if n_pi == 0:
        el.append(Delay(T))
    else:
        tau = T / n_pi
        el.append(Delay(tau / 2))
        for i in range(n_pi):
            el.append(Pulse("y", np.pi))
            el.append(Delay(tau if i < n_pi - 1 else tau / 2))
    el.append(Acquire(n_acq, dwell))
    return Sequence(f"cpmg:{n_pi}:{T:g}", tuple(el))


def fid_delay(T: float, n_acq: int, dwell: float) -> Sequence:
    return Sequence(f"fid:{T:g}", (Pulse("x", np.pi / 2), Delay(T), Acquire(n_acq, dwell)))


def inversion_recovery(T: float, n_acq: int, dwell: float) -> Sequence:
    return Sequence(f"ir:{T:g}", (Pulse("x", np.pi), Delay(T), Pulse("x", np.pi / 2),
                                  Acquire(n_acq, dwell)))


# --- execution -----------------------------------------------------------------
class Backend(Protocol):
    def init(self): ...
    def pulse(self, state, axis: str, angle: float): ...
    def free(self, state, t: float): ...
    def acquire(self, state, n: int, dwell: float): ...


def run(seq: Sequence, backend: Backend, state=None):
    """Execute ``seq`` on ``backend``; returns whatever ``backend.acquire`` returns."""
    R = backend.init() if state is None else state
    for e in seq.elements:
        if isinstance(e, Pulse):
            R = backend.pulse(R, e.axis, e.angle)
        elif isinstance(e, Delay):
            R = backend.free(R, e.t)
        else:
            return backend.acquire(R, e.n, e.dwell)
