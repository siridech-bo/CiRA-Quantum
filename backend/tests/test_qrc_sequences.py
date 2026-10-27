"""Module C: pulse sequences as an engine primitive.

The refactored M0 probe (``Batch.probe`` -> ``sequences.cpmg`` -> ``sequences.run``) must
reproduce the pre-refactor implementation exactly, and sequences must round-trip through
the hardware event format.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("qutip")
torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from app.qrc import sequences as sq  # noqa: E402


def _legacy_probe(B, m, n_pi, T):
    """Frozen copy of the Milestone-0 Batch.probe (commit 09e3bfd) before the refactor."""
    R = B.init() @ B.s.P90.T
    if n_pi == 0:
        R = B.free(R, T)
    else:
        tau = T / n_pi
        R = B.free(R, tau / 2)
        for i in range(n_pi):
            R = R @ B.s.P180.T
            R = B.free(R, tau if i < n_pi - 1 else tau / 2)
    det = [ci for ci, (_, ix) in enumerate(B.s.classes) if torch.abs(B.s.o[ix]).max() > 0]
    keep = np.concatenate([B.s.classes[ci][1] for ci in det])
    out = []
    for k in range(m.N_ACQ):
        if k:
            R = B.free(R, m.DWELL, cls=det)
        out.append(R[:, :, keep].sum(1) @ B.s.o[keep])
    return torch.stack(out, 1)


@pytest.mark.parametrize("system", ["S1", "R3"])
def test_refactored_probe_matches_legacy(system):
    import qrc_noisespec_m0 as m
    sp = m.Spins(m.make_systems()[system])
    B = m.Batch(sp, [[(150.0, 2e-3)], [(40.0, 0.3e-3)]], 6)
    for n_pi, t_ms in [(0, 2), (1, 5), (4, 20), (32, 200)]:
        new = B.probe(n_pi, t_ms * 1e-3).numpy()
        old = _legacy_probe(B, m, n_pi, t_ms * 1e-3).numpy()
        assert np.abs(new - old).max() <= 1e-12 * np.abs(old).max()


def test_cpmg_structure_and_roundtrip():
    s = sq.cpmg(4, 20e-3, 512, 50e-6)
    kinds = [type(e).__name__ for e in s.elements]
    assert kinds == ["Pulse", "Delay"] + ["Pulse", "Delay"] * 4 + ["Acquire"]
    assert abs(s.prep_duration - 20e-3) < 1e-15
    ev = s.to_dict()["events"]
    pulse_t = [e["t"] for e in ev if e["type"] == "pulse"][1:]
    assert np.allclose(pulse_t, [2.5e-3, 7.5e-3, 12.5e-3, 17.5e-3])
    assert ev[-1]["t"] == pytest.approx(20e-3)
    assert sq.Sequence.from_dict(s.to_dict()) == s
    assert sq.cpmg(0, 5e-3, 8, 1e-5).elements[1] == sq.Delay(5e-3)


def test_sequence_validation():
    with pytest.raises(ValueError):
        sq.Sequence("bad", (sq.Pulse("x", 1.0), sq.Delay(1e-3)))
    with pytest.raises(ValueError):
        sq.Pulse("z", 1.0)
