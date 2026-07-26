"""Tests for ``app.qrc`` — the Quantum Reservoir Computing simulator.

Two tiers:

* NumPy-only tests (config presets, encoding angle math, metrics) that
  run everywhere — these guard the lightweight surface the rest of the
  backend can import without the heavy quantum stack.
* Physics tests gated behind ``importorskip("qutip")`` — the
  fading-memory gate (the one result that has to be right, plan §5.5) and
  a tiny scaling sanity check.

The physics tests use small systems / short sequences so they stay fast;
they assert *qualitative* correctness (memory decays smoothly; steady
state is reached), not exact published numbers.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.qrc.config import (
    QRCConfig,
    SimConfig,
    crotonic_acid_like,
    generic_nqubit,
    make_system_config,
    spinq_3qubit,
)
from app.qrc.encoding import angle_for
from app.qrc.tasks import delayed_targets, narma_sequence
from app.qrc.utils import nmse, normalize, r2_score, squared_correlation

# --------------------------------------------------------------------------
# NumPy-only surface
# --------------------------------------------------------------------------


def test_presets_are_well_formed():
    for cfg in (spinq_3qubit(), crotonic_acid_like(), generic_nqubit(5)):
        n = cfg.n_qubits
        assert len(cfg.chemical_shifts) == n
        assert len(cfg.t1) == n == len(cfg.t2)
        j = np.asarray(cfg.j_coupling)
        assert j.shape == (n, n)
        assert np.allclose(j, j.T)               # symmetric
        assert np.allclose(np.diag(j), 0.0)      # no self-coupling


def test_generic_nqubit_scales():
    for n in (1, 3, 9, 12):
        cfg = generic_nqubit(n)
        assert cfg.n_qubits == n


def test_make_system_config_dispatch():
    assert make_system_config("spinq3").n_qubits == 3
    assert make_system_config(7).n_qubits == 7
    with pytest.raises(KeyError):
        make_system_config("nope")


def test_repro_hash_is_stable_and_sensitive():
    a = QRCConfig(system=spinq_3qubit())
    b = QRCConfig(system=spinq_3qubit())
    assert a.repro_hash() == b.repro_hash()
    c = QRCConfig(system=spinq_3qubit(), sim=SimConfig(tau=0.05))
    assert c.repro_hash() != a.repro_hash()


def test_encoding_angle_ranges():
    # Paper 4 arcsin(√s): [0, π/2]; s=1 → π/2, s=0 → 0.
    assert angle_for("arcsin_sqrt", 0.0) == pytest.approx(0.0)
    assert angle_for("arcsin_sqrt", 1.0) == pytest.approx(np.pi / 2)
    # Paper 3 arccos(2s-1): [0, π]; s=0.5 → π/2.
    assert angle_for("arccos", 0.5) == pytest.approx(np.pi / 2)
    # Linear θ = sπ.
    assert angle_for("linear", 0.5) == pytest.approx(np.pi / 2)


def test_delayed_targets_shift():
    u = np.arange(10.0)
    tgt = delayed_targets(u, max_delay=3)
    assert np.array_equal(tgt[0], u)
    assert tgt[2][2] == u[0] and tgt[2][5] == u[3]   # y_k = u_{k-2}


def test_metrics_sane():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert r2_score(y, y) == pytest.approx(1.0)
    assert squared_correlation(y, 2 * y) == pytest.approx(1.0)
    assert nmse(y, y) == pytest.approx(0.0)
    s, lo, hi = normalize(np.array([10.0, 40.0]))
    assert (lo, hi) == (10.0, 40.0) and s[0] == 0.0 and s[1] == 1.0


def test_narma_finite():
    u, y = narma_sequence(200, order=2, seed=0)
    assert np.all(np.isfinite(y)) and len(u) == len(y) == 200
