"""QRC physics tests — require QuTiP (``pip install ".[qrc]"``).

The whole module is skipped when QuTiP isn't installed. These assert
*qualitative* correctness on small, fast systems: the steady state is
physical, the fading-memory gate holds (plan §5.5), and memory capacity
does not shrink as qubits are added (the proof-of-benefit claim, §10.4).
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("qutip")

from app.qrc.config import QRCConfig, SimConfig, TrainingConfig, make_system_config


def _fast_cfg(n=3):
    return QRCConfig(
        system=make_system_config("spinq3") if n == 3 else make_system_config(n),
        sim=SimConfig(tau=0.03, n_virtual=10),
        training=TrainingConfig(
            device="cpu", washout=40, n_train=200, n_test=80, cv_folds=3,
        ),
    )


def test_steady_state_is_reached():
    from app.qrc.system import QRCSystem

    cfg = _fast_cfg()
    sysm = QRCSystem(cfg.system, cfg.sim)
    ss = sysm.steady_state()
    assert ss.tr() == pytest.approx(1.0, abs=1e-6)      # trace preserved
    assert (ss - ss.dag()).norm() < 1e-8                # Hermitian


def test_hamiltonian_is_hermitian():
    from app.qrc.system import QRCSystem

    sysm = QRCSystem(_fast_cfg().system, _fast_cfg().sim)
    assert (sysm.H - sysm.H.dag()).norm() < 1e-9


def test_action_backend_matches_propagator():
    """The exact sparse Krylov 'action' backend (used for N>=7) must agree
    with the dense 'propagator' backend on a small system where both run."""
    from app.qrc.benchmarks import run_memory_capacity

    common = dict(max_delay=8, n_steps=250)
    tr = TrainingConfig(device="cpu", washout=40, n_train=140, n_test=60,
                        cv_folds=3)
    mc = {}
    for mode in ("propagator", "action"):
        cfg = QRCConfig(
            system=make_system_config(4),
            sim=SimConfig(tau=0.03, n_virtual=8, evolution_mode=mode),
            training=tr,
        )
        mc[mode] = run_memory_capacity(cfg, **common).total_mc
    assert mc["action"] == pytest.approx(mc["propagator"], rel=1e-3)


def test_gpu_backend_matches_action():
    """The GPU Taylor-exponential backend must match the CPU-exact
    'action' backend. Skipped when no CUDA torch is available."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    from app.qrc.benchmarks import run_memory_capacity

    common = dict(max_delay=8, n_steps=250)
    tr = TrainingConfig(device="cpu", washout=40, n_train=140, n_test=60,
                        cv_folds=3)
    mc = {}
    for mode in ("action", "gpu"):
        cfg = QRCConfig(
            system=make_system_config(5),
            sim=SimConfig(tau=0.03, n_virtual=8, evolution_mode=mode),
            training=tr,
        )
        mc[mode] = run_memory_capacity(cfg, **common).total_mc
    assert mc["gpu"] == pytest.approx(mc["action"], rel=2e-3)


def test_memory_capacity_gate():
    """The core physics assertion: fading memory decays smoothly."""
    from app.qrc.benchmarks import run_memory_capacity

    res = run_memory_capacity(_fast_cfg(), max_delay=12, n_steps=400)
    assert res.per_delay[0] >= 0.7                       # recalls current input
    ds = [res.per_delay[d] for d in sorted(res.per_delay)]
    assert np.mean(np.diff(ds)) <= 0.02                 # non-increasing on avg
    assert res.per_delay[max(res.per_delay)] < res.per_delay[0]  # actually fades


@pytest.mark.slow
def test_scaling_memory_does_not_shrink_with_qubits():
    """Larger reservoir → at least as much memory capacity (§10.4)."""
    from app.qrc.benchmarks import scaling_study

    res = scaling_study(
        qubit_counts=(3, 5),
        sim=SimConfig(tau=0.03, n_virtual=10),
        training=TrainingConfig(device="cpu", washout=40, n_train=200,
                                n_test=80, cv_folds=3),
        max_delay=12,
        mc_steps=400,
        verbose=False,
    )
    assert res.memory_capacity[-1] >= res.memory_capacity[0]
