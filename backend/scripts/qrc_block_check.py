"""Validate the block-structured propagator (app/qrc/block_propagator.py).

CPU-only, small systems. Checks, in complex128:
  1. generator: the reassembled sub-block generators equal QuTiP's full
     Liouvillian entry-for-entry (both dephasing models);
  2. step: block-mode ``step_diff`` (features + end state) matches the
     existing exact dense-propagator path over a multi-step sequence, for the
     dense pulse U and for the per-spin 2x2 factor pulse;
  3. gradient: d(loss)/d(angles) through the sequence matches.

Run: python backend/scripts/qrc_block_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.qrc.block_propagator import BlockPropagator  # noqa: E402
from app.qrc.config import SimConfig, make_system_config  # noqa: E402
from app.qrc.system import QRCSystem  # noqa: E402
from qrc_learnable_9spin import _rx1, init_state, per_spin_pulse  # noqa: E402

C128 = torch.complex128


def check_generator(sysc, model):
    sim = SimConfig(tau=0.03, n_virtual=4, dephasing_model=model)
    sysm = QRCSystem(sysc, sim)
    Lq = np.asarray(sysm._ensure_liouvillian().todense())
    bp = BlockPropagator(sysc, sim, cdtype=C128)
    d = bp.d
    Lb = np.zeros((d * d, d * d), dtype=complex)
    for g in bp.groups:
        idx = (g["A"] ^ g["P"][:, None]) * d + g["A"]
        Lg = bp.generator(g).numpy()
        for c in range(idx.shape[0]):
            Lb[np.ix_(idx[c], idx[c])] = Lg[c]
    err = np.abs(Lb - Lq).max() / np.abs(Lq).max()
    print(f"  generator n={sysc.n_qubits} {model:8s}: max rel err {err:.2e}")
    return err < 1e-12


def run_seq(sysm, g, thetas, factors):
    vec = init_state(sysm, "cpu", C128)
    rows = []
    for th in thetas:
        U = [_rx1(t, C128) for t in th] if factors else per_spin_pulse(th, sysm.n, C128)
        f, vec = sysm.step_diff(vec, U, g)
        rows.append(f)
    return torch.stack(rows), vec


def check_step(sysc, steps=6, V=8):
    sim = SimConfig(tau=0.03, n_virtual=V)
    sysm = QRCSystem(sysc, sim)
    gd = sysm.ensure_diff(device="cpu", cdtype=C128, dense=True)
    gb = sysm.ensure_diff(device="cpu", cdtype=C128, block=True)
    rng = np.random.default_rng(1)
    th0 = torch.tensor(rng.uniform(0, np.pi, (steps, sysm.n)), dtype=torch.float64)
    ok = True
    for factors in (False, True):
        thd = th0.clone().requires_grad_(True)
        thb = th0.clone().requires_grad_(True)
        Fd, vd = run_seq(sysm, gd, thd, False)
        Fb, vb = run_seq(sysm, gb, thb, factors)
        (Fd ** 2).sum().backward()
        (Fb ** 2).sum().backward()
        ef = (Fb - Fd).abs().max() / Fd.abs().max()
        ev = (vb - vd).abs().max() / vd.abs().max()
        eg = (thb.grad - thd.grad).abs().max() / thd.grad.abs().max()
        tag = "2x2-factor pulse" if factors else "dense-U pulse   "
        print(f"  step n={sysm.n} V={V} {tag}: feats {ef:.1e}  state {ev:.1e}  grad {eg:.1e}")
        ok &= bool(ef < 1e-10 and ev < 1e-10 and eg < 1e-10)
    return ok


def main():
    ok = True
    print("1) generator vs QuTiP Liouvillian")
    for sysc in (make_system_config("spinq3"), make_system_config(4), make_system_config(5)):
        for model in ("physical", "simple"):
            ok &= check_generator(sysc, model)
    print("2+3) block step vs dense-propagator step (features, state, gradient)")
    for sysc in (make_system_config("spinq3"), make_system_config(4), make_system_config(5)):
        ok &= check_step(sysc)
    print("ALL PASS" if ok else "FAILURES ABOVE")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
