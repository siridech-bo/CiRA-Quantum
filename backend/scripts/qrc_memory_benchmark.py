"""Memory-capacity benchmark (STM vs Parity-Check) with time-multiplexing on the
NMR spin reservoir — an API-launchable runner.

Reproduces the Das-Giorgi-Zambrini (PRR 2026) STM vs Parity-Check protocol on our
coupled-spin reservoir, sweeping virtual nodes V. Time-multiplexing = sampling the
reservoir observables at V equally-spaced instants within each input window tau.

Two readouts (SOP sec 0 / sec 3.2 — every result is LABELED with its readout + D):
  observable   : single-qubit <sx,sy,sz> x V (the differentiable-path readout).
  fid_reduced  : physics-informed reduced FID. Set n_virtual=M dense samples in tau,
                 form the transverse-magnetization FID, and project onto the D_eff
                 analytic single-quantum lines (app.qrc.spectral_lines) -> 2*D_eff
                 real features (Re/Im at each known line frequency). Cheap, labeled,
                 and comparable to the 653-FID standard's physics content.

Tasks (fixed arcsin broadcast encoding, ridge readout):
  STM (linear memory): input u ~ U(0,1); target y^tau_i = u_{i-tau}.
  PC  (nonlinear):     input u ~ {0,1};  target y^tau_i = (sum_{j=1..tau} u_{i-j}) mod 2.
Capacity C(tau) = squared Pearson correlation on the test split; total = sum_tau C.

Forward-only, CPU, no GPU. Launchable via /api/qrc (task "memory") or standalone:
  python backend/scripts/qrc_memory_benchmark.py --run-dir <dir> --out <results.json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.qrc.spectral_lines import physics_informed_lines  # noqa: E402
from qrc_learnable_9spin import build_system, forward_features  # noqa: E402
from qrc_progress import ProgressLogger  # noqa: E402

RDT = torch.float64
CDT = torch.complex128
FIG_OUT = Path(__file__).resolve().parents[2] / "docs" / "QRC" / "figures" / "fig19_memory_multiplex.png"


def arcsin_broadcast(u, n):
    return [torch.full((n,), float(np.arcsin(np.sqrt(min(max(float(s), 0.0), 1.0)))), dtype=RDT)
            for s in u]


def stm_targets(u, taus):
    T = len(u)
    return {t: np.array([u[i - t] if i - t >= 0 else 0.0 for i in range(T)]) for t in taus}


def pc_targets(u, taus):
    T = len(u)
    out = {}
    for t in taus:
        y = np.zeros(T)
        for i in range(T):
            if i >= t:
                y[i] = int(sum(u[i - j] for j in range(1, t + 1))) % 2
        out[t] = y
    return out


def _sqcorr(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = (a.std() * b.std())
    return 0.0 if d < 1e-12 else float((a @ b / len(a)) ** 2 / (a.var() * b.var()))


def capacity(F, targets, washout, n_tr, alpha=1e-2):
    F = np.hstack([F, np.ones((len(F), 1))])[washout:]
    total, curve = 0.0, {}
    A = F[:n_tr].T @ F[:n_tr] + alpha * np.eye(F.shape[1])
    for tau, y in targets.items():
        yv = y[washout:]
        w = np.linalg.solve(A, F[:n_tr].T @ yv[:n_tr])
        C = _sqcorr(F[n_tr:] @ w, yv[n_tr:])
        curve[tau] = C; total += C
    return total, curve


# ---- readouts -------------------------------------------------------------


def features_observable(u, V, n_spins, coupling_scale):
    """Standard observable readout: <sx,sy,sz> x V (D = 3*n*V)."""
    sysm = build_system(n_spins, n_virtual=V, coupling_scale=coupling_scale)
    g = sysm.ensure_diff(device="cpu", cdtype=CDT)
    F = forward_features(arcsin_broadcast(u, sysm.n), sysm, g, "cpu", CDT)
    return F.detach().numpy().real, F.shape[1]


def features_fid_reduced(u, M, n_spins, coupling_scale, readout_qubits=None):
    """Physics-informed reduced FID: dense FID over M nodes in tau, projected onto
    the D_eff analytic lines. Returns (T, 2*D_eff) real features (Re/Im per line)."""
    sysm = build_system(n_spins, n_virtual=M, coupling_scale=coupling_scale)
    g = sysm.ensure_diff(device="cpu", cdtype=CDT)
    # transverse magnetization FID = sum_k <sx_k> + i<sy_k> over the M nodes.
    # forward_features returns [obs axes x spins x V]; reconstruct FID per step.
    F = forward_features(arcsin_broadcast(u, sysm.n), sysm, g, "cpu", CDT).detach().numpy().real
    n = sysm.n
    # feature layout (see FeatureExtractor.observable_names): v-major, then axis, then spin
    F = F.reshape(F.shape[0], M, 3, n)            # (T, V, axis{x,y,z}, spin)
    fid = F[:, :, 0, :].sum(axis=2) + 1j * F[:, :, 1, :].sum(axis=2)   # (T, M)
    ls = physics_informed_lines(sysm.system, readout_qubits)
    dwell = sysm.sim.tau / M
    phi = ls.projection(M, dwell)                 # (M, D_eff) complex
    X = fid @ phi                                 # (T, D_eff) complex
    feats = np.concatenate([X.real, X.imag], axis=1)   # (T, 2*D_eff)
    return feats, feats.shape[1], ls


def run(args, log: ProgressLogger):
    T, washout, n_tr = args.T, args.washout, args.n_train
    rng = np.random.default_rng(args.seed)
    u_stm = rng.random(T)
    u_pc = rng.integers(0, 2, T).astype(float)
    stm_taus = list(range(0, args.kmax + 1))
    pc_taus = list(range(1, max(2, args.kmax - 2)))
    sweep = args.Vs if args.readout == "observable" else args.Ms
    label = "V" if args.readout == "observable" else "M"

    log.event("config", f"readout={args.readout} system={args.n_spins}-spin "
              f"coupling={args.coupling_scale}x T={T} train={n_tr} sweep={sweep}")
    print(f"{args.n_spins}-qubit reservoir | readout={args.readout} | T={T} "
          f"washout={washout} train={n_tr} test={T-washout-n_tr}")
    print(f"{label:>4s} {'D':>6s} {'STM totMC':>10s} {'PC totCap':>10s}")

    results = {}
    d_eff = None
    total_steps = len(sweep)
    for i, s in enumerate(sweep):
        log.status(phase=f"{args.readout} {label}={s}", step=i, total=total_steps,
                   eta_s=None)
        if args.readout == "observable":
            F_stm, D = features_observable(u_stm, s, args.n_spins, args.coupling_scale)
            F_pc, _ = features_observable(u_pc, s, args.n_spins, args.coupling_scale)
        else:
            F_stm, D, ls = features_fid_reduced(u_stm, s, args.n_spins, args.coupling_scale)
            F_pc, _, _ = features_fid_reduced(u_pc, s, args.n_spins, args.coupling_scale)
            d_eff = ls.d_eff
        stm_tot, stm_c = capacity(F_stm, stm_targets(u_stm, stm_taus), washout, n_tr)
        pc_tot, pc_c = capacity(F_pc, pc_targets(u_pc, pc_taus), washout, n_tr)
        results[int(s)] = {"D": int(D), "stm_curve": {int(k): float(v) for k, v in stm_c.items()},
                           "pc_curve": {int(k): float(v) for k, v in pc_c.items()},
                           "stm_totMC": float(stm_tot), "pc_totCap": float(pc_tot)}
        log.result(f"{label}={s}", {"D": int(D), "STM_totMC": round(stm_tot, 3),
                                    "PC_totCap": round(pc_tot, 3)})
        print(f"{s:4d} {D:6d} {stm_tot:10.3f} {pc_tot:10.3f}")

    payload = {
        "experiment": "memory_benchmark",
        "readout": args.readout,
        "D_eff": d_eff,
        "system": f"{args.n_spins}-spin",
        "coupling_scale": args.coupling_scale,
        "T": T, "washout": washout, "n_train": n_tr,
        "sweep_key": label, "sweep": [int(s) for s in sweep],
        "stm_taus": stm_taus, "pc_taus": pc_taus,
        "results": results,
    }
    _figure(results, stm_taus, pc_taus, sweep, label, args.readout)
    return payload


def _figure(results, stm_taus, pc_taus, sweep, label, readout):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    palette = ["#9aa7b4", "#4da3ff", "#e07b3c", "#cb4b4b", "#7bd88f", "#c98bdb"]
    colors = {s: palette[i % len(palette)] for i, s in enumerate(sweep)}
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.4))
    for s in sweep:
        r = results[int(s)]
        axL.plot(stm_taus, [r["stm_curve"][t] for t in stm_taus], marker="o", color=colors[s],
                 label=f"{label}={s} (totMC {r['stm_totMC']:.2f})")
        axR.plot(pc_taus, [r["pc_curve"][t] for t in pc_taus], marker="s", color=colors[s],
                 label=f"{label}={s} (totCap {r['pc_totCap']:.2f})")
    axL.set_xlabel("delay tau"); axL.set_ylabel("capacity C (r2)")
    axL.set_title("A - STM (linear memory)"); axL.legend(fontsize=8); axL.grid(alpha=0.25)
    axR.set_xlabel("delay tau"); axR.set_ylabel("capacity C (r2)")
    axR.set_title("B - Parity-Check (nonlinear memory)"); axR.legend(fontsize=8); axR.grid(alpha=0.25)
    fig.suptitle(f"Time-multiplexing on the NMR spin reservoir (readout={readout})",
                 y=1.02, fontsize=11)
    fig.tight_layout()
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_OUT, dpi=140, bbox_inches="tight")
    print(f"\nwrote {FIG_OUT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=None, help="progress/events dir (launcher-managed)")
    ap.add_argument("--out", default=None, help="results JSON path (launcher-managed)")
    ap.add_argument("--readout", choices=["observable", "fid_reduced"], default="observable")
    ap.add_argument("--n-spins", type=int, default=6)
    ap.add_argument("--coupling-scale", type=float, default=1.0)
    ap.add_argument("--T", type=int, default=800)
    ap.add_argument("--washout", type=int, default=80)
    ap.add_argument("--n-train", type=int, default=420)
    ap.add_argument("--kmax", type=int, default=8, help="max STM delay tau")
    ap.add_argument("--Vs", type=int, nargs="+", default=[1, 2, 5, 10],
                    help="virtual-node sweep for the observable readout")
    ap.add_argument("--Ms", type=int, nargs="+", default=[64, 128, 256],
                    help="FID sample sweep for the fid_reduced readout")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    run_dir = args.run_dir or str(Path(__file__).resolve().parents[1] / "artifacts" / "qrc_runs" / "memory_local")
    log = ProgressLogger(run_dir=run_dir, title=f"QRC memory benchmark ({args.readout})")
    try:
        payload = run(args, log)
        out = args.out or str(Path(run_dir) / "results.json")
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        log.event("results", f"wrote {out}")
        log.close("done")
    except Exception as e:  # noqa: BLE001
        log.event("error", f"{type(e).__name__}: {e}")
        log.close("error")
        raise


if __name__ == "__main__":
    main()
