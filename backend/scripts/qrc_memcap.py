"""Memory Capacity / Information-Processing Capacity for encoding comparison.

Weather-R² is a poor metric to *screen* encodings — it needs near-full fidelity.
Memory capacity (MC) is the standard intrinsic reservoir metric: drive the
reservoir with a random i.i.d. input and measure how well a linear readout can
reconstruct **delayed** (linear MC) and **nonlinearly transformed** (nonlinear
MC / IPC) versions of that input. It characterises the reservoir's memory +
nonlinearity directly, needs no downstream task, and is far less fidelity-fragile.

This module has two independent pieces (so the expensive part is never coupled
to the metric — per the "persist the waveform" rule):

  * :func:`memory_capacity` — **offline**: given a saved reservoir readout matrix
    ``X`` and the driving input ``u``, compute linear + quadratic MC with a
    train/test split and a PCA-conditioned readout (so p<n, no overfit inflation).
  * :func:`run_encoding_memcap` — drives the reservoir once with a random input
    for a given encoding, **saves the raw waveform trace**, then scores MC.

Self-test: ``python scripts/qrc_memcap.py --selftest`` (no GPU) checks the MC
math on a synthetic delay-line reservoir with known capacity.
"""
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import Legendre
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from app.qrc.encoding import Encoder
from app.qrc.evolution import Reservoir
from app.qrc.feature_methods import build_features
from app.qrc.features import FeatureConfig
from app.qrc.system import QRCSystem
from qrc_gen_traces import _CKPT_ROOT, StreamingTrace, _ckpt_hash, _resolve_system
from qrc_phase2 import (  # reuse encoding configs + sweep plumbing
    ENCODINGS,
    QUICK_ENCODINGS,
    _RESULTS_ROOT,
    _sweep_key,
    build_cfg,
)
from qrc_progress import ProgressLogger


def _sq_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Squared Pearson correlation (the per-target capacity, in [0, 1])."""
    yt = y_true - y_true.mean()
    yp = y_pred - y_pred.mean()
    den = float(np.sqrt((yt @ yt) * (yp @ yp)))
    if den < 1e-12:
        return 0.0
    c = float(yt @ yp) / den
    return c * c


def _legendre_target(u_shift: np.ndarray, degree: int) -> np.ndarray:
    """Degree-``degree`` Legendre polynomial of the (assumed in [-1,1]) input —
    the orthogonal basis used for information-processing capacity."""
    coeffs = [0.0] * degree + [1.0]
    return Legendre(coeffs)(u_shift)


def memory_capacity(
    X: np.ndarray, u: np.ndarray, *, kmax: int = 30, washout: int = 50,
    n_train: int | None = None, degrees=(1, 2), n_pca: int = 50, seed: int = 0,
) -> dict:
    """Linear + nonlinear memory capacity from a reservoir readout matrix.

    ``X`` is ``[n_steps, n_features]`` (e.g. the FID magnitudes), ``u`` the
    driving input in roughly ``[-1, 1]``. For each delay ``k`` and Legendre
    degree ``d`` we fit a linear readout on the train split to reconstruct
    ``P_d(u[t-k])`` and measure its **test** squared-correlation (the capacity).
    Capacities are summed over delays. A PCA fit on the train split conditions
    the readout so p<n (in-sample MC would be overfit-inflated)."""
    X = np.asarray(X, dtype=float)
    u = np.asarray(u, dtype=float).ravel()
    n = X.shape[0]
    if n_train is None:
        n_train = int((n - washout) * 0.7)
    tr = np.arange(washout, washout + n_train)
    te = np.arange(washout + n_train, n)

    # PCA-condition the readout (fit on train only).
    k_pca = int(min(n_pca, X.shape[1], len(tr) - 1))
    pca = PCA(n_components=k_pca, random_state=seed).fit(X[tr])
    Xtr, Xte = pca.transform(X[tr]), pca.transform(X[te])

    caps = {int(d): [] for d in degrees}
    for d in degrees:
        for k in range(1, kmax + 1):
            # target at time t is P_d(u[t-k]); valid where t-k >= 0
            ytr = _legendre_target(u[tr - k], d)
            yte = _legendre_target(u[te - k], d)
            model = LinearRegression().fit(Xtr, ytr)
            caps[int(d)].append(_sq_corr(yte, model.predict(Xte)))

    lin = float(np.sum(caps[1])) if 1 in caps else 0.0
    nonlin = float(sum(np.sum(caps[d]) for d in degrees if d >= 2))
    return {
        "linear_MC": lin,
        "nonlinear_MC": nonlin,
        "total_MC": lin + nonlin,
        "kmax": kmax, "n_pca": k_pca, "degrees": list(degrees),
        "caps_by_degree": {str(d): [float(c) for c in caps[d]] for d in caps},
    }


KMAX = 30


def run_encoding_memcap(cfg, label, *, kmax=KMAX, logger=None, phase_label=""):
    """Drive the reservoir once with a random input for ``cfg``'s encoding, SAVE
    the raw waveform trace, then score memory capacity. The reservoir input is
    ``u01 ∈ [0,1]`` (what the encoder needs); MC targets use ``u = 2·u01−1``."""
    tr = cfg.training
    n_steps = tr.washout + tr.n_train + tr.n_test
    system = QRCSystem(cfg.system, cfg.sim)
    encoder = Encoder(system, cfg.encoding)
    res = Reservoir(system, encoder, FeatureConfig(readout="fid", n_peaks=653))
    u01 = np.random.default_rng(cfg.sim.seed).uniform(0.0, 1.0, n_steps)

    chash = _ckpt_hash(cfg, f"memcap::{label}")
    chk = StreamingTrace(_CKPT_ROOT / chash, n_steps, cfg.sim.fid_points, chash,
                         total=n_steps, logger=logger, phase=phase_label or label,
                         fid_dwell=cfg.sim.fid_dwell)
    start_step, state0 = chk.try_resume()
    if start_step:
        print(f"  [{label}] resuming from step {start_step}/{n_steps}", flush=True)
    res.run(u01, fid_cb=chk, start_step=start_step, state0=state0,
            checkpoint_cb=chk.checkpoint)
    fids = chk.finalize()

    # PERSIST the waveform (never discard reservoir compute) — a random-input
    # trace so MC / any other intrinsic metric can be recomputed offline.
    # Display name distinguishes variants of the same function (fn stays
    # 'arcsin_sqrt'): '_pa' marks R_z(2πs)·R_x(θ) enabled; '_protons' marks a
    # spin-subset (target_qubits) pulse. Keeps trace filenames/results distinct
    # from the all-spins baseline so the judge separates them.
    disp = (cfg.encoding.fn
            + ("_pa" if cfg.encoding.phase_amplitude else "")
            + ("_protons" if cfg.encoding.target_qubits else ""))
    traces_dir = _CKPT_ROOT.parent
    traces_dir.mkdir(parents=True, exist_ok=True)
    trace_path = traces_dir / f"memcap_{disp}_{chash}.npz"
    meta = {"task": "memcap", "fn": disp, "base_fn": cfg.encoding.fn,
            "phase_amplitude": cfg.encoding.phase_amplitude, "kmax": kmax,
            "encoding": asdict(cfg.encoding), "sim": asdict(cfg.sim),
            "training": asdict(cfg.training), "system": asdict(cfg.system)}
    np.savez_compressed(
        trace_path, fids=fids, fid_dwell=np.float64(cfg.sim.fid_dwell),
        task=np.str_("memcap"), split=np.asarray([tr.washout, tr.n_train, tr.n_test], np.int64),
        seed=np.int64(cfg.sim.seed), memcap_input=np.asarray(u01, np.float64),
        meta=np.str_(json.dumps(meta, default=float)),
    )
    print(f"[memcap] saved waveform trace -> {trace_path.name}", flush=True)
    chk.cleanup()

    X, _ = build_features(fids, "magnitude653", n_peaks=653, select="first")
    mc = memory_capacity(X, 2.0 * u01 - 1.0, kmax=kmax, washout=tr.washout,
                         n_train=tr.n_train, seed=cfg.sim.seed)
    print(f"[memcap] {label:<24} linMC={mc['linear_MC']:.2f}  "
          f"nlMC={mc['nonlinear_MC']:.2f}  totMC={mc['total_MC']:.2f}", flush=True)
    return {"label": label, "fn": disp,
            "phase_amplitude": cfg.encoding.phase_amplitude,
            "n_steps": n_steps, "fid_points": cfg.sim.fid_points,
            "trace": trace_path.name,
            "linear_MC": mc["linear_MC"], "nonlinear_MC": mc["nonlinear_MC"],
            "total_MC": mc["total_MC"], "n_pca": mc["n_pca"]}


def _plan(experiment, fidelity, seed, system):
    if experiment == "phaseamp":
        # 2.2: phase-amplitude encoding ON for the best fixed encoding
        # (arcsin_sqrt). The phase-amp OFF baseline is the existing
        # ``memcap_arcsin_sqrt`` trace, so only this one new run is needed;
        # the judge compares both saved waveforms.
        return [("mc_arcsin_sqrt_pa",
                 build_cfg("arcsin_sqrt", fidelity, seed=seed, system=system,
                           phase_amplitude=True))]
    if experiment == "protons":
        # 2.3: encode the input only into the proton spins (labels starting
        # "H") instead of all spins — the protons are the detected readout
        # nuclei; the carbons are a bath. The all-spins baseline is the
        # existing ``memcap_arcsin_sqrt`` trace, so only this one new run is
        # needed; the judge compares both saved waveforms.
        sys_cfg = _resolve_system(system)
        protons = [i for i, lbl in enumerate(sys_cfg.labels)
                   if lbl.upper().startswith("H")]
        if not protons or len(protons) >= len(sys_cfg.labels):
            raise SystemExit(
                f"system '{system}' has no proton subset to target "
                f"(labels={sys_cfg.labels})")
        return [("mc_arcsin_sqrt_protons",
                 build_cfg("arcsin_sqrt", fidelity, seed=seed, system=system,
                           target_qubits=protons))]
    fns = QUICK_ENCODINGS if experiment == "quick" else ENCODINGS
    return [(f"mc_{fn}", build_cfg(fn, fidelity, seed=seed, system=system)) for fn in fns]


def _write_manifest(out: Path, results: list[dict]) -> None:
    """Link this run to its saved waveform traces so the FID viewer can browse
    them after completion (run-dir has no trace.npz; the traces live in
    artifacts/traces/ under hashed names)."""
    try:
        (out / "traces.json").write_text(json.dumps({"traces": [
            {"fn": r["fn"], "name": r.get("trace")} for r in results if r.get("trace")
        ]}, indent=2), encoding="utf-8")
    except Exception:  # noqa: BLE001 - manifest is best-effort
        pass


def sweep_main():
    ap = argparse.ArgumentParser(description="QRC memory-capacity encoding sweep")
    ap.add_argument("--experiment", choices=["all", "quick", "phaseamp", "protons"], default="all")
    ap.add_argument("--fidelity", default="screen")
    ap.add_argument("--system", default="crotonic9_paper4")
    ap.add_argument("--kmax", type=int, default=KMAX)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="artifacts/qrc_memcap")
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()

    out = Path(args.run_dir) if args.run_dir else Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    logger = ProgressLogger(run_dir=args.run_dir, title=f"memcap {args.experiment}") if args.run_dir else None
    cache = _RESULTS_ROOT / ("mc_" + _sweep_key(args.experiment, args.fidelity, args.seed, args.system))
    cache.mkdir(parents=True, exist_ok=True)

    plan = _plan(args.experiment, args.fidelity, args.seed, args.system)
    n = len(plan)
    print(f"memcap {args.experiment} @ {args.fidelity}: {n} encodings", flush=True)
    if logger:
        logger.event("config", f"memcap {args.experiment} @ {args.fidelity}: {n} encodings")

    results = []
    try:
        for i, (label, cfg) in enumerate(plan):
            cpath, rpath = cache / f"{label}.json", out / f"result_{label}.json"
            if cpath.exists():
                res = json.loads(cpath.read_text(encoding="utf-8"))
                rpath.write_text(json.dumps(res, indent=2), encoding="utf-8")
                results.append(res)
                _write_manifest(out, results)
                print(f"[memcap] {label}: cached, skipping", flush=True)
                continue
            pl = f"{i + 1}/{n} {cfg.encoding.fn}"
            if logger:
                logger.event("encoding", f"encoding {pl}", phase=pl)
            res = run_encoding_memcap(cfg, label, kmax=args.kmax, logger=logger, phase_label=pl)
            res["fidelity"] = args.fidelity
            blob = json.dumps(res, indent=2)
            cpath.write_text(blob, encoding="utf-8")
            rpath.write_text(blob, encoding="utf-8")
            results.append(res)
            _write_manifest(out, results)
    except Exception as exc:  # noqa: BLE001
        if logger:
            logger.event("error", f"memcap failed: {type(exc).__name__}: {exc}")
        raise

    order = sorted(range(len(results)), key=lambda i: -results[i]["total_MC"])
    summary = {"generated_utc": datetime.now(UTC).isoformat(), "experiment": args.experiment,
               "fidelity": args.fidelity, "seed": args.seed, "kmax": args.kmax,
               "ranking": [results[i]["label"] for i in order], "results": results}
    (out / "memcap_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nRanking by total memory capacity:", flush=True)
    for i in order:
        r = results[i]
        print(f"  {r['fn']:<14} totMC={r['total_MC']:.2f} (lin {r['linear_MC']:.2f} + nl {r['nonlinear_MC']:.2f})", flush=True)
    _memcap_figure(results, out / "memcap_encoding.png")
    shutil.rmtree(cache, ignore_errors=True)
    if logger:
        logger.status(phase="done", eta_s=0)
        logger.close("done", f"memcap {args.experiment} complete")
    print(f"\nDone. Outputs in {out.resolve()}")


def _memcap_figure(results, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    order = sorted(results, key=lambda r: -r["total_MC"])
    fns = [r["fn"] for r in order]
    lin = [r["linear_MC"] for r in order]
    nl = [r["nonlinear_MC"] for r in order]
    x = np.arange(len(fns))
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.bar(x, lin, label="linear MC", color="#2563eb")
    ax.bar(x, nl, bottom=lin, label="nonlinear MC", color="#16a34a")
    ax.set_xticks(x); ax.set_xticklabels(fns, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("memory capacity"); ax.set_title("Phase-2: encoding memory capacity (intrinsic)")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[memcap] wrote {path}", flush=True)


# --------------------------------------------------------------------------
# Self-test (no GPU): a synthetic delay-line reservoir has linear MC ≈ its
# number of independent delay taps, and ~0 nonlinear MC.
# --------------------------------------------------------------------------
def _selftest() -> None:
    rng = np.random.default_rng(0)
    n, taps = 3000, 8
    u = rng.uniform(-1, 1, n)
    # X[t] = [u[t-1], ..., u[t-taps]] + small noise — a pure linear delay line.
    X = np.zeros((n, taps))
    for k in range(1, taps + 1):
        X[k:, k - 1] = u[:-k]
    X += 0.01 * rng.standard_normal(X.shape)
    r = memory_capacity(X, u, kmax=20, n_pca=taps, degrees=(1, 2))
    print(f"[selftest] delay-line taps={taps} -> linear_MC={r['linear_MC']:.2f} "
          f"(expect ~{taps}), nonlinear_MC={r['nonlinear_MC']:.2f} (expect ~0)")
    ok = abs(r["linear_MC"] - taps) < 1.5 and r["nonlinear_MC"] < 1.5
    print("[selftest]", "PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        _selftest()
    else:
        sweep_main()
