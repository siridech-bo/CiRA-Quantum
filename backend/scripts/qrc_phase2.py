"""QRC Phase-2 — encoding sweep (re-evolves the reservoir per setting).

Unlike Phase 1 (which re-scores one cached trace), each encoding choice changes
the reservoir dynamics, so every setting needs its *own* reservoir pass. This
runner sweeps encoding settings, holding everything else fixed, and scores each
with the same rigorous protocol as Phase-1 v2 (magnitude-653 readout, per-fit
RidgeCV, blocked time-respecting CV → mean ± std, all horizons).

Experiments (``--experiment``):
  * ``2.1`` — the 7 encoding functions (arcsin_sqrt … exponential).
  * ``2.2`` — phase-amplitude encoding off vs on (with the Paper-4 arcsin_sqrt).
  * ``all`` — 2.1 then 2.2.

Robustness: each setting's trace streams to disk and resumes on crash (reuses
``StreamingTrace``); the *sweep* is resumable too — a setting whose result JSON
already exists is skipped. Fidelity presets keep the sweep affordable:
``--fidelity screen`` (fast, for ranking) vs ``full`` (confirm the winner).

Run from ``backend``::

    python scripts/qrc_phase2.py --experiment 2.1 --fidelity screen \
        --out-dir artifacts/qrc_phase2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from app.qrc.benchmarks import run_weather_reservoir
from app.qrc.config import EncodingConfig, QRCConfig, SimConfig, TrainingConfig
from app.qrc.feature_methods import build_features
from app.qrc.features import FeatureConfig
from app.qrc.tasks import load_weather
from app.qrc.utils import normalize
from qrc_gen_traces import (
    _CKPT_ROOT,
    StreamingTrace,
    _ckpt_hash,
    _resolve_system,
    _weather_indices,
)
from qrc_phase1_v2 import blocked_folds, ridge_r2
from qrc_progress import ProgressLogger

ENCODINGS = ["arcsin_sqrt", "arccos", "linear", "sinusoidal",
             "logarithmic", "polynomial", "exponential"]
# A representative 3-way subset for a quick sanity sweep (baseline + two very
# different curvatures): Paper-4 arcsin√, plain linear, mid-emphasising sinusoidal.
QUICK_ENCODINGS = ["arcsin_sqrt", "linear", "sinusoidal"]
HORIZONS = [1, 10, 20, 30, 45]

# Fidelity presets. 'screen' ranks settings affordably (~40 min/setting on the
# 9-spin GPU); 'full' confirms the winner at Paper-4 resolution (~hours).
FIDELITY = {
    # ~1.5–1.8 h for 3 encodings at the measured ~10 s/step — a fast sanity
    # ranking (small blocked-CV, enough to see the spread across encodings).
    "quick": {"fid_points": 512, "n_virtual": 10, "splits": (30, 110, 70)},
    "screen": {"fid_points": 512, "n_virtual": 10, "splits": (100, 300, 200)},
    "full": {"fid_points": 2048, "n_virtual": 25, "splits": (374, 600, 500)},
    "tiny": {"fid_points": 64, "n_virtual": 4, "splits": (10, 20, 10)},  # smoke test
}


def build_cfg(fn, fidelity, *, phase_amplitude=False, target_qubits=None,
              seed=42, system="crotonic9_paper4", tau=0.03) -> QRCConfig:
    f = FIDELITY[fidelity]
    wo, ntr, nte = f["splits"]
    return QRCConfig(
        system=_resolve_system(system),
        sim=SimConfig(tau=tau, n_virtual=f["n_virtual"], evolution_mode="gpu",
                      seed=seed, fid_points=f["fid_points"]),
        training=TrainingConfig(washout=wo, n_train=ntr, n_test=nte, seed=seed),
        encoding=EncodingConfig(fn=fn, phase_amplitude=phase_amplitude,
                                target_qubits=list(target_qubits or [])),
    )


def _weather_series(cfg, weather_train, weather_test):
    tr = cfg.training
    n_steps = tr.washout + tr.n_train + tr.n_test
    n_days = n_steps + max(HORIZONS)
    raw = load_weather(weather_train)
    if weather_test and Path(weather_test).exists():
        raw = np.vstack([raw, load_weather(weather_test)])
    cols = [normalize(raw[:, j])[0] for j in range(raw.shape[1])]
    weather = np.column_stack(cols)[:n_days]
    if len(weather) < n_days:
        raise ValueError(f"weather series too short: need {n_days}, have {len(weather)}")
    return weather, n_steps


def run_setting(cfg: QRCConfig, label: str, weather_train, weather_test,
                *, logger: ProgressLogger | None = None, phase_label: str = "") -> dict:
    """Evolve the reservoir for one encoding setting and score it (blocked CV)."""
    weather, n_steps = _weather_series(cfg, weather_train, weather_test)
    proton_idx, carbon_idx = _weather_indices(cfg.system.n_qubits)

    # ``_ckpt_hash`` keys on system/sim/training; fold the encoding label into
    # the task string so different encodings get distinct, resumable checkpoints.
    chash = _ckpt_hash(cfg, f"phase2::{label}")
    chk = StreamingTrace(_CKPT_ROOT / chash, n_steps, cfg.sim.fid_points, chash,
                         total=n_steps, logger=logger, phase=phase_label or label,
                         fid_dwell=cfg.sim.fid_dwell)
    start_step, state0 = chk.try_resume()
    if start_step:
        print(f"  [{label}] resuming from step {start_step}/{n_steps}", flush=True)
    run_weather_reservoir(
        cfg, weather, n_steps, feature_cfg=FeatureConfig(readout="fid", n_peaks=653),
        proton_idx=proton_idx, carbon_idx=carbon_idx, fid_cb=chk,
        start_step=start_step, state0=state0, checkpoint_cb=chk.checkpoint,
    )
    fids = chk.finalize()
    tr = cfg.training

    # PERSIST the raw waveform — NEVER discard reservoir compute. Save a full
    # §1-schema trace .npz (like weather_full.npz) so ANY metric (R², memory
    # capacity, NARMA, other readouts) can be computed offline later WITHOUT
    # re-evolving. The reservoir pass is the expensive part; keep its output.
    traces_dir = _CKPT_ROOT.parent
    traces_dir.mkdir(parents=True, exist_ok=True)
    trace_path = traces_dir / f"enc_{cfg.encoding.fn}_{chash}.npz"
    meta = {"task": "weather", "fn": cfg.encoding.fn,
            "encoding": asdict(cfg.encoding), "sim": asdict(cfg.sim),
            "training": asdict(cfg.training), "system": asdict(cfg.system),
            "horizons": HORIZONS}
    np.savez_compressed(
        trace_path,
        fids=fids, fid_dwell=np.float64(cfg.sim.fid_dwell), task=np.str_("weather"),
        split=np.asarray([tr.washout, tr.n_train, tr.n_test], dtype=np.int64),
        seed=np.int64(cfg.sim.seed), weather_norm=np.asarray(weather, dtype=np.float64),
        horizons=np.asarray(HORIZONS, dtype=np.int64),
        meta=np.str_(json.dumps(meta, default=float)),
    )
    print(f"[phase2] saved waveform trace -> {trace_path.name}", flush=True)
    chk.cleanup()  # remove the checkpoint SCRATCH only — the saved trace stays

    X, _ = build_features(fids, "magnitude653", n_peaks=653, select="first")
    temp = weather[:, 0]
    folds = blocked_folds(tr.washout, n_steps, 5)
    by_h = {}
    for h in HORIZONS:
        y = temp[np.arange(n_steps) + h]
        cv = [ridge_r2(X[a], y[a], X[b], y[b]) for a, b in folds]
        by_h[f"h{h}"] = {"cv_mean": float(np.mean(cv)), "cv_std": float(np.std(cv))}
    row = "  ".join(f"h{h}={by_h[f'h{h}']['cv_mean']:.3f}±{by_h[f'h{h}']['cv_std']:.3f}"
                    for h in HORIZONS)
    print(f"[phase2] {label:<26} {row}", flush=True)
    return {"label": label, "fn": cfg.encoding.fn,
            "phase_amplitude": cfg.encoding.phase_amplitude,
            "target_qubits": list(cfg.encoding.target_qubits),
            "n_steps": n_steps, "fid_points": cfg.sim.fid_points,
            "trace": trace_path.name, "by_horizon": by_h}


# Config-keyed cache of finished per-encoding results, so a crash/reboot +
# relaunch (which gets a fresh run-dir) skips already-finished encodings rather
# than re-running hours of them. Sibling to the trace-gen checkpoint root.
_RESULTS_ROOT = _CKPT_ROOT.parent / ".phase2_results"


def _write_trace_manifest(out: Path, results: list[dict]) -> None:
    """Link this run to its saved waveform traces (run-dir/traces.json) so the
    FID viewer can browse them per encoding after completion."""
    try:
        (out / "traces.json").write_text(json.dumps({"traces": [
            {"fn": r["fn"], "name": r.get("trace")} for r in results if r.get("trace")
        ]}, indent=2), encoding="utf-8")
    except Exception:  # noqa: BLE001 - manifest is best-effort
        pass


def _sweep_key(experiment: str, fidelity: str, seed: int, system: str) -> str:
    blob = json.dumps({"e": experiment, "f": fidelity, "s": seed, "sys": system},
                      sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def settings_for(experiment: str, fidelity: str, seed: int,
                 system: str = "crotonic9_paper4") -> list[tuple[str, QRCConfig]]:
    out = []
    if experiment == "2.1_quick":
        for fn in QUICK_ENCODINGS:
            out.append((f"2.1_{fn}", build_cfg(fn, fidelity, seed=seed, system=system)))
        return out
    if experiment in ("2.1", "all"):
        for fn in ENCODINGS:
            out.append((f"2.1_{fn}", build_cfg(fn, fidelity, seed=seed, system=system)))
    if experiment in ("2.2", "all"):
        out.append(("2.2_phaseamp_off", build_cfg("arcsin_sqrt", fidelity, seed=seed,
                                                   phase_amplitude=False, system=system)))
        out.append(("2.2_phaseamp_on", build_cfg("arcsin_sqrt", fidelity, seed=seed,
                                                  phase_amplitude=True, system=system)))
    return out


def make_figure(results: list[dict], out: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    order = sorted(results, key=lambda r: -np.mean([r["by_horizon"][f"h{h}"]["cv_mean"] for h in HORIZONS]))
    cmap = plt.get_cmap("tab10")
    for i, r in enumerate(order):
        m = np.array([r["by_horizon"][f"h{h}"]["cv_mean"] for h in HORIZONS])
        s = np.array([r["by_horizon"][f"h{h}"]["cv_std"] for h in HORIZONS])
        ax.plot(HORIZONS, m, "o-", color=cmap(i % 10), lw=1.8, ms=5, label=r["label"])
        ax.fill_between(HORIZONS, m - s, m + s, color=cmap(i % 10), alpha=0.08)
    ax.set_xlabel("forecast horizon (days ahead)")
    ax.set_ylabel("weather R²  (blocked-CV mean ± std)")
    ax.set_title("Phase-2: encoding sweep (weather, magnitude-653 readout)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[phase2] wrote {out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="QRC Phase-2 encoding sweep")
    ap.add_argument("--experiment", choices=["2.1", "2.1_quick", "2.2", "all"], default="2.1")
    ap.add_argument("--fidelity", choices=list(FIDELITY), default="screen")
    ap.add_argument("--system", default="crotonic9_paper4",
                    help="preset name or integer qubit count (use a small int for smoke tests)")
    ap.add_argument("--out-dir", default="artifacts/qrc_phase2")
    ap.add_argument("--run-dir", default=None,
                    help="managed-run dir (launcher). When set, --out-dir defaults "
                         "to it and live progress is emitted for the UI.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--weather-train", default="data/weather/DailyDelhiClimateTrain.csv")
    ap.add_argument("--weather-test", default="data/weather/DailyDelhiClimateTest.csv")
    ap.add_argument("--only", nargs="*", default=None,
                    help="restrict to these encoding fn names (for a quick subset)")
    args = ap.parse_args()

    # A managed run (--run-dir) writes its outputs into the run-dir and emits
    # live progress for the UI; a bare CLI run uses --out-dir and stays quiet.
    out = Path(args.run_dir) if args.run_dir else Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    logger = None
    if args.run_dir:
        logger = ProgressLogger(run_dir=args.run_dir, title=f"phase2 {args.experiment}")

    plan = settings_for(args.experiment, args.fidelity, args.seed, args.system)
    if args.only:
        plan = [(lbl, cfg) for lbl, cfg in plan if cfg.encoding.fn in args.only
                or "phaseamp" in lbl]
    n_set = len(plan)
    # Config-keyed result cache survives a relaunch (fresh run-dir); the run-dir
    # copy is for local inspection / the UI.
    cache_dir = _RESULTS_ROOT / _sweep_key(args.experiment, args.fidelity, args.seed, args.system)
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Phase-2 {args.experiment} @ {args.fidelity}: {n_set} settings", flush=True)
    if logger is not None:
        logger.event("config", f"phase2 {args.experiment} @ {args.fidelity}: {n_set} encodings",
                     experiment=args.experiment, fidelity=args.fidelity, n_settings=n_set)

    results = []
    try:
        for i, (label, cfg) in enumerate(plan):
            cpath = cache_dir / f"result_{label}.json"          # durable (resume)
            rpath = out / f"result_{label}.json"                # run-dir (UI/local)
            if cpath.exists():                                  # sweep-level resume
                print(f"[phase2] {label}: cached, skipping", flush=True)
                res = json.loads(cpath.read_text(encoding="utf-8"))
                rpath.write_text(json.dumps(res, indent=2), encoding="utf-8")
                results.append(res)
                _write_trace_manifest(out, results)
                if logger is not None:
                    logger.event("cached", f"encoding {i + 1}/{n_set} {cfg.encoding.fn}: cached, skipped")
                continue
            phase_label = f"{i + 1}/{n_set} {cfg.encoding.fn}"
            if logger is not None:
                logger.event("encoding", f"encoding {phase_label}", phase=phase_label)
            res = run_setting(cfg, label, args.weather_train, args.weather_test,
                              logger=logger, phase_label=phase_label)
            res["fidelity"] = args.fidelity
            blob = json.dumps(res, indent=2)
            cpath.write_text(blob, encoding="utf-8")            # durable first
            rpath.write_text(blob, encoding="utf-8")
            results.append(res)
            _write_trace_manifest(out, results)
    except Exception as exc:  # surface into the event log, then re-raise
        if logger is not None:
            logger.event("error", f"phase2 failed: {type(exc).__name__}: {exc}")
        raise

    # summary + figure
    key = np.mean([[r["by_horizon"][f"h{h}"]["cv_mean"] for h in HORIZONS] for r in results], axis=1)
    order = np.argsort(-key)
    summary = {
        "generated_utc": datetime.now(UTC).isoformat(),
        "experiment": args.experiment, "fidelity": args.fidelity, "seed": args.seed,
        "ranking": [results[i]["label"] for i in order],
        "results": results,
    }
    (out / "phase2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nRanking (mean CV-R² across horizons):", flush=True)
    for i in order:
        print(f"  {results[i]['label']:<26} {key[i]:.3f}", flush=True)
    make_figure(results, out / "phase2_encoding.png")
    # Full sweep succeeded — the resume cache is no longer needed.
    shutil.rmtree(cache_dir, ignore_errors=True)
    print(f"\nDone. Outputs in {out.resolve()}")
    if logger is not None:
        logger.status(phase="done", eta_s=0)
        logger.event("ranking", "sweep complete — ranking: " + ", ".join(summary["ranking"]))
        logger.close("done", f"phase2 {args.experiment} complete")


if __name__ == "__main__":
    main()
