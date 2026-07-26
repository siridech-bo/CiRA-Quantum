"""Reproduce Hou et al. 2026 (PRL 136, 120602) NARMA results (plan §4).

Drives the 9-spin ¹³C crotonic-acid reservoir (``crotonic9_paper4`` preset,
owned by Coder A) with the Paper-4 multi-tone sine input, reads out the
653-peak FID spectrum, fits a ridge readout, and reports the NMSE for
NARMA orders 2/5/10/15/20 next to the paper's Table I values. A classical
ESN sweep (500/1000/5000/10000) provides the baseline. Results are saved
to JSON.

Run from ``d:\\CiRA Quantum\\backend`` with a CUDA torch build::

    PYTHONIOENCODING=utf-8 python scripts/qrc_reproduce_paper4.py

The physics (FID readout mode in ``evolution.py``, ``QRCSystem.fid_signal``,
and the ``crotonic9_paper4`` preset) is owned by Coder A; this runner codes
to the §0 contracts and will work once those land.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from qrc_progress import ProgressLogger

from app.qrc.benchmarks import esn_sweep, run_narma_multitask
from app.qrc.config import (
    QRCConfig,
    SimConfig,
    TrainingConfig,
    make_system_config,
)
from app.qrc.features import FeatureConfig
from app.qrc.tasks import narma_sequence_sine

# Paper 4, Table I best-case NARMA NMSE (FID-653 readout). The higher
# orders report a range (observable-only -> FID); we track the best (FID)
# value and keep the observable-only figure for context.
PAPER_TABLE_I = {
    2: {"best": 1.74e-7, "obs_only": 1.74e-7},
    5: {"best": 4.44e-5, "obs_only": 4.44e-5},
    10: {"best": 5.84e-5, "obs_only": 3.39e-4},
    15: {"best": 6.37e-5, "obs_only": 5.64e-4},
    20: {"best": 4.34e-5, "obs_only": 7.15e-4},
}

NARMA_ORDERS = (2, 5, 10, 15, 20)


def build_config(
    tau: float = 0.01,
    seed: int = 42,
    fid_points: int = 2048,
    n_train: int = 400,
    n_test: int = 100,
    washout: int = 100,
) -> QRCConfig:
    """QRCConfig for the Paper-4 reproduction: crotonic9, τ=0.01 s, GPU.

    Defaults are paper-scale (a multi-hour run). Lower ``fid_points`` /
    ``n_train`` for a faster reduced pass — but note the readout advantage
    only shows when ``n_train`` comfortably exceeds ``n_peaks`` and
    ``fid_points`` is large enough for fine spectral resolution."""
    return QRCConfig(
        system=make_system_config("crotonic9_paper4"),
        sim=SimConfig(tau=tau, evolution_mode="gpu", seed=seed, fid_points=fid_points),
        training=TrainingConfig(
            washout=washout, n_train=n_train, n_test=n_test, device="cuda", seed=seed
        ),
    )


def _step_cb(log: ProgressLogger, phase: str):
    """Return a reservoir progress callback that logs step-level ETA.

    Emits a status update every ~5% of steps (cheap; re-renders the HTML)
    without flooding the console/JSONL."""
    state = {"t0": time.time(), "last": -1}

    def cb(done: int, total: int) -> None:
        milestone = max(1, total // 20)
        if done % milestone and done != total:
            return
        if done == state["last"]:
            return
        state["last"] = done
        rate = done / max(time.time() - state["t0"], 1e-9)
        eta = (total - done) / rate if rate > 0 else None
        log.status(phase=phase, step=done, total=total, eta_s=eta)

    return cb


def run_narma_suite(
    cfg: QRCConfig,
    feature_cfg: FeatureConfig,
    log: ProgressLogger,
    orders=NARMA_ORDERS,
    out_path: Path | None = None,
) -> dict[int, dict]:
    """Emulate all NARMA orders from ONE reservoir pass (multitasking).

    The NARMA input is order-independent, so the (expensive) reservoir
    readout is computed once and a separate ridge readout is fit per order
    — matching Paper 4's multitasking and ~len(orders)× faster than running
    the reservoir per order. Step-level ETA covers the single pass; each
    order's result is logged, and the whole set is checkpointed."""
    orders = list(orders)
    log.event("reservoir_start",
              f"single reservoir pass (sine, FID-{feature_cfg.n_peaks}) "
              f"for orders {orders} — multitask")
    t0 = time.time()
    results, _ = run_narma_multitask(
        cfg, orders, feature_cfg=feature_cfg, input_kind="sine",
        progress_cb=_step_cb(log, "reservoir"),
    )
    log.event("reservoir_done",
              f"reservoir pass done in {(time.time()-t0)/60:.1f} min; "
              f"fitting {len(orders)} readouts")
    for order in orders:
        m = results[order]
        paper = PAPER_TABLE_I.get(order, {}).get("best")
        log.result(f"NARMA{order}", {
            "nmse_paper": m["nmse_paper"], "r2": m["r2"], "paper_best": paper,
        })
        log.event(
            "order_done",
            f"NARMA{order}: NMSE_paper={m['nmse_paper']:.3e} "
            f"(paper {paper:.2e}) R2={m['r2']:.4f}",
            order=order, **m,
        )
    if out_path is not None:
        _save(out_path, cfg, feature_cfg, results, {})
    return results


def run_esn_baseline(cfg: QRCConfig, log: ProgressLogger, order: int = 10) -> dict[int, dict]:
    """Classical ESN sweep on the same NARMA task for comparison."""
    tr = cfg.training
    n_steps = tr.washout + tr.n_train + tr.n_test
    u, y = narma_sequence_sine(n_steps, order, seed=cfg.sim.seed)
    log.event("esn_start", f"ESN sweep (NARMA{order}, sizes 500-10000) starting")
    esn = esn_sweep(u, y, tr_cfg=tr, seed=cfg.sim.seed)
    for size, m in sorted(esn.items()):
        log.result(f"ESN({size})",
                   {"nmse_paper": m["nmse_paper"], "r2": m["r2"], "paper_best": None,
                    "secs": None})
    log.event("esn_done", f"ESN sweep done ({len(esn)} sizes)")
    return esn


def _save(out: Path, cfg: QRCConfig, feature_cfg: FeatureConfig,
          qrc: dict, esn: dict) -> None:
    """Write the consolidated results JSON (used for the final save and for
    per-order checkpointing)."""
    payload = {
        "generated_utc": datetime.now(UTC).isoformat(),
        "repro_hash": cfg.repro_hash(),
        "config": {
            "feature_cfg": asdict(feature_cfg),
            "sim": asdict(cfg.sim),
            "training": asdict(cfg.training),
        },
        "paper_table_i": PAPER_TABLE_I,
        "qrc_narma": {str(k): v for k, v in qrc.items()},
        "esn_sweep": {str(k): v for k, v in esn.items()},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")


def run_weather(cfg: QRCConfig, feature_cfg: FeatureConfig, csv_path: str) -> dict:
    """Weather-forecasting hook (plan §10.2). NARMA is the priority; this
    stub wires the structure for a follow-up. Splits per §4: 374 washout /
    600 train / 600 test."""
    raise NotImplementedError(
        "Weather reproduction is a follow-up hook; NARMA is the priority. "
        "Wire load_weather() -> reservoir.run() -> train_readout() here."
    )


def print_narma_table(qrc: dict[int, dict]) -> None:
    """Print QRC NMSE next to Paper-4 Table I."""
    print("\nNARMA reproduction vs Paper 4, Table I")
    print(f"{'order':>5} | {'QRC NMSE_paper':>15} | {'paper best':>11} | "
          f"{'paper obs-only':>14} | {'QRC R2':>8}")
    print("-" * 66)
    for order in NARMA_ORDERS:
        if order not in qrc:
            continue
        m = qrc[order]
        p = PAPER_TABLE_I.get(order, {})
        print(
            f"{order:>5} | {m['nmse_paper']:>15.3e} | "
            f"{p.get('best', float('nan')):>11.2e} | "
            f"{p.get('obs_only', float('nan')):>14.2e} | {m['r2']:>8.4f}"
        )


def print_esn_table(esn: dict[int, dict]) -> None:
    print("\nClassical ESN baseline sweep (NARMA10)")
    print(f"{'size':>6} | {'NMSE_paper':>12} | {'NMSE':>12} | {'R2':>8}")
    print("-" * 46)
    for size in sorted(esn):
        m = esn[size]
        print(f"{size:>6} | {m['nmse_paper']:>12.3e} | "
              f"{m['nmse']:>12.3e} | {m['r2']:>8.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Paper-4 NARMA reproduction runner")
    ap.add_argument("--tau", type=float, default=0.01, help="evolution time (s)")
    ap.add_argument("--n-peaks", type=int, default=653, help="FID spectral peaks")
    ap.add_argument("--fid-points", type=int, default=2048, help="FID samples/step")
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--n-test", type=int, default=100)
    ap.add_argument("--orders", type=int, nargs="+", default=list(NARMA_ORDERS),
                    help="NARMA orders to run")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-esn", action="store_true", help="skip the ESN baseline")
    ap.add_argument(
        "--out",
        default="artifacts/qrc_paper4_narma.json",
        help="JSON output path (checkpointed after every order)",
    )
    ap.add_argument(
        "--run-dir",
        default="artifacts/paper4_run",
        help="dir for the live event log + progress.html trace",
    )
    args = ap.parse_args()

    cfg = build_config(tau=args.tau, seed=args.seed, fid_points=args.fid_points,
                       n_train=args.n_train, n_test=args.n_test)
    feature_cfg = FeatureConfig(readout="fid", n_peaks=args.n_peaks)
    out = Path(args.out)
    orders = tuple(args.orders)

    log = ProgressLogger(run_dir=args.run_dir, title="QRC Paper-4 NARMA reproduction")
    log.event("config",
              f"crotonic9 tau={args.tau}s FID-{args.n_peaks} "
              f"fid_points={cfg.sim.fid_points} n_train={cfg.training.n_train} "
              f"orders={list(orders)}")
    try:
        qrc = run_narma_suite(cfg, feature_cfg, log, orders=orders, out_path=out)
        esn = {} if args.no_esn else run_esn_baseline(cfg, log)
        _save(out, cfg, feature_cfg, qrc, esn)
        print_narma_table(qrc)
        if esn:
            print_esn_table(esn)
        log.event("saved", f"results saved to {out.resolve()}")
        log.close("done", "reproduction complete")
    except Exception as exc:                       # keep the log on failure
        log.event("error", f"{type(exc).__name__}: {exc}")
        raise
    print(f"\nSaved results to {out.resolve()}")
    print(f"Live trace: {(Path(args.run_dir) / 'progress.html').resolve()}")


if __name__ == "__main__":
    main()
