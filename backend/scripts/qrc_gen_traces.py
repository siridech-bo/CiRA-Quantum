"""Generate the raw-FID trace cache (Stage A Phase 0.A).

Drives one reservoir pass — weather or NARMA — with the Paper-4 FID readout and
dumps the *per-step raw complex FID* to ``backend/artifacts/traces/<name>.npz``
using the Stage A §1 schema. Downstream, Coder B's feature builders + the
:mod:`app.qrc.feature_lab` harness re-extract and score features from this cache
in minutes, instead of re-running the ~15 h reservoir evolution.

The pass reuses the existing runners (:func:`run_weather_reservoir`,
:func:`run_narma_multitask`); a ``fid_cb`` hook on the reservoir loop collects
each step's FID with no change to the normal readout.

Run from ``d:\\CiRA Quantum\\backend``::

    # Full weather trace (paper scale, GPU) — the ~14.6 h one-time run:
    PYTHONIOENCODING=utf-8 python scripts/qrc_gen_traces.py --task weather \
        --out artifacts/traces/weather.npz

    # Full NARMA trace (~4 h):
    PYTHONIOENCODING=utf-8 python scripts/qrc_gen_traces.py --task narma \
        --out artifacts/traces/narma.npz

    # Tiny CPU smoke test (validates the .npz schema in seconds):
    PYTHONIOENCODING=utf-8 python scripts/qrc_gen_traces.py --task narma \
        --system 4 --evolution-mode action --fid-points 32 \
        --splits 5 10 5 --orders 2 10 --out artifacts/traces/narma_tiny.npz
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from qrc_progress import ProgressLogger

from app.qrc.benchmarks import run_narma_multitask, run_weather_reservoir
from app.qrc.config import (
    QRCConfig,
    SimConfig,
    TrainingConfig,
    make_system_config,
)
from app.qrc.features import FeatureConfig
from app.qrc.tasks import load_weather, narma_sequence_sine
from app.qrc.utils import normalize

WEATHER_HORIZONS = (1, 10, 20, 30, 45)
NARMA_ORDERS = (2, 5, 10, 15, 20)


class FidCollector:
    """Accumulate the per-step complex FID emitted by the reservoir loop.

    When a :class:`ProgressLogger` + ``total`` step count are supplied, each
    call also updates the live status snapshot (phase/step/total/ETA) that the
    control API's ``GET /runs/<id>/progress`` reads — so a UI-launched
    trace-gen shows a moving progress bar + ETA instead of a silent "running".
    ETA is a rolling estimate from mean per-step wall time so far.
    """

    def __init__(
        self,
        total: int | None = None,
        logger: ProgressLogger | None = None,
        phase: str = "reservoir",
    ) -> None:
        self.rows: list[np.ndarray] = []
        self.total = total
        self.logger = logger
        self.phase = phase
        self._t0 = time.time()

    def __call__(self, step: int, fid: np.ndarray) -> None:
        self.rows.append(np.asarray(fid, dtype=np.complex64))
        if self.logger is not None:
            n = len(self.rows)
            elapsed = time.time() - self._t0
            per_step = elapsed / n if n else 0.0
            eta = per_step * (self.total - n) if self.total else None
            self.logger.status(phase=self.phase, step=n, total=self.total, eta_s=eta)

    def stack(self) -> np.ndarray:
        if not self.rows:
            return np.empty((0, 0), dtype=np.complex64)
        return np.vstack(self.rows).astype(np.complex64)


def _resolve_system(name_or_n: str):
    """Accept an int qubit count or a preset name for ``--system``."""
    try:
        return make_system_config(int(name_or_n))
    except ValueError:
        return make_system_config(name_or_n)


def build_config(args) -> QRCConfig:
    washout, n_train, n_test = args.splits
    return QRCConfig(
        system=_resolve_system(args.system),
        sim=SimConfig(
            tau=args.tau,
            n_virtual=args.n_virtual,
            evolution_mode=args.evolution_mode,
            seed=args.seed,
            fid_points=args.fid_points,
        ),
        training=TrainingConfig(
            washout=washout, n_train=n_train, n_test=n_test, seed=args.seed
        ),
    )


def _weather_indices(n: int):
    """Default proton/carbon encoding qubits for an ``n``-spin system.

    Matches the Paper-4 9-spin layout (protons 4–8, carbons 0–3) when
    ``n >= 9``; otherwise splits the spins in half so smaller smoke-test
    systems drive valid indices."""
    if n >= 9:
        return (4, 5, 6, 7, 8), (0, 1, 2, 3)
    half = max(1, n // 2)
    return tuple(range(half, n)), tuple(range(half))


def _synthetic_weather(n_days: int, seed: int) -> np.ndarray:
    """A random [0,1] (temp, humidity) series for schema smoke tests."""
    rng = np.random.default_rng(seed)
    return rng.random((n_days, 2))


def gen_weather(
    cfg: QRCConfig, feature_cfg: FeatureConfig, args,
    logger: ProgressLogger | None = None,
) -> dict:
    horizons = list(args.horizons)
    tr = cfg.training
    n_steps = tr.washout + tr.n_train + tr.n_test
    n_days = n_steps + max(horizons)

    if args.synthetic:
        weather = _synthetic_weather(n_days, args.seed)
    else:
        raw = load_weather(args.weather_train)
        if args.weather_test and Path(args.weather_test).exists():
            raw = np.vstack([raw, load_weather(args.weather_test)])
        cols = [normalize(raw[:, j])[0] for j in range(raw.shape[1])]
        weather = np.column_stack(cols)
        if len(weather) < n_days:
            raise ValueError(
                f"weather series too short: need {n_days} days "
                f"(split {tuple(args.splits)} + max horizon {max(horizons)}) "
                f"but have {len(weather)}"
            )
    weather = weather[:n_days]

    proton_idx, carbon_idx = _weather_indices(cfg.system.n_qubits)
    collector = FidCollector(total=n_steps, logger=logger, phase="reservoir")
    print(f"[gen] weather pass: {n_steps} steps, fid_points={cfg.sim.fid_points}, "
          f"proton_idx={proton_idx} carbon_idx={carbon_idx}", flush=True)
    if logger is not None:
        logger.event("reservoir", f"weather reservoir pass: {n_steps} steps",
                      phase="reservoir", step=0, total=n_steps)
    run_weather_reservoir(
        cfg, weather, n_steps, feature_cfg=feature_cfg,
        proton_idx=proton_idx, carbon_idx=carbon_idx, fid_cb=collector,
    )
    fids = collector.stack()
    meta = {
        "task": "weather",
        "generated_utc": datetime.now(UTC).isoformat(),
        "repro_hash": cfg.repro_hash(),
        "code_version": cfg.code_version,
        "proton_idx": list(proton_idx),
        "carbon_idx": list(carbon_idx),
        "synthetic": bool(args.synthetic),
        "sim": asdict(cfg.sim),
        "training": asdict(cfg.training),
        "encoding": asdict(cfg.encoding),
        "system": asdict(cfg.system),
        "feature_cfg": asdict(feature_cfg),
        "horizons": horizons,
    }
    return {
        "fids": fids,
        "fid_dwell": np.float64(cfg.sim.fid_dwell),
        "task": np.str_("weather"),
        "split": np.asarray([tr.washout, tr.n_train, tr.n_test], dtype=np.int64),
        "seed": np.int64(cfg.sim.seed),
        "weather_norm": np.asarray(weather, dtype=np.float64),
        "horizons": np.asarray(horizons, dtype=np.int64),
        "meta": np.str_(json.dumps(meta, default=float)),
    }


def gen_narma(
    cfg: QRCConfig, feature_cfg: FeatureConfig, args,
    logger: ProgressLogger | None = None,
) -> dict:
    orders = list(args.orders)
    tr = cfg.training
    n_steps = tr.washout + tr.n_train + tr.n_test
    # The driving input is order-independent (depends only on n_steps/seed),
    # exactly the sequence run_narma_multitask feeds the reservoir.
    narma_input, _ = narma_sequence_sine(n_steps, orders[0], seed=cfg.sim.seed)

    collector = FidCollector(total=n_steps, logger=logger, phase="reservoir")
    print(f"[gen] narma pass: {n_steps} steps, fid_points={cfg.sim.fid_points}, "
          f"orders={orders}", flush=True)
    if logger is not None:
        logger.event("reservoir", f"narma reservoir pass: {n_steps} steps",
                      phase="reservoir", step=0, total=n_steps)
    run_narma_multitask(
        cfg, orders, feature_cfg=feature_cfg, input_kind="sine",
        fid_cb=collector,
    )
    fids = collector.stack()
    meta = {
        "task": "narma",
        "generated_utc": datetime.now(UTC).isoformat(),
        "repro_hash": cfg.repro_hash(),
        "code_version": cfg.code_version,
        "input_kind": "sine",
        "sim": asdict(cfg.sim),
        "training": asdict(cfg.training),
        "encoding": asdict(cfg.encoding),
        "system": asdict(cfg.system),
        "feature_cfg": asdict(feature_cfg),
        "orders": orders,
    }
    return {
        "fids": fids,
        "fid_dwell": np.float64(cfg.sim.fid_dwell),
        "task": np.str_("narma"),
        "split": np.asarray([tr.washout, tr.n_train, tr.n_test], dtype=np.int64),
        "seed": np.int64(cfg.sim.seed),
        "narma_input": np.asarray(narma_input, dtype=np.float64),
        "orders": np.asarray(orders, dtype=np.int64),
        "meta": np.str_(json.dumps(meta, default=float)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="QRC raw-FID trace-cache generator")
    ap.add_argument("--task", choices=["weather", "narma"], required=True)
    ap.add_argument("--system", default="crotonic9_paper4",
                    help="preset name or integer qubit count")
    ap.add_argument("--evolution-mode",
                    choices=["auto", "propagator", "action", "mesolve", "gpu"],
                    default="gpu")
    ap.add_argument("--tau", type=float, default=0.01,
                    help="evolution time (s); paper uses 0.01 NARMA / 0.03 weather")
    ap.add_argument("--n-virtual", type=int, default=25)
    ap.add_argument("--fid-points", type=int, default=2048)
    ap.add_argument("--n-peaks", type=int, default=653,
                    help="FID spectral peaks (readout; not stored in the trace)")
    ap.add_argument("--splits", type=int, nargs=3, default=(100, 400, 100),
                    metavar=("WASHOUT", "N_TRAIN", "N_TEST"))
    ap.add_argument("--horizons", type=int, nargs="+", default=list(WEATHER_HORIZONS),
                    help="forecast horizons (task=weather)")
    ap.add_argument("--orders", type=int, nargs="+", default=list(NARMA_ORDERS),
                    help="NARMA orders (task=narma)")
    ap.add_argument("--weather-train", default="data/weather/DailyDelhiClimateTrain.csv")
    ap.add_argument("--weather-test", default="data/weather/DailyDelhiClimateTest.csv")
    ap.add_argument("--synthetic", action="store_true",
                    help="use a random weather series (schema smoke test only)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None, help="output .npz path")
    ap.add_argument("--run-dir", default=None,
                    help="managed-run dir (launcher). When set and --out is "
                         "omitted, the trace is written to <run-dir>/trace.npz "
                         "so the control API's /fid endpoint resolves it.")
    args = ap.parse_args()

    cfg = build_config(args)
    feature_cfg = FeatureConfig(readout="fid", n_peaks=args.n_peaks)
    if args.out:
        out = Path(args.out)
    elif args.run_dir:
        out = Path(args.run_dir) / "trace.npz"
    else:
        out = Path(f"artifacts/traces/{args.task}.npz")
    out.parent.mkdir(parents=True, exist_ok=True)

    # When launched as a managed run (--run-dir set), emit live progress into
    # the run-dir so GET /runs/<id>/progress surfaces phase/step/total/ETA +
    # an event log. Run directly (no --run-dir) → console-only, no logger.
    logger = None
    if args.run_dir:
        logger = ProgressLogger(run_dir=args.run_dir, title=f"trace-gen {args.task}")
        logger.event("config", "trace-gen configured", task=args.task,
                      system=args.system, evolution_mode=args.evolution_mode,
                      fid_points=args.fid_points, splits=list(args.splits))

    try:
        payload = (gen_weather if args.task == "weather" else gen_narma)(
            cfg, feature_cfg, args, logger=logger
        )
        if logger is not None:
            # Keep step/total at their final (100%) values — only advance the
            # phase label — so the bar doesn't snap back during the brief write.
            logger.status(phase="writing", eta_s=0)
            logger.event("writing", f"writing trace to {out.name}")
        np.savez_compressed(out, **payload)
    except Exception as exc:  # surface the failure into the event log, then re-raise
        if logger is not None:
            logger.event("error", f"trace-gen failed: {type(exc).__name__}: {exc}")
        raise

    fids = payload["fids"]
    print(f"[gen] wrote {out.resolve()}")
    print(f"[gen]   fids: shape={fids.shape} dtype={fids.dtype}")
    print(f"[gen]   keys: {sorted(payload)}")
    print(f"[gen]   split={payload['split'].tolist()} "
          f"fid_dwell={float(payload['fid_dwell'])} seed={int(payload['seed'])}")

    if logger is not None:
        logger.event("wrote", f"trace written: {tuple(fids.shape)} FID rows")
        logger.close("done", f"trace-gen {args.task} complete")


if __name__ == "__main__":
    main()
