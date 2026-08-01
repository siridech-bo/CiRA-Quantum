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
import hashlib
import json
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from qrc_progress import ProgressLogger

# Config-keyed checkpoint root — stable across re-launches (each managed run
# gets a fresh run-dir, so the checkpoint can't live there or resume would
# never find it). Anchored to the backend dir, independent of cwd.
_CKPT_ROOT = Path(__file__).resolve().parents[1] / "artifacts" / "traces" / ".ckpt"

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


def _ckpt_hash(cfg, task: str) -> str:
    """Stable short hash of everything that determines the per-step FID, so a
    re-launch with the *same* config finds and resumes its checkpoint, and a
    changed config does not."""
    payload = {
        "task": task,
        "system": asdict(cfg.system),
        "sim": asdict(cfg.sim),          # tau, n_virtual, evolution_mode, seed, fid_points
        "training": asdict(cfg.training),  # washout/n_train/n_test (the split → n_steps + input)
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


class StreamingTrace:
    """Stream each step's FID straight to an on-disk memmap (never hold the
    whole run in RAM), checkpoint the GPU state periodically, and support
    resume — so a crash costs at most one checkpoint interval, not the whole
    multi-hour run.

    Serves three reservoir hooks:
      * ``__call__(k, fid)``  → ``fid_cb`` — write row ``k`` to the memmap +
        update the live progress snapshot.
      * ``checkpoint(k, state_fn)`` → ``checkpoint_cb`` — every ``every`` steps,
        flush the memmap and atomically save {state, step} for resume.
      * ``try_resume()`` → ``(start_step, state0)`` from a matching checkpoint.
    """

    def __init__(
        self, ckpt_dir: Path, n_steps: int, fid_points: int, config_hash: str,
        *, total: int, logger: ProgressLogger | None = None, every: int = 20,
        phase: str = "reservoir", fid_dwell: float = 3e-4,
    ) -> None:
        self.dir = Path(ckpt_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.n_steps = n_steps
        self.fid_points = fid_points
        self.hash = config_hash
        self.total = total
        self.logger = logger
        self.every = max(1, every)
        self.phase = phase
        self.fid_dwell = fid_dwell
        self.fids_path = self.dir / "fids.dat"
        self.ckpt_path = self.dir / "ckpt.npz"
        # Must end in .npz — np.savez appends '.npz' to any other name, which
        # would make the subsequent atomic replace target a missing file.
        self.tmp_path = self.dir / "ckpt.tmp.npz"
        self._t0 = time.time()
        self._start_step = 0

        # A stale memmap of the wrong size (config changed) can't be reused.
        expected = n_steps * fid_points * np.dtype(np.complex64).itemsize
        if self.fids_path.exists() and self.fids_path.stat().st_size != expected:
            self.fids_path.unlink()
            self.ckpt_path.unlink(missing_ok=True)
        mode = "r+" if self.fids_path.exists() else "w+"
        self.mmap = np.memmap(
            self.fids_path, dtype=np.complex64, mode=mode, shape=(n_steps, fid_points)
        )

        # Point the run-dir at THIS live memmap so GET /runs/<id>/fid can read
        # the waveform *as it streams* (before any .npz is saved). Best-effort.
        if logger is not None:
            try:
                (Path(logger.dir) / "live_fid.json").write_text(json.dumps({
                    "fids_path": str(self.fids_path), "n_steps": n_steps,
                    "fid_points": fid_points, "fid_dwell": float(fid_dwell),
                    "label": phase,
                }), encoding="utf-8")
            except Exception:  # noqa: BLE001 - live view is best-effort
                pass

    def try_resume(self) -> tuple[int, np.ndarray | None]:
        """Resume from a saved checkpoint iff it matches this config. Returns
        ``(start_step, state0)``; ``(0, None)`` for a fresh run."""
        if not self.ckpt_path.exists():
            return 0, None
        try:
            with np.load(self.ckpt_path, allow_pickle=False) as z:
                if str(z["config_hash"]) != self.hash:
                    return 0, None
                step = int(z["step"])
                state = np.asarray(z["state"]) if "state" in z.files else None
            if state is None or not (0 <= step < self.n_steps):
                return 0, None
            self._start_step = step + 1
            return self._start_step, state
        except Exception:  # noqa: BLE001 - a corrupt checkpoint just means fresh
            return 0, None

    def __call__(self, k: int, fid: np.ndarray) -> None:  # fid_cb
        self.mmap[k] = np.asarray(fid, dtype=np.complex64)
        if self.logger is not None:
            done_this_session = k - self._start_step + 1
            elapsed = time.time() - self._t0
            per = elapsed / done_this_session if done_this_session > 0 else 0.0
            eta = per * (self.n_steps - (k + 1))
            self.logger.status(phase=self.phase, step=k + 1, total=self.total, eta_s=eta)

    def checkpoint(self, k: int, state_fn) -> None:  # checkpoint_cb
        """Every ``every`` steps (and on the last), durably persist {state,step}
        for resume. Best-effort: a checkpoint write must never crash the run."""
        if (k + 1) % self.every != 0 and k != self.n_steps - 1:
            return
        try:
            state = state_fn()
            if state is None:
                return  # backend has no serialisable state → stream-only
            self.mmap.flush()
            np.savez(
                self.tmp_path, step=np.int64(k), config_hash=np.str_(self.hash),
                n_steps=np.int64(self.n_steps), fid_points=np.int64(self.fid_points),
                state=np.asarray(state, dtype=np.complex64),
            )
            self.tmp_path.replace(self.ckpt_path)
        except Exception:  # noqa: BLE001 - checkpointing is best-effort
            pass

    def finalize(self) -> np.ndarray:
        """The complete [n_steps, fid_points] FID array from disk."""
        self.mmap.flush()
        return np.array(self.mmap[: self.n_steps], dtype=np.complex64)

    def cleanup(self) -> None:
        """Drop the checkpoint scratch once the final trace is safely written."""
        try:
            self.mmap.flush()
            del self.mmap
        except Exception:  # noqa: BLE001
            pass
        for p in (self.fids_path, self.ckpt_path, self.tmp_path):
            try:
                p.unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                pass
        try:
            self.dir.rmdir()
        except OSError:
            pass


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
) -> tuple[dict, "StreamingTrace"]:
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
    chash = _ckpt_hash(cfg, "weather")
    chk = StreamingTrace(_CKPT_ROOT / chash, n_steps, cfg.sim.fid_points, chash,
                         total=n_steps, logger=logger)
    start_step, state0 = chk.try_resume()
    print(f"[gen] weather pass: {n_steps} steps, fid_points={cfg.sim.fid_points}, "
          f"proton_idx={proton_idx} carbon_idx={carbon_idx}"
          f"{f' (RESUMING from step {start_step})' if start_step else ''}", flush=True)
    if logger is not None:
        if start_step:
            logger.event("resume", f"resuming weather pass from step {start_step}/{n_steps}",
                          phase="reservoir", step=start_step, total=n_steps)
        else:
            logger.event("reservoir", f"weather reservoir pass: {n_steps} steps",
                          phase="reservoir", step=0, total=n_steps)
    run_weather_reservoir(
        cfg, weather, n_steps, feature_cfg=feature_cfg,
        proton_idx=proton_idx, carbon_idx=carbon_idx, fid_cb=chk,
        start_step=start_step, state0=state0, checkpoint_cb=chk.checkpoint,
    )
    fids = chk.finalize()
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
    payload = {
        "fids": fids,
        "fid_dwell": np.float64(cfg.sim.fid_dwell),
        "task": np.str_("weather"),
        "split": np.asarray([tr.washout, tr.n_train, tr.n_test], dtype=np.int64),
        "seed": np.int64(cfg.sim.seed),
        "weather_norm": np.asarray(weather, dtype=np.float64),
        "horizons": np.asarray(horizons, dtype=np.int64),
        "meta": np.str_(json.dumps(meta, default=float)),
    }
    return payload, chk


def gen_narma(
    cfg: QRCConfig, feature_cfg: FeatureConfig, args,
    logger: ProgressLogger | None = None,
) -> tuple[dict, "StreamingTrace"]:
    orders = list(args.orders)
    tr = cfg.training
    n_steps = tr.washout + tr.n_train + tr.n_test
    # The driving input is order-independent (depends only on n_steps/seed),
    # exactly the sequence run_narma_multitask feeds the reservoir.
    narma_input, _ = narma_sequence_sine(n_steps, orders[0], seed=cfg.sim.seed)

    chash = _ckpt_hash(cfg, "narma")
    chk = StreamingTrace(_CKPT_ROOT / chash, n_steps, cfg.sim.fid_points, chash,
                         total=n_steps, logger=logger)
    start_step, state0 = chk.try_resume()
    print(f"[gen] narma pass: {n_steps} steps, fid_points={cfg.sim.fid_points}, "
          f"orders={orders}{f' (RESUMING from step {start_step})' if start_step else ''}",
          flush=True)
    if logger is not None:
        if start_step:
            logger.event("resume", f"resuming narma pass from step {start_step}/{n_steps}",
                          phase="reservoir", step=start_step, total=n_steps)
        else:
            logger.event("reservoir", f"narma reservoir pass: {n_steps} steps",
                          phase="reservoir", step=0, total=n_steps)
    run_narma_multitask(
        cfg, orders, feature_cfg=feature_cfg, input_kind="sine",
        fid_cb=chk, start_step=start_step, state0=state0, checkpoint_cb=chk.checkpoint,
    )
    fids = chk.finalize()
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
    payload = {
        "fids": fids,
        "fid_dwell": np.float64(cfg.sim.fid_dwell),
        "task": np.str_("narma"),
        "split": np.asarray([tr.washout, tr.n_train, tr.n_test], dtype=np.int64),
        "seed": np.int64(cfg.sim.seed),
        "narma_input": np.asarray(narma_input, dtype=np.float64),
        "orders": np.asarray(orders, dtype=np.int64),
        "meta": np.str_(json.dumps(meta, default=float)),
    }
    return payload, chk


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

    chk = None
    try:
        payload, chk = (gen_weather if args.task == "weather" else gen_narma)(
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

    # Trace is safely written — the streamed checkpoint scratch can go now.
    if chk is not None:
        chk.cleanup()

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
