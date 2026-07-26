"""Publication-grade QRC scaling sweep, N = 3 .. 9 (durable, incremental).

Runs memory-capacity + NARMA-2 (with a size-matched classical ESN
comparison) across qubit counts, holding everything else fixed so the
only variable is the reservoir's Hilbert-space dimension. Uses the
auto-selected evolution backend (dense propagator for N<=6, exact sparse
Krylov ``action`` for N>=7) so N=9 (512x512 density matrix) is feasible.

Each N's result is appended to a JSONL file *as it completes*, so a long
run (N=9 alone is ~70-90 min) yields durable partial results even if
interrupted. A final summary + consolidated JSON are written at the end.

Run:  PYTHONIOENCODING=utf-8 python scripts/qrc_scaling_run.py
Override qubit list:  ... scripts/qrc_scaling_run.py 3 4 5 6 7 8 9
"""
from __future__ import annotations

import json
import sys
import time

from app.qrc.benchmarks import (
    ScalingResult,
    esn_baseline,
    run_memory_capacity,
    run_narma,
)
from app.qrc.config import (
    EncodingConfig,
    QRCConfig,
    SimConfig,
    TrainingConfig,
    make_system_config,
)
from app.qrc.features import FeatureConfig
from app.qrc.tasks import narma_sequence

JSONL = "qrc_scaling_progress.jsonl"
OUT_JSON = "qrc_scaling_results.json"

# Fair scaling design: n_train must exceed the readout dimension
# (3 * N * V) at every N, otherwise MC and NARMA are capped by the
# training-set size rather than the reservoir — the confound that made
# the first (n_train=150) sweep decline past N=5. At V=8 the readout is
# 3*9*8 = 216 features at N=9, so n_train=350 keeps ~1.6x headroom.
#
# Backend: the GPU exp(t·L) path is ~28x faster than CPU at N=9
# (~4 min/point vs ~1.75 h), so if a CUDA torch is present we use it for
# N>=6 and the whole N=3..9 sweep finishes in ~15 min. Falls back to the
# CPU 'auto' path otherwise.
SIM = dict(tau=0.03, n_virtual=8)
TRAIN = dict(device="cpu", washout=100, n_train=350, n_test=120, cv_folds=3)
MAX_DELAY = 25
NARMA_ORDER = 2
MC_STEPS = TRAIN["washout"] + TRAIN["n_train"] + TRAIN["n_test"]  # 570


def _mode_for(n: int) -> str:
    """GPU for N>=6 when CUDA is available, else the CPU auto path."""
    try:
        import torch
        if torch.cuda.is_available() and n >= 6:
            return "gpu"
    except Exception:
        pass
    return "auto"


def run_one(n: int) -> dict:
    cfg = QRCConfig(
        system=make_system_config(n),
        sim=SimConfig(**SIM, evolution_mode=_mode_for(n)),
        encoding=EncodingConfig(fn="arcsin_sqrt"),
        training=TrainingConfig(**TRAIN),
    )
    fc = FeatureConfig(observables=("x", "y", "z"))
    t0 = time.time()
    mc = run_memory_capacity(cfg, max_delay=MAX_DELAY, n_steps=MC_STEPS, feature_cfg=fc)
    na = run_narma(cfg, order=NARMA_ORDER, feature_cfg=fc)
    # size-matched ESN on the same NARMA task
    n_res = 3 * n * cfg.sim.n_virtual
    n_steps = TRAIN["washout"] + TRAIN["n_train"] + TRAIN["n_test"]
    u, y = narma_sequence(n_steps, NARMA_ORDER, seed=cfg.sim.seed)
    esn = esn_baseline(u, y, n_reservoir=n_res, tr_cfg=cfg.training, seed=cfg.sim.seed)
    return {
        "n_qubits": n,
        "hilbert_dim": 2**n,
        "memory_capacity": mc.total_mc,
        "mc_gate_passed": mc.passed_gate,
        "mc_per_delay": mc.per_delay,
        "narma_nmse": na.metrics["nmse"],
        "esn_narma_nmse": esn["nmse"],
        "backend": mc.backend,
        "seconds": time.time() - t0,
    }


def main() -> None:
    ns = [int(x) for x in sys.argv[1:]] or [3, 4, 5, 6, 7, 8, 9]
    open(JSONL, "w").close()  # truncate
    results = []
    for n in ns:
        print(f"[scaling] N={n} (dim={2**n}) running ...", flush=True)
        r = run_one(n)
        results.append(r)
        with open(JSONL, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(r, default=float) + "\n")
        print(
            f"[scaling] N={n}: MC={r['memory_capacity']:.3f} "
            f"QRC NMSE={r['narma_nmse']:.3e} ESN NMSE={r['esn_narma_nmse']:.3e} "
            f"gate={'ok' if r['mc_gate_passed'] else 'FAIL'} "
            f"({r['seconds']/60:.1f} min)",
            flush=True,
        )

    sr = ScalingResult(
        qubit_counts=[r["n_qubits"] for r in results],
        memory_capacity=[r["memory_capacity"] for r in results],
        narma_nmse=[r["narma_nmse"] for r in results],
        mc_gate_passed=[r["mc_gate_passed"] for r in results],
        esn_narma_nmse=[r["esn_narma_nmse"] for r in results],
        details=results,
    )
    print("\n" + sr.summary())
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump({"summary": sr.summary(), "details": results}, fh, indent=2,
                  default=float)
    print(f"\nsaved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
