"""QRC CLI — run the pipeline end to end.

Examples
--------
    # Fading-memory gate on the 3-qubit SPINQ system (run this first):
    python -m app.qrc.main memory --system spinq3

    # NARMA-2 on SPINQ:
    python -m app.qrc.main narma --system spinq3 --order 2

    # The headline scaling study (3 → 9 qubits) proving benefit of size:
    python -m app.qrc.main scaling --qubits 3 5 7 9

    # Use the GPU (JAX) backend for the larger systems:
    python -m app.qrc.main scaling --qubits 5 7 9 --backend jax

Results print as tables and are optionally saved to JSON (--out).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys

from app.qrc.config import (
    EncodingConfig,
    QRCConfig,
    SimConfig,
    TrainingConfig,
    make_system_config,
)
from app.qrc.features import FeatureConfig


def _sim_from_args(a) -> SimConfig:
    return SimConfig(
        tau=a.tau, n_virtual=a.virtual, backend=a.backend,
        evolution_mode=a.evolution, seed=a.seed,
    )


def _feature_cfg(a) -> FeatureConfig:
    obs = tuple(a.observables)
    return FeatureConfig(
        observables=obs,
        spectral=a.multimodal,
        time_domain=a.multimodal,
        wavelet=a.multimodal,
        nonlinear=a.multimodal,
    )


def _dump(obj, path):
    def default(o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        raise TypeError(o)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=default)
    print(f"\nsaved → {path}")


def cmd_memory(a):
    from app.qrc.benchmarks import run_memory_capacity

    cfg = QRCConfig(
        system=make_system_config(a.system if a.system else a.n),
        sim=_sim_from_args(a),
        encoding=EncodingConfig(fn=a.encoding),
        training=TrainingConfig(device=a.device, seed=a.seed),
    )
    print(f"config hash: {cfg.repro_hash()}")
    res = run_memory_capacity(cfg, max_delay=a.max_delay, n_steps=a.steps,
                              feature_cfg=_feature_cfg(a))
    print(f"\nMemory capacity (N={res.n_qubits}, backend={res.backend}):")
    print(f"  total MC = {res.total_mc:.3f}")
    print(f"  gate     = {'PASS' if res.passed_gate else 'FAIL'}")
    print("  corr^2 by delay:")
    for d in sorted(res.per_delay):
        bar = "#" * int(res.per_delay[d] * 40)
        print(f"    d={d:>2}: {res.per_delay[d]:.3f} {bar}")
    if a.out:
        _dump(res, a.out)


def cmd_narma(a):
    from app.qrc.benchmarks import run_narma

    cfg = QRCConfig(
        system=make_system_config(a.system if a.system else a.n),
        sim=_sim_from_args(a),
        encoding=EncodingConfig(fn=a.encoding),
        training=TrainingConfig(device=a.device, seed=a.seed),
    )
    print(f"config hash: {cfg.repro_hash()}")
    res = run_narma(cfg, order=a.order, feature_cfg=_feature_cfg(a))
    print(f"\nNARMA{res.order} (N={res.n_qubits}):")
    for k, v in res.metrics.items():
        print(f"  {k:>7}: {v}")
    if a.out:
        _dump(res, a.out)


def cmd_scaling(a):
    from app.qrc.benchmarks import scaling_study

    res = scaling_study(
        qubit_counts=tuple(a.qubits),
        sim=_sim_from_args(a),
        encoding=EncodingConfig(fn=a.encoding),
        training=TrainingConfig(device=a.device, seed=a.seed),
        feature_cfg=_feature_cfg(a),
        max_delay=a.max_delay,
        narma_order=a.order,
    )
    print("\n" + res.summary())
    if a.out:
        _dump({"details": res.details, "summary": res.summary()}, a.out)


def build_parser() -> argparse.ArgumentParser:
    # Plain-ASCII description: the module docstring contains Unicode
    # (arrows) that a Windows cp1252 console can't encode when argparse
    # prints --help.
    p = argparse.ArgumentParser(
        prog="app.qrc.main",
        description="Quantum Reservoir Computing simulator CLI "
        "(memory-capacity gate, NARMA, N-qubit scaling study).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--system", type=str, default="spinq3",
                        help="preset name (spinq3, crotonic9) or empty to use --n")
        sp.add_argument("--n", type=int, default=3, help="qubit count if no preset")
        sp.add_argument("--tau", type=float, default=0.03)
        sp.add_argument("--virtual", type=int, default=25)
        sp.add_argument("--backend", choices=["numpy", "jax"], default="numpy")
        sp.add_argument("--evolution",
                        choices=["auto", "propagator", "action", "mesolve", "gpu"],
                        default="auto",
                        help="time-evolution backend; 'gpu' needs CUDA torch "
                             "and is ~28x faster than 'action' at N=9")
        sp.add_argument("--encoding", default="arcsin_sqrt")
        sp.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
        sp.add_argument("--observables", nargs="+", default=["x", "y", "z"],
                        help="readout axes; sz alone is frozen by the ZZ "
                             "Hamiltonian, so transverse sx/sy are essential")
        sp.add_argument("--multimodal", action="store_true",
                        help="add spectral/time-domain/wavelet/entropy features")
        sp.add_argument("--seed", type=int, default=42)
        sp.add_argument("--out", type=str, default=None, help="save results JSON")

    sp = sub.add_parser("memory", help="memory-capacity fading-memory gate")
    common(sp)
    sp.add_argument("--max-delay", type=int, default=30)
    sp.add_argument("--steps", type=int, default=1200)
    sp.set_defaults(func=cmd_memory)

    sp = sub.add_parser("narma", help="NARMA-n benchmark")
    common(sp)
    sp.add_argument("--order", type=int, default=2)
    sp.set_defaults(func=cmd_narma)

    sp = sub.add_parser("scaling", help="N-qubit scaling study (proof of benefit)")
    common(sp)
    sp.add_argument("--qubits", nargs="+", type=int, default=[3, 5, 7, 9])
    sp.add_argument("--max-delay", type=int, default=30)
    sp.add_argument("--order", type=int, default=2)
    sp.set_defaults(func=cmd_scaling)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
