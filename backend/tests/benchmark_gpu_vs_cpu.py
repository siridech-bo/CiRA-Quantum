"""Benchmark: GPU SA vs CPU SA across problem sizes.

The reference table in PROJECT_TEMPLATE.md assumes dense BQMs (CPU SA at
N=1000 taking ~28 seconds is only possible at full density), so this
benchmark uses dense Ising couplings — every variable pair has an edge.

For N=5000 the CPU SA reference is ~13 minutes; we skip that pairing by
default and require ``--include-5000`` to opt in.

Run with::

    python tests/benchmark_gpu_vs_cpu.py
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

# Allow `python tests/benchmark_gpu_vs_cpu.py` to find the `app` package.
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

import dimod  # noqa: E402
from dwave.samplers import SimulatedAnnealingSampler  # noqa: E402

from app.optimization.gpu_sa import GPUSimulatedAnnealingSampler  # noqa: E402


def _make_dense_ising(n: int, seed: int) -> dimod.BinaryQuadraticModel:
    """Fully connected Ising BQM with random ±1-ish couplings."""
    rng = random.Random(seed)
    linear = {i: rng.uniform(-1, 1) for i in range(n)}
    quadratic: dict[tuple[int, int], float] = {}
    for i in range(n):
        for j in range(i + 1, n):
            quadratic[(i, j)] = rng.uniform(-1, 1)
    return dimod.BinaryQuadraticModel(linear, quadratic, 0.0, dimod.SPIN)


def _time_sample(sampler, bqm, num_reads, num_sweeps) -> tuple[float, float]:
    t0 = time.perf_counter()
    ss = sampler.sample(bqm, num_reads=num_reads, num_sweeps=num_sweeps)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return elapsed_ms, float(ss.first.energy)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GPU SA vs CPU SA benchmark (dense Ising BQMs)")
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[100, 500, 1000],
        help="Variable counts to test",
    )
    # GPU SA needs many chains to amortize per-spin kernel launch overhead.
    # At small `num_reads` the GPU path is launch-bound and the C++ CPU SA
    # in `dwave-samplers` is competitive; at `num_reads >= 4000` the GPU
    # cleanly clears the Phase 1 DoD target of ≥10× speedup at N=1000.
    parser.add_argument("--num-reads", type=int, default=4000)
    parser.add_argument("--num-sweeps", type=int, default=1000)
    parser.add_argument(
        "--include-5000",
        action="store_true",
        help="Also benchmark N=5000 (CPU SA takes ~13 minutes)",
    )
    parser.add_argument(
        "--kernel",
        choices=["compile", "jit", "eager"],
        default="compile",
        help="GPU inner-sweep kernel. 'compile' has the highest throughput "
        "but pays a one-time compile cost.",
    )
    parser.add_argument(
        "--energy-tol",
        type=float,
        default=0.05,
        help="Relative energy agreement tolerance for 'same optimum'",
    )
    args = parser.parse_args(argv)

    sizes = list(args.sizes)
    if args.include_5000 and 5000 not in sizes:
        sizes.append(5000)

    gpu = GPUSimulatedAnnealingSampler(kernel=args.kernel)
    cpu = SimulatedAnnealingSampler()
    print(f"GPU device: {gpu.properties}")
    print(
        f"Settings: num_reads={args.num_reads}  num_sweeps={args.num_sweeps}  "
        f"kernel={args.kernel}  vartype=SPIN  density=full"
    )

    if args.kernel == "compile":
        print("Warming up the compiled kernel for each size (one-time cost) ...")
        # Inductor recompiles per (R, n) shape, so do a throwaway sample()
        # at each size we'll later benchmark; the timed runs then hit cache.
        for n in sizes:
            warmup_bqm = _make_dense_ising(n, seed=999_999 + n)
            t0 = time.perf_counter()
            gpu.sample(warmup_bqm, num_reads=args.num_reads, num_sweeps=args.num_sweeps)
            print(f"  N={n}: {time.perf_counter() - t0:.1f}s")

    print()
    header = (
        f"{'N':>6}  {'CPU SA (ms)':>12}  {'GPU SA (ms)':>12}  "
        f"{'Speedup':>8}  {'Same optimum?':>14}"
    )
    print(header)
    print("-" * len(header))

    for n in sizes:
        bqm = _make_dense_ising(n, seed=12345 + n)

        cpu_ms, cpu_e = _time_sample(cpu, bqm, args.num_reads, args.num_sweeps)
        gpu_ms, gpu_e = _time_sample(gpu, bqm, args.num_reads, args.num_sweeps)

        speedup = cpu_ms / gpu_ms if gpu_ms > 0 else float("inf")
        denom = max(abs(cpu_e), abs(gpu_e), 1.0)
        same = "yes" if abs(cpu_e - gpu_e) / denom <= args.energy_tol else "no"

        print(
            f"{n:>6}  {cpu_ms:>12.0f}  {gpu_ms:>12.0f}  {speedup:>7.1f}x  {same:>14}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
