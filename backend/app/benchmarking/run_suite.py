"""CLI Benchmark runner.

Run a registered solver over every instance in a registered suite, write
a ``RunRecord`` per instance to ``benchmarks/archive/``, and print a
summary table.

Usage::

    python -m app.benchmarking.run_suite \
        --solver gpu_sa --instances knapsack/small

Notes
-----
The runner converts each instance's CQM to a BQM via
``dimod.cqm_to_bqm`` before sampling, except for ``exact_cqm`` which
consumes the CQM natively. The ``--lagrange`` flag controls the BQM
penalty multiplier; tuning it is part of the Benchmark methodology and
the chosen value is faithfully recorded in the per-instance RunRecord
parameters dict so future benchmarks can reproduce or revise it.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any

import dimod

from app.benchmarking import record_run
from app.benchmarking.instances import InstanceMetadata, get_suite, list_suites
from app.benchmarking.registry import get_solver
from app.optimization.compiler import compile_cqm_json


def run_suite(
    *,
    solver_name: str,
    suite_id: str,
    num_reads: int,
    num_sweeps: int,
    seed: int | None,
    lagrange: float,
    kernel: str | None = None,
    archive_samples: bool = True,
) -> list:
    """Iterate every instance in ``suite_id`` and record one run per
    instance. Returns the list of ``RunRecord``s."""
    # Validate solver and suite up front so a missing argument doesn't burn
    # an instance's worth of solver time before failing.
    get_solver(solver_name)
    instances = get_suite(suite_id)

    records = []
    for inst in instances:
        cqm_json = inst.load_cqm_json()
        cqm, _registry, sense = compile_cqm_json(cqm_json)

        params: dict[str, Any] = {}
        # `num_reads`/`num_sweeps` are SA-class kwargs; the exact CQM
        # solver, CP-SAT, and HiGHS don't take them. Seed is more
        # universal — most solvers accept it (CP-SAT, HiGHS both do).
        if solver_name in ("gpu_sa", "cpu_sa_neal"):
            params.update({
                "num_reads": num_reads,
                "num_sweeps": num_sweeps,
            })
        if solver_name in ("gpu_sa", "cpu_sa_neal", "cpsat", "highs") and seed is not None:
            params["seed"] = seed
        # `kernel` only applies to GPU SA. Recording it in `parameters`
        # makes the kernel mode reproducible per the v2 Phase-2 note.
        if solver_name == "gpu_sa" and kernel is not None:
            params["kernel"] = kernel

        if solver_name == "exact_cqm":
            bqm = dimod.BinaryQuadraticModel.empty(dimod.BINARY)  # not used
        else:
            bqm, _invert = dimod.cqm_to_bqm(cqm, lagrange_multiplier=lagrange)

        record = record_run(
            solver_name=solver_name,
            instance_id=inst.instance_id,
            bqm=bqm,
            parameters=params,
            archive_samples=archive_samples,
            cqm=cqm,
            sense=sense,
            expected_optimum=inst.expected_optimum,
        )
        records.append((inst, record))
    return records


def _print_summary(records: list[tuple[InstanceMetadata, Any]]) -> None:
    print(
        f"{'instance':<48} {'best_user':>12} {'expected':>10} {'conv':>5} "
        f"{'feasible':>9} {'time(ms)':>10}  record_id"
    )
    print("-" * 116)
    for inst, rec in records:
        bu = rec.results.get("best_user_energy")
        bu_s = f"{bu:.4g}" if bu is not None else "-"
        exp = inst.expected_optimum
        exp_s = f"{exp:.4g}" if exp is not None else "?"
        conv = rec.results.get("converged_to_expected")
        # ✓ when known optimum reached; ✗ when miss confirmed; "?" when no ground truth.
        conv_s = "?" if conv is None else ("yes" if conv else "no")
        feas = rec.results.get("num_feasible")
        feas_s = "-" if feas is None else str(feas)
        elapsed = rec.results.get("elapsed_ms", 0.0)
        print(
            f"{inst.instance_id:<48} {bu_s:>12} {exp_s:>10} {conv_s:>5} "
            f"{feas_s:>9} {elapsed:>10.0f}  {rec.record_id}"
        )


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.benchmarking.run_suite",
        description="Run a registered solver over a registered instance suite.",
    )
    parser.add_argument("--solver", required=True, help="Registered solver name")
    parser.add_argument(
        "--instances",
        required=True,
        help=f"Suite ID. Known suites: {', '.join(list_suites())}",
    )
    parser.add_argument("--num-reads", type=int, default=1000)
    parser.add_argument("--num-sweeps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lagrange", type=float, default=10.0)
    parser.add_argument(
        "--kernel",
        choices=["jit", "compile", "eager"],
        default="jit",
        help="Inner-sweep kernel (GPU SA only). 'compile' has highest "
        "throughput but pays a one-time compile cost.",
    )
    parser.add_argument(
        "--no-archive-samples",
        action="store_true",
        help="Skip writing the gzipped SampleSet alongside each record.",
    )
    args = parser.parse_args(argv)

    t0 = time.perf_counter()
    try:
        records = run_suite(
            solver_name=args.solver,
            suite_id=args.instances,
            num_reads=args.num_reads,
            num_sweeps=args.num_sweeps,
            seed=args.seed,
            lagrange=args.lagrange,
            kernel=args.kernel,
            archive_samples=not args.no_archive_samples,
        )
    except KeyError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    elapsed = time.perf_counter() - t0

    _print_summary(records)
    print(f"\nWrote {len(records)} records in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
