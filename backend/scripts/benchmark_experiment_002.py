"""Benchmark Experiment #2 — Phase 9C tiers.

Runs the new quantum-inspired classical solvers (``parallel_tempering``
and ``simulated_bifurcation``) against every Phase-2 registered
instance, with 5 seeds for each, identical hyperparameters within
each solver class.

This populates the previously-empty "quantum-inspired classical"
column in the Findings dashboard. The first time the Phase 9C
results land, the dashboard renders all four solver-tier categories
the v2 spec promises (classical / QUBO-heuristic / quantum-inspired /
quantum).
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

import dimod  # noqa: E402

from app.benchmarking import bootstrap_default_solvers  # noqa: E402
from app.benchmarking.instances import get_suite, list_suites  # noqa: E402
from app.benchmarking.records import record_run  # noqa: E402
from app.optimization.compiler import compile_cqm_json  # noqa: E402

SEEDS = [42, 1, 2, 3, 4]
LAGRANGE_MULTIPLIER = 10.0

SOLVER_PLAN: list[dict[str, Any]] = [
    {
        "name": "parallel_tempering",
        # PT shares information across replicas via swap proposals,
        # so each sweep is more valuable than an isolated SA sweep.
        # 20k sweeps × 8 replicas = 160k spin-flips, vs. SA's
        # 500k flips at num_reads=500 × num_sweeps=1000 — roughly
        # fair after factoring PT's replica-exchange efficiency.
        "init_kwargs": {"num_replicas": 8, "beta_range": [0.1, 5.0]},
        "sample_kwargs_per_seed": lambda seed: {
            "num_reads": 50, "num_sweeps": 20000, "seed": seed,
        },
    },
    {
        "name": "simulated_bifurcation",
        "init_kwargs": {"mode": "discrete"},
        "sample_kwargs_per_seed": lambda seed: {
            "agents": 64, "max_steps": 2000, "num_reads": 100, "seed": seed,
        },
    },
]


def main() -> int:
    bootstrap_default_solvers()
    suites = list_suites()
    started = time.perf_counter()

    completed = 0
    failed = 0

    print(f"=== Benchmark Experiment #2 — Phase 9C tiers — {len(suites)} suites ===\n")

    for suite_id in suites:
        instances = get_suite(suite_id)
        for inst in instances:
            print(f"\n--- {inst.instance_id} ---")
            cqm_json = inst.load_cqm_json()
            cqm, _registry, sense = compile_cqm_json(cqm_json)
            bqm, _invert = dimod.cqm_to_bqm(
                cqm, lagrange_multiplier=LAGRANGE_MULTIPLIER
            )

            for plan in SOLVER_PLAN:
                for seed in SEEDS:
                    params = {**plan["init_kwargs"], **plan["sample_kwargs_per_seed"](seed)}
                    try:
                        record = record_run(
                            solver_name=plan["name"],
                            instance_id=inst.instance_id,
                            bqm=bqm,
                            parameters=params,
                            archive_samples=True,
                            cqm=cqm,
                            sense=sense,
                            expected_optimum=inst.expected_optimum,
                        )
                        bu = record.results.get("best_user_energy")
                        ms = record.results.get("elapsed_ms")
                        conv = record.results.get("converged_to_expected")
                        conv_flag = (
                            "[match]" if conv else
                            ("[no-match]" if conv is False else "[no-ground-truth]")
                        )
                        print(
                            f"  {plan['name']:24s} seed={seed}: "
                            f"best={bu} {conv_flag}  ({ms:.1f} ms)  "
                            f"id={record.record_id[-6:]}"
                        )
                        completed += 1
                    except Exception as e:  # noqa: BLE001
                        msg = str(e).encode("ascii", errors="replace").decode("ascii")
                        print(
                            f"  {plan['name']:24s} seed={seed}: "
                            f"FAIL - {type(e).__name__}: {msg}"
                        )
                        traceback.print_exc()
                        failed += 1

    elapsed = time.perf_counter() - started
    print(
        f"\n=== Experiment #2 done in {elapsed:.1f}s — "
        f"{completed} completed, {failed} failed ===\n"
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
