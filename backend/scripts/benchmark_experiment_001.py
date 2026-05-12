"""Benchmark Experiment #1 — controlled sweep across all 6 solver tiers.

Runs every registered solver against every Phase-2 registered instance
under deliberate, consistent settings. The output is ~95 new
``RunRecord``s appended to ``benchmarks/archive/``, and a printed
progress log. The findings are interpreted in ``BENCHMARK_REPORT_001.md``
at the repo root.

Experimental settings (chosen to match the Phase-5C dashboard's defaults
and to give stochastic solvers a fair shake):

* Deterministic solvers (``exact_cqm``, ``cpsat``, ``highs``): 1 run
  each, ``seed=42``, ``time_limit=60s``.
* Stochastic solvers (``gpu_sa``, ``cpu_sa_neal``, ``qaoa_sim``): 5
  seeds (``42, 1, 2, 3, 4``) per instance, identical per-class
  hyperparameters. SA-class: ``num_reads=500, num_sweeps=1000``.
  QAOA: ``layer=3, optimizer=SLSQP``.

Instances on which each solver is *skipped*:

* ``exact_cqm`` skips anything that would take > 60s or run out of
  memory (JSS, knapsack_20, maxcut_50). Detected via ``ValueError`` or
  timeout from the solver itself.
* ``highs`` skips quadratic-objective CQMs (``maxcut/*``) — adapter
  rejects with a clear error.
* ``qaoa_sim`` skips CQMs that lower beyond its 20-qubit budget
  (everything except setcover_4item, knapsack_5item, maxcut_6node,
  graphcoloring_4node).

The script tolerates per-(solver, instance, seed) failures: it logs
them and moves on. The archive will reflect the same "honest gaps"
the dashboard already surfaces.

Run with:

    cd backend
    python scripts/benchmark_experiment_001.py
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path
from typing import Any

# Allow `python scripts/benchmark_experiment_001.py` from backend/ root.
sys.path.insert(0, str(Path(__file__).parent.parent))

import dimod  # noqa: E402

from app.benchmarking import bootstrap_default_solvers  # noqa: E402
from app.benchmarking.instances import get_suite, list_suites  # noqa: E402
from app.benchmarking.records import record_run  # noqa: E402
from app.optimization.compiler import compile_cqm_json  # noqa: E402

# ---- Experiment configuration --------------------------------------------

SEEDS_STOCHASTIC = [42, 1, 2, 3, 4]
SEED_DETERMINISTIC = 42

SA_KWARGS = {
    "num_reads": 500,
    "num_sweeps": 1000,
}
GPU_SA_INIT_KWARGS = {"kernel": "jit"}
QAOA_INIT_KWARGS = {"layer": 3, "optimizer": "SLSQP"}

LAGRANGE_MULTIPLIER = 10.0


# Solver category → settings spec. Drives the runner loop.
SOLVER_PLAN: list[dict[str, Any]] = [
    {
        "name": "exact_cqm",
        "kind": "deterministic",
        "init_kwargs": {},
        "sample_kwargs": {},
    },
    {
        "name": "cpsat",
        "kind": "deterministic",
        "init_kwargs": {"num_workers": 4},
        "sample_kwargs": {"time_limit": 60.0, "seed": SEED_DETERMINISTIC},
    },
    {
        "name": "highs",
        "kind": "deterministic",
        "init_kwargs": {"presolve": True},
        "sample_kwargs": {"time_limit": 60.0, "seed": SEED_DETERMINISTIC},
    },
    {
        "name": "gpu_sa",
        "kind": "stochastic",
        "init_kwargs": GPU_SA_INIT_KWARGS,
        "sample_kwargs_per_seed": lambda seed: {**SA_KWARGS, "seed": seed},
    },
    {
        "name": "cpu_sa_neal",
        "kind": "stochastic",
        "init_kwargs": {},
        "sample_kwargs_per_seed": lambda seed: {**SA_KWARGS, "seed": seed},
    },
    {
        "name": "qaoa_sim",
        "kind": "stochastic",
        "init_kwargs": QAOA_INIT_KWARGS,
        "sample_kwargs_per_seed": lambda seed: {"seed": seed},
    },
]


# ---- Runner ---------------------------------------------------------------


def main() -> int:
    bootstrap_default_solvers()

    suites = list_suites()
    started = time.perf_counter()

    completed = 0
    skipped = 0
    failed = 0

    print(f"=== Benchmark Experiment #1 — {len(suites)} suites ===\n")

    for suite_id in suites:
        instances = get_suite(suite_id)
        for inst in instances:
            print(f"\n--- {inst.instance_id} ---")
            cqm_json = inst.load_cqm_json()
            cqm, _registry, sense = compile_cqm_json(cqm_json)
            bqm, _invert = dimod.cqm_to_bqm(cqm, lagrange_multiplier=LAGRANGE_MULTIPLIER)

            for plan in SOLVER_PLAN:
                runs = (
                    [(SEED_DETERMINISTIC, plan["sample_kwargs"])]
                    if plan["kind"] == "deterministic"
                    else [(seed, plan["sample_kwargs_per_seed"](seed)) for seed in SEEDS_STOCHASTIC]
                )

                for seed, sample_kwargs in runs:
                    params = {**plan["init_kwargs"], **sample_kwargs}
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
                            f"  {plan['name']:12s} seed={seed}: "
                            f"best={bu} {conv_flag}  ({ms:.1f} ms)  "
                            f"id={record.record_id[-6:]}"
                        )
                        completed += 1
                    except UnicodeError as e:  # narrower than ValueError
                        # Print-encoding glitches mustn't masquerade as
                        # adapter rejections — surface them loudly.
                        print(
                            f"  {plan['name']:12s} seed={seed}: "
                            f"PRINT-ENCODE-FAIL ({e}) but record may have been written"
                        )
                        failed += 1
                    except ValueError as e:
                        # Honest gap — adapter rejected this (qubit budget,
                        # quadratic objective, REAL variables, ...).
                        msg = str(e).encode("ascii", errors="replace").decode("ascii")
                        print(f"  {plan['name']:12s} seed={seed}: SKIP - {msg}")
                        skipped += 1
                    except Exception as e:  # noqa: BLE001
                        msg = str(e).encode("ascii", errors="replace").decode("ascii")
                        print(
                            f"  {plan['name']:12s} seed={seed}: "
                            f"FAIL - {type(e).__name__}: {msg}"
                        )
                        traceback.print_exc()
                        failed += 1

    elapsed = time.perf_counter() - started
    print(
        f"\n=== Experiment #1 done in {elapsed:.1f}s — "
        f"{completed} completed, {skipped} skipped (honest gaps), {failed} failed ===\n"
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
