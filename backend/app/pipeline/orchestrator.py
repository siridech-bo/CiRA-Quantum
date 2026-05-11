"""Five-stage solve pipeline.

Stages, in order, with the status string emitted onto the event bus:

  1. formulating — the LLM provider turns prose into cqm_v1 JSON.
  2. compiling   — JSON → ``dimod.ConstrainedQuadraticModel``.
  3. validating  — three-layer validation harness (Layer A oracle when
                   the CQM is small enough; Layer C constraint coverage
                   always; Layer B *skipped by default* for live solves
                   to keep latency reasonable — Phase 5C's dashboard is
                   where multi-solver agreement is surfaced).
  4. solving     — CQM → BQM (Lagrange penalty) → sample. The sampler
                   is injected so tests can pass a mock; production
                   uses ``GPUSimulatedAnnealingSampler(kernel="jit")``.
  5. complete    — final job row update + the interpreted solution.

Any stage failure flips the job to ``status='error'`` with the message
attached. The pipeline never re-raises — the orchestrator is the only
code that knows about the job row, so it owns the failure mode.

Synchronous-in-one-process: Phase 4 ships this as ``async def run(...)``
launched on a background thread per request. Phase 6 swaps the thread
for a Redis-backed RQ worker with the same public interface.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any

import dimod

from app.formulation import get_provider
from app.optimization.compiler import compile_cqm_json
from app.optimization.interpreter import interpret_solution
from app.optimization.validation import validate_cqm
from app.pipeline.events import EventBus


class PipelineError(Exception):
    """Stage failure inside the orchestrator. Carries the stage name in
    its message so the user-facing error string is informative."""


# Lagrange multiplier for CQM → BQM lowering. Phase 2 settled on 10.0 as
# the production default; surfacing it here so callers (and the Phase 5C
# dashboard's reproduction tooling) can override per-run.
DEFAULT_LAGRANGE = 10.0


class Orchestrator:
    """The five-stage pipeline. Holds the injectable dependencies that
    tests want to mock; production constructs with no arguments."""

    def __init__(
        self,
        *,
        sampler: Any = None,
        provider_resolver: Callable[[str], Any] | None = None,
        lagrange_multiplier: float = DEFAULT_LAGRANGE,
        num_reads: int = 1000,
        num_sweeps: int = 1000,
    ):
        self.sampler = sampler
        self.provider_resolver = provider_resolver or get_provider
        self.lagrange_multiplier = lagrange_multiplier
        self.num_reads = num_reads
        self.num_sweeps = num_sweeps

    def _build_sampler(self) -> Any:
        """Lazy-construct the production sampler so a CUDA-less host
        running the unit-test suite never instantiates it."""
        if self.sampler is not None:
            return self.sampler
        # Imported lazily so importing this module doesn't require CUDA.
        from app.optimization.gpu_sa import GPUSimulatedAnnealingSampler
        self.sampler = GPUSimulatedAnnealingSampler(kernel="jit")
        return self.sampler

    async def run(
        self,
        *,
        job_id: str,
        user_id: int,
        problem_statement: str,
        provider_name: str,
        api_key: str,
        event_bus: EventBus,
    ) -> None:
        """End-to-end pipeline. Updates the ``jobs`` row at each stage
        and emits SSE events; never raises — every failure path lands
        on the ``error`` status."""
        # Import models here, not at module load, so the orchestrator
        # module is import-safe without an initialized SQLite schema
        # (matters for the auto-bootstrap behavior in ``app.formulation``).
        from app.models import update_job

        try:
            # ---- Stage 1: formulate -----------------------------------------
            event_bus.emit(job_id, "formulating")
            update_job(job_id, status="formulating")

            provider = self.provider_resolver(provider_name)
            formulation = await provider.formulate(problem_statement, api_key)

            update_job(
                job_id,
                cqm_json=json.dumps(formulation.cqm_json),
                variable_registry=json.dumps(formulation.variable_registry),
            )

            # ---- Stage 2: compile -------------------------------------------
            event_bus.emit(job_id, "compiling")
            update_job(job_id, status="compiling")

            cqm, registry, sense = compile_cqm_json(formulation.cqm_json)
            update_job(
                job_id,
                num_variables=len(cqm.variables),
                num_constraints=len(cqm.constraints),
            )
            # Counts land in the DB and are surfaced by the subsequent
            # validation/solving events; we don't double-emit "compiling"
            # because the SSE consumer expects one event per stage.

            # ---- Stage 3: validate ------------------------------------------
            event_bus.emit(job_id, "validating")
            update_job(job_id, status="validating")

            expected = (
                formulation.cqm_json.get("test_instance") or {}
            ).get("expected_optimum")
            # Layer B (multi-solver agreement) is intentionally skipped here
            # — it'd add 30-60 s per request. Phase 5C's dashboard runs the
            # full three-solver agreement on archived RunRecords offline.
            report = validate_cqm(
                cqm,
                expected_optimum=expected,
                sense=sense,
                skip_layer_b=True,
            )
            report_payload = _serialize_validation_report(report)
            update_job(job_id, validation_report=json.dumps(report_payload))

            if not report.passed:
                first_reason = next(iter(report.warnings), "validation failed")
                raise PipelineError(f"validation: {first_reason}")

            # ---- Stage 4: solve ---------------------------------------------
            event_bus.emit(job_id, "solving")
            update_job(job_id, status="solving")

            t0 = time.perf_counter()
            bqm, invert = dimod.cqm_to_bqm(
                cqm, lagrange_multiplier=self.lagrange_multiplier
            )
            sampler = self._build_sampler()
            sampleset = sampler.sample(
                bqm, num_reads=self.num_reads, num_sweeps=self.num_sweeps,
            )
            elapsed_ms = int((time.perf_counter() - t0) * 1000)

            # Map BQM-space sample back to CQM space.
            best_bqm = sampleset.first.sample
            cqm_sample = invert(best_bqm)

            # ---- Stage 5: interpret + complete ------------------------------
            interpreted = interpret_solution(cqm_sample, registry, cqm, sense=sense)
            # JSON keys must be strings, even when the CQM uses int variable
            # labels — keep the conversion explicit. dimod returns numpy
            # scalars (int8 for binaries, int64 for integers) which the
            # default JSON encoder rejects, so we coerce to Python natives.
            solution_payload = {
                str(k): (v.item() if hasattr(v, "item") else v)
                for k, v in cqm_sample.items()
            }
            update_job(
                job_id,
                status="complete",
                solution=json.dumps(solution_payload),
                interpreted_solution=interpreted,
                solve_time_ms=elapsed_ms,
                completed_at=datetime.utcnow().isoformat(),
            )
            event_bus.emit(job_id, "complete", solve_time_ms=elapsed_ms)

        except PipelineError as e:
            update_job(
                job_id,
                status="error",
                error=str(e),
                completed_at=datetime.utcnow().isoformat(),
            )
            event_bus.emit(job_id, "error", error=str(e))
        except Exception as e:
            # Anything else (LLM HTTP failure, compiler ValueError, sampler
            # timeout, programmer error) becomes a job-level error too —
            # the route handler never sees a raw exception.
            msg = f"{type(e).__name__}: {e}"
            update_job(
                job_id,
                status="error",
                error=msg,
                completed_at=datetime.utcnow().isoformat(),
            )
            event_bus.emit(job_id, "error", error=msg)


def _serialize_validation_report(report: Any) -> dict[str, Any]:
    """Coerce a ``ValidationReport`` dataclass into a JSON-friendly dict.

    The ``passed`` field is a property (computed from the three layer
    booleans), so ``asdict`` skips it; we add it back explicitly.
    """
    base = asdict(report) if is_dataclass(report) else dict(report)
    base["passed"] = bool(getattr(report, "passed", False))
    return base
