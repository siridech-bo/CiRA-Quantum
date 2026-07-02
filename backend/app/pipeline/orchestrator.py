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

# Per-solver wall-clock timeouts (seconds). If a sampler hangs past
# this, the orchestrator abandons it and moves on. Critical for
# qaoa_originqc — pyqpanda3's ``backend.run()`` can block indefinitely
# on certain Origin cloud-side states (Phase 9D investigation in
# progress). The orphaned worker thread keeps running in the
# background; it'll free its resources whenever pyqpanda3 finally
# returns or the Flask process restarts.
_SOLVER_WALLCLOCK_TIMEOUT_S: dict[str, float] = {
    "qaoa_originqc": 90.0,
    "qaoa_sim": 60.0,
    # M1 places a generous cap so sync sample() works for early
    # testing. M2 will short-circuit qaoa_ibmq into the async path
    # which doesn't need a sync timeout at all (the orchestrator
    # returns immediately after submit_async).
    "qaoa_ibmq": 900.0,
}
_DEFAULT_SOLVER_TIMEOUT_S: float = 30.0


def _classify_solver_failure(exc: BaseException) -> tuple[str, str]:
    """Decide whether a sampler exception is a hard error or a soft skip.

    QAOA tiers raise ``ValueError`` with a specific shape when the
    lowered BQM is bigger than the tier's qubit cap (local simulator
    RAM ceiling, configured IBMQ cap, Origin backend budget). That
    isn't a bug — the sampler is correctly refusing a workload that
    wouldn't fit on its qubits. Classify those rows as ``skipped`` so
    the comparison table doesn't render red error chips for every
    medium-sized instance that hits an over-budget QAOA tier.

    Returns ``(status, message)``. The skipped message is the
    sampler's own human-readable string verbatim (no ``ValueError:``
    prefix); the error message keeps the type prefix so operator
    debugging stays informative.
    """
    msg = str(exc)
    if isinstance(exc, ValueError) and "exceeds" in msg and (
        "qubit cap" in msg
        or "qubit budget" in msg
        or "configured cap" in msg
    ):
        return "skipped", msg
    return "error", f"{type(exc).__name__}: {exc}"


# Per-solver parameters used by the multi-solver fan-out (Phase 5D).
# Conservative defaults — the goal is interactive-latency comparison
# across the whole registry, not the best possible per-solver score
# (the Benchmark dashboard's Experiment scripts already do the latter).
# Keys must match each sampler's actual __init__ / sample() signatures
# exactly; _split_parameters routes init-only kwargs vs sample kwargs
# from records._INIT_ONLY_KWARGS_BY_SOLVER.
_DEFAULT_PARAMS_BY_SOLVER: dict[str, dict[str, Any]] = {
    "gpu_sa":                {"kernel": "jit", "num_reads": 200, "num_sweeps": 500},
    "cpu_sa_neal":           {"num_reads": 200, "num_sweeps": 500},
    "parallel_tempering":    {"num_reads": 50, "num_sweeps": 1000, "num_replicas": 8},
    "simulated_bifurcation": {"max_steps": 500, "agents": 128},
    "qaoa_sim":              {"layer": 1, "max_qubits": 12},
    "qaoa_originqc":         {"layer": 1, "max_qubits": 7, "shots": 200},
    "qaoa_ibmq":             {"layer": 1, "max_qubits": 20, "shots": 200},
    "cpsat":                 {"time_limit": 5.0},
    "highs":                 {"time_limit": 5.0},
    "exact_cqm":             {},
}


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

    def _run_multi_solver(
        self,
        cqm: dimod.ConstrainedQuadraticModel,
        sense: str,
        solver_names: list[str],
        user_id: int,
        *,
        job_id: str,
        event_bus: EventBus,
        solver_params_overrides: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Run each registered solver on the same problem and return a
        per-solver summary dict. CQM-native solvers consume the CQM
        directly; everything else gets the BQM lowered once and reused.
        One solver failing is captured per-row — it never aborts the
        fan-out.

        ``user_id`` is needed so BYOK-bearing solvers (``qaoa_originqc``)
        can look up the user's stored credential at runtime instead of
        relying on a server-level env var.

        ``solver_params_overrides`` is a per-request escape hatch — a
        ``{solver_name: {param: value}}`` dict that gets merged on top of
        ``_DEFAULT_PARAMS_BY_SOLVER[name]``. Used by the frontend's
        "Origin backend" selector to route a single solve to Wukong /
        Hanyuan instead of the default cloud simulator, without touching
        the module-level defaults every other request depends on.
        """
        overrides = solver_params_overrides or {}
        # Lazy imports keep this module free of CUDA / dwave hard deps.
        from app.benchmarking.records import _split_parameters
        from app.benchmarking.registry import bootstrap_default_solvers, get_solver

        bootstrap_default_solvers()

        # Lower CQM → BQM once. The penalty is applied here so every
        # BQM-style solver sees the same lifted problem.
        bqm, invert = dimod.cqm_to_bqm(
            cqm, lagrange_multiplier=self.lagrange_multiplier
        )

        from app.models import update_job

        def _publish(partial: dict[str, dict[str, Any]]) -> None:
            """Persist a snapshot of the per-solver results dict + emit
            an SSE so the frontend can render live progress. Strips the
            private _cqm_sample copies before serializing."""
            public = {
                k: {kk: vv for kk, vv in v.items() if kk != "_cqm_sample"}
                for k, v in partial.items()
            }
            payload = json.dumps({
                "solvers": public,
                "primary": None,  # picked at the end of stage 4
                "sense": sense,
                "in_progress": True,
            })
            try:
                update_job(job_id, solver_results=payload)
            except Exception:  # pragma: no cover — defensive
                pass
            event_bus.emit(job_id, "solver_progress")

        results: dict[str, dict[str, Any]] = {
            # Seed pending rows for every requested solver up-front so
            # the frontend can render the checklist immediately.
            n: {"status": "pending"} for n in solver_names
        }
        _publish(results)

        for name in solver_names:
            t0 = time.perf_counter()
            # Mark this solver as actively running so the UI can show a
            # spinner on the right row.
            results[name] = {"status": "running"}
            _publish(results)
            try:
                ident, sampler_cls = get_solver(name)
            except KeyError as e:
                results[name] = {
                    "status": "error",
                    "error": str(e),
                    "elapsed_ms": 0,
                }
                _publish(results)
                continue

            is_cqm_native = (
                sampler_cls.__name__ == "ExactCQMSolver"
                or getattr(sampler_cls, "_CQM_NATIVE", False)
            )
            params = dict(_DEFAULT_PARAMS_BY_SOLVER.get(name, {}))
            # Per-solve overrides (e.g. origin_backend=WK_C180 for a
            # Wukong run). Merged on top of the defaults so the module
            # constant stays canonical.
            if overrides.get(name):
                params.update(overrides[name])
            # Per-backend qubit budget for qaoa_originqc. The module
            # constant's ``max_qubits: 7`` matches Origin's cloud
            # ``full_amplitude`` simulator budget. Real superconducting
            # chips (Wukong 180, 180-2, Hanyuan) have 180 physical
            # qubits — realistically usable for QAOA at reasonable
            # depth is ~10-15 before transpile blows up. We use 12 as
            # a safe upper bound that fits every gallery template
            # marked qpu_ready without pushing compile times. Bumping
            # only happens when the user hasn't set max_qubits
            # explicitly via overrides (they can't today, but this
            # keeps the logic future-proof).
            if (
                name == "qaoa_originqc"
                and params.get("backend_name") in {
                    "WK_C180", "WK_C180_2", "HanYuan_01",
                }
                and (overrides.get(name, {}) or {}).get("max_qubits") is None
            ):
                params["max_qubits"] = 12

            # BYOK injection. qaoa_originqc requires an Origin Quantum
            # API key stored under provider "qpanda" in the api_keys
            # table; the key is decrypted just-in-time and passed to
            # the sampler constructor. Missing key → friendly row
            # error so the other solvers in the fan-out still run.
            byok_err = _inject_byok_credentials(name, params, user_id)
            if byok_err is not None:
                results[name] = {
                    "status": "error",
                    "error": byok_err,
                    "elapsed_ms": int((time.perf_counter() - t0) * 1000),
                    "tier_source": ident.source,
                    "version": ident.version,
                    "hardware": ident.hardware,
                }
                _publish(results)
                continue

            init_kwargs, sample_kwargs = _split_parameters(name, params)

            # Phase 11 M2 — async submit for cloud solvers that
            # implement ``submit_async``. The orchestrator submits the
            # cloud job, stashes the cloud_job_id + materialize context
            # in the pending-jobs queue, and records a "queued" row
            # without ever blocking. The pending-jobs poller picks the
            # job up later and fills in the row when the cloud is done.
            #
            # This avoids the pain of long IBM queue times (30 min to
            # hours on the free Open Plan) and Origin's intermittent
            # pilot-task rejections. Either way, the solve UI completes
            # in seconds with classical results; cloud rows materialize
            # whenever they're ready.
            if hasattr(sampler_cls, "submit_async"):
                if _try_submit_async(
                    name=name, sampler_cls=sampler_cls,
                    init_kwargs=init_kwargs, sample_kwargs=sample_kwargs,
                    bqm=bqm, ident=ident, results=results,
                    job_id=job_id, sense=sense, t0=t0,
                ):
                    _publish(results)
                    continue
                # If the async submit itself fails (e.g. auth error
                # detected immediately at submit time), the helper
                # records the error row and we move on. No exception
                # propagation needed — _try_submit_async returns True
                # in both the queued and error-row cases.

            # Wall-clock timeout per solver. If the sampler blocks past
            # this, we abandon it (the worker thread keeps running in
            # the background; it'll exit whenever pyqpanda3 / the
            # underlying lib finally returns). The orchestrator thread
            # stays free to continue the fan-out.
            timeout_s = _SOLVER_WALLCLOCK_TIMEOUT_S.get(
                name, _DEFAULT_SOLVER_TIMEOUT_S,
            )

            def _run_sampler() -> Any:
                sampler = sampler_cls(**init_kwargs)
                if is_cqm_native:
                    return sampler.sample_cqm(cqm, **sample_kwargs)
                return sampler.sample(bqm, **sample_kwargs)

            try:
                import concurrent.futures as _cf
                with _cf.ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix=f"solver-{name}",
                ) as pool:
                    future = pool.submit(_run_sampler)
                    try:
                        sampleset = future.result(timeout=timeout_s)
                    except _cf.TimeoutError:
                        # We're abandoning the future intentionally —
                        # the C-extension thread will keep running, but
                        # we can't kill it cooperatively from Python.
                        raise RuntimeError(
                            f"Solver {name!r} did not return within "
                            f"{timeout_s:.0f} s (wall-clock cap). The "
                            f"underlying call may still be running; "
                            f"the orchestrator moves on."
                        )
            except Exception as e:
                status_, error_msg = _classify_solver_failure(e)
                results[name] = {
                    "status": status_,
                    "error": error_msg,
                    "elapsed_ms": int((time.perf_counter() - t0) * 1000),
                    "tier_source": ident.source,
                    "version": ident.version,
                    "hardware": ident.hardware,
                }
                _publish(results)
                continue

            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            raw_energy = _finite_or_none(sampleset.first.energy)
            best_sample = dict(sampleset.first.sample)
            cqm_sample = best_sample if is_cqm_native else invert(best_sample)

            # Honest feasibility + user-facing energy. The BQM energy from
            # the SampleSet includes the Lagrange penalty for violated
            # constraints; we report the CQM objective energy so all rows
            # are comparable on the same scale. Non-converging stochastic
            # samplers (e.g. simulated_bifurcation reporting "no agent
            # converged") can emit NaN/inf; those have to be stripped
            # BEFORE the row goes anywhere near json.dumps, or the
            # background thread dies silently with the job stuck at
            # status='solving' forever.
            try:
                feasible = bool(cqm.check_feasible(cqm_sample))
            except Exception:  # pragma: no cover — defensive
                feasible = False
            try:
                obj_energy = _finite_or_none(cqm.objective.energy(cqm_sample))
            except Exception:  # pragma: no cover
                obj_energy = raw_energy
            if obj_energy is None:
                user_energy = None
            else:
                user_energy = obj_energy if sense == "minimize" else -obj_energy

            row: dict[str, Any] = {
                "status": "complete",
                "energy": user_energy,
                "raw_energy": raw_energy,
                "feasible": feasible,
                "elapsed_ms": elapsed_ms,
                "tier_source": ident.source,
                "version": ident.version,
                "hardware": ident.hardware,
                # Private — stripped before persistence; used only to
                # populate the legacy ``solution`` column from the winner.
                "_cqm_sample": cqm_sample,
            }

            # Phase 10A — capture QAOA-specific telemetry so the
            # explainer panel can render the trained circuit, the
            # measurement histogram, and the top-K classical filter
            # step. Only the QAOA samplers populate ``ss.info`` with
            # these keys, so this is a no-op for everything else.
            qaoa_extras = _build_qaoa_extras(
                sampleset, bqm, variables_in_bqm=list(bqm.variables),
                sense=sense,
                num_logical_vars=len(cqm.variables),
            )
            if qaoa_extras is not None:
                row["qaoa_extras"] = qaoa_extras

            results[name] = row
            _publish(results)

        return results

    async def run(
        self,
        *,
        job_id: str,
        user_id: int,
        problem_statement: str,
        provider_name: str,
        api_key: str,
        event_bus: EventBus,
        solvers: list[str] | None = None,
        require_approval: bool = False,
        solver_params_overrides: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """End-to-end pipeline. Updates the ``jobs`` row at each stage
        and emits SSE events; never raises — every failure path lands
        on the ``error`` status.

        ``solvers`` is the Phase 5D fan-out list. When ``None`` or empty,
        the legacy single-solver path is used (GPU SA via ``_build_sampler``).
        When supplied, stage 4 runs each solver sequentially and stage 5
        interprets the winning result while preserving the per-solver
        comparison map in ``solver_results``.

        ``require_approval=True`` pauses the pipeline after stage 3
        (validation) with ``status='awaiting_approval'`` so the user can
        review the LLM-emitted CQM + qubit estimate before any solver
        burns time. A subsequent ``POST /api/jobs/<id>/approve`` calls
        :meth:`resume_after_approval` to resume from stage 4. Without
        this gate the LLM's choice of encoding can silently inflate the
        qubit count and skip every QAOA tier (Max-Cut shipped to demo
        customers 2026-06-30 with 6 expected vs. 69 actual qubits).
        """
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

            # ---- Approval gate (optional) -----------------------------------
            # When require_approval is set, pause here so the user can
            # review the LLM-emitted CQM + validation report + lowered-
            # qubit estimate before any solver runs. NB: this comes
            # BEFORE the strict validation hard-abort on purpose. A
            # failed validation is exactly the case where the user
            # most wants to see what the LLM emitted (the validator
            # caught something — show it). The approval panel renders
            # the validation report as advisory; the user can still
            # approve and run solvers on the bad CQM (useful for demos
            # where seeing the solver output matters more than oracle
            # agreement) or cancel and re-formulate.
            #
            # The pipeline thread exits cleanly; the bus closes via the
            # awaiting_approval terminal event. A subsequent
            # POST /api/jobs/<id>/approve calls resume_after_approval()
            # which picks up at stage 4 unconditionally — the user
            # already saw the validation report at the gate and made an
            # informed decision.
            if require_approval:
                preflight = self._compute_preflight(
                    cqm, solver_params_overrides=solver_params_overrides,
                )
                # Persist the overrides alongside the paused row so the
                # resume path (POST /api/jobs/<id>/approve) picks them up
                # from the DB — the resume thread runs in a fresh
                # Orchestrator instance and doesn't share memory with
                # this one.
                pause_fields: dict[str, Any] = {
                    "status": "awaiting_approval",
                    "preflight": json.dumps(preflight),
                }
                if solver_params_overrides:
                    pause_fields["solver_params_overrides"] = json.dumps(
                        solver_params_overrides,
                    )
                update_job(job_id, **pause_fields)
                event_bus.emit(job_id, "awaiting_approval", preflight=preflight)
                return

            if not report.passed:
                first_reason = next(iter(report.warnings), "validation failed")
                raise PipelineError(f"validation: {first_reason}")

            # ---- Stages 4 + 5: solve + interpret ----------------------------
            await self._solve_and_interpret(
                job_id=job_id, user_id=user_id,
                cqm=cqm, registry=registry, sense=sense,
                solvers=solvers, event_bus=event_bus,
                solver_params_overrides=solver_params_overrides,
                # Threaded through so stage-5 can call
                # provider.summarize_solution() with the same key that
                # ran the formulation; None on the resume path (see
                # resume_after_approval).
                problem_statement=problem_statement,
                provider_name=provider_name,
                api_key=api_key,
            )

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

    async def resume_after_approval(
        self,
        *,
        job_id: str,
        user_id: int,
        event_bus: EventBus,
    ) -> None:
        """Resume a paused (``awaiting_approval``) job at stage 4.

        Rebuilds the CQM + registry + sense from the cqm_json already
        saved on the row, then dispatches to :meth:`_solve_and_interpret`
        — the same helper :meth:`run` uses, so the post-approval path
        gets identical event/DB semantics to the legacy auto-solve path.

        Errors land on ``status='error'`` with the same outer try/except
        contract as :meth:`run`. Callers (the approve route) only need
        to verify the job is in ``awaiting_approval`` before calling.
        """
        from app.models import get_job, update_job

        try:
            job = get_job(job_id, user_id=user_id)
            if job is None:
                raise PipelineError(f"job {job_id!r} not found")
            if job.get("status") != "awaiting_approval":
                raise PipelineError(
                    f"job is in status {job.get('status')!r}, not awaiting_approval",
                )
            cqm_json_raw = job.get("cqm_json")
            if not cqm_json_raw:
                raise PipelineError("job has no cqm_json — cannot resume")
            cqm, registry, sense = compile_cqm_json(json.loads(cqm_json_raw))
            solvers_raw = job.get("solvers_requested") or "[]"
            solvers = json.loads(solvers_raw) or None
            # Rehydrate per-solve overrides (Origin backend selection,
            # future shot-count tweaks, etc.) from the paused row so
            # stage 4 sees the same params the user picked pre-approval.
            overrides_raw = job.get("solver_params_overrides")
            solver_params_overrides = (
                json.loads(overrides_raw) if overrides_raw else None
            )
            # Rehydrate the LLM-summary inputs — the paused row already
            # carries problem_statement + provider; the BYOK key comes
            # from the user's stored credential (best-effort — if the
            # key was rotated between pause and resume, the summary
            # will silently skip and the technical view remains).
            resumed_problem_statement = job.get("problem_statement")
            resumed_provider = job.get("provider")
            resumed_api_key: str | None = None
            if resumed_provider and resumed_provider != "local":
                try:
                    from app.config import KEY_ENCRYPTION_SECRET
                    from app.crypto import decrypt_api_key
                    from app.models import get_api_key_ciphertext
                    cipher = get_api_key_ciphertext(user_id, resumed_provider)
                    if cipher is not None:
                        resumed_api_key = decrypt_api_key(
                            cipher, KEY_ENCRYPTION_SECRET,
                        )
                except Exception:  # noqa: BLE001
                    resumed_api_key = None
            # The previous bus session emitted a terminal
            # ``awaiting_approval`` event; the channel is now closed and
            # ``emit`` would silently drop our new events. Reset it so
            # post-approval events flow to fresh subscribers.
            event_bus.reset_channel(job_id)
            await self._solve_and_interpret(
                job_id=job_id, user_id=user_id,
                cqm=cqm, registry=registry, sense=sense,
                solvers=solvers, event_bus=event_bus,
                solver_params_overrides=solver_params_overrides,
                problem_statement=resumed_problem_statement,
                provider_name=resumed_provider,
                api_key=resumed_api_key,
            )
        except PipelineError as e:
            update_job(
                job_id,
                status="error",
                error=str(e),
                completed_at=datetime.utcnow().isoformat(),
            )
            event_bus.emit(job_id, "error", error=str(e))
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            update_job(
                job_id,
                status="error",
                error=msg,
                completed_at=datetime.utcnow().isoformat(),
            )
            event_bus.emit(job_id, "error", error=msg)

    async def _solve_and_interpret(
        self,
        *,
        job_id: str,
        user_id: int,
        cqm: dimod.ConstrainedQuadraticModel,
        registry: dict[str, str],
        sense: str,
        solvers: list[str] | None,
        event_bus: EventBus,
        solver_params_overrides: dict[str, dict[str, Any]] | None = None,
        # Optional — enables the plain-English post-processing call. On
        # the approval resume path we look these up from the DB.
        problem_statement: str | None = None,
        provider_name: str | None = None,
        api_key: str | None = None,
    ) -> None:
        """Stages 4 + 5 — solve, pick winner, interpret, write completion.

        Extracted so :meth:`run` (no-approval path) and
        :meth:`resume_after_approval` (post-approval path) share the
        same downstream logic. Raises ``PipelineError`` on a hard
        failure; the caller's outer try/except converts that into the
        terminal ``error`` event.
        """
        from app.models import update_job

        event_bus.emit(job_id, "solving")
        update_job(job_id, status="solving")

        if solvers:
            solver_results = self._run_multi_solver(
                cqm, sense, solvers, user_id,
                job_id=job_id, event_bus=event_bus,
                solver_params_overrides=solver_params_overrides,
            )
            primary_name, primary = _pick_primary(solver_results, sense)
            if primary is None or primary.get("status") != "complete":
                first_err = next(
                    (r.get("error") for r in solver_results.values()
                     if r.get("error")),
                    "all solvers failed",
                )
                raise PipelineError(f"solving: {first_err}")
            cqm_sample = primary["_cqm_sample"]
            elapsed_ms = int(primary["elapsed_ms"])
            public_solver_results = {
                name: {k: v for k, v in r.items() if k != "_cqm_sample"}
                for name, r in solver_results.items()
            }
        else:
            # Legacy single-solver path (GPU SA via _build_sampler).
            t0 = time.perf_counter()
            bqm, invert = dimod.cqm_to_bqm(
                cqm, lagrange_multiplier=self.lagrange_multiplier
            )
            sampler = self._build_sampler()
            sampleset = sampler.sample(
                bqm, num_reads=self.num_reads, num_sweeps=self.num_sweeps,
            )
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            cqm_sample = invert(sampleset.first.sample)
            public_solver_results = None
            primary_name = "gpu_sa"

        interpreted = interpret_solution(cqm_sample, registry, cqm, sense=sense)
        # Plain-English summary. Best-effort — a failure at this stage
        # must never abort a successful solve; the technical
        # ``interpreted`` above is the safety net. When ``problem_statement``
        # + ``api_key`` are missing (legacy callers) or the provider
        # returns None, ``plain_english`` stays None and the frontend
        # falls back to the technical view.
        plain_english: str | None = None
        if problem_statement and provider_name and api_key:
            try:
                provider = self.provider_resolver(provider_name)
                plain_english = await provider.summarize_solution(
                    problem_statement=problem_statement,
                    interpreted_solution=interpreted,
                    api_key=api_key,
                )
            except Exception:  # noqa: BLE001
                # Any provider-side failure (HTTP, timeout, malformed
                # response) is swallowed — the interpretation is
                # informational, not load-bearing.
                plain_english = None
        # JSON keys must be strings even when the CQM uses int variable
        # labels; coerce numpy scalars to Python natives so the default
        # JSON encoder doesn't reject them.
        solution_payload = {
            str(k): (v.item() if hasattr(v, "item") else v)
            for k, v in cqm_sample.items()
        }
        completion_fields: dict[str, Any] = {
            "status": "complete",
            "solution": json.dumps(solution_payload),
            "interpreted_solution": interpreted,
            "solve_time_ms": elapsed_ms,
            "completed_at": datetime.utcnow().isoformat(),
        }
        if plain_english:
            completion_fields["plain_english_solution"] = plain_english
        if public_solver_results is not None:
            completion_fields["solver_results"] = json.dumps({
                "solvers": public_solver_results,
                "primary": primary_name,
                "sense": sense,
            })
        update_job(job_id, **completion_fields)
        event_bus.emit(
            job_id, "complete",
            solve_time_ms=elapsed_ms,
            primary_solver=primary_name,
        )

    def _compute_preflight(
        self,
        cqm: dimod.ConstrainedQuadraticModel,
        solver_params_overrides: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Estimate the lowered qubit count + per-tier verdict so the
        approval UI can warn the user *before* a solver burns time.

        The lowering is done via ``dimod.cqm_to_bqm`` with the same
        Lagrange multiplier the real solve uses, then we read
        ``num_variables`` off the BQM. Caps are pulled from the QAOA
        tier table the orchestrator already enforces — if a tier's
        cap is exceeded, the user sees a "would be skipped" verdict
        upfront instead of discovering it after submitting.

        When ``solver_params_overrides`` picks a real Origin backend
        (Wukong / Hanyuan), also computes ``qpu_footprint`` — a rough
        (low, high) range of compute-seconds. Queue time is deliberately
        NOT part of the estimate: Origin and IBM both bill compute time
        only, so queue time is a latency concern, not a cost concern.
        """
        try:
            bqm, _invert = dimod.cqm_to_bqm(
                cqm, lagrange_multiplier=self.lagrange_multiplier,
            )
            lowered_qubits = int(bqm.num_variables)
            linear_count = sum(1 for c in bqm.linear.values() if c != 0.0)
            quadratic_count = sum(1 for c in bqm.quadratic.values() if c != 0.0)
        except Exception as e:  # noqa: BLE001
            return {
                "cqm_variables": int(len(cqm.variables)),
                "cqm_constraints": int(len(cqm.constraints)),
                "lowered_qubits": None,
                "lowering_error": f"{type(e).__name__}: {e}",
                "tier_verdicts": {},
                "qpu_footprint": None,
            }

        # Mirror the caps the QAOA samplers actually enforce. Source of
        # truth is the per-sampler config; we duplicate the numbers
        # here so the UI verdict matches what the sampler will do.
        #
        # ``qaoa_originqc``'s cap is backend-dependent. The module
        # default (7) matches Origin's ``full_amplitude`` cloud
        # simulator budget; picking a real Wukong / Hanyuan backend in
        # ProblemInput bumps to 12 inside ``_run_multi_solver``. We
        # mirror that logic here so the preflight card doesn't
        # under-report (showing "would skip 10 > 7") when the runtime
        # cap will actually be 12 — before this fix, users on the Max
        # Independent Set 5-node template with Wukong selected saw a
        # "would skip" that never actually happened.
        origin_backend = (
            (solver_params_overrides or {})
            .get("qaoa_originqc", {})
            .get("backend_name")
        )
        real_origin = {"WK_C180", "WK_C180_2", "HanYuan_01"}
        origin_default_cap = _DEFAULT_PARAMS_BY_SOLVER.get(
            "qaoa_originqc", {},
        ).get("max_qubits", 7)
        origin_effective_cap = (
            12 if origin_backend in real_origin else origin_default_cap
        )
        tier_caps: dict[str, int] = {
            "qaoa_sim": _DEFAULT_PARAMS_BY_SOLVER.get("qaoa_sim", {}).get("max_qubits", 12),
            "qaoa_originqc": origin_effective_cap,
            "qaoa_ibmq": _DEFAULT_PARAMS_BY_SOLVER.get("qaoa_ibmq", {}).get("max_qubits", 20),
        }
        verdicts = {
            name: {
                "cap": cap,
                "fits": lowered_qubits <= cap,
            }
            for name, cap in tier_caps.items()
        }

        qpu_footprint = self._estimate_qpu_footprint(
            num_qubits=lowered_qubits,
            linear_count=linear_count,
            quadratic_count=quadratic_count,
            solver_params_overrides=solver_params_overrides,
        )

        return {
            "cqm_variables": int(len(cqm.variables)),
            "cqm_constraints": int(len(cqm.constraints)),
            "lowered_qubits": lowered_qubits,
            "tier_verdicts": verdicts,
            "qpu_footprint": qpu_footprint,
        }

    def _estimate_qpu_footprint(
        self,
        *,
        num_qubits: int,
        linear_count: int,
        quadratic_count: int,
        solver_params_overrides: dict[str, dict[str, Any]] | None,
    ) -> dict[str, Any] | None:
        """Rough compute-time estimate for the selected quantum backend.

        Returns ``None`` when no real-QPU backend is selected — we only
        surface the footprint card when it's actually decision-relevant
        (a Wukong / Hanyuan submission that decrements the user's
        quota). Simulator submissions are effectively free.

        The estimate is a range on purpose. Ground-truth is one
        observed data point (6 qubits, 200 shots, 1 layer, 7 edges =
        ~2.5 s end-to-end on Wukong, 2026-06-30). The formula
        back-solves from that point but the actual per-shot cost
        varies with chip calibration, so we present a low/high range
        rather than a single misleading number.
        """
        # Wukong-family real hardware only — matches
        # QAOACloudSampler._REAL_HARDWARE_BACKENDS.
        origin_backend = (
            (solver_params_overrides or {})
            .get("qaoa_originqc", {})
            .get("backend_name")
        )
        real_origin = {"WK_C180", "WK_C180_2", "HanYuan_01"}
        if origin_backend not in real_origin:
            return None

        origin_params = _DEFAULT_PARAMS_BY_SOLVER.get("qaoa_originqc", {})
        shots = int(origin_params.get("shots", 200))
        layers = int(origin_params.get("layer", 1))

        # Circuit depth per layer: H prep + measure + per-layer body.
        # Per layer: (quadratic ×3) + linear + qubits (mixer).
        gates_per_layer = (quadratic_count * 3) + linear_count + num_qubits
        total_gates = num_qubits + (layers * gates_per_layer) + num_qubits

        # Back-solved from the 2026-06-30 Wukong data point:
        #   observed = 2.5 s for shots=200, gates≈46, layers=1
        # Overhead dominates over per-shot cost, so most of the range
        # comes from queue-processing / classical / network variance.
        fixed_low = 1.0
        fixed_high = 3.0
        per_shot_low = 0.003   # 3 ms per shot at the low end
        per_shot_high = 0.010  # 10 ms per shot at the high end
        # Small linear penalty for very deep circuits (irrelevant at
        # our current 6-qubit demo sizes, but keeps the estimate honest
        # if someone runs a wider template).
        depth_penalty = max(0.0, (total_gates - 50) * 0.005)

        est_low = fixed_low + shots * per_shot_low + depth_penalty
        est_high = fixed_high + shots * per_shot_high + depth_penalty

        return {
            "backend": origin_backend,
            "compute_seconds_low": round(est_low, 2),
            "compute_seconds_high": round(est_high, 2),
            "assumptions": {
                "shots": shots,
                "qaoa_layers": layers,
                "estimated_gates": total_gates,
                "num_qubits": num_qubits,
            },
            # Explicit note for the frontend to render — makes the
            # "queue-time is not billed" story load-bearing in the UI.
            "note": (
                "Origin bills compute time only. Queue wait is a "
                "latency concern, not a cost concern — those seconds "
                "do not decrement your quota."
            ),
        }


def _try_submit_async(
    *,
    name: str,
    sampler_cls: type,
    init_kwargs: dict[str, Any],
    sample_kwargs: dict[str, Any],
    bqm: dimod.BinaryQuadraticModel,
    ident: Any,
    results: dict[str, dict[str, Any]],
    job_id: str,
    sense: str,
    t0: float,
) -> bool:
    """Submit this solver asynchronously and record a 'queued' row.

    Returns True when handling is complete (either queued successfully
    OR errored in a way the orchestrator should treat as terminal for
    this row). Returns False if we couldn't even attempt the async
    submit (e.g. constructor failed) and the caller should fall back
    to the sync path.
    """
    from app.benchmarking import pending_jobs as pj

    # Construct the sampler. This shouldn't take long for any of our
    # cloud samplers — heavy work happens in submit_async (training).
    try:
        sampler = sampler_cls(**init_kwargs)
    except Exception as e:  # noqa: BLE001
        results[name] = {
            "status": "error",
            "error": f"sampler init failed: {type(e).__name__}: {e}",
            "elapsed_ms": int((time.perf_counter() - t0) * 1000),
            "tier_source": getattr(ident, "source", ""),
            "version": getattr(ident, "version", ""),
            "hardware": getattr(ident, "hardware", ""),
        }
        return True

    # Submit. submit_async signature is (bqm, **kwargs_subset) where
    # kwargs include at most ``seed``. Filter sample_kwargs accordingly.
    seed = sample_kwargs.get("seed")
    try:
        submission = sampler.submit_async(bqm, seed=seed)
    except Exception as e:  # noqa: BLE001
        # Auth errors, immediate API errors, etc. — record as a row
        # error and let the user retry. Async submit shouldn't take
        # long (just the LLM-free training step + a network round-trip).
        # Capacity overflows (CQM too big for the configured cap) are
        # classified as 'skipped' rather than 'error' — same logic as
        # the sync path. The async wrapper prefix is only added for
        # genuine errors so operators can tell "never reached cloud"
        # from later sync failures; skipped messages are already
        # self-explanatory.
        status_, error_msg = _classify_solver_failure(e)
        if status_ == "error":
            error_msg = f"async submit failed: {error_msg}"
        results[name] = {
            "status": status_,
            "error": error_msg,
            "elapsed_ms": int((time.perf_counter() - t0) * 1000),
            "tier_source": getattr(ident, "source", ""),
            "version": getattr(ident, "version", ""),
            "hardware": getattr(ident, "hardware", ""),
        }
        return True

    cloud_job_id = submission.get("cloud_job_id", "")
    if submission.get("empty"):
        # The empty-BQM short-circuit. Skip the queue entirely.
        results[name] = {
            "status": "complete",
            "energy": 0.0,
            "feasible": True,
            "elapsed_ms": int((time.perf_counter() - t0) * 1000),
            "tier_source": getattr(ident, "source", ""),
            "version": getattr(ident, "version", ""),
            "hardware": getattr(ident, "hardware", ""),
        }
        return True

    # Persist the BQM so the poller can recompute per-bitstring energies
    # later without re-running anything. Stored once, reused on every
    # poll cycle until materialization.
    bqm_blob = _serialize_bqm(bqm)
    materialize_ctx = {
        "bqm": bqm_blob,
        "sense": sense,
        "solver_name": name,
        "parent_job_id": job_id,
        "trained_gammas": submission.get("trained_gammas", []),
        "trained_betas": submission.get("trained_betas", []),
        "layer": submission.get("layer"),
        "shots": submission.get("shots"),
        "backend_name": submission.get("backend_name"),
    }
    pending = pj.PendingJob(
        job_id=cloud_job_id,
        solver_name=name,
        instance_id=f"solve-job:{job_id}",
        parameters={
            k: v for k, v in init_kwargs.items()
            if k not in ("api_key",)  # never persist credentials
        },
        lagrange_multiplier=0.0,  # not used by solve-job materializer
        submitted_at=datetime.utcnow().isoformat() + "Z",
        notes=f"async cloud submit from solve job {job_id[:8]}",
        target=f"solve_job:{job_id}:{name}",
        materialize_context=materialize_ctx,
    )
    pj.add(pending)

    # Stamp a "pre-execution" qaoa_extras onto the queued row so the
    # frontend explainer panel can render the circuit + submitted code
    # + metadata *while* the cloud job is still running. The histogram
    # / top-K panels degrade gracefully to "waiting for cloud" when the
    # measurement fields are still empty; when materialization fires
    # later, the same block gets replaced with the full one that
    # includes top_bitstrings + top_probabilities + top_energies.
    pre_extras = _build_qaoa_extras_pre_execution(
        bqm=bqm,
        solver_name=name,
        submission=submission,
    )

    results[name] = {
        "status": "queued",
        "cloud_job_id": cloud_job_id,
        "backend_name": submission.get("backend_name"),
        "elapsed_ms": int((time.perf_counter() - t0) * 1000),
        "tier_source": getattr(ident, "source", ""),
        "version": getattr(ident, "version", ""),
        "hardware": getattr(ident, "hardware", ""),
        "qaoa_extras": pre_extras,
    }
    return True


def _build_qaoa_extras_pre_execution(
    *,
    bqm: dimod.BinaryQuadraticModel,
    solver_name: str,
    submission: dict[str, Any],
) -> dict[str, Any]:
    """Build the QAOA telemetry block WITHOUT sampled outcomes.

    Used at cloud-submit time so the frontend can show the circuit +
    metadata + submitted-code preview while the row is still ``queued``.
    Shape matches :func:`_build_qaoa_extras` — same fields, but the
    ``top_*`` measurement lists start empty. The pending-jobs
    materializer replaces this whole block later with the sampled
    version (see :func:`_row_from_originqc_result`).
    """
    bqm_bin = bqm.change_vartype(dimod.BINARY, inplace=False)
    variables = list(bqm_bin.variables)
    var_index = {v: i for i, v in enumerate(variables)}

    h_terms: list[tuple[int, float]] = []
    for v, c in bqm_bin.linear.items():
        fc = _finite_or_none(c)
        if fc is None or fc == 0.0:
            continue
        h_terms.append((var_index[v], fc))

    j_terms: list[tuple[int, int, float]] = []
    for (u, v), c in bqm_bin.quadratic.items():
        fc = _finite_or_none(c)
        if fc is None or fc == 0.0:
            continue
        i, j = var_index[u], var_index[v]
        j_terms.append((min(i, j), max(i, j), fc))

    backend_name = submission.get("backend_name") or ""
    # Rough classification: Origin's "full_amplitude" (and "cpu_qvm",
    # "single_amplitude", "partial_amplitude") are cloud simulators;
    # "wu_yuan" / "wukong" / real-QPU backends are hardware. IBM's
    # backend names distinguish simulator vs. real by containing
    # "_simulator" or "simulator". Anything the sampler didn't
    # explicitly tag is treated as simulator (safer default — we don't
    # want to overclaim a real-QPU run).
    lower = backend_name.lower()
    is_real_hardware = bool(
        submission.get("is_real_hardware")
        # Origin real-QPU naming conventions we know about:
        # ``WK_C180`` / ``WK_C180_2`` (Wukong 180 family), ``HanYuan_01``,
        # and any future "wukong"-prefixed marketing name Origin might
        # ship. Kept generous on purpose — false positives (marking a
        # simulator as real hardware) are less harmful than the reverse.
        or ("wk_c180" in lower or "wukong" in lower or "hanyuan" in lower)
        or (solver_name == "qaoa_ibmq" and "simulator" not in lower and lower)
    )

    return {
        "layer": submission.get("layer"),
        "trained_gammas": [_finite_or_none(g) for g in submission.get("trained_gammas", [])],
        "trained_betas": [_finite_or_none(b) for b in submission.get("trained_betas", [])],
        "num_qubits": len(variables),
        "num_logical_vars": None,
        "linear_terms": h_terms,
        "quadratic_terms": j_terms,
        # Empty until the cloud comes back — the frontend renders a
        # "waiting on cloud" placeholder where the histogram lives.
        "top_bitstrings": [],
        "top_probabilities": [],
        "top_energies": [],
        "train_loss": _finite_or_none(submission.get("train_loss")),
        "train_optimizer": submission.get("train_optimizer"),
        "backend_name": backend_name,
        "is_real_hardware": is_real_hardware,
        "job_id": submission.get("cloud_job_id", ""),
        # Extra fields the pre-execution panel uses that the completed
        # block doesn't carry (shots is only meaningful at submit time
        # for the metadata card; the completed sampleset knows its own
        # shot count via the probability distribution).
        "shots": submission.get("shots"),
    }


def _serialize_bqm(bqm: dimod.BinaryQuadraticModel) -> dict[str, Any]:
    """JSON-safe snapshot of a BQM. The poller deserializes back via
    ``dimod.BinaryQuadraticModel.from_serializable``-style reconstruction
    in the materializer."""
    variables = [str(v) for v in bqm.variables]
    return {
        "linear": {str(v): float(c) for v, c in bqm.linear.items()},
        "quadratic": {
            f"{u}\x1f{v}": float(c)  # \x1f is a safe in-string separator
            for (u, v), c in bqm.quadratic.items()
        },
        "offset": float(bqm.offset),
        "vartype": "BINARY" if bqm.vartype == dimod.BINARY else "SPIN",
        "variables": variables,
    }


def deserialize_bqm(blob: dict[str, Any]) -> dimod.BinaryQuadraticModel:
    """Rebuild a BQM from a ``_serialize_bqm`` snapshot."""
    vartype = dimod.BINARY if blob["vartype"] == "BINARY" else dimod.SPIN
    linear = blob["linear"]
    quadratic = {}
    for k, v in blob["quadratic"].items():
        u_str, v_str = k.split("\x1f", 1)
        quadratic[(u_str, v_str)] = float(v)
    bqm = dimod.BinaryQuadraticModel(linear, quadratic, float(blob["offset"]), vartype)
    return bqm


def _build_qaoa_extras(
    sampleset: dimod.SampleSet,
    bqm: dimod.BinaryQuadraticModel,
    *,
    variables_in_bqm: list,
    sense: str,
    num_logical_vars: int | None = None,
) -> dict[str, Any] | None:
    """Return the QAOA-specific telemetry the Explainer panel needs, or
    ``None`` if the sampleset isn't from a QAOA solver.

    The QAOA samplers (Phase 9A local, Phase 9B cloud) stamp
    ``sampleset.info`` with the trained ``(γ, β)`` angles, the layer
    count, and the top-K measurement bitstrings + probabilities.
    Detecting any of those keys is the cheapest "is this QAOA?" check —
    we don't want to plumb a solver-name allow-list through to this
    helper.
    """
    info = getattr(sampleset, "info", None) or {}
    bitstrings = info.get("qaoa_top_bitstrings")
    if bitstrings is None:
        return None  # not a QAOA result — nothing to surface

    probs = info.get("qaoa_top_probabilities") or []
    gammas = info.get("qaoa_trained_gammas") or []
    betas = info.get("qaoa_trained_betas") or []

    # Compute the user-facing energy for each top bitstring so the
    # frontend can colour the histogram bars + run the top-K filter
    # without ever seeing the BQM. ``cqm_to_bqm`` keeps binary
    # variables 1:1 with the CQM, so a bitstring evaluated against the
    # BQM gives the same energy you'd get plugging into the CQM
    # objective (modulo Lagrange penalty for infeasible samples, which
    # is exactly the "is this state feasible?" signal we want).
    bqm_bin = bqm.change_vartype(dimod.BINARY, inplace=False)
    n = len(variables_in_bqm)
    energies: list[float | None] = []
    for bitstring in bitstrings:
        padded = str(bitstring).zfill(n)[-n:]
        sample = {v: int(padded[i]) for i, v in enumerate(variables_in_bqm)}
        raw = _finite_or_none(bqm_bin.energy(sample))
        if raw is None:
            energies.append(None)
        else:
            energies.append(raw if sense == "minimize" else -raw)

    # Capture the BQM coupling structure so the explainer can render a
    # realistic gate-level diagram (linear coefficients → RZ gates on
    # each qubit; quadratic couplings → CNOT–RZ–CNOT decompositions
    # between qubit pairs). Strings are used as variable labels so the
    # frontend can map them to the qubit indices in ``variables_in_bqm``.
    var_index = {v: i for i, v in enumerate(variables_in_bqm)}
    h_terms: list[tuple[int, float]] = []
    for v, c in bqm_bin.linear.items():
        fc = _finite_or_none(c)
        if fc is None or fc == 0.0:
            continue
        h_terms.append((var_index[v], fc))
    j_terms: list[tuple[int, int, float]] = []
    for (u, v), c in bqm_bin.quadratic.items():
        fc = _finite_or_none(c)
        if fc is None or fc == 0.0:
            continue
        i, j = var_index[u], var_index[v]
        j_terms.append((min(i, j), max(i, j), fc))

    return {
        "layer": info.get("qaoa_layer"),
        "trained_gammas": [_finite_or_none(g) for g in gammas],
        "trained_betas": [_finite_or_none(b) for b in betas],
        "num_qubits": n,
        "num_logical_vars": num_logical_vars,
        "linear_terms": h_terms,
        "quadratic_terms": j_terms,
        "top_bitstrings": [str(b) for b in bitstrings],
        "top_probabilities": [_finite_or_none(p) for p in probs],
        "top_energies": energies,
        "train_loss": _finite_or_none(info.get("qaoa_train_loss")),
        "train_optimizer": info.get("qaoa_optimizer"),
        "backend_name": info.get("cloud_backend"),
        "is_real_hardware": bool(info.get("cloud_is_real_hardware", False)),
        "job_id": info.get("cloud_job_id"),
    }


def _finite_or_none(x: Any) -> float | None:
    """Coerce a scalar to a JSON-safe float, returning ``None`` for
    NaN/inf so the orchestrator's persistence layer (json.dumps) never
    chokes on a non-converging sampler's output. This used to be the
    silent-thread-death culprit before Phase 10 ship: simulated
    bifurcation could emit NaN energies, json.dumps rejected the dict,
    the orchestrator thread crashed, and the job row sat at
    status='solving' forever with no error trail.
    """
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    import math
    if math.isnan(v) or math.isinf(v):
        return None
    return v


# Solvers that need a per-user credential from the api_keys table at
# run time. Map from solver name → BYOK provider string. The
# orchestrator decrypts the stored ciphertext and injects the plaintext
# as ``api_key`` into the parameters dict. Plus a fallback to the
# server-side ``QPANDA_API_KEY_FILE`` env var for benchmark-script
# parity — that path is admin-controlled and only fires when no
# user-stored key is found.
#
# The provider string matches the ``ApiKeyManager`` dropdown value
# ("originqc") so users see the same label across the Keys tab and the
# error messages in MultiSolverResultDisplay.
_BYOK_BY_SOLVER: dict[str, str] = {
    "qaoa_originqc": "originqc",
    "qaoa_ibmq": "ibm_quantum",
}


def _inject_byok_credentials(
    solver_name: str, params: dict[str, Any], user_id: int,
) -> str | None:
    """Inject ``api_key`` into ``params`` for BYOK-bearing solvers.
    Returns ``None`` on success or a user-facing error string when the
    user has no key on file. Mutates ``params`` in place.
    """
    provider = _BYOK_BY_SOLVER.get(solver_name)
    if provider is None:
        return None  # not a BYOK solver — nothing to do

    # Lazy imports — keeps this module free of route/crypto deps at load.
    from app.config import KEY_ENCRYPTION_SECRET
    from app.crypto import decrypt_api_key
    from app.models import get_api_key_ciphertext

    cipher = get_api_key_ciphertext(user_id, provider)
    if cipher is not None:
        try:
            params["api_key"] = decrypt_api_key(cipher, KEY_ENCRYPTION_SECRET)
            return None
        except ValueError as e:
            return f"stored {provider!r} key is unreadable: {e}"

    # No user key — try the server-side env-file fallback so admin-side
    # benchmark scripts that import this orchestrator still work.
    import os
    from pathlib import Path
    env_path = os.environ.get("QPANDA_API_KEY_FILE")
    if env_path and Path(env_path).exists():
        try:
            params["api_key"] = Path(env_path).read_text(encoding="utf-8").strip()
            return None
        except OSError:  # pragma: no cover — defensive
            pass

    return (
        "no Origin Quantum API key on file — save your credential in "
        "the API Keys tab (provider \"originqc\") to enable this solver"
    )


def _pick_primary(
    results: dict[str, dict[str, Any]],
    sense: str,
) -> tuple[str | None, dict[str, Any] | None]:
    """Pick the row that drives the legacy ``solution`` / ``interpreted_solution``
    columns. "Best" depends on the CQM's optimization sense — for
    minimize problems we want the row with the lowest energy; for
    maximize problems, the highest. Preference order:
    complete-and-feasible-with-best-energy, falling back to
    complete-with-best-energy, then to the first error row.
    Returns ``(name, row)`` or ``(None, None)`` when no rows exist."""
    if not results:
        return None, None

    # For maximize problems, ``user_energy`` retains the original sign
    # (we negated back inside ``_run_multi_solver``), so "best" is the
    # largest value. For minimize, smallest.
    is_max = sense == "maximize"
    sentinel = float("-inf") if is_max else float("inf")

    def keyfn(kv: tuple[str, dict[str, Any]]) -> float:
        energy = kv[1].get("energy")
        if energy is None:
            return sentinel  # rows with no usable energy can't win
        return -energy if is_max else energy

    # Rows where the sampler produced a non-finite energy (NaN, inf)
    # are downgraded — they're "complete" by status but can't be primary.
    complete_feasible = [
        (name, r) for name, r in results.items()
        if r.get("status") == "complete"
        and r.get("feasible")
        and r.get("energy") is not None
    ]
    if complete_feasible:
        return min(complete_feasible, key=keyfn)

    complete_any = [
        (name, r) for name, r in results.items()
        if r.get("status") == "complete" and r.get("energy") is not None
    ]
    if complete_any:
        return min(complete_any, key=keyfn)

    # No completes — pass back the first row so the caller can surface the error.
    return next(iter(results.items()))


def _serialize_validation_report(report: Any) -> dict[str, Any]:
    """Coerce a ``ValidationReport`` dataclass into a JSON-friendly dict.

    The ``passed`` field is a property (computed from the three layer
    booleans), so ``asdict`` skips it; we add it back explicitly.
    """
    base = asdict(report) if is_dataclass(report) else dict(report)
    base["passed"] = bool(getattr(report, "passed", False))
    return base
