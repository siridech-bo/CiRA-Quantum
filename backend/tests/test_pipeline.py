"""Phase 4 — solve pipeline orchestrator tests.

We exercise the five-stage pipeline end-to-end against an in-process
SQLite (Phase 0 fixture), a stub LLM provider (deterministic CQM
output), and a mock sampler (returns a canned SampleSet). No GPU, no
network — every dependency is injected at construction time.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import dimod
import pytest

from app.formulation.base import FormulationError, FormulationResult

# --- Test fixtures ------------------------------------------------------------


@pytest.fixture
def isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point ``DATABASE_PATH`` at a tmp SQLite, reload the relevant modules,
    and create the schema so ``models.create_job`` / ``update_job`` work."""
    db_path = tmp_path / "pipeline.db"
    monkeypatch.setenv("SECRET_KEY", "phase4-test-secret")

    from app import config as config_module
    monkeypatch.setattr(config_module, "DATABASE_PATH", str(db_path))

    from app import models as models_module
    importlib.reload(models_module)
    monkeypatch.setattr(models_module, "DATABASE_PATH", str(db_path))
    models_module.init_db()
    # Seed a non-admin user we'll attribute jobs to.
    user = models_module.create_user("pipeline_user", "verysecret123")
    return models_module, user


def _knapsack_cqm_json() -> dict:
    """The Phase-2 knapsack_5item fixture, inline so tests are self-contained."""
    return {
        "version": "1",
        "variables": [
            {"name": f"x_{i}", "type": "binary",
             "description": f"Item {i} selected"}
            for i in range(5)
        ],
        "objective": {
            "sense": "maximize",
            "linear": {"x_0": 3, "x_1": 4, "x_2": 5, "x_3": 8, "x_4": 10},
            "quadratic": {},
        },
        "constraints": [{
            "label": "capacity",
            "type": "inequality_le",
            "linear": {"x_0": 2, "x_1": 3, "x_2": 4, "x_3": 5, "x_4": 9},
            "quadratic": {},
            "rhs": 20,
        }],
        "test_instance": {"description": "knapsack", "expected_optimum": 26},
    }


class _StubProvider:
    """Returns a canned ``FormulationResult`` without touching the network."""

    name = "stub"

    def __init__(self, cqm_json: dict | None = None, *, raise_on_call: bool = False):
        self._cqm_json = cqm_json or _knapsack_cqm_json()
        self._raise = raise_on_call

    async def formulate(self, problem_statement, api_key, timeout=60):
        if self._raise:
            raise FormulationError("simulated formulation failure")
        return FormulationResult(
            cqm_json=self._cqm_json,
            variable_registry={
                v["name"]: v.get("description", "")
                for v in self._cqm_json["variables"]
            },
            raw_llm_output=json.dumps(self._cqm_json),
            tokens_used=42,
            model="stub-model",
        )

    def estimate_cost(self, problem_statement: str) -> float:
        return 0.0


class _MockSampler:
    """Returns the canned best sample on every call. The orchestrator passes
    it a BQM lowered from the CQM; we don't care about the BQM here."""

    def __init__(self, cqm_sample: dict[str, int] | None = None):
        # Best sample for the knapsack fixture (items {0,2,3,4}, value 26).
        self.cqm_sample = cqm_sample or {
            "x_0": 1, "x_1": 0, "x_2": 1, "x_3": 1, "x_4": 1,
        }
        self.calls = 0

    def sample(self, bqm, **kwargs):
        self.calls += 1
        # Map our CQM-space sample back into the BQM's variable namespace.
        # cqm_to_bqm uses the same variable names for binaries, so this
        # works for the knapsack fixture (no integer expansion).
        bqm_sample = {v: int(self.cqm_sample.get(v, 0)) for v in bqm.variables}
        return dimod.SampleSet.from_samples_bqm(bqm_sample, bqm=bqm)


# --- The four spec'd test cases -----------------------------------------------


def test_pipeline_end_to_end_with_mocked_llm(isolated_db):
    models, user = isolated_db
    from app.pipeline.events import EventBus
    from app.pipeline.orchestrator import Orchestrator

    job_id = models.create_job(user["id"], "pack a knapsack", "stub")
    bus = EventBus()

    orch = Orchestrator(sampler=_MockSampler(), provider_resolver=lambda _: _StubProvider())
    import asyncio
    asyncio.run(orch.run(
        job_id=job_id, user_id=user["id"],
        problem_statement="pack a knapsack",
        provider_name="stub", api_key="",
        event_bus=bus,
    ))

    job = models.get_job(job_id, user_id=user["id"])
    assert job is not None
    assert job["status"] == "complete", job.get("error")
    assert job["num_variables"] == 5
    assert job["num_constraints"] == 1
    assert "Item 0" in job["interpreted_solution"]
    # Event bus saw the canonical 5-stage sequence.
    events = list(bus.subscribe(job_id))
    statuses = [e["status"] for e in events]
    assert statuses == [
        "formulating", "compiling", "validating", "solving", "complete",
    ]


def test_pipeline_handles_formulation_error(isolated_db):
    models, user = isolated_db
    from app.pipeline.events import EventBus
    from app.pipeline.orchestrator import Orchestrator

    job_id = models.create_job(user["id"], "p", "stub")
    bus = EventBus()
    failing = _StubProvider(raise_on_call=True)

    import asyncio
    asyncio.run(Orchestrator(
        sampler=_MockSampler(), provider_resolver=lambda _: failing,
    ).run(
        job_id=job_id, user_id=user["id"], problem_statement="p",
        provider_name="stub", api_key="", event_bus=bus,
    ))

    job = models.get_job(job_id, user_id=user["id"])
    assert job["status"] == "error"
    assert "simulated formulation failure" in (job["error"] or "")
    events = list(bus.subscribe(job_id))
    statuses = [e["status"] for e in events]
    assert statuses == ["formulating", "error"]


def test_pipeline_handles_validation_failure(isolated_db):
    """A CQM whose claimed expected_optimum disagrees with the oracle
    surfaces as an 'error' job, with the validation report attached so
    the user can see *what* the harness caught."""
    models, user = isolated_db
    from app.pipeline.events import EventBus
    from app.pipeline.orchestrator import Orchestrator

    bad = _knapsack_cqm_json()
    # Lie about the optimum: real one is 26.
    bad["test_instance"]["expected_optimum"] = 999

    job_id = models.create_job(user["id"], "lying knapsack", "stub")
    bus = EventBus()
    orch = Orchestrator(
        sampler=_MockSampler(), provider_resolver=lambda _: _StubProvider(bad),
    )
    import asyncio
    asyncio.run(orch.run(
        job_id=job_id, user_id=user["id"], problem_statement="x",
        provider_name="stub", api_key="", event_bus=bus,
    ))

    job = models.get_job(job_id, user_id=user["id"])
    assert job["status"] == "error", job
    assert "validation" in (job["error"] or "").lower()
    assert job["validation_report"] is not None
    report = json.loads(job["validation_report"])
    assert report["oracle_agreement"] is False


def test_pipeline_multi_solver_fanout(isolated_db):
    """Phase 5D — when ``solvers=[...]`` is passed, the orchestrator
    runs each registered solver on the same compiled CQM/BQM and stores
    a per-solver comparison map. The legacy ``solution`` columns are
    populated from the best (feasible, min-energy) row.
    """
    models, user = isolated_db
    from app.pipeline.events import EventBus
    from app.pipeline.orchestrator import Orchestrator

    job_id = models.create_job(user["id"], "knapsack multi", "stub")
    bus = EventBus()
    # cpu_sa_neal is in-tree, pure-Python, no GPU/network. Two reps of
    # the same solver lets us assert the fan-out structure without
    # depending on multiple optional registry entries.
    orch = Orchestrator(provider_resolver=lambda _: _StubProvider())
    import asyncio
    asyncio.run(orch.run(
        job_id=job_id, user_id=user["id"],
        problem_statement="knapsack",
        provider_name="stub", api_key="", event_bus=bus,
        solvers=["cpu_sa_neal"],
    ))

    job = models.get_job(job_id, user_id=user["id"])
    assert job["status"] == "complete", job.get("error")
    assert job["solver_results"], "multi-solver fan-out should populate solver_results"
    payload = json.loads(job["solver_results"])
    assert "solvers" in payload and "primary" in payload
    assert "cpu_sa_neal" in payload["solvers"]
    row = payload["solvers"]["cpu_sa_neal"]
    assert row["status"] == "complete"
    assert isinstance(row["energy"], (int, float))
    assert isinstance(row["elapsed_ms"], int)
    assert "_cqm_sample" not in row, "private sample copy must be stripped before persisting"
    # The winning solver should be the only one we ran.
    assert payload["primary"] == "cpu_sa_neal"


def test_pipeline_handles_solver_timeout(isolated_db):
    """A sampler that raises mid-solve must be caught and surfaced as
    an 'error' job — not propagate to the route handler."""
    models, user = isolated_db
    from app.pipeline.events import EventBus
    from app.pipeline.orchestrator import Orchestrator

    class _ExplodingSampler:
        def sample(self, *args, **kwargs):
            raise TimeoutError("simulated solver timeout")

    job_id = models.create_job(user["id"], "knapsack", "stub")
    bus = EventBus()
    orch = Orchestrator(
        sampler=_ExplodingSampler(), provider_resolver=lambda _: _StubProvider(),
    )
    import asyncio
    asyncio.run(orch.run(
        job_id=job_id, user_id=user["id"], problem_statement="x",
        provider_name="stub", api_key="", event_bus=bus,
    ))

    job = models.get_job(job_id, user_id=user["id"])
    assert job["status"] == "error"
    assert "timeout" in (job["error"] or "").lower()


# --- Failure classifier: capacity skip vs hard error -------------------------


class TestClassifySolverFailure:
    """``_classify_solver_failure`` turns QAOA qubit-cap ValueErrors into
    ``skipped`` rows so the UI doesn't render them as red errors. Every
    other exception is still an ``error``. Keep these assertions tight
    on the exact phrases the three QAOA samplers raise, so this safety
    net catches a future drift in their messages."""

    def test_qaoa_sim_overflow_is_skipped(self):
        from app.pipeline.orchestrator import _classify_solver_failure

        exc = ValueError(
            "QAOASampler: this CQM lowered to 24 variables, which exceeds "
            "the local-simulator qubit cap of 12."
        )
        status, msg = _classify_solver_failure(exc)
        assert status == "skipped"
        # The sampler's own message is preserved verbatim — no type prefix.
        assert msg.startswith("QAOASampler:")
        assert "ValueError" not in msg

    def test_qaoa_ibmq_overflow_is_skipped(self):
        from app.pipeline.orchestrator import _classify_solver_failure

        exc = ValueError(
            "QAOAIBMQSampler: this CQM lowered to 24 variables, which "
            "exceeds the configured cap of 20."
        )
        status, _ = _classify_solver_failure(exc)
        assert status == "skipped"

    def test_qaoa_originqc_overflow_is_skipped(self):
        from app.pipeline.orchestrator import _classify_solver_failure

        exc = ValueError(
            "QAOACloudSampler: this CQM lowered to 24 variables, which "
            "exceeds backend 'full_amplitude''s qubit budget of 7."
        )
        status, _ = _classify_solver_failure(exc)
        assert status == "skipped"

    def test_other_value_error_is_hard_error(self):
        from app.pipeline.orchestrator import _classify_solver_failure

        # ValueError that isn't a capacity overflow — should still
        # surface as an error so operators see it.
        status, msg = _classify_solver_failure(ValueError("bad config: foo"))
        assert status == "error"
        assert msg.startswith("ValueError:")

    def test_non_value_error_is_hard_error(self):
        from app.pipeline.orchestrator import _classify_solver_failure

        status, msg = _classify_solver_failure(RuntimeError("boom"))
        assert status == "error"
        assert msg == "RuntimeError: boom"


# --- Approval gate: pause after stage 3, resume from stage 4 -----------------


class TestApprovalGate:
    """``Orchestrator.run(require_approval=True)`` pauses after
    validation; ``Orchestrator.resume_after_approval`` picks up at
    stage 4. Together they let the user review the LLM's CQM before
    burning solver time — the surface that catches Max-Cut-style
    encoding regressions."""

    def test_pause_at_awaiting_approval(self, isolated_db):
        models, user = isolated_db
        from app.pipeline.events import EventBus
        from app.pipeline.orchestrator import Orchestrator

        job_id = models.create_job(user["id"], "x", "stub")
        bus = EventBus()
        orch = Orchestrator(
            sampler=_MockSampler(), provider_resolver=lambda _: _StubProvider(),
        )

        import asyncio
        asyncio.run(orch.run(
            job_id=job_id, user_id=user["id"],
            problem_statement="x", provider_name="stub", api_key="",
            event_bus=bus, require_approval=True,
        ))

        job = models.get_job(job_id, user_id=user["id"])
        assert job["status"] == "awaiting_approval"
        # CQM was formulated + validation report was written before pause.
        assert job["cqm_json"] is not None
        assert job["validation_report"] is not None
        # Preflight summary populated for the approval UI.
        preflight = json.loads(job["preflight"])
        assert preflight["cqm_variables"] == 5
        assert preflight["cqm_constraints"] == 1
        assert preflight["lowered_qubits"] is not None
        assert "qaoa_originqc" in preflight["tier_verdicts"]
        # SSE saw the awaiting_approval terminal event.
        events = list(bus.subscribe(job_id))
        assert events[-1]["status"] == "awaiting_approval"

    def test_resume_runs_stages_4_and_5(self, isolated_db):
        models, user = isolated_db
        from app.pipeline.events import EventBus
        from app.pipeline.orchestrator import Orchestrator

        job_id = models.create_job(user["id"], "x", "stub")

        # First pass: pause at approval.
        pause_bus = EventBus()
        orch = Orchestrator(
            sampler=_MockSampler(), provider_resolver=lambda _: _StubProvider(),
        )
        import asyncio
        asyncio.run(orch.run(
            job_id=job_id, user_id=user["id"],
            problem_statement="x", provider_name="stub", api_key="",
            event_bus=pause_bus, require_approval=True,
        ))
        assert models.get_job(job_id, user_id=user["id"])["status"] == "awaiting_approval"

        # Second pass: resume — new bus mirrors what the approve route does.
        resume_bus = EventBus()
        asyncio.run(orch.resume_after_approval(
            job_id=job_id, user_id=user["id"], event_bus=resume_bus,
        ))

        job = models.get_job(job_id, user_id=user["id"])
        assert job["status"] == "complete", job.get("error")
        assert job["interpreted_solution"] is not None
        # Resume bus saw stage 4-5 events only.
        events = list(resume_bus.subscribe(job_id))
        statuses = [e["status"] for e in events]
        assert statuses == ["solving", "complete"]

    def test_pre_execution_qaoa_extras_shape(self):
        """The queued row must carry a partial ``qaoa_extras`` block so
        the frontend's explainer panel can render the submitted circuit
        + code + metadata while the cloud job is still running. The
        block matches the shape :func:`_build_qaoa_extras` produces at
        completion, but with the ``top_*`` measurement lists empty —
        the materializer replaces the block with the sampled version
        when the cloud finishes."""
        import dimod

        from app.pipeline.orchestrator import _build_qaoa_extras_pre_execution

        bqm = dimod.BinaryQuadraticModel(
            {"x_0": 1.0, "x_1": 2.0},
            {("x_0", "x_1"): -1.5},
            0.0,
            dimod.BINARY,
        )
        submission = {
            "cloud_job_id": "TEST_JOB_123",
            "trained_gammas": [0.7],
            "trained_betas": [1.9],
            "layer": 1,
            "shots": 200,
            "backend_name": "full_amplitude",
        }
        extras = _build_qaoa_extras_pre_execution(
            bqm=bqm, solver_name="qaoa_originqc", submission=submission,
        )

        assert extras["num_qubits"] == 2
        assert extras["layer"] == 1
        assert extras["trained_gammas"] == [0.7]
        assert extras["trained_betas"] == [1.9]
        assert extras["shots"] == 200
        assert extras["backend_name"] == "full_amplitude"
        assert extras["job_id"] == "TEST_JOB_123"
        # full_amplitude is a cloud simulator, not real hardware.
        assert extras["is_real_hardware"] is False
        # Measurement fields present but empty — frontend renders
        # the "waiting on cloud" placeholder.
        assert extras["top_bitstrings"] == []
        assert extras["top_probabilities"] == []
        assert extras["top_energies"] == []
        # Circuit-defining fields populated so the gate-level SVG renders.
        assert len(extras["linear_terms"]) == 2
        assert len(extras["quadratic_terms"]) == 1

    def test_run_multi_solver_merges_per_request_overrides(
        self, isolated_db, monkeypatch: pytest.MonkeyPatch,
    ):
        """``solver_params_overrides`` flows from the route → launcher →
        orchestrator → ``_run_multi_solver`` and is merged on top of
        ``_DEFAULT_PARAMS_BY_SOLVER`` so a single solve can flip the
        Origin backend from ``full_amplitude`` to Wukong without
        touching the module constant. Verified by intercepting the
        merged params before ``_split_parameters`` runs — that's the
        seam that carries the final dict into the sampler's
        ``__init__`` and ``sample()`` kwargs."""
        from app.benchmarking import records as _records
        from app.pipeline.orchestrator import Orchestrator, _DEFAULT_PARAMS_BY_SOLVER

        # gpu_sa is the safest solver to run through the pipeline in
        # tests (no cloud, no GPU required — it falls back to CPU JIT
        # or the SA implementation registered in bootstrap). We just
        # want to see the merged params, not the actual solve output.
        captured_params: list[dict] = []
        real_split = _records._split_parameters

        def spy_split(solver_name: str, params: dict) -> tuple[dict, dict]:
            captured_params.append({"solver": solver_name, "params": dict(params)})
            return real_split(solver_name, params)
        monkeypatch.setattr(_records, "_split_parameters", spy_split)

        # Pass ``fake_backend`` as a made-up override to prove the
        # merge happens regardless of what _INIT_ONLY_KWARGS decides
        # to do with it downstream — we're testing the merge, not the
        # split.
        default_gpu_sa = _DEFAULT_PARAMS_BY_SOLVER["gpu_sa"]
        assert "backend_name" not in default_gpu_sa

        import dimod

        from app.pipeline.events import EventBus

        cqm = dimod.ConstrainedQuadraticModel()
        cqm.add_variable("BINARY", "x_0")
        cqm.objective.add_linear("x_0", -1)  # trivial: pick x_0 = 1

        orch = Orchestrator(
            sampler=_MockSampler(), provider_resolver=lambda _: _StubProvider(),
        )
        try:
            orch._run_multi_solver(
                cqm, "minimize", ["gpu_sa"], 1,
                job_id="test-overrides", event_bus=EventBus(),
                solver_params_overrides={
                    "gpu_sa": {"backend_name": "WK_C180"},
                },
            )
        except Exception:
            # gpu_sa might not initialize cleanly in a test env (no
            # CUDA, no CPU JIT registered). We don't care about the
            # solve result — only that _split_parameters saw our merged
            # override. Swallow any downstream sampler failure.
            pass

        assert captured_params, "spy_split was never called"
        gpu_sa_call = next(
            (c for c in captured_params if c["solver"] == "gpu_sa"), None,
        )
        assert gpu_sa_call is not None
        # Defaults still there:
        for k, v in default_gpu_sa.items():
            assert gpu_sa_call["params"][k] == v
        # Override applied on top:
        assert gpu_sa_call["params"]["backend_name"] == "WK_C180"

    def test_qpu_footprint_only_populated_for_real_origin_backends(self):
        """The preflight card should only show the footprint estimate
        when a real Origin backend was selected. Simulator + no-override
        cases return ``None`` — the frontend hides the whole section."""
        import dimod

        from app.pipeline.orchestrator import Orchestrator

        cqm = dimod.ConstrainedQuadraticModel()
        for i in range(6):
            cqm.add_variable("BINARY", f"x_{i}")

        orch = Orchestrator(
            sampler=_MockSampler(), provider_resolver=lambda _: _StubProvider(),
        )

        # No overrides at all → simulator path, no footprint.
        p = orch._compute_preflight(cqm)
        assert p["qpu_footprint"] is None

        # Explicit simulator override → still no footprint.
        p = orch._compute_preflight(
            cqm,
            solver_params_overrides={
                "qaoa_originqc": {"backend_name": "full_amplitude"},
            },
        )
        assert p["qpu_footprint"] is None

        # Wukong → footprint present with a sane range.
        p = orch._compute_preflight(
            cqm,
            solver_params_overrides={
                "qaoa_originqc": {"backend_name": "WK_C180"},
            },
        )
        fp = p["qpu_footprint"]
        assert fp is not None
        assert fp["backend"] == "WK_C180"
        # Range must be a valid ordered interval and include the
        # 2026-06-30 observed data point (2.5 s for shots=200).
        assert 0 < fp["compute_seconds_low"] < fp["compute_seconds_high"]
        assert fp["compute_seconds_low"] <= 2.5 <= fp["compute_seconds_high"]
        # Cost/latency distinction gets a first-class note so the
        # frontend doesn't have to invent the copy.
        assert "queue" in fp["note"].lower()
        assert "cost" in fp["note"].lower() or "quota" in fp["note"].lower()
        # Assumptions dict carries the inputs for user sanity-checking.
        assert fp["assumptions"]["shots"] == 200
        assert fp["assumptions"]["num_qubits"] > 0

    def test_pre_execution_extras_flags_wukong_as_real_hardware(self):
        """Origin's ``wukong`` backend is the real QPU, not a simulator.
        The frontend header chip uses this flag to render a red
        "Real QPU" badge vs. the yellow "Simulator" badge."""
        import dimod

        from app.pipeline.orchestrator import _build_qaoa_extras_pre_execution

        bqm = dimod.BinaryQuadraticModel({"x_0": 1.0}, {}, 0.0, dimod.BINARY)
        submission = {
            "cloud_job_id": "WK_1",
            "trained_gammas": [0.5],
            "trained_betas": [1.0],
            "layer": 1,
            "shots": 100,
            "backend_name": "wukong",
        }
        extras = _build_qaoa_extras_pre_execution(
            bqm=bqm, solver_name="qaoa_originqc", submission=submission,
        )
        assert extras["is_real_hardware"] is True

    def test_resume_rejects_jobs_not_in_awaiting_approval(self, isolated_db):
        models, user = isolated_db
        from app.pipeline.events import EventBus
        from app.pipeline.orchestrator import Orchestrator

        # Job sits at status='queued' — never paused, so resume must fail.
        job_id = models.create_job(user["id"], "x", "stub")
        bus = EventBus()
        orch = Orchestrator(
            sampler=_MockSampler(), provider_resolver=lambda _: _StubProvider(),
        )

        import asyncio
        asyncio.run(orch.resume_after_approval(
            job_id=job_id, user_id=user["id"], event_bus=bus,
        ))
        job = models.get_job(job_id, user_id=user["id"])
        assert job["status"] == "error"
        assert "awaiting_approval" in (job["error"] or "")
