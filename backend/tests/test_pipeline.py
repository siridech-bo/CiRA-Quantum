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
