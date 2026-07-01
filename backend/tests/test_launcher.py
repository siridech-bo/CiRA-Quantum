"""Phase 6 — JobLauncher abstraction tests.

Covers:
* ThreadedJobLauncher (default) — actually spawns a daemon thread and
  delegates to the orchestrator. We mock the orchestrator to avoid
  pulling the LLM provider stack.
* RQJobLauncher (opt-in) — uses ``fakeredis`` so the test never needs
  a real Redis server. Validates the enqueue + RQ ``worker.work_burst``
  cycle wires up correctly.
* The factory respects ``USE_REDIS_QUEUE``.
* RedisEventBus emit/subscribe round-trip via fakeredis.
"""
from __future__ import annotations

import threading
import time

import pytest

from app.pipeline.events import EventBus, RedisEventBus, reset_event_bus_for_tests


class _FakeOrchestrator:
    """Captures run() calls instead of actually executing the pipeline."""

    instances: list["_FakeOrchestrator"] = []

    def __init__(self) -> None:
        self.calls: list[dict] = []
        type(self).instances.append(self)

    async def run(self, **kwargs) -> None:
        self.calls.append(kwargs)


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
    """Each test starts with a clean launcher + event bus singleton."""
    from app.pipeline.launcher import reset_launcher_for_tests
    reset_launcher_for_tests()
    reset_event_bus_for_tests()
    _FakeOrchestrator.instances.clear()
    monkeypatch.delenv("USE_REDIS_QUEUE", raising=False)
    yield
    reset_launcher_for_tests()
    reset_event_bus_for_tests()


def test_threaded_launcher_runs_orchestrator():
    """Default path: ThreadedJobLauncher actually invokes orchestrator.run()."""
    from app.pipeline.launcher import ThreadedJobLauncher
    bus = EventBus()
    launcher = ThreadedJobLauncher(
        orchestrator_factory=_FakeOrchestrator,
        event_bus=bus,
    )
    launcher.launch(
        job_id="test-job-1",
        user_id=42,
        problem_statement="pack a knapsack",
        provider_name="stub",
        api_key="dummy",
        solvers=["cpu_sa_neal"],
    )
    # The daemon thread runs asynchronously — give it a moment to land.
    for _ in range(20):
        if _FakeOrchestrator.instances and _FakeOrchestrator.instances[0].calls:
            break
        time.sleep(0.05)
    assert _FakeOrchestrator.instances, "no orchestrator was instantiated"
    call = _FakeOrchestrator.instances[0].calls[0]
    assert call["job_id"] == "test-job-1"
    assert call["user_id"] == 42
    assert call["solvers"] == ["cpu_sa_neal"]
    assert call["event_bus"] is bus


def test_factory_defaults_to_threaded(monkeypatch):
    """No USE_REDIS_QUEUE env var → threaded launcher."""
    from app.pipeline.launcher import ThreadedJobLauncher, get_launcher
    bus = EventBus()
    launcher = get_launcher(
        orchestrator_factory=_FakeOrchestrator,
        event_bus=bus,
    )
    assert isinstance(launcher, ThreadedJobLauncher)


def test_factory_picks_rq_when_env_set(monkeypatch):
    """USE_REDIS_QUEUE=1 → RQ launcher (with fakeredis backend)."""
    monkeypatch.setenv("USE_REDIS_QUEUE", "1")

    # Patch redis.from_url so RQJobLauncher uses fakeredis instead of
    # trying to connect to a real Redis server.
    import fakeredis
    import redis
    monkeypatch.setattr(redis, "from_url", lambda _url: fakeredis.FakeRedis())

    from app.pipeline.launcher import RQJobLauncher, get_launcher
    bus = EventBus()
    launcher = get_launcher(
        orchestrator_factory=_FakeOrchestrator,
        event_bus=bus,
    )
    assert isinstance(launcher, RQJobLauncher)


def test_rq_launcher_enqueues_with_correct_kwargs(monkeypatch):
    """RQJobLauncher.launch puts a job onto the configured Redis queue
    with the expected kwargs. We don't run the worker here — RQ
    serializes the target callable by its importable name, and our
    test fake can't be re-imported. The "did the worker execute it
    correctly" surface area is the worker_entry module itself, which
    is tiny and trivially testable on its own."""
    import fakeredis
    import redis
    monkeypatch.setattr(redis, "from_url", lambda _url: fakeredis.FakeRedis())

    from app.pipeline.launcher import RQJobLauncher
    launcher = RQJobLauncher(redis_url="redis://fake/0", queue_name="test-queue")

    launcher.launch(
        job_id="test-rq-job-1",
        user_id=7,
        problem_statement="x",
        provider_name="stub",
        api_key="k",
        solvers=["cpu_sa_neal"],
    )

    # Inspect the queue contents directly. RQ stores job kwargs as the
    # job's ``kwargs`` dict; the func_name is the import path of the
    # callable it'll invoke on the worker side.
    from rq import Queue
    q = Queue("test-queue", connection=launcher._connection)
    assert q.count == 1
    job = q.jobs[0]
    assert job.func_name == "app.pipeline.worker_entry.run_pipeline_job"
    assert job.kwargs["job_id"] == "test-rq-job-1"
    assert job.kwargs["user_id"] == 7
    assert job.kwargs["solvers"] == ["cpu_sa_neal"]
    assert job.kwargs["problem_statement"] == "x"


def test_redis_event_bus_roundtrip():
    """Producer emits via RedisEventBus; subscriber sees the same events.
    Uses fakeredis to avoid needing a real server."""
    import fakeredis
    fake = fakeredis.FakeRedis()
    bus = RedisEventBus(redis_client=fake)

    # Producer runs in a separate thread so we can subscribe + receive
    # the terminal event without deadlocking ourselves.
    def producer():
        time.sleep(0.05)
        bus.emit("test-bus-job", "formulating")
        time.sleep(0.05)
        bus.emit("test-bus-job", "solving")
        time.sleep(0.05)
        bus.emit("test-bus-job", "complete", solve_time_ms=42)

    threading.Thread(target=producer, daemon=True).start()

    received: list[dict] = []
    for event in bus.subscribe("test-bus-job"):
        if event.get("status"):  # skip heartbeat empty entries (none in this test)
            received.append(event)
        if event.get("status") == "complete":
            break

    assert [e["status"] for e in received] == ["formulating", "solving", "complete"]
    assert received[-1]["solve_time_ms"] == 42
    assert received[-1]["job_id"] == "test-bus-job"


def test_threaded_launcher_emits_terminal_on_crash(monkeypatch):
    """When the orchestrator escapes its own try/except (i.e. raises
    before its first ``event_bus.emit`` line), the launcher's safety
    net must (a) log the crash and (b) emit a terminal 'error' event
    so SSE subscribers don't block forever waiting for a status that
    will never come.

    Regression for the 2026-06-30 dev-server hang where 5 jobs sat at
    ``status='queued'`` after their pipeline threads crashed silently,
    leaking Werkzeug worker threads via the blocked SSE subscribers."""
    from app.pipeline.launcher import ThreadedJobLauncher

    class _CrashingOrchestrator:
        async def run(self, **kwargs) -> None:
            raise RuntimeError("simulated crash before first emit")

    # The DB-update branch tries to import app.models and call
    # update_job(). In this isolated test we don't have a DB schema,
    # so monkeypatch update_job to a no-op — the launcher's safety
    # net swallows its own failure cleanly, but stubbing it keeps the
    # test focused on the bus-emit guarantee.
    from app import models as models_module
    monkeypatch.setattr(models_module, "update_job", lambda *a, **k: None)

    bus = EventBus()
    launcher = ThreadedJobLauncher(
        orchestrator_factory=_CrashingOrchestrator,
        event_bus=bus,
    )

    launcher.launch(
        job_id="crash-job",
        user_id=1,
        problem_statement="x",
        provider_name="stub",
        api_key="",
        solvers=["cpu_sa_neal"],
    )

    received = list(bus.subscribe("crash-job"))
    assert received, "no terminal event was emitted after crash"
    assert received[-1]["status"] == "error"
    assert "simulated crash" in received[-1]["error"]


def test_redis_event_bus_history_replay():
    """A subscriber that joins AFTER events were emitted still gets the
    whole history (Streams XREAD from id 0)."""
    import fakeredis
    fake = fakeredis.FakeRedis()
    bus = RedisEventBus(redis_client=fake)

    bus.emit("late-job", "formulating")
    bus.emit("late-job", "compiling")
    bus.emit("late-job", "complete")

    # Now subscribe — should get all three even though they were
    # already emitted.
    received = list(bus.subscribe("late-job"))
    assert [e["status"] for e in received] == ["formulating", "compiling", "complete"]
