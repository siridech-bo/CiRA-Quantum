"""Tests for the QML-8 RQ-vs-threaded launcher path.

The optimization-side launcher is tested in test_launcher.py; this is
the QML mirror. Strategy: use ``fakeredis`` so we don't need a real
Redis, and inspect the queue's job kwargs rather than actually
executing the worker function (RQ serializes by import path + the
fake worker would need PennyLane + sklearn loaded just to verify
arg passing).
"""
from __future__ import annotations

import pytest


pytest.importorskip("rq")
pytest.importorskip("fakeredis")


@pytest.fixture
def fake_redis_module(monkeypatch):
    """Substitute ``redis.from_url`` with the fakeredis implementation
    so ``launch_training_job`` can construct an RQ queue without a
    live Redis."""
    import fakeredis
    server = fakeredis.FakeServer()
    fake_client = fakeredis.FakeStrictRedis(server=server)

    import redis
    monkeypatch.setattr(redis, "from_url", lambda *a, **kw: fake_client)
    return fake_client


def test_threaded_path_when_env_unset(monkeypatch):
    """Default (no USE_REDIS_QUEUE) → returns 'thread' and the launch
    function calls the threaded variant. We monkeypatch the threaded
    helper to a no-op so we don't actually spawn a daemon."""
    monkeypatch.delenv("USE_REDIS_QUEUE", raising=False)
    from app.qml import trainer as trainer_mod

    called: dict = {}
    def _fake_thread(**kwargs):
        called.update(kwargs)
    monkeypatch.setattr(trainer_mod, "launch_training_thread", _fake_thread)

    from app.pipeline.events import EventBus
    bus = EventBus()
    result = trainer_mod.launch_training_job(
        job_id="job-xyz",
        dataset_id="moons",
        hyperparameters={"n_epochs": 2},
        event_bus=bus,
    )
    assert result == "thread"
    assert called["job_id"] == "job-xyz"
    assert called["dataset_id"] == "moons"


def test_queue_path_when_env_set(monkeypatch, fake_redis_module):
    """USE_REDIS_QUEUE=1 → returns 'queue' and the job lands on the
    RQ queue with the expected kwargs."""
    monkeypatch.setenv("USE_REDIS_QUEUE", "1")
    monkeypatch.setenv("QUEUE_NAME", "test-qml-queue")
    from app.qml import trainer as trainer_mod

    # Sentinel: if the fallback fires, this gets called. The test
    # asserts it doesn't.
    def _should_not_fire(**_kwargs):
        raise AssertionError("threaded fallback should not have fired")
    monkeypatch.setattr(trainer_mod, "launch_training_thread", _should_not_fire)

    from app.pipeline.events import EventBus
    result = trainer_mod.launch_training_job(
        job_id="job-abc",
        dataset_id="iris",
        hyperparameters={"n_epochs": 3, "n_qubits": 2},
        event_bus=EventBus(),
    )
    assert result == "queue"

    # Inspect the queue contents.
    from rq import Queue
    queue = Queue("test-qml-queue", connection=fake_redis_module)
    assert queue.count == 1
    job = queue.jobs[0]
    assert job.kwargs["job_id"] == "job-abc"
    assert job.kwargs["dataset_id"] == "iris"
    assert job.kwargs["hyperparameters"]["n_qubits"] == 2
    # RQ uses the function's import path; this is what the worker
    # process re-imports + calls.
    assert "run_qml_training_job" in job.func_name


def test_queue_path_falls_back_on_redis_error(monkeypatch):
    """Redis init failure → falls back to threaded so a misconfigured
    env doesn't user-visibly break training."""
    monkeypatch.setenv("USE_REDIS_QUEUE", "1")
    # Force redis.from_url to blow up.
    import redis
    def _boom(*a, **kw):
        raise RuntimeError("synthetic redis init failure")
    monkeypatch.setattr(redis, "from_url", _boom)

    from app.qml import trainer as trainer_mod
    called: dict = {}
    def _fake_thread(**kwargs):
        called.update(kwargs)
    monkeypatch.setattr(trainer_mod, "launch_training_thread", _fake_thread)

    from app.pipeline.events import EventBus
    result = trainer_mod.launch_training_job(
        job_id="job-fall",
        dataset_id="moons",
        hyperparameters={},
        event_bus=EventBus(),
    )
    # Returns "thread" so the route logs the fallback.
    assert result == "thread"
    assert called["job_id"] == "job-fall"
