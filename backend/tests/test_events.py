"""Phase 4 — SSE event-bus mechanics tests."""

from __future__ import annotations

import threading
import time

import pytest

from app.pipeline.events import EventBus


@pytest.fixture
def bus() -> EventBus:
    return EventBus()


def test_subscribe_after_emit_replays_history(bus):
    """A subscriber that connects *after* the producer has already emitted
    must see the full history — exactly the case where a Phase-5 client
    opens an SSE stream a beat too late."""
    bus.emit("job-1", "formulating")
    bus.emit("job-1", "compiling", num_variables=5)
    events = []
    for event in bus.subscribe("job-1"):
        events.append(event)
        if event["status"] == "complete" or len(events) >= 3:
            break
        if len(events) == 2:
            # Producer emits the terminal event after subscription:
            bus.emit("job-1", "complete")
    statuses = [e["status"] for e in events]
    assert statuses == ["formulating", "compiling", "complete"]
    # Extra fields ride along.
    assert events[1]["num_variables"] == 5


def test_terminal_event_closes_subscription(bus):
    bus.emit("job-2", "error", error="boom")
    events = list(bus.subscribe("job-2"))
    assert events == [{"job_id": "job-2", "status": "error", "error": "boom"}]


def test_subscriptions_are_per_job(bus):
    bus.emit("job-A", "complete")
    bus.emit("job-B", "error", error="other failure")
    a = list(bus.subscribe("job-A"))
    b = list(bus.subscribe("job-B"))
    assert [e["status"] for e in a] == ["complete"]
    assert [e["status"] for e in b] == ["error"]
    assert b[0]["error"] == "other failure"


def test_producer_thread_and_consumer_thread_interleave(bus):
    """Concurrent producer + consumer threads converge on the same
    ordered event sequence."""
    seen: list[dict] = []

    def consume() -> None:
        for event in bus.subscribe("job-async"):
            seen.append(event)
            if event["status"] in {"complete", "error"}:
                break

    consumer = threading.Thread(target=consume)
    consumer.start()
    # Give the consumer a fair shot at subscribing first.
    time.sleep(0.05)
    for status in ("formulating", "compiling", "validating", "solving", "complete"):
        bus.emit("job-async", status)
        time.sleep(0.01)
    consumer.join(timeout=2.0)
    assert [e["status"] for e in seen] == [
        "formulating", "compiling", "validating", "solving", "complete",
    ]


def test_emit_unknown_job_id_creates_queue(bus):
    """``emit`` for a never-subscribed job is a no-op-safe path; the
    later subscriber still sees the event."""
    bus.emit("job-new", "formulating")
    bus.emit("job-new", "complete")
    events = list(bus.subscribe("job-new"))
    assert [e["status"] for e in events] == ["formulating", "complete"]


def test_get_event_bus_returns_singleton():
    """The module-level default bus is the same instance everywhere — the
    route handler and the pipeline thread share it."""
    from app.pipeline.events import get_event_bus
    a = get_event_bus()
    b = get_event_bus()
    assert a is b


def test_subscribe_yields_heartbeat_when_producer_is_silent(
    bus, monkeypatch: pytest.MonkeyPatch,
):
    """A stuck or never-emitting producer must NOT pin the consumer's
    thread indefinitely. Regression for the 2026-06-30 dev-server hang
    where 5 zombie ``queued`` jobs leaked Werkzeug threads because
    ``queue.get()`` blocked forever. After the fix, subscribe() emits a
    heartbeat sentinel each ``_HEARTBEAT_INTERVAL_S`` so the SSE
    handler can write to the response stream and detect dead clients."""
    from app.pipeline import events as events_module

    # Short interval so the test doesn't have to wait the default 15 s.
    monkeypatch.setattr(events_module, "_HEARTBEAT_INTERVAL_S", 0.1)

    seen: list[dict] = []
    iterator = bus.subscribe("job-stuck")

    def consume() -> None:
        for event in iterator:
            seen.append(event)
            if len(seen) >= 3:
                break

    consumer = threading.Thread(target=consume, daemon=True)
    consumer.start()
    consumer.join(timeout=2.0)

    # We expect at least 3 heartbeat sentinels — the producer never
    # emitted, so every yield must be a heartbeat (no real events).
    assert len(seen) >= 3
    assert all(e.get(events_module.HEARTBEAT_MARKER) for e in seen), seen
    # The sentinels carry no "status" field, so a misconfigured SSE
    # consumer that forwards them would not corrupt the frontend's
    # status state.
    assert all("status" not in e for e in seen)


def test_subscribe_recovers_to_real_event_after_heartbeats(
    bus, monkeypatch: pytest.MonkeyPatch,
):
    """A producer that emits *after* a quiet period should still see its
    event delivered. Heartbeats don't poison the channel."""
    from app.pipeline import events as events_module
    monkeypatch.setattr(events_module, "_HEARTBEAT_INTERVAL_S", 0.05)

    seen: list[dict] = []

    def consume() -> None:
        for event in bus.subscribe("job-late"):
            seen.append(event)
            if event.get("status") in {"complete", "error"}:
                break

    consumer = threading.Thread(target=consume, daemon=True)
    consumer.start()
    time.sleep(0.25)  # let several heartbeats fire
    bus.emit("job-late", "complete")
    consumer.join(timeout=2.0)

    statuses = [e.get("status") for e in seen if "status" in e]
    assert statuses == ["complete"]
    # Heartbeats and the real event coexisted in the iterator output.
    heartbeat_count = sum(
        1 for e in seen if e.get(events_module.HEARTBEAT_MARKER)
    )
    assert heartbeat_count >= 2, f"expected ≥2 heartbeats before complete, got {seen}"
