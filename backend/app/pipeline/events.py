"""Per-job event bus for the SSE stream.

Phase 4 ships a thread-safe in-process bus. A producer thread (the
pipeline orchestrator running in a background thread) calls
``emit(job_id, status, **fields)``; a consumer thread (the Flask SSE
route handler) iterates ``subscribe(job_id)`` to receive events.

Design notes:

* **History replay.** A subscriber that connects *after* events have
  been emitted (a common SSE race when the frontend opens the stream a
  beat too late) still gets the full history. The bus keeps a buffer
  per job until the terminal event arrives.
* **Terminal events close the subscription.** Statuses ``complete``
  and ``error`` end the iteration cleanly. Subscribers don't have to
  guess when to stop.
* **One-process scope.** Phase 4's orchestrator runs in a background
  thread of the same Flask process. Phase 6's queue (Redis + RQ)
  replaces this with a cross-process mechanism; the public interface
  here is small enough that swap-out is a 2-file change.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterator
from typing import Any

_TERMINAL_STATUSES = frozenset({"complete", "error", "awaiting_approval"})

# How long ``subscribe()`` blocks on a single ``queue.get()`` before
# yielding a heartbeat. Heartbeats let the SSE handler write to the
# response stream periodically, which is how Werkzeug detects a
# disconnected client and reclaims the request thread. Without this,
# a stuck producer (e.g., an orchestrator crashed without emitting
# terminal) would pin Flask threads forever — exactly the hang we hit
# on 2026-06-30. 15 s is short enough that dead clients are noticed
# quickly and long enough that healthy clients don't see noisy traffic.
_HEARTBEAT_INTERVAL_S: float = 15.0

# Sentinel marker for heartbeat events. The SSE handler turns these
# into ``: heartbeat\n\n`` comment lines (which browsers ignore) so the
# frontend's status handler never sees them.
HEARTBEAT_MARKER = "_heartbeat"


class _JobChannel:
    """One queue + history buffer per job_id."""

    def __init__(self) -> None:
        self.queue: queue.Queue = queue.Queue()
        self.history: list[dict] = []
        self.closed = False
        self.lock = threading.Lock()


class EventBus:
    """Thread-safe, in-process publish-subscribe for job status events.

    Singleton-friendly: there's one default bus per process, accessible
    via :func:`get_event_bus`. Tests instantiate their own.
    """

    def __init__(self) -> None:
        self._channels: dict[str, _JobChannel] = {}
        self._channels_lock = threading.Lock()

    def _channel(self, job_id: str) -> _JobChannel:
        with self._channels_lock:
            ch = self._channels.get(job_id)
            if ch is None:
                ch = _JobChannel()
                self._channels[job_id] = ch
            return ch

    def reset_channel(self, job_id: str) -> None:
        """Wipe any existing channel state for ``job_id`` so future
        :meth:`emit` calls land on a fresh history. Used by the approval
        gate when resuming a paused job — the prior session emitted a
        terminal ``awaiting_approval`` event that closed the channel,
        and the resume's stages need a clean bus session so their
        events aren't silently dropped by ``if ch.closed: return`` in
        emit()."""
        with self._channels_lock:
            self._channels.pop(job_id, None)

    def emit(self, job_id: str, status: str, **fields: Any) -> None:
        """Push an event into ``job_id``'s queue. Replayed by any later
        subscriber until the channel is dropped."""
        event = {"job_id": job_id, "status": status, **fields}
        ch = self._channel(job_id)
        with ch.lock:
            if ch.closed:
                return
            ch.history.append(event)
            ch.queue.put(event)
            if status in _TERMINAL_STATUSES:
                ch.closed = True
                # Sentinel so any blocked subscribers wake up.
                ch.queue.put(None)

    def subscribe(self, job_id: str) -> Iterator[dict]:
        """Iterate events for ``job_id``.

        Yields the full event history first (so late subscribers don't
        miss anything), then blocks waiting for new events, finally
        terminates on the ``complete`` or ``error`` event.
        """
        ch = self._channel(job_id)
        # Snapshot the history *before* we start consuming the live queue
        # so we don't double-deliver events the producer emits during the
        # snapshot.
        with ch.lock:
            replayed = list(ch.history)
            already_closed = ch.closed and not ch.queue.queue
        for event in replayed:
            yield event
            if event["status"] in _TERMINAL_STATUSES:
                return

        if already_closed:
            return

        # Drain anything new that landed after the snapshot, then keep
        # blocking until the terminal event arrives. Use a heartbeat
        # timeout so a stuck producer (or a forever-queued job whose
        # orchestrator thread silently died) can't pin this consumer's
        # thread indefinitely — the heartbeat yields let the SSE
        # handler write to the response stream periodically, which is
        # how Werkzeug detects disconnected clients.
        import queue as _queue_mod

        while True:
            try:
                event = ch.queue.get(timeout=_HEARTBEAT_INTERVAL_S)
            except _queue_mod.Empty:
                # No event in the heartbeat window — emit a sentinel
                # the SSE handler turns into a comment line. Browsers
                # ignore comment-only SSE chunks; the frontend's status
                # listener never sees them.
                yield {HEARTBEAT_MARKER: True}
                continue
            if event is None:
                # Sentinel posted by emit() after terminal status — return.
                return
            if event in replayed:
                # Already delivered during the snapshot phase.
                continue
            yield event
            if event["status"] in _TERMINAL_STATUSES:
                return


class RedisEventBus:
    """Phase 6 — cross-process event bus via Redis Streams.

    Same public interface as :class:`EventBus` (``emit`` / ``subscribe``)
    so callers don't need to know which backend is in play. Events are
    written to a stream named ``cira:events:{job_id}``; subscribers
    XREAD from id 0 to get full history + blocking wait for future events.

    Used when ``USE_REDIS_QUEUE=1`` is set in the environment — the
    worker process emits, the Flask SSE process subscribes. Streams
    naturally give us history replay (a late subscriber still sees all
    earlier events) and a consistent terminal signal.

    Streams are auto-expired after the terminal event is seen, so we
    don't leak Redis memory for finished jobs.
    """

    def __init__(self, redis_client=None):
        import json
        self._json = json
        if redis_client is None:
            import os
            import redis
            url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
            redis_client = redis.from_url(url)
        self._r = redis_client

    @staticmethod
    def _stream_key(job_id: str) -> str:
        return f"cira:events:{job_id}"

    def reset_channel(self, job_id: str) -> None:
        """Delete the underlying Redis Stream for ``job_id`` so the
        next ``emit`` starts a fresh stream. Same semantics as
        :meth:`EventBus.reset_channel` — used by the approval gate's
        resume path so post-approval events don't replay alongside
        the pre-approval history."""
        try:
            self._r.delete(self._stream_key(job_id))
        except Exception:  # noqa: BLE001
            # Redis unreachable / transient — the worst case is the
            # next SSE subscriber sees stale history; not worth raising.
            pass

    def emit(self, job_id: str, status: str, **fields: Any) -> None:
        event = {"job_id": job_id, "status": status, **fields}
        payload = self._json.dumps(event)
        key = self._stream_key(job_id)
        self._r.xadd(key, {"data": payload})
        if status in _TERMINAL_STATUSES:
            # Keep finished streams around for an hour so very late
            # subscribers still get the history; after that, Redis
            # reclaims the memory.
            self._r.expire(key, 3600)

    def subscribe(self, job_id: str) -> Iterator[dict]:
        key = self._stream_key(job_id)
        last_id = "0"  # start from the beginning of the stream
        while True:
            # XREAD with BLOCK waits indefinitely until new events
            # arrive (or a millisecond timeout if we want to poll).
            # Block for up to 30 s; if nothing arrives, loop and try
            # again so the Flask SSE handler can periodically check
            # whether the client has disconnected.
            entries = self._r.xread({key: last_id}, block=30_000, count=64)
            if not entries:
                # No events in 30 s — yield a heartbeat sentinel so the
                # SSE handler can write a comment line to the response
                # stream, which is how Werkzeug detects a disconnected
                # client. (Fixed 2026-06-30 — this branch previously
                # only ``continue``d, which kept the connection alive
                # in Redis but never yielded to the Flask SSE handler,
                # so dead clients still leaked threads.)
                yield {HEARTBEAT_MARKER: True}
                continue
            # entries: [(stream_key_bytes, [(entry_id_bytes, {b"data": b"..."}), ...])]
            for _stream_name, items in entries:
                for entry_id, fields in items:
                    last_id = (
                        entry_id.decode() if isinstance(entry_id, bytes)
                        else entry_id
                    )
                    raw = fields.get(b"data") or fields.get("data") or b""
                    if isinstance(raw, bytes):
                        raw = raw.decode()
                    try:
                        event = self._json.loads(raw)
                    except Exception:  # pragma: no cover — defensive
                        continue
                    yield event
                    if event.get("status") in _TERMINAL_STATUSES:
                        return


_default_bus: EventBus | RedisEventBus | None = None
_default_bus_lock = threading.Lock()


def get_event_bus() -> EventBus | RedisEventBus:
    """Module-level singleton used by the Flask app + pipeline thread /
    worker process. When ``USE_REDIS_QUEUE=1`` the bus is Redis-backed
    so events from a worker process reach the SSE handler in the Flask
    process; otherwise the in-process queue is used (legacy path).
    """
    global _default_bus
    with _default_bus_lock:
        if _default_bus is None:
            import os
            use_redis = (os.environ.get("USE_REDIS_QUEUE") or "").strip().lower() in (
                "1", "true", "yes", "on",
            )
            if use_redis:
                try:
                    _default_bus = RedisEventBus()
                except Exception:
                    # Redis unreachable — fall back to in-process so the
                    # platform still works. Operator will see this in the
                    # Flask log.
                    import logging
                    logging.getLogger(__name__).exception(
                        "RedisEventBus init failed; falling back to in-process",
                    )
                    _default_bus = EventBus()
            else:
                _default_bus = EventBus()
        return _default_bus


def reset_event_bus_for_tests() -> None:
    """Clear the singleton so tests that flip env vars get a fresh bus."""
    global _default_bus
    with _default_bus_lock:
        _default_bus = None
