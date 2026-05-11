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

_TERMINAL_STATUSES = frozenset({"complete", "error"})


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
        # blocking until the terminal event arrives.
        while True:
            event = ch.queue.get()
            if event is None:
                # Sentinel posted by emit() after terminal status — return.
                return
            if event in replayed:
                # Already delivered during the snapshot phase.
                continue
            yield event
            if event["status"] in _TERMINAL_STATUSES:
                return


_default_bus: EventBus | None = None
_default_bus_lock = threading.Lock()


def get_event_bus() -> EventBus:
    """Module-level singleton used by the Flask app + pipeline thread."""
    global _default_bus
    with _default_bus_lock:
        if _default_bus is None:
            _default_bus = EventBus()
        return _default_bus
