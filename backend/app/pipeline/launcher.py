"""Phase 6 — job launcher abstraction.

Defines the seam between the Flask route layer ("a user clicked Solve;
go run this pipeline") and the actual execution mechanism. Two
implementations ship out of the box:

* :class:`ThreadedJobLauncher` (default) — spawns a daemon thread in
  the Flask process and runs the orchestrator synchronously. No
  infrastructure dependencies; ideal for development and small
  classroom deployments.

* :class:`RQJobLauncher` (opt-in) — enqueues onto a Redis-backed
  RQ queue. A separate ``python -m rq.cli worker`` process picks
  jobs off the queue and runs them. The event bus migrates to
  Redis pub/sub so SSE in the Flask web process still sees status
  updates emitted from the worker process. Required for any
  deployment that needs:
    - concurrent solves > the GPU's per-process capacity
    - durability across Flask restarts
    - horizontal scaling of workers

Selection is driven by environment variables:

* ``USE_REDIS_QUEUE=1`` (case-insensitive ``1`` / ``true`` / ``yes``)
  enables the RQ launcher; anything else falls back to threaded.
* ``REDIS_URL`` (default ``redis://localhost:6379/0``) is the
  connection string both the launcher and the worker use.

The interface is intentionally narrow — ``launch(...)`` is the only
method routes call. Both implementations have the same blocking-
in-Flask cost: ~milliseconds (enqueue or thread-spawn).
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from abc import ABC, abstractmethod

from app.pipeline.events import EventBus

logger = logging.getLogger(__name__)


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in ("1", "true", "yes", "on")


def _redis_url() -> str:
    return os.environ.get("REDIS_URL", "redis://localhost:6379/0")


def _use_redis_queue() -> bool:
    return _truthy(os.environ.get("USE_REDIS_QUEUE"))


# Default queue name. Operator can override via QUEUE_NAME env var if
# they're running multiple instances against the same Redis.
def _queue_name() -> str:
    return os.environ.get("QUEUE_NAME", "cira-quantum-solves")


class JobLauncher(ABC):
    """Two-method abstraction the solve route depends on."""

    @abstractmethod
    def launch(
        self,
        *,
        job_id: str,
        user_id: int,
        problem_statement: str,
        provider_name: str,
        api_key: str,
        solvers: list[str],
    ) -> None:
        """Kick off the pipeline. Should return in milliseconds."""

    def name(self) -> str:
        return type(self).__name__


# ---- Threaded (default) -----------------------------------------------------


class ThreadedJobLauncher(JobLauncher):
    """Spawn a daemon thread that runs the orchestrator inline. This
    is the legacy Phase-4 path, kept as the default so a fresh install
    doesn't need Redis."""

    def __init__(self, *, orchestrator_factory, event_bus: EventBus):
        self._orch_factory = orchestrator_factory
        self._event_bus = event_bus

    def launch(
        self,
        *,
        job_id: str,
        user_id: int,
        problem_statement: str,
        provider_name: str,
        api_key: str,
        solvers: list[str],
    ) -> None:
        orchestrator = self._orch_factory()
        bus = self._event_bus

        def target() -> None:
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(
                    orchestrator.run(
                        job_id=job_id, user_id=user_id,
                        problem_statement=problem_statement,
                        provider_name=provider_name, api_key=api_key,
                        event_bus=bus, solvers=solvers,
                    )
                )
            except Exception:
                logger.exception("pipeline thread for job %s crashed", job_id)
            finally:
                loop.close()

        threading.Thread(target=target, daemon=True).start()


# ---- RQ-backed (opt-in) -----------------------------------------------------


class RQJobLauncher(JobLauncher):
    """Enqueue onto a Redis-backed RQ queue. The actual orchestrator
    runs in a separate worker process started via the worker.py
    entry point (or ``python -m rq.cli worker``).

    The launcher itself is import-deferred: ``rq`` and ``redis`` are
    only imported on construction, so installs that don't pull the
    ``[queue]`` extra never load them.
    """

    def __init__(self, *, redis_url: str | None = None, queue_name: str | None = None):
        import redis
        from rq import Queue
        url = redis_url or _redis_url()
        self._connection = redis.from_url(url)
        self._queue = Queue(
            queue_name or _queue_name(),
            connection=self._connection,
            default_timeout=3600,  # 1 hr — covers a slow LLM + slow solver
        )

    def launch(
        self,
        *,
        job_id: str,
        user_id: int,
        problem_statement: str,
        provider_name: str,
        api_key: str,
        solvers: list[str],
    ) -> None:
        # The worker imports this same module path and runs the
        # function with the kwargs. Strings + ints + lists serialize
        # cleanly via pickle (RQ's default).
        from app.pipeline.worker_entry import run_pipeline_job
        self._queue.enqueue(
            run_pipeline_job,
            kwargs={
                "job_id": job_id,
                "user_id": user_id,
                "problem_statement": problem_statement,
                "provider_name": provider_name,
                "api_key": api_key,
                "solvers": solvers,
            },
            job_id=job_id,  # use our DB id so the RQ job is greppable
            job_timeout=3600,
        )


# ---- Factory ---------------------------------------------------------------


_LAUNCHER_INSTANCE: JobLauncher | None = None


def get_launcher(*, orchestrator_factory, event_bus: EventBus) -> JobLauncher:
    """Return the singleton JobLauncher for this Flask process. Selects
    between threaded / RQ based on env. Cached so we don't reconnect to
    Redis on every solve."""
    global _LAUNCHER_INSTANCE
    if _LAUNCHER_INSTANCE is None:
        if _use_redis_queue():
            try:
                _LAUNCHER_INSTANCE = RQJobLauncher()
                logger.info("JobLauncher: RQ-backed, queue=%s", _queue_name())
            except Exception:
                logger.exception(
                    "RQ launcher init failed; falling back to threaded. "
                    "Check REDIS_URL or unset USE_REDIS_QUEUE."
                )
                _LAUNCHER_INSTANCE = ThreadedJobLauncher(
                    orchestrator_factory=orchestrator_factory,
                    event_bus=event_bus,
                )
        else:
            _LAUNCHER_INSTANCE = ThreadedJobLauncher(
                orchestrator_factory=orchestrator_factory,
                event_bus=event_bus,
            )
            logger.info("JobLauncher: threaded (in-process daemon)")
    return _LAUNCHER_INSTANCE


def reset_launcher_for_tests() -> None:
    """Clear the singleton — used by tests that flip env vars between
    cases."""
    global _LAUNCHER_INSTANCE
    _LAUNCHER_INSTANCE = None
