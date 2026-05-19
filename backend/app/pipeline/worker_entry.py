"""Phase 6 — worker entry point for RQ-backed launches.

When :class:`app.pipeline.launcher.RQJobLauncher` enqueues a job, the
worker process pulls it off the queue and calls :func:`run_pipeline_job`
with the same kwargs the Flask route would have passed to a daemon
thread. The function:

1. Builds an Orchestrator
2. Acquires the event bus (Redis-backed when running under a worker;
   in-process otherwise — useful for tests)
3. Drives the pipeline to completion

Lives in its own module (separate from ``launcher.py``) so the import
graph is straightforward: ``rq`` workers do
``from app.pipeline.worker_entry import run_pipeline_job`` without
pulling the entire Flask app context.
"""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)


def run_pipeline_job(
    *,
    job_id: str,
    user_id: int,
    problem_statement: str,
    provider_name: str,
    api_key: str,
    solvers: list[str],
) -> None:
    """RQ worker callable. Same shape as ThreadedJobLauncher's target()."""
    # Lazy imports — keeps the worker startup light. The worker process
    # IS still a Python process that needs to load the orchestrator,
    # samplers, etc., but doing it on first call rather than at module
    # load lets ``rq worker`` come up fast.
    from app.pipeline import get_event_bus
    from app.pipeline.orchestrator import Orchestrator

    orchestrator = Orchestrator()
    bus = get_event_bus()
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
        logger.exception(
            "RQ worker pipeline run crashed for job %s", job_id,
        )
        raise
    finally:
        loop.close()
