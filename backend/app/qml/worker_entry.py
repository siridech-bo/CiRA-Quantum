"""QML-8 — RQ worker entry for QML training jobs.

Mirrors :mod:`app.pipeline.worker_entry` but for the QML training
pipeline. When ``USE_REDIS_QUEUE=1`` is set, the QML training route
enqueues this function on the shared Redis queue; an ``rq`` worker
process picks the job up and runs the trainer there instead of in
the Flask process. The event bus is the Redis-backed one (selected
automatically by :func:`app.pipeline.events.get_event_bus` under the
same env flag), so SSE subscribers in the web process still see live
training events.

Lazy imports inside the function so worker startup stays fast — RQ
imports the entry-point module once per job; the trainer's heavier
deps (PennyLane + scikit-learn) only load when training actually
begins.
"""

from __future__ import annotations

from typing import Any


def run_qml_training_job(
    *,
    job_id: str,
    dataset_id: str,
    hyperparameters: dict[str, Any],
) -> None:
    """Worker-side entry point. Hard-wires to the canonical event bus
    (whichever one ``get_event_bus`` returns in this process) so a
    cross-process Redis setup picks up the same broker."""
    from app.pipeline.events import get_event_bus
    from app.qml.trainer import run_training_job

    run_training_job(
        job_id=job_id,
        dataset_id=dataset_id,
        hyperparameters=hyperparameters,
        event_bus=get_event_bus(),
    )
