"""Phase 6 — RQ worker entry point.

Run this in a separate process when ``USE_REDIS_QUEUE=1`` is set on
the Flask web process. The worker picks jobs off the configured Redis
queue and runs the full pipeline (formulate / compile / validate /
solve / interpret).

Usage::

    # Terminal 1 — start Redis (Memurai on Windows, ``redis-server``
    # on Linux, or `docker run -p 6379:6379 redis:7`).

    # Terminal 2 — start the worker:
    set REDIS_URL=redis://localhost:6379/0          # PowerShell
    set USE_REDIS_QUEUE=1
    python worker.py

    # Terminal 3 — start the Flask app with the same env vars:
    set REDIS_URL=redis://localhost:6379/0
    set USE_REDIS_QUEUE=1
    python run.py

When ``USE_REDIS_QUEUE`` is unset, the Flask app does NOT enqueue —
it runs the pipeline in a daemon thread inside the web process, and
this worker is unused.

Multiple workers can run against the same queue for concurrent solve
throughput; they coordinate via Redis.
"""

from __future__ import annotations

import logging
import os
import sys


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        import redis
        from rq import Connection, Queue, SimpleWorker, Worker
        from rq.timeouts import BaseDeathPenalty
    except ImportError:
        print(
            "rq / redis not installed. Install the queue extras:\n"
            "  pip install -e \".[queue]\"",
            file=sys.stderr,
        )
        sys.exit(1)

    url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    queue_name = os.environ.get("QUEUE_NAME", "cira-quantum-solves")
    conn = redis.from_url(url)

    # Pick the right Worker class for the platform:
    # * Linux/macOS: default Worker forks per job (best isolation).
    # * Windows: no os.fork → SimpleWorker (single process). Also no
    #   signal.SIGALRM → patch out the death-penalty timeout. Operator
    #   loses per-job hard-kill on Windows; recommendation is to run
    #   the worker under WSL / Docker for production deployments.
    if hasattr(os, "fork"):
        worker_cls = Worker
    else:
        class _NoTimeout(BaseDeathPenalty):
            def setup_death_penalty(self):
                pass
            def cancel_death_penalty(self):
                pass

        class _WindowsWorker(SimpleWorker):
            death_penalty_class = _NoTimeout

        worker_cls = _WindowsWorker

    print(
        f"RQ worker starting — queue={queue_name!r} url={url!r} "
        f"class={worker_cls.__name__}",
        file=sys.stderr,
    )
    with Connection(conn):
        worker = worker_cls([Queue(queue_name)])
        # Pre-import the heavy modules so the first job doesn't pay the
        # full import cost; subsequent jobs reuse the imports.
        from app.pipeline import worker_entry  # noqa: F401
        worker.work(with_scheduler=False)


if __name__ == "__main__":
    main()
