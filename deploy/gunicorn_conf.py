"""Gunicorn configuration for the CiRA Quantum backend in production.

Tuned for the current shipping profile:
  * Single-container deployment (Cloudflare Containers, Fly.io, or a
    bare VPS behind nginx). No pre-fork worker pool coordination across
    hosts — each replica is independent.
  * Modest concurrency (research customers, dozens of active users).
  * Long-lived SSE connections on ``/api/jobs/<id>/events`` and the
    approval-gate resume path.
  * Outbound HTTP calls to Anthropic/OpenAI/Origin QC / IBM Quantum
    that can legitimately take 30–60 s per call.

Worker model
------------
We use ``gthread`` workers rather than ``sync`` because SSE holds
one connection per streaming client for minutes at a time; a sync
pool would exhaust its worker slots. ``gthread`` gives us N cooperative
threads per worker without needing an async runtime shim on top of
Flask.

Numbers below are safe defaults for a 1-vCPU / 1 GB RAM instance
(Cloudflare Containers small / Fly shared-cpu-1x). Bump ``workers``
proportionally with the vCPU count via the ``GUNICORN_WORKERS`` env
var if you scale up the container class.
"""

from __future__ import annotations

import os

# ---- Bind ----

# Cloudflare Containers passes the listener port via ``PORT``; other
# hosts (Fly, plain Docker) can override too. Default matches the dev
# server so ``docker run`` locally hits the same address.
_port = int(os.environ.get("PORT", "5009"))
bind = f"0.0.0.0:{_port}"

# ---- Concurrency ----

# One process is enough on a 1-vCPU container — pyqpanda3 and dimod
# do most heavy work in C extensions that release the GIL, and adding
# more workers on a single core just increases context switching.
# Scale-out is horizontal (more container replicas), not vertical.
workers = int(os.environ.get("GUNICORN_WORKERS", "1"))

# gthread: cooperative threading, one per client. Keeps SSE cheap.
worker_class = "gthread"
threads = int(os.environ.get("GUNICORN_THREADS", "16"))

# ---- Timeouts ----

# Long timeout because a QAOA sim can legitimately take ~60 s and the
# formulation LLM call can take 30–45 s. Anything longer than 300 s is
# almost certainly hung — killing the worker frees the slot for the
# next request rather than letting a wedged solver block traffic.
timeout = int(os.environ.get("GUNICORN_TIMEOUT", "300"))

# graceful_timeout controls how long a worker gets to finish in-flight
# requests during a deploy or restart. 30 s covers the slowest
# summarize_solution call comfortably.
graceful_timeout = 30

# Keep TCP connections alive briefly so the SSE reconnect burst after
# a deploy doesn't hammer the accept queue.
keepalive = 5

# ---- Logging ----

# Both stdout so Cloudflare Containers picks them up automatically
# (their log collector reads stdout/stderr, no file path plumbing).
accesslog = "-"
errorlog = "-"
loglevel = os.environ.get("GUNICORN_LOG_LEVEL", "info")

# Access log format: request-id-friendly, includes upstream time so we
# can spot slow LLM calls in the logs without tracing tooling.
access_log_format = (
    '%(h)s "%(r)s" %(s)s %(b)s %(L)ss "%(f)s" "%(a)s"'
)

# ---- Process naming ----

proc_name = "cira-quantum-backend"

# ---- Preload ----

# Load the app in the master process before forking workers. Speeds up
# worker start (no per-worker imports) at the cost of losing hot reload
# — which we don't want in production anyway.
preload_app = True
