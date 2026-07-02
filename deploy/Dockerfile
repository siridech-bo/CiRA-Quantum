# CiRA Quantum backend — production Docker image.
#
# Multi-stage build:
#   1. builder — installs dependencies against a full-toolchain image so
#      C-extension wheels (pyqpanda3, dwave-samplers, ortools, highspy)
#      can compile if a prebuilt wheel isn't available for the target.
#   2. runtime — copies the wheels + app source into a slim image so the
#      final container is under ~500 MB.
#
# Target platform: linux/amd64 (Cloudflare Containers today; Origin's
# pyqpanda3 wheels are amd64-only, so this is not optional).
#
# GPU simulated annealing is intentionally excluded from this image —
# CPSAT beats it on the instance sizes we actually ship (verified by
# backend/tests/test_cpsat_beats_gpu_sa.py). Skipping torch + cuda
# saves ~2 GB of base layers.

ARG PYTHON_VERSION=3.12

# ---- Stage 1: builder -----------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Toolchain for any C/C++ extension that doesn't have a prebuilt wheel.
# libgomp1 is pulled in by ortools; libstdc++6 is required by pyqpanda3
# at runtime too (kept in the runtime stage below via the base image).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy dependency manifests first so the wheels layer caches across
# source-only changes (fastest iteration on backend/app edits).
COPY backend/pyproject.toml backend/requirements.txt* ./
COPY backend/README.md ./

# Install the base app + the two extras we need in production:
#   quantum      — pyqpanda3 for local + Origin QAOA
#   ibm-quantum  — qiskit runtime for the IBMQ tier
#
# We deliberately DO NOT install:
#   [dev]        — pytest/ruff/fakeredis; not needed in prod
#   [queue]      — Redis-RQ; opt-in via USE_REDIS_QUEUE at runtime
#   [qml]        — heavy (PennyLane + scikit-learn + matplotlib);
#                  the QML app can install this in a separate image later
#   [qldpc]      — heavy; same reasoning as qml
#
# torch is NOT installed because we removed the GPU SA solver from the
# production lineup. If you re-enable it later, add a separate stage
# that installs torch from the cu128 wheel index (see backend/README.md).
COPY backend/ ./backend/

RUN pip wheel --wheel-dir /wheels \
        "./backend[quantum,ibm-quantum]" \
        gunicorn

# ---- Stage 2: runtime -----------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=5009 \
    # Prevent Flask from binding only to localhost in production.
    HOST=0.0.0.0

# libgomp1 for ortools, libstdc++6 for pyqpanda3, curl for the
# healthcheck. Everything else the app needs is pure Python.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libstdc++6 \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user so the container doesn't run as root — small
# but non-zero blast-radius win if any dependency turns out to have an
# arbitrary-file-write bug.
RUN useradd --create-home --uid 10001 cira

WORKDIR /app

# Install the prebuilt wheels — no compilation happens in this stage,
# so the runtime image stays small and cold-start-friendly.
COPY --from=builder /wheels /wheels
RUN pip install --no-index --find-links=/wheels \
        cira-quantum-backend[quantum,ibm-quantum] \
        gunicorn \
    && rm -rf /wheels

# Strip pyqpanda_alg's over-declared dev tooling — sphinx-autoapi,
# mypy, astroid, and the sphinx docs chain got pulled in as runtime
# install_requires when they should be dev_requires upstream. None of
# them are imported at pyqpanda_alg init time (verified via
# ``python -c "import pyqpanda_alg"`` after uninstall in dev), so
# removing them saves ~200 MB without breaking anything the
# production solve path needs. Add packages here if a future
# pyqpanda_alg version tightens its deps and one of these turns out
# to actually be required — boot will fail loudly with
# ModuleNotFoundError, which is the right signal.
RUN pip uninstall -y \
        sphinx \
        sphinx-autoapi \
        sphinxcontrib-applehelp \
        sphinxcontrib-devhelp \
        sphinxcontrib-htmlhelp \
        sphinxcontrib-jsmath \
        sphinxcontrib-qthelp \
        sphinxcontrib-serializinghtml \
        alabaster \
        babel \
        docutils \
        imagesize \
        roman-numerals \
        snowballstemmer \
        Pygments \
        mypy \
        mypy_extensions \
        astroid \
        ast-serialize \
        dill

COPY --chown=cira:cira backend/ /app/backend/
COPY --chown=cira:cira deploy/gunicorn_conf.py /app/gunicorn_conf.py

USER cira
WORKDIR /app/backend

# Cloudflare Containers respect the PORT env var; local runs default
# to 5009 to match the dev port. Bind to 0.0.0.0 so the platform's
# ingress can reach the container.
EXPOSE 5009

# The healthcheck lets Cloudflare's scheduler distinguish "starting"
# from "hung" — /api/health returns 200 as soon as create_app() finishes.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl --fail --silent "http://127.0.0.1:${PORT}/api/health" || exit 1

# gunicorn runs the Flask app via create_app(); see gunicorn_conf.py
# for worker count, timeout, and access-log format.
CMD ["gunicorn", "--config", "/app/gunicorn_conf.py", "app:create_app()"]
