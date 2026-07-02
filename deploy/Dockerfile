# CiRA Quantum backend — production Docker image.
#
# Multi-stage build:
#   1. frontend-builder — Node stage that builds the Vue SPA. Output
#      lives at /build/frontend/dist and is copied into the runtime
#      image so Flask serves the SPA directly via CIRA_SPA_DIR.
#   2. builder — installs backend Python deps against a full-toolchain
#      image so C-extension wheels (pyqpanda3, dwave-samplers, ortools,
#      highspy) can compile if a prebuilt wheel isn't available.
#   3. runtime — copies both the wheels and the built SPA into a slim
#      image so the final container is under ~2.5 GB uncompressed
#      (~750 MB compressed).
#
# Target platform: linux/amd64 (Cloudflare Containers today; Origin's
# pyqpanda3 wheels are amd64-only, so this is not optional). The
# Windows deploy on .110 runs this same image via Docker Desktop's
# WSL2 backend, which transparently runs Linux containers.
#
# GPU simulated annealing is intentionally excluded from this image —
# CPSAT beats it on the instance sizes we actually ship (verified by
# backend/tests/test_cpsat_beats_gpu_sa.py). Skipping torch + cuda
# saves ~2 GB of base layers.

ARG PYTHON_VERSION=3.12
# Node 20 (LTS) rather than 22 because vue-tsc's TypeScript-internals
# monkey-patch is incompatible with Node 22.23+'s bundled TS ("Search
# string not found: /supportedTSExtensions = .*(?=;)/"). Dev on Node 22
# works fine because the local vue-tsc pins a compatible TS at
# npm ci time, but the Docker build context resolves a different
# TS version that trips the patch. Node 20 avoids the issue entirely.
ARG NODE_VERSION=20

# ---- Stage 1: frontend-builder -------------------------------------
# Builds the Vue SPA once at image-build time. Output at
# /build/frontend/dist is later copied into the runtime image, and
# Flask serves it via the CIRA_SPA_DIR env var (see backend/app/__init__.py).
FROM node:${NODE_VERSION}-slim AS frontend-builder

WORKDIR /build/frontend

# Copy package manifests first so `npm ci` caches across source-only
# frontend changes.
COPY frontend/package.json frontend/package-lock.json* ./

# `--ignore-scripts` skips any postinstall hooks that pull down binaries
# we don't want in the deterministic build. Vite + Vue3 don't have any
# essential postinstall steps, so this is safe.
RUN npm ci --ignore-scripts

COPY frontend/ ./

# Vite writes production build output to ./dist by default. If the
# project ever changes vite.config to a different outDir, update
# the COPY --from=frontend-builder path in the runtime stage too.
#
# We bypass the ``npm run build`` script and invoke vite directly
# because the project's build script wraps ``vue-tsc --noEmit &&
# vite build``, and vue-tsc 1.8.x's TypeScript-internals monkey-patch
# breaks against the TS 5.4+ resolved into the Docker install (dev
# happens to pin a compatible TS via the local lockfile). Type-check
# happens at commit time in the developer's workflow; the Docker
# image just needs to produce dist/. Upgrading vue-tsc to 2.x is the
# proper fix and lives in a follow-up PR — see docs/OPTIMIZATION.md
# roadmap section.
RUN npx vite build \
    && test -f dist/index.html \
    || (echo "vite build produced no dist/index.html — vite config change?" && exit 1)

# ---- Stage 2: builder ----------------------------------------------
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

# ---- Stage 3: runtime -----------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Default to 5209 so the NSSM-wrapped ``docker run`` on .110 lines
    # up with the Cloudflare Tunnel ingress rule out of the box. Override
    # via ``-e PORT=…`` when running the container in a different context.
    PORT=5209 \
    # Prevent Flask from binding only to localhost in production.
    HOST=0.0.0.0 \
    # Tell the Flask factory where the built SPA lives; ``_register_spa()``
    # mounts index.html at / with SPA fallback so Vue Router owns
    # client-side routing.
    CIRA_SPA_DIR=/app/frontend/dist

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

# The built Vue SPA from stage 1. CIRA_SPA_DIR (set above) points here,
# so Flask's _register_spa() picks it up automatically at boot and
# serves index.html at / with SPA fallback.
COPY --from=frontend-builder --chown=cira:cira /build/frontend/dist /app/frontend/dist

USER cira
WORKDIR /app/backend

# Bind to 0.0.0.0 so a ``docker run -p 5209:5209`` port map works
# regardless of the host platform.
EXPOSE 5209

# The healthcheck lets Cloudflare's scheduler (or Docker Desktop /
# NSSM's ``sc query`` polling) distinguish "starting" from "hung" —
# /api/health returns 200 as soon as create_app() finishes.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl --fail --silent "http://127.0.0.1:${PORT}/api/health" || exit 1

# gunicorn runs the Flask app via create_app(); see gunicorn_conf.py
# for worker count, timeout, and access-log format. gunicorn is fine
# here (Linux container, no Windows fork limitations) — waitress in
# the venv is used only for the (deprecated) native-Python NSSM path.
CMD ["gunicorn", "--config", "/app/gunicorn_conf.py", "app:create_app()"]
