"""Backend configuration.

All values fall back to placeholder defaults so the test suite and the
local dev server start without env wiring. Production overrides ride on
top via environment variables — that's the contract the v2 spec's
"Configuration" section pins.

Hard rule (from the v2 AI Coder workflow notes): no real secrets in
committed code. The defaults below are deliberately *unreal* — they
look like the kind of placeholder string an operator would immediately
replace.
"""

from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path

# ---- Layout ----

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATABASE_PATH = str(DATA_DIR / "app.db")

# ---- Session ----

SECRET_KEY = os.environ.get("SECRET_KEY", "change-in-production")
SESSION_LIFETIME = timedelta(hours=8)
SESSION_COOKIE_SAMESITE = "Lax"
# `SESSION_COOKIE_SECURE = True` only in production with HTTPS — Phase 0 ships
# with the dev-friendly default; Phase 7's hardening flips it.
SESSION_COOKIE_SECURE = os.environ.get("SESSION_COOKIE_SECURE", "false").lower() == "true"

# ---- BYOK encryption (already in use in app/crypto.py since Phase 3) ----

KEY_ENCRYPTION_SECRET = os.environ.get(
    "KEY_ENCRYPTION_SECRET",
    "change-this-32-byte-secret-now!!!",
)

# ---- Roles + default admin ----

ROLE_ADMIN = "admin"
ROLE_USER = "user"
DEFAULT_ADMIN_USERNAME = "admin"
DEFAULT_ADMIN_PASSWORD = "admin123"  # change immediately after first login

# ---- CORS allow-list (Vite dev server origins) ----
#
# Spec defaults are 3011/3012 but this host has those ports occupied, so
# 3070 is the operative dev port. We keep the spec defaults in the list
# too so a teammate cloning the repo on a clean machine doesn't have to
# change config.
CORS_ORIGINS = [
    "http://localhost:3070",
    "http://localhost:3011",
    "http://localhost:3012",
]

# ---- Optimization defaults (already informally settled in Phase 2/3 work) ----

GPU_DEVICE = os.environ.get("GPU_DEVICE", "cuda:0")
DEFAULT_NUM_READS = 1000
DEFAULT_NUM_SWEEPS = 1000
MAX_PROBLEM_VARIABLES = 50000
SOLVE_TIMEOUT_SECONDS = 300

# ---- Rate limits (per user, enforced in Phase 7 — declared here so the
# constants are discoverable from one place) ----

MAX_CONCURRENT_JOBS = 1
MAX_JOBS_PER_HOUR = 10
MAX_PROMPT_LENGTH = 8000

# ---- Local LLM endpoint ----

LOCAL_LLM_ENDPOINT = os.environ.get("LOCAL_LLM_ENDPOINT", "http://localhost:11434")

# ---- Dev server port (declared so Phase 4's smoke tests can reference it) ----

DEV_PORT = int(os.environ.get("PORT", "5009"))
