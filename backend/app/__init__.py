"""Flask application factory.

The factory pattern is preserved verbatim from the v1 CiRA Oculus
pattern the v2 spec inherits: blueprints carry the route surface; the
factory wires session config + CORS + DB init; tests instantiate their
own app with overrides via ``create_app(test_config=...)``.

Phase 0 mounts ``auth`` and ``health``. Phase 4 will mount ``solve``
and ``keys``; Phase 5B will mount ``templates``; Phase 7 will mount
``admin``. Each addition is a single ``register_blueprint`` line.

2026-07-02: the NSSM Windows deploy at ``quantum.cira-core.com``
serves the built Vue SPA from this same process (one-deployable
pattern, matching Oculus). Set ``CIRA_SPA_DIR`` to point at the
frontend build output (typically ``D:\\services\\cira-quantum\\frontend\\dist``)
and the factory mounts a catch-all handler at ``/`` that returns
``index.html`` for any non-``/api`` GET, so Vue Router owns
client-side routing. Left unset in dev — Vite dev server on
``localhost:3070`` handles the frontend during ``npm run dev``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from flask import Flask, send_from_directory
from flask_cors import CORS

from app.config import (
    CORS_ORIGINS,
    SECRET_KEY,
    SESSION_COOKIE_SAMESITE,
    SESSION_COOKIE_SECURE,
    SESSION_LIFETIME,
)
from app.models import init_db
from app.routes.admin import admin_bp
from app.routes.auth import auth_bp
from app.routes.benchmarks import benchmarks_bp
from app.routes.health import health_bp
from app.routes.keys import keys_bp
from app.routes.qldpc import qldpc_bp
from app.routes.qml import qml_bp
from app.routes.solve import solve_bp
from app.routes.templates import solve_from_template_bp, templates_bp


def create_app(test_config: dict[str, Any] | None = None) -> Flask:
    app = Flask(__name__)

    app.config.update(
        SECRET_KEY=SECRET_KEY,
        PERMANENT_SESSION_LIFETIME=SESSION_LIFETIME,
        SESSION_COOKIE_SAMESITE=SESSION_COOKIE_SAMESITE,
        SESSION_COOKIE_SECURE=SESSION_COOKIE_SECURE,
        # Tests opt in via test_config to use an in-memory or tmp DB.
        TESTING=False,
    )

    if test_config:
        app.config.update(test_config)

    CORS(app, supports_credentials=True, origins=CORS_ORIGINS)

    # Initialize the schema + default admin. Tests that want isolation
    # override DATABASE_PATH via ``CIRA_DB_PATH`` env var before the
    # process starts (see conftest fixture for the auth tests).
    with app.app_context():
        init_db()

    app.register_blueprint(auth_bp, url_prefix="/api/auth")
    app.register_blueprint(health_bp, url_prefix="/api")
    app.register_blueprint(keys_bp, url_prefix="/api/keys")
    app.register_blueprint(solve_bp, url_prefix="/api")
    app.register_blueprint(templates_bp, url_prefix="/api/templates")
    app.register_blueprint(solve_from_template_bp, url_prefix="/api")
    app.register_blueprint(benchmarks_bp, url_prefix="/api/benchmarks")
    app.register_blueprint(admin_bp, url_prefix="/api/admin")
    app.register_blueprint(qml_bp, url_prefix="/api/qml")
    app.register_blueprint(qldpc_bp, url_prefix="/api/qldpc")

    _register_spa(app)

    return app


def _register_spa(app: Flask) -> None:
    """Mount the built Vue SPA when ``CIRA_SPA_DIR`` is set.

    Two routes:
      * ``/assets/<path:filename>`` and any other hashed static file
        served directly from the SPA directory.
      * ``/`` and any non-``/api`` GET → returns ``index.html`` so
        Vue Router owns client-side routing (deep-links like
        ``/solve/jobs/abc123`` refresh cleanly instead of 404-ing).

    Env var only — no ``[vars]`` in the code path unless the operator
    explicitly opts in. Kept off by default so ``python run.py`` in
    dev doesn't try to serve a stale ``dist/`` when the frontend is
    live-reloading via Vite.
    """
    spa_dir_env = os.environ.get("CIRA_SPA_DIR")
    if not spa_dir_env:
        return

    spa_dir = Path(spa_dir_env).resolve()
    if not spa_dir.exists():
        # Fail loudly at boot rather than serving 404s at runtime — an
        # operator who set CIRA_SPA_DIR but forgot to run ``npm run
        # build`` should see the misconfiguration in the service log.
        raise RuntimeError(
            f"CIRA_SPA_DIR points at {spa_dir!s} which does not exist; "
            "did you forget to `npm run build` in frontend/?"
        )

    index_html = spa_dir / "index.html"
    if not index_html.exists():
        raise RuntimeError(
            f"CIRA_SPA_DIR ({spa_dir!s}) is missing index.html; "
            "the build likely failed or the wrong directory was passed."
        )

    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def _spa_or_asset(path: str):
        # /api/* is owned by blueprints registered above; Flask's
        # route ordering means those already match before we get
        # here, so the check is defensive against future refactors.
        if path.startswith("api/"):
            return app.response_class(status=404)

        # If the requested path names a real file under the SPA dir
        # (hashed asset, favicon, robots.txt), serve it directly.
        candidate = spa_dir / path
        if path and candidate.is_file():
            return send_from_directory(str(spa_dir), path)

        # Otherwise return index.html so Vue Router handles the URL.
        return send_from_directory(str(spa_dir), "index.html")
