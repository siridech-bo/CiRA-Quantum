"""Flask application factory.

The factory pattern is preserved verbatim from the v1 CiRA Oculus
pattern the v2 spec inherits: blueprints carry the route surface; the
factory wires session config + CORS + DB init; tests instantiate their
own app with overrides via ``create_app(test_config=...)``.

Phase 0 mounts ``auth`` and ``health``. Phase 4 will mount ``solve``
and ``keys``; Phase 5B will mount ``templates``; Phase 7 will mount
``admin``. Each addition is a single ``register_blueprint`` line.
"""

from __future__ import annotations

from typing import Any

from flask import Flask
from flask_cors import CORS

from app.config import (
    CORS_ORIGINS,
    SECRET_KEY,
    SESSION_COOKIE_SAMESITE,
    SESSION_COOKIE_SECURE,
    SESSION_LIFETIME,
)
from app.models import init_db
from app.routes.auth import auth_bp
from app.routes.benchmarks import benchmarks_bp
from app.routes.health import health_bp
from app.routes.keys import keys_bp
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

    return app
