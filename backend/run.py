"""Dev-server entry point.

In production this is replaced by ``gunicorn -w 4 -b 0.0.0.0:5009 'app:create_app()'``;
the file exists so ``python run.py`` works for local development and
matches the v2 spec's Getting Started instructions.
"""

from __future__ import annotations

from app import create_app
from app.config import DEV_PORT


def main() -> None:
    app = create_app()
    # debug=False on purpose — the dev server still auto-reloads if you wrap
    # with `flask run`, and Flask's debugger pin behavior interferes with
    # session cookies during interactive smoke tests.
    app.run(host="127.0.0.1", port=DEV_PORT, debug=False)


if __name__ == "__main__":
    main()
