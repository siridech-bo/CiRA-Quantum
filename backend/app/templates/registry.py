"""Template registry.

The template library is a directory of JSON files — no DB table. Each
file conforms to ``schemas/template_v1.json``. The registry loads the
whole directory at module import, validates every file against the
schema (so a malformed template fails at process startup, not at the
first user request), and exposes lookup / aggregation helpers.

Why JSON files and not a DB row:

  * Templates are *version-controlled artifacts*. A reviewer can read a
    template in a PR diff. A DB row is invisible to git.
  * The Phase-10 contribution pipeline expects "drop a JSON file into
    the library and open a PR" as the contribution shape.
  * The registry is tiny — even 1000 templates fit in memory trivially.
"""

from __future__ import annotations

import json
import logging
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any

import jsonschema

logger = logging.getLogger(__name__)

_LIBRARY_DIR = Path(__file__).parent / "library"
_SCHEMA_PATH = Path(__file__).parent / "schemas" / "template_v1.json"

# The 10 IDs the v2 spec calls out as ship-required.
REQUIRED_TEMPLATE_IDS: frozenset[str] = frozenset({
    "knapsack_classic", "number_partitioning", "set_cover_simple",
    "max_cut_6node", "tsp_5cities", "graph_coloring_planar",
    "bin_packing_basic", "jss_3job_3machine", "nurse_rostering_small",
    "portfolio_3asset",
})

_REGISTRY: list[dict[str, Any]] | None = None
_REGISTRY_LOCK = threading.Lock()


def load_all_templates(force: bool = False) -> list[dict[str, Any]]:
    """Read every JSON file in the library directory, validate it
    against ``template_v1.json``, and return the list.

    Cached after the first call. Pass ``force=True`` to re-read (used in
    a few tests; not exposed via the routes).
    """
    global _REGISTRY
    with _REGISTRY_LOCK:
        if _REGISTRY is not None and not force:
            return list(_REGISTRY)

        with open(_SCHEMA_PATH) as f:
            schema = json.load(f)

        templates: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        for path in sorted(_LIBRARY_DIR.glob("*.json")):
            with open(path) as f:
                doc = json.load(f)
            try:
                jsonschema.validate(doc, schema)
            except jsonschema.ValidationError as e:
                # Loud failure at process startup is the entire point of
                # eager validation — operators must not ship a broken
                # template file silently.
                raise RuntimeError(
                    f"Template {path.name} failed schema validation at "
                    f"/{'/'.join(str(p) for p in e.absolute_path)}: {e.message}"
                ) from e

            if doc["id"] in seen_ids:
                raise RuntimeError(f"duplicate template id: {doc['id']}")
            seen_ids.add(doc["id"])
            templates.append(doc)

        # Sanity check: every spec-required template must be present.
        missing = REQUIRED_TEMPLATE_IDS - seen_ids
        if missing:
            logger.warning("template library missing required IDs: %s", missing)

        _REGISTRY = templates
        return list(templates)


def list_templates() -> list[dict[str, Any]]:
    """Return every template in the library."""
    return load_all_templates()


def get_template(template_id: str) -> dict[str, Any] | None:
    """Look up one template by ID. Returns ``None`` for unknown IDs."""
    for t in load_all_templates():
        if t["id"] == template_id:
            return t
    return None


def aggregate_categories() -> list[dict[str, Any]]:
    """Group templates by category and count them. Used by the gallery's
    filter-chip row in the frontend."""
    counts: dict[str, int] = defaultdict(int)
    for t in load_all_templates():
        counts[t["category"]] += 1
    return [
        {"name": name, "count": count}
        for name, count in sorted(counts.items())
    ]


def list_modules() -> dict[str, list[dict[str, Any]]]:
    """Group templates that carry a ``module`` field by ``module_id``,
    ordered by the per-lesson ``order`` value. Templates without a
    ``module`` are excluded — they appear in the gallery but not in
    the Modules-library view.
    """
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for t in load_all_templates():
        module = t.get("module")
        if not module:
            continue
        grouped[module["module_id"]].append(t)
    for lessons in grouped.values():
        lessons.sort(key=lambda lesson: lesson["module"]["order"])
    return dict(grouped)
