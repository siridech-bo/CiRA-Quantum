"""Curated problem-template library (Phase 5B).

Public surface::

    list_templates()        — full list (summary fields enough for the gallery)
    get_template(id)        — full template object, or None
    aggregate_categories()  — [{"name", "count"}, ...] for filter chips
    list_modules()          — {module_id: [lesson_template, ...]} for the
                              Modules-library view (v2 addition)
    REQUIRED_TEMPLATE_IDS   — the 10 IDs the v2 spec pins as ship-required
"""

from __future__ import annotations

from app.templates.registry import (
    REQUIRED_TEMPLATE_IDS,
    aggregate_categories,
    get_template,
    list_modules,
    list_templates,
    load_all_templates,
)

__all__ = [
    "REQUIRED_TEMPLATE_IDS",
    "aggregate_categories",
    "get_template",
    "list_modules",
    "list_templates",
    "load_all_templates",
]
