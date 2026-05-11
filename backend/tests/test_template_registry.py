"""Phase 5B — template registry tests."""

from __future__ import annotations

from app.templates.registry import (
    REQUIRED_TEMPLATE_IDS,
    aggregate_categories,
    get_template,
    list_modules,
    list_templates,
    load_all_templates,
)


def test_loads_all_template_files_at_startup():
    templates = list_templates()
    assert len(templates) >= 10
    ids = {t["id"] for t in templates}
    assert REQUIRED_TEMPLATE_IDS.issubset(ids)


def test_validates_each_template_against_schema(tmp_path):
    """Loading a fresh registry from disk must not raise. The loader
    runs schema validation eagerly so a malformed template fails the
    process startup, not the first request."""
    fresh = load_all_templates()
    assert len(fresh) >= 10


def test_unique_ids():
    ids = [t["id"] for t in list_templates()]
    assert len(ids) == len(set(ids)), f"duplicate template ids in registry: {ids}"


def test_required_templates_present():
    """The 10 ids the spec calls out must each exist."""
    ids = {t["id"] for t in list_templates()}
    for required in REQUIRED_TEMPLATE_IDS:
        assert required in ids, f"missing required template: {required}"


def test_get_by_id_returns_full_object():
    t = get_template("knapsack_classic")
    assert t is not None
    assert t["id"] == "knapsack_classic"
    # Full object includes the fields needed by the detail modal.
    for field in (
        "title", "category", "difficulty", "summary", "problem_statement",
        "real_world_example", "expected_pattern", "expected_optimum",
        "expected_solution_summary", "learning_notes",
    ):
        assert field in t, f"missing field {field!r} in {t['id']}"


def test_get_by_id_unknown_returns_none():
    assert get_template("does_not_exist") is None


def test_categories_aggregated_correctly():
    cats = aggregate_categories()
    # Every category must have at least one template.
    counts = {c["name"]: c["count"] for c in cats}
    for category in ("allocation", "scheduling", "routing", "graph", "finance", "logic"):
        assert counts.get(category, 0) >= 1, f"category {category!r} has zero templates"


# ---- v2 additions ----------------------------------------------------------


def test_at_least_five_templates_have_module_metadata():
    """v2 DoD: ≥ 5 of the 10 required templates carry a `module` field,
    forming a coherent introductory module."""
    templates_with_module = [t for t in list_templates() if t.get("module")]
    assert len(templates_with_module) >= 5


def test_modules_grouping_is_coherent():
    """The module view groups templates by module_id and orders them by
    the per-lesson ``order`` field — the prerequisite chain must be
    consistent (every prerequisite id must be a real lesson_id in the
    same module)."""
    modules = list_modules()
    assert "qubo_foundations" in modules
    lessons = modules["qubo_foundations"]
    assert len(lessons) >= 5
    # Lessons in the module are sorted by `order`.
    orders = [lesson["module"]["order"] for lesson in lessons]
    assert orders == sorted(orders)
    # Every prerequisite name resolves to a real lesson template id.
    all_ids = {t["id"] for t in list_templates()}
    for lesson in lessons:
        for prereq in lesson["module"].get("prerequisites", []):
            assert prereq in all_ids, (
                f"prerequisite {prereq!r} in lesson {lesson['id']!r} "
                f"is not a known template id"
            )


def test_v1_only_templates_still_load():
    """The schema's `module` field is optional — templates without it
    must load without error and appear in the gallery list."""
    no_module = [t for t in list_templates() if "module" not in t]
    assert len(no_module) >= 1
