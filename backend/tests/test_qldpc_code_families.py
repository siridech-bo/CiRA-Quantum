"""Sprint 0 registry shape tests.

These keep the static registry well-formed so the frontend gallery
can rely on the same field set across every card. Sprint 1 will swap
the canned ``n, k, d`` for ``qldpc``-computed values — at that point
these tests become regressions guards for the schema.
"""

from __future__ import annotations

from app.qldpc import get_code_family, list_code_families


REQUIRED_FIELDS = {
    "id",
    "title",
    "category",
    "regime",
    "n",
    "k",
    "d",
    "summary",
    "discovered_by",
    "key_property",
    "use_case",
    "best_known_threshold_pct",
}

ALLOWED_CATEGORIES = {"css_classical", "css_product", "topological"}
ALLOWED_REGIMES = {"zero-rate", "finite-rate"}


def test_list_code_families_returns_at_least_four():
    families = list_code_families()
    assert len(families) >= 4
    ids = {f["id"] for f in families}
    # Sprint 0 ships exactly these four.
    assert {"bicycle", "surface", "hypergraph_product", "toric"} <= ids


def test_every_family_has_required_fields():
    for fam in list_code_families():
        missing = REQUIRED_FIELDS - set(fam.keys())
        assert not missing, f"{fam['id']} missing fields: {missing}"


def test_field_value_constraints():
    for fam in list_code_families():
        assert fam["category"] in ALLOWED_CATEGORIES, fam["id"]
        assert fam["regime"] in ALLOWED_REGIMES, fam["id"]
        # The math: distance can't exceed n, k must be < n, both positive.
        assert fam["n"] > 0, fam["id"]
        assert 0 < fam["k"] < fam["n"], fam["id"]
        assert 0 < fam["d"] <= fam["n"], fam["id"]
        # Thresholds are percentages, ~0.5%–1.5% for known qLDPC codes.
        assert 0 < fam["best_known_threshold_pct"] < 5, fam["id"]


def test_get_code_family_known_id():
    fam = get_code_family("surface")
    assert fam is not None
    assert fam["id"] == "surface"
    assert fam["category"] == "topological"


def test_get_code_family_unknown_id_returns_none():
    assert get_code_family("does-not-exist") is None


def test_list_returns_copies_not_references():
    """Callers mutating the returned dicts mustn't corrupt the registry."""
    fams_a = list_code_families()
    fams_a[0]["title"] = "MUTATED"
    fams_b = list_code_families()
    assert fams_b[0]["title"] != "MUTATED"
