"""qLDPC code family registry.

Each entry is a small, self-describing record. The blueprint exposes
them as JSON so the frontend Code Family Gallery can render cards
without the backend importing any of the heavy math libs by default —
the parity-check matrices and live distance verification land via
``generators.py`` + ``serializers.py`` (Sprint 1) once the ``qldpc``
PyPI package is installed and the ``/api/qldpc/health`` capability
flag flips to True.

Four families ship, chosen because they're the most cited in the
qLDPC research literature and span the design space the pitched
customers (QI theorists, QEC architects) work in day-to-day:

* Bicycle (BB)        — Bravyi et al. (2024); IBM "gross code" ⟦144, 12, 12⟧
* Surface             — Kitaev (2003); rotated d=13 patch     ⟦169, 1, 13⟧
* Hypergraph product  — Tillich & Zémor (2014); HGP of Hamming(7,4) ⟦58, 16, 3⟧
* Toric               — Kitaev (1997); unrotated 10×10        ⟦200, 2, 10⟧

The ``n, k, d`` values here are the canned fallback used when the
``qldpc`` lib isn't installed. When it is, the routes overlay
``list_code_families_live()`` which calls into ``generators.py`` to
compute exact ``n, k`` and an upper bound on ``d`` per family. The
canned values are kept aligned with the live computation so the
gallery looks consistent across both states.
"""

from __future__ import annotations

from typing import Any

# Category tags drive the gallery's filter chips. "topological" carries
# 2D-geometry-baked-in codes; "css_product" carries codes built by
# combining classical LDPC codes via algebraic constructions; "css_classical"
# carries codes built directly from a single classical LDPC code.
_CODE_FAMILIES: list[dict[str, Any]] = [
    {
        "id": "bicycle",
        "title": "Bicycle code (bivariate)",
        "category": "css_classical",
        "regime": "finite-rate",
        # Sprint 1 live: IBM "gross code" — Bravyi, Cross, Gambetta,
        # Maslov, Rall & Yoder, Nature 2024. ⟦144, 12, 12⟧.
        "n": 144,
        "k": 12,
        "d": 12,
        "params": {"orders": {"x": 12, "y": 6}, "poly_a": "x^3 + y + y^2", "poly_b": "y^3 + x + x^2"},
        "summary": (
            "Bivariate bicycle (BB) code with the parameters from "
            "Bravyi et al. (2024) — the now-famous 'gross code'. 12 "
            "logical qubits on 144 physical with distance 12: roughly "
            "10× the encoding rate of an equivalent surface code. "
            "Non-local stabilizers — a worked example of why qLDPC "
            "needs Phase-2 routing analysis to land on real hardware."
        ),
        "discovered_by": "Bravyi, Cross, Gambetta, Maslov, Rall & Yoder (Nature 2024)",
        "key_property": "Constant rate k/n at fixed Tanner-graph degree",
        "use_case": "Memory storage at 10× lower overhead than surface codes",
        "best_known_threshold_pct": 0.7,
    },
    {
        "id": "surface",
        "title": "Surface code",
        "category": "topological",
        "regime": "zero-rate",
        # Rotated surface(13): ⟦169, 1, 13⟧.
        "n": 169,
        "k": 1,
        "d": 13,
        "params": {"rows": 13, "rotated": True},
        "summary": (
            "The most-studied topological code: data qubits live on "
            "the edges of a 2D lattice with open boundaries, with X- "
            "and Z-stabilizers on the plaquettes and vertices. "
            "Strictly local weight-4 checks make it the easiest qLDPC "
            "code to fabricate on a planar superconducting chip — at "
            "the cost of zero rate (one logical qubit per patch)."
        ),
        "discovered_by": "Kitaev (2003); Bravyi & Kitaev (1998)",
        "key_property": "Geometrically local weight-4 stabilizers on a 2D grid",
        "use_case": "Near-term fault tolerance on superconducting hardware",
        "best_known_threshold_pct": 1.0,
    },
    {
        "id": "hypergraph_product",
        "title": "Hypergraph product code",
        "category": "css_product",
        "regime": "finite-rate",
        # HGP of classical Hamming(7,4): 49 + 9 = 58 physical, 16 logical.
        # Distance is modest at small sizes; the family showcases the
        # algebraic construction, not a fault-tolerance candidate here.
        "n": 58,
        "k": 16,
        "d": 3,
        "params": {"base_code": "Hamming(7,4)"},
        "summary": (
            "Tillich-Zémor's algebraic construction: take two classical "
            "LDPC codes and combine them into a quantum code whose "
            "parameters track the classical ones. This instance is "
            "HGP applied to the (7,4) Hamming code — small enough to "
            "fit on screen but big enough (k=16) to show why HGP "
            "unlocked finite-rate qLDPC."
        ),
        "discovered_by": "Tillich & Zémor (2014)",
        "key_property": "Distance d scales as Θ(√n) with k = Θ(n)",
        "use_case": "Reducing qubit overhead vs surface codes at larger sizes",
        "best_known_threshold_pct": 0.7,
    },
    {
        "id": "toric",
        "title": "Toric code",
        "category": "topological",
        "regime": "zero-rate",
        # Unrotated 10×10: n = 2L² = 200, k = 2, d = 10.
        "n": 200,
        "k": 2,
        "d": 10,
        "params": {"rows": 10, "cols": 10, "rotated": False},
        "summary": (
            "Kitaev's original topological code: the surface code on a "
            "torus (periodic boundaries) instead of a square. Two "
            "logical qubits per patch instead of one, and the canonical "
            "pedagogical example for stabilizer formalism. Hard to "
            "fabricate (you need a topological torus), so it's mostly "
            "a theoretical reference point against which the surface "
            "code is measured."
        ),
        "discovered_by": "Kitaev (1997)",
        "key_property": "Two logical qubits from the torus's two non-contractible loops",
        "use_case": "Theory benchmark; reference for surface-code comparisons",
        "best_known_threshold_pct": 1.1,
    },
]

_BY_ID = {c["id"]: c for c in _CODE_FAMILIES}

# Cache live metadata per family_id so the gallery is fast across page
# loads. Cleared by ``invalidate_live_cache()`` if a generator changes.
_LIVE_CACHE: dict[str, dict[str, Any]] = {}


def list_code_families() -> list[dict[str, Any]]:
    """All registered code families with canned metadata, in display order.

    This is the always-available view: independent of whether the
    ``qldpc`` PyPI lib is installed. Use ``list_code_families_live()``
    to overlay live ``n, k, d`` values.
    """
    return [dict(c) for c in _CODE_FAMILIES]


def get_code_family(family_id: str) -> dict[str, Any] | None:
    """Canned metadata for one family, or ``None`` if unknown."""
    c = _BY_ID.get(family_id)
    return dict(c) if c else None


def list_code_families_live() -> list[dict[str, Any]]:
    """Code families with live ``n, k, d`` overlaid when ``qldpc`` is installed.

    Falls back to canned metadata on any per-family error so a partial
    failure in one factory doesn't blank the whole gallery.
    """
    return [_overlay_live(dict(c)) for c in _CODE_FAMILIES]


def get_code_family_live(family_id: str) -> dict[str, Any] | None:
    """Single-family equivalent of ``list_code_families_live()``."""
    c = _BY_ID.get(family_id)
    if c is None:
        return None
    return _overlay_live(dict(c))


def _overlay_live(family_dict: dict[str, Any]) -> dict[str, Any]:
    family_id = family_dict["id"]
    live = _compute_live_metadata(family_id)
    if live is None:
        family_dict["live"] = False
        return family_dict
    family_dict.update(live)
    family_dict["live"] = True
    return family_dict


def _compute_live_metadata(family_id: str) -> dict[str, Any] | None:
    """Best-effort live ``n, k, d`` computation. Cached per family."""
    if family_id in _LIVE_CACHE:
        return _LIVE_CACHE[family_id]
    try:
        from app.qldpc.generators import get_factory
        from app.qldpc.serializers import compute_distance
        factory = get_factory(family_id)
        if factory is None:
            return None
        code = factory()
        dist = compute_distance(code, exact=False)
        live = {
            "n": int(code.num_qudits),
            "k": int(code.dimension),
            "d": dist["distance"],
            "d_mode": dist["mode"],
        }
        _LIVE_CACHE[family_id] = live
        return live
    except Exception:
        # qldpc not installed or factory blew up; route falls back to canned.
        return None


def invalidate_live_cache() -> None:
    """Clear the live-metadata cache. Tests + Sprint 1+ hot-swaps use this."""
    _LIVE_CACHE.clear()
