"""qLDPC (Quantum Low-Density Parity-Check codes) sister app — package root.

This package houses the qLDPC scaffolding that ships in Sprint 0 (foundation)
and grows through Sprint 1 (Phase 1 math) → Sprint 4 (Phase 3 hardware
execution). It deliberately mirrors the QML side's layout — a
``code_families`` registry plays the same role here that ``datasets``
plays in the QML app.

Sprint 0 shipped the code-family registry + blueprint shell with
three public endpoints. Sprint 1 added live matrix generation via the
``qldpc`` PyPI lib, CSS commutativity verification, Tanner-graph JSON
export, and distance computation (with a fast upper-bound default
and an opt-in exact path).
"""

from app.qldpc.code_families import (
    get_code_family,
    get_code_family_live,
    list_code_families,
    list_code_families_live,
)

__all__ = [
    "get_code_family",
    "get_code_family_live",
    "list_code_families",
    "list_code_families_live",
]
