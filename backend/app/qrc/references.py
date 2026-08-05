"""Named reference points (benchmarks & tiers) a run is compared against.

A run isn't judged in isolation: the setup page picks a **benchmark** (a pinned
reference result — the paper's numbers, the arcsin baseline, a classical control)
or a **tier** (a named target level — readout tier / training regime, SOP §0/§3.1)
to score against, and the compare page shows runs side by side with these.

(Distinct from ``app.qrc.benchmarks``, which *runs* experiments; this module only
stores reference metadata.) Storage is a small JSON file
(``artifacts/qrc_runs/references.json``); it is seeded with the built-ins on first
read so the selector is never empty, and user-saved references are appended
(auth-gated route). Comparability metadata (``task``/``readout``/``D``) travels
with each entry so the compare view can flag apples-to-oranges (different D or
task — SOP §0).
"""

from __future__ import annotations

import json
from typing import Any

from app.qrc.launcher import RUNS_ROOT

_PATH = RUNS_ROOT / "references.json"

# Built-in references. Metrics are labeled with the readout + D they were measured
# on so the compare guard can enforce same-readout/same-task comparisons.
_SEED: list[dict[str, Any]] = [
    {
        "id": "paper-djc-stm-pc", "kind": "benchmark",
        "label": "Das-Giorgi-Zambrini JC (STM/PC)",
        "task": "memory", "readout": "moments", "D": None, "regime": "A (readout-only)",
        "metrics": {"note": "qubit-boson reservoir; PC rises with V, STM flat"},
        "source": "PRR 2026 (arXiv JC-QRC)", "builtin": True,
    },
    {
        "id": "arcsin-baseline", "kind": "tier",
        "label": "arcsin(sqrt) baseline (regime A)",
        "task": "learnable", "readout": "observable", "D": None,
        "regime": "A (fixed encoding + ridge readout)",
        "metrics": {"note": "the honest baseline any learned encoding must beat"},
        "source": "SOP §3.1", "builtin": True,
    },
    {
        "id": "tier-653-fid", "kind": "tier",
        "label": "653-FID standard readout",
        "task": "*", "readout": "fid653", "D": 653, "regime": "A",
        "metrics": {"note": "the standard readout (SOP §0) — production comparisons"},
        "source": "SOP §0", "builtin": True,
    },
    {
        "id": "tier-fid-reduced", "kind": "tier",
        "label": "Physics-informed reduced FID (D_eff)",
        "task": "memory", "readout": "fid_reduced", "D": None, "regime": "A/B",
        "metrics": {"crotonic9_D_eff": 100, "six_spin_D_eff": 137},
        "source": "SOP §3.2", "builtin": True,
    },
    {
        "id": "classical-ridge-control", "kind": "benchmark",
        "label": "Classical ridge control (window)",
        "task": "*", "readout": "classical", "D": None, "regime": "control",
        "metrics": {"note": "linear+quadratic ridge on the raw window — the τ→0 control"},
        "source": "SOP §6.2", "builtin": True,
    },
]


def _load_raw() -> list[dict[str, Any]]:
    if not _PATH.exists():
        return []
    try:
        data = json.loads(_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    return data if isinstance(data, list) else []


def list_references() -> list[dict[str, Any]]:
    """Seeded built-ins + any user-saved references (built-ins first)."""
    saved = [b for b in _load_raw() if not b.get("builtin")]
    return _SEED + saved


def add_reference(entry: dict[str, Any]) -> dict[str, Any]:
    """Append a user reference. Requires ``id`` + ``label``; ``builtin`` forced
    False. Overwrites a prior user entry with the same id."""
    if not isinstance(entry, dict) or not entry.get("id") or not entry.get("label"):
        raise ValueError("reference needs 'id' and 'label'")
    entry = {**entry, "builtin": False}
    saved = [b for b in _load_raw() if not b.get("builtin") and b.get("id") != entry["id"]]
    saved.append(entry)
    _PATH.parent.mkdir(parents=True, exist_ok=True)
    _PATH.write_text(json.dumps(saved, indent=2, default=str), encoding="utf-8")
    return entry


def get_reference(rid: str) -> dict[str, Any] | None:
    return next((b for b in list_references() if b.get("id") == rid), None)
