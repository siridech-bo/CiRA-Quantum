"""Instance suite registry — read access to the manifest of Benchmark
instances and their suites.

The manifest lives at ``app/benchmarking/instances/manifest.json``;
each entry follows the ``instance_v1`` schema. Paths inside it are
relative to the backend directory so a contributor can drop a new
instance file alongside the existing ones and add a single manifest
entry pointing to it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_BACKEND_ROOT = Path(__file__).resolve().parents[2]
_MANIFEST_PATH = Path(__file__).parent / "instances" / "manifest.json"


@dataclass(frozen=True)
class InstanceMetadata:
    instance_id: str
    suite: str
    problem_class: str
    cqm_path: Path
    expected_optimum: float | None
    expected_optimum_kind: str
    tags: tuple[str, ...]
    source: str

    def load_cqm_json(self) -> dict:
        """Return the cqm_v1 JSON for this instance as a plain dict."""
        with open(self.cqm_path) as f:
            return json.load(f)


def _load_manifest() -> dict[str, Any]:
    with open(_MANIFEST_PATH) as f:
        return json.load(f)


def _to_metadata(entry: dict) -> InstanceMetadata:
    return InstanceMetadata(
        instance_id=entry["instance_id"],
        suite=entry["suite"],
        problem_class=entry["problem_class"],
        cqm_path=_BACKEND_ROOT / entry["cqm_path"],
        expected_optimum=entry.get("expected_optimum"),
        expected_optimum_kind=entry.get("expected_optimum_kind", "unknown"),
        tags=tuple(entry.get("tags", [])),
        source=entry.get("source", "unknown"),
    )


def list_suites() -> list[str]:
    return sorted(_load_manifest()["suites"].keys())


def get_suite(suite_id: str) -> list[InstanceMetadata]:
    """Return all instances belonging to ``suite_id``."""
    suites = _load_manifest()["suites"]
    if suite_id not in suites:
        raise KeyError(f"Unknown suite: {suite_id!r}. Known: {sorted(suites.keys())}")
    return [_to_metadata(entry) for entry in suites[suite_id]]


def get_instance(instance_id: str) -> InstanceMetadata:
    """Look up a single instance by ID across all suites."""
    suites = _load_manifest()["suites"]
    for entries in suites.values():
        for entry in entries:
            if entry["instance_id"] == instance_id:
                return _to_metadata(entry)
    raise KeyError(f"Unknown instance: {instance_id!r}")
