"""Run comparison — assemble N runs (+ references) into a side-by-side view.

Loads each run's registry entry + results JSON, extracts its
``(task, readout, D, regime)`` badge and a flat metric bundle, diffs the configs,
and applies the **comparability guard**: results are only comparable within the
same task *and* the same readout/``D`` (SOP §0 — absolute NMSE/capacity is
readout-dependent, ``MC ≤ D``). Cross-readout or cross-task selections are
surfaced as warnings, not silently mixed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.qrc import launcher, references


def _load_results(entry: dict[str, Any]) -> dict[str, Any]:
    rp = entry.get("results_path")
    if rp and Path(rp).exists():
        try:
            return json.loads(Path(rp).read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _badge(entry: dict[str, Any], results: dict[str, Any]) -> dict[str, Any]:
    """The (task, readout, D, regime) identity used for the comparability guard."""
    task = entry.get("task")
    readout = results.get("readout")
    D = results.get("D_eff") or results.get("feature_count")
    regime = results.get("regime")
    if readout is None:
        # infer from task defaults
        readout = {"narma": "fid653", "weather": "fid653", "trace-gen": "fid653"}.get(task)
    return {"task": task, "readout": readout, "D": D, "regime": regime}


def _metrics(entry: dict[str, Any], results: dict[str, Any]) -> dict[str, float]:
    """Flat scalar-metric bundle, robust across task result shapes."""
    out: dict[str, float] = {}
    # top-level scalar floats/ints (skip obvious config echoes)
    skip = {"T", "washout", "n_train", "seed", "coupling_scale", "D_eff", "feature_count"}
    for k, v in results.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool) and k not in skip:
            out[k] = float(v)
    # learnable: per-condition test NMSE
    for cond, rec in (results.get("conditions") or {}).items():
        if isinstance(rec, dict) and "test_nmse" in rec:
            out[f"{cond}_test_nmse"] = float(rec["test_nmse"])
    # memory: best totMC / totCap across the sweep
    mem = results.get("results")
    if isinstance(mem, dict) and mem:
        stm = [r.get("stm_totMC") for r in mem.values() if isinstance(r, dict)]
        pc = [r.get("pc_totCap") for r in mem.values() if isinstance(r, dict)]
        if any(x is not None for x in stm):
            out["best_STM_totMC"] = float(max(x for x in stm if x is not None))
        if any(x is not None for x in pc):
            out["best_PC_totCap"] = float(max(x for x in pc if x is not None))
    return out


def compare(run_ids: list[str], reference_ids: list[str] | None = None) -> dict[str, Any]:
    reference_ids = reference_ids or []
    items: list[dict[str, Any]] = []

    for rid in run_ids:
        entry = launcher.get_run(rid)
        if entry is None:
            items.append({"id": rid, "kind": "run", "missing": True})
            continue
        results = _load_results(entry)
        badge = _badge(entry, results)
        items.append({
            "id": rid, "kind": "run", "label": rid, "status": entry.get("status"),
            "config": entry.get("config") or {}, **badge,
            "metrics": _metrics(entry, results),
        })

    for ref_id in reference_ids:
        ref = references.get_reference(ref_id)
        if ref is None:
            continue
        items.append({
            "id": ref_id, "kind": "reference", "label": ref.get("label"),
            "task": ref.get("task"), "readout": ref.get("readout"),
            "D": ref.get("D"), "regime": ref.get("regime"),
            "config": {}, "metrics": {k: v for k, v in (ref.get("metrics") or {}).items()
                                      if isinstance(v, (int, float)) and not isinstance(v, bool)},
            "note": (ref.get("metrics") or {}).get("note"),
        })

    # config diff: only keys whose value differs across the runs (references have none)
    run_items = [it for it in items if it["kind"] == "run" and not it.get("missing")]
    all_keys: set[str] = set()
    for it in run_items:
        all_keys |= set(it["config"].keys())
    diff: dict[str, dict[str, Any]] = {}
    for key in sorted(all_keys):
        vals = {it["id"]: it["config"].get(key) for it in run_items}
        if len({json.dumps(v, default=str, sort_keys=True) for v in vals.values()}) > 1:
            diff[key] = vals

    # comparability guard (over the runs; references marked task "*" are universal)
    warnings: list[str] = []
    comparable = True
    cmp_items = [it for it in items if not it.get("missing")
                 and it.get("task") not in (None, "*")]
    tasks = {it["task"] for it in cmp_items}
    if len(tasks) > 1:
        comparable = False
        warnings.append(f"Different tasks {sorted(tasks)} — metrics are not comparable.")
    readouts = {it.get("readout") for it in cmp_items if it.get("readout")}
    if len(readouts) > 1:
        comparable = False
        warnings.append(f"Different readouts {sorted(readouts)} — absolute NMSE/capacity is "
                        "readout-dependent (MC ≤ D); compare within one readout (SOP §0).")
    ds = {it.get("D") for it in cmp_items if it.get("D")}
    if len(ds) > 1:
        warnings.append(f"Different feature counts D {sorted(ds)} — capacities scale with D.")

    # union of metric names, for the table columns
    metric_keys: list[str] = []
    for it in items:
        for k in it.get("metrics", {}):
            if k not in metric_keys:
                metric_keys.append(k)

    return {
        "items": items,
        "config_diff": diff,
        "metric_keys": metric_keys,
        "comparable": comparable,
        "warnings": warnings,
    }
