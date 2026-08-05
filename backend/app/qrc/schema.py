"""UI form schema for the QRC experiment-setup page (/qrc/new).

Single source of truth for *what a run accepts* stays in ``launcher`` (the
allow-list); this module layers **UI metadata** (labels, groups, defaults, help,
dataset roles, GPU flag) on top and emits a per-task field list the Vue form
renders directly. Because every field's flag/type/range is pulled from
``launcher._PARAM_SPEC``/``_CHOICE_SPEC``, a form built from this schema can never
submit a config the launcher would 400 — the two cannot drift.

Also surfaces the **readout catalogue** with each readout's feature count ``D``
(653-FID standard; observable ``3·n·V``; physics-informed reduced FID ``D_eff``
computed per system via ``spectral_lines``) so the form shows the physics of the
selection, per SOP §0/§3.2.
"""

from __future__ import annotations

from typing import Any

from app.qrc import launcher
from app.qrc.config import crotonic_acid_paper4, generic_nqubit
from app.qrc.spectral_lines import physics_informed_lines

# Field groups shown as sections in the form (order matters).
GROUPS = [
    {"key": "task", "label": "Task & dataset"},
    {"key": "reservoir", "label": "Reservoir"},
    {"key": "encoding", "label": "Encoding"},
    {"key": "readout", "label": "Readout"},
    {"key": "splits", "label": "Splits & training"},
    {"key": "rigor", "label": "Rigor controls"},
]

# Per-param UI metadata. type/min/max come from launcher specs; here we add the
# human label, its group, a default, help text, and (for the dataset dropdown)
# a role marker. Keyed by the launcher config key.
PARAM_META: dict[str, dict[str, Any]] = {
    # task / dataset
    "subtask": {"label": "Dataset", "group": "task", "role": "dataset", "default": "weather",
                "help": "Which dataset the reservoir is driven with."},
    "task_learn": {"label": "Dataset (task)", "group": "task", "role": "dataset", "default": "narma2",
                   "help": "NARMA-2 = encoding-sensitive; NARMA-10 = memory-bound."},
    "readout": {"label": "Readout", "group": "readout", "role": "readout", "default": "observable",
                "help": "observable = <sigma>xV; fid_reduced = physics-informed D_eff lines (SOP 3.2)."},
    "mc_experiment": {"label": "Experiment", "group": "task", "default": "quick",
                      "help": "Memory-capacity encoding sweep variant."},
    "experiment": {"label": "Experiment", "group": "task", "default": "2.1_quick",
                   "help": "Phase-2 encoding-sweep variant."},
    "select": {"label": "Feature selector", "group": "readout", "default": "mean"},
    "orders": {"label": "NARMA orders", "group": "task", "default": [2, 10], "list": True},
    "horizons": {"label": "Forecast horizons", "group": "task", "default": [1, 3, 7], "list": True},
    "splits": {"label": "Washout / train / test", "group": "splits", "default": [100, 400, 100], "list": True},
    # reservoir
    "n_spins": {"label": "Spins (qubits)", "group": "reservoir", "default": 6},
    "system_learn": {"label": "System", "group": "reservoir", "default": "6"},
    "tau": {"label": "tau (s) — free-evolution per step", "group": "reservoir", "default": 0.01},
    "n_virtual": {"label": "Virtual nodes V", "group": "reservoir", "default": 25},
    "coupling_scale": {"label": "Coupling scale (xJ)", "group": "reservoir", "default": 1.0,
                       "help": "Multiplies all J-couplings. 2.0 = strong coupling."},
    "fid_points": {"label": "FID points", "group": "readout", "default": 2048},
    "n_peaks": {"label": "Spectral peaks (D)", "group": "readout", "default": 653,
                "help": "Standard 653-FID readout keeps the 653 largest peaks."},
    # sweeps (memory)
    "Vs": {"label": "Virtual-node sweep (observable)", "group": "readout", "default": [1, 2, 5, 10], "list": True},
    "Ms": {"label": "FID-sample sweep (fid_reduced)", "group": "readout", "default": [64, 128, 256], "list": True},
    "kmax": {"label": "Max delay tau", "group": "splits", "default": 8},
    # splits / training
    "n_train": {"label": "Train size", "group": "splits", "default": 400},
    "n_test": {"label": "Test size", "group": "splits", "default": 100},
    "washout": {"label": "Washout", "group": "splits", "default": 100},
    "T": {"label": "Sequence length T", "group": "splits", "default": 300},
    "steps": {"label": "Adam steps", "group": "splits", "default": 100},
    "lr": {"label": "Learning rate", "group": "splits", "default": 0.02},
    "seed": {"label": "Seed", "group": "splits", "default": 42},
    "max_days": {"label": "Max days (weather)", "group": "task", "default": 1500},
    "key_horizon": {"label": "Key horizon", "group": "task", "default": 1},
    "conditions": {"label": "Conditions", "group": "encoding", "default": "arcsin,perspin",
                   "help": "Encoding conditions to compare (regime B, SOP 3.1)."},
    "device": {"label": "Device", "group": "reservoir", "default": "cuda"},
    # flags
    "no_esn": {"label": "Skip ESN baseline", "group": "rigor", "default": False},
    "no_memory": {"label": "tau->0 ablation (reset each step)", "group": "rigor", "default": False,
                  "help": "Disables cross-step quantum memory (the mandatory ablation, SOP 6.1)."},
    "correlation": {"label": "2-body correlation readout", "group": "readout", "default": False},
}

# Human titles / descriptions per task.
TASK_META: dict[str, dict[str, str]] = {
    "narma": {"label": "NARMA (Paper-4 reproduction)",
              "desc": "Reservoir NARMA benchmark on the 653-FID readout."},
    "weather": {"label": "Weather forecasting (Delhi)",
                "desc": "Horizon-h climate forecast; fidelity-fragile R2 (653-FID)."},
    "trace-gen": {"label": "Trace generation (evolve + cache FID)",
                  "desc": "Evolve the reservoir once and cache the raw FID .npz (SOP persist rule)."},
    "phase1": {"label": "Phase-1 feature selection",
               "desc": "Feature-reduction sweep over a cached trace."},
    "phase2": {"label": "Phase-2 encoding sweep",
               "desc": "Re-evolve per encoding; rank encodings."},
    "memcap": {"label": "Memory capacity (encoding sweep)",
               "desc": "Intrinsic MC scoring — fidelity-robust encoding ranking."},
    "memory": {"label": "Memory benchmark (STM / Parity-Check)",
               "desc": "Time-multiplexing STM + nonlinear PC capacity. Observable or "
                       "physics-informed reduced-FID readout. Forward-only, CPU."},
    "learnable": {"label": "Learnable encoding (regime B)",
                  "desc": "Train the input encoding by backprop through the reservoir vs "
                          "the arcsin baseline (leakage-free 3-way split). GPU."},
}


def _field(name: str) -> dict[str, Any] | None:
    """Build one field descriptor from the launcher specs + PARAM_META."""
    meta = PARAM_META.get(name, {"label": name, "group": "task"})
    if name in launcher._CHOICE_SPEC:
        _flag, opts = launcher._CHOICE_SPEC[name]
        return {"name": name, "kind": "choice", "options": list(opts),
                "label": meta["label"], "group": meta["group"],
                "default": meta.get("default", opts[0]),
                "help": meta.get("help"), "role": meta.get("role")}
    if name in launcher._FLAG_SPEC:
        return {"name": name, "kind": "flag", "label": meta["label"],
                "group": meta["group"], "default": bool(meta.get("default", False)),
                "help": meta.get("help")}
    if name in launcher._PARAM_SPEC:
        _flag, typ, lo, hi = launcher._PARAM_SPEC[name]
        return {"name": name, "kind": "list" if meta.get("list") else "number",
                "ntype": "float" if typ is float else "int",
                "min": lo, "max": hi, "label": meta["label"], "group": meta["group"],
                "default": meta.get("default", lo), "help": meta.get("help"),
                "role": meta.get("role")}
    return None


def _task_fields(task: str) -> list[dict[str, Any]]:
    spec = launcher._TASKS[task]
    order = (list(spec.get("choices", ())) + list(spec["numeric"])
             + list(spec["lists"]) + list(spec["flags"])
             + (["trace"] if spec.get("trace_required") else []))
    fields = [_field(n) for n in order]
    return [f for f in fields if f]


def _readout_catalogue() -> dict[str, Any]:
    """Feature count D per readout, incl. physics-informed D_eff per system."""
    systems = {
        "crotonic9_paper4": (crotonic_acid_paper4(), tuple(range(4, 9))),
        "6": (generic_nqubit(6, seed=4), None),
        "3": (generic_nqubit(3, seed=4), None),
    }
    fid_reduced = {}
    for key, (sysc, ro) in systems.items():
        ls = physics_informed_lines(sysc, ro)
        fid_reduced[key] = {"D_eff": ls.d_eff, "n_raw": ls.n_raw,
                            "linewidth_hz": round(ls.linewidth_hz, 3),
                            "f_max_hz": round(ls.f_max_hz, 1),
                            "D_features": 2 * ls.d_eff,
                            "suggest_M": ls.suggest_samples(),
                            "suggest_dwell_ms": round(ls.suggest_dwell() * 1e3, 4)}
    return {
        "fid653": {"label": "653-FID spectrum (standard)", "D": 653,
                   "note": "SOP §0 standard readout — report on this unless labeled otherwise."},
        "observable": {"label": "Observable <sigma>xV", "D_formula": "3 * n_spins * V",
                       "note": "Differentiable-path readout; label D."},
        "fid_reduced": {"label": "Physics-informed reduced FID (D_eff)", "per_system": fid_reduced,
                        "note": "SOP §3.2 — analytic single-quantum lines merged within 1/(pi T2)."},
    }


def build_schema() -> dict[str, Any]:
    """The full form schema consumed by GET /api/qrc/schema."""
    tasks = []
    for key in launcher._TASKS:
        meta = TASK_META.get(key, {"label": key, "desc": ""})
        fields = _task_fields(key)
        dataset_field = next((f["name"] for f in fields if f.get("role") == "dataset"), None)
        tasks.append({
            "key": key, "label": meta["label"], "desc": meta["desc"],
            "gpu": key in launcher.GPU_TASKS,
            "dataset_field": dataset_field,
            "fields": fields,
        })
    return {
        "groups": GROUPS,
        "tasks": tasks,
        "readouts": _readout_catalogue(),
        "single_active_job": True,
    }
