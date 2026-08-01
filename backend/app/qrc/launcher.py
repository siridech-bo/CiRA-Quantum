"""Managed-subprocess launcher + JSON run registry for QRC Stage A.

The QRC control API (``app/routes/qrc.py``, Coder C) needs to kick off the
long-running reproduction / phase-1 / trace-generation runners as *managed
subprocesses* and track them across HTTP requests. This module owns that
machinery so the route layer stays thin:

* a JSON registry under ``artifacts/qrc_runs/registry.json`` mapping
  ``id -> {task, run_dir, pid, status, config, ...}`` (§3 contract);
* per-run directories (``artifacts/qrc_runs/<id>/``) holding the runner's
  ``events.jsonl`` live log + ``results.json`` output;
* a **config allow-list** with numeric range checks so a request can never
  smuggle arbitrary flags — every argv is built as a list (``shell=False``);
* a **single active heavy job** guard (the route returns 409 when one is
  already running);
* start / stop / status-refresh helpers keyed on the recorded PID.

Importable without a GPU or any heavy scientific dep — it only ever *spawns*
subprocesses; it never imports the physics stack itself. The subprocess is
what needs CUDA / QuTiP.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.config import BASE_DIR

# ---- Layout ---------------------------------------------------------------

# Both roots honour env overrides so tests can point them at a tmp dir
# without touching the real ``artifacts/`` tree.
RUNS_ROOT = Path(os.environ.get("QRC_RUNS_DIR") or (BASE_DIR / "artifacts" / "qrc_runs"))
TRACES_ROOT = Path(os.environ.get("QRC_TRACES_DIR") or (BASE_DIR / "artifacts" / "traces"))

# Serialize registry read-modify-write and launch decisions within the
# process. Cross-process safety isn't required for Stage A (single Flask
# process, single heavy job at a time).
_LOCK = threading.RLock()

# In-process handles to spawned subprocesses, keyed by run id. Lets us
# ``poll()`` for liveness / exit code and ``terminate()`` cleanly. A run
# recorded in the registry but absent here (e.g. after a server restart)
# falls back to a best-effort PID liveness probe.
_PROCS: dict[str, subprocess.Popen] = {}

# Statuses that mean "this run still holds the single-job lock".
_ACTIVE_STATUSES = frozenset({"running"})


class ConfigError(ValueError):
    """Raised when a launch config violates the allow-list / range checks."""


# ---- Task → runner mapping + config allow-list ----------------------------

# task -> (script path relative to backend, fixed leading args, allowed params).
# Only ``narma``/``weather`` runners exist today (Coder A's
# ``qrc_reproduce_paper4.py``); ``phase1``/``trace-gen`` are being built by
# Coders A/B, so we code to their expected ``--run-dir`` / ``--out`` CLI and
# guard on the script's presence at launch time.
_TASKS: dict[str, dict[str, Any]] = {
    "narma": {
        "script": "scripts/qrc_reproduce_paper4.py",
        "fixed": ["--task", "narma"],
        "numeric": ("tau", "n_peaks", "fid_points", "n_train", "n_test", "washout", "seed"),
        "lists": ("orders",),
        "flags": ("no_esn",),
    },
    "weather": {
        "script": "scripts/qrc_reproduce_paper4.py",
        "fixed": ["--task", "weather"],
        "numeric": (
            "tau", "n_peaks", "fid_points", "n_train", "n_test",
            "washout", "seed", "max_days",
        ),
        "lists": ("horizons",),
        "flags": ("no_esn",),
    },
    # phase1 operates on an ALREADY-CACHED trace (not a fresh reservoir run),
    # so its CLI is ``--trace <npz> --out-dir <dir>`` (no --run-dir/--out).
    # ``trace`` is a sanitised name resolved under TRACES_ROOT.
    "phase1": {
        "script": "scripts/qrc_phase1.py",
        "fixed": [],
        "numeric": ("n_peaks", "key_horizon"),
        "lists": (),
        "flags": (),
        "choices": ("select",),
        "output": "out_dir",
        "trace_required": True,
    },
    # trace-gen maps to Coder A's ``qrc_gen_traces.py``, whose CLI takes
    # ``--task weather|narma`` (via the ``subtask`` choice) and a single
    # ``--splits WASHOUT N_TRAIN N_TEST`` triple (not separate flags). The
    # produced ``.npz`` path is registered at launch so ``/fid`` resolves it.
    "trace-gen": {
        "script": "scripts/qrc_gen_traces.py",
        "fixed": [],
        "numeric": ("seed", "fid_points", "n_peaks", "tau", "n_virtual"),
        "lists": ("splits", "horizons", "orders"),
        "flags": (),
        "choices": ("subtask",),
    },
    # phase2 re-evolves the reservoir per encoding setting. Its CLI takes
    # ``--experiment``/``--fidelity`` (choices) and writes results + live
    # progress into ``--run-dir`` (so the UI shows a moving bar per encoding).
    "phase2": {
        "script": "scripts/qrc_phase2.py",
        "fixed": [],
        "numeric": ("seed",),
        "lists": (),
        "flags": (),
        "choices": ("experiment", "fidelity"),
        "output": "run_dir",
    },
    # memcap: intrinsic memory-capacity encoding sweep (random-input reservoir
    # passes + MC scoring; saves waveforms). Fidelity-robust metric for ranking
    # encodings where weather-R² is too fidelity-hungry.
    "memcap": {
        "script": "scripts/qrc_memcap.py",
        "fixed": [],
        "numeric": ("seed", "kmax"),
        "lists": (),
        "flags": (),
        "choices": ("mc_experiment", "fidelity"),
        "output": "run_dir",
    },
}

# name -> (cli flag, python type, min, max). Applies to both scalar
# ``numeric`` params and per-element ``lists`` params.
_PARAM_SPEC: dict[str, tuple[str, type, float, float]] = {
    "tau": ("--tau", float, 1e-4, 10.0),
    "n_peaks": ("--n-peaks", int, 1, 8192),
    "fid_points": ("--fid-points", int, 16, 65536),
    "n_train": ("--n-train", int, 1, 1_000_000),
    "n_test": ("--n-test", int, 1, 1_000_000),
    "washout": ("--washout", int, 0, 1_000_000),
    "seed": ("--seed", int, 0, 2**31 - 1),
    "max_days": ("--max-days", int, 1, 1_000_000),
    "orders": ("--orders", int, 1, 100),
    "horizons": ("--horizons", int, 1, 100_000),
    "n_virtual": ("--n-virtual", int, 1, 4096),
    "splits": ("--splits", int, 0, 1_000_000),
    "key_horizon": ("--key-horizon", int, 1, 100_000),
    "kmax": ("--kmax", int, 1, 500),
}

_FLAG_SPEC: dict[str, str] = {
    "no_esn": "--no-esn",
}

# name -> (cli flag, allowed string values). String params can never smuggle
# shell/args: the value must be one of a fixed, hard-coded allow-list.
_CHOICE_SPEC: dict[str, tuple[str, tuple[str, ...]]] = {
    "subtask": ("--task", ("weather", "narma")),
    "select": ("--select", ("first", "mean")),
    "experiment": ("--experiment", ("2.1", "2.1_quick", "2.2", "all")),
    "mc_experiment": ("--experiment", ("all", "quick")),
    "fidelity": ("--fidelity", ("quick", "screen", "full", "tiny")),
}

# Only names/dots/dashes — never a path separator or "..". Blocks traversal
# when resolving a phase1 ``trace`` name to a file under TRACES_ROOT.
_TRACE_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def _resolve_trace(name: Any) -> Path:
    """Resolve a phase1 ``trace`` name to ``TRACES_ROOT/<name>.npz``, safely."""
    if not isinstance(name, str) or not _TRACE_NAME_RE.match(name) or ".." in name:
        raise ConfigError(f"invalid trace name {name!r} (use [A-Za-z0-9._-])")
    stem = name[:-4] if name.endswith(".npz") else name
    path = (TRACES_ROOT / f"{stem}.npz").resolve()
    if TRACES_ROOT.resolve() not in path.parents:
        raise ConfigError("trace path escapes the traces directory")
    if not path.exists():
        raise ConfigError(f"trace {stem!r}.npz not found; generate it first")
    return path


def _coerce_numeric(name: str, raw: Any) -> int | float:
    """Type-and-range-check a single numeric value against ``_PARAM_SPEC``."""
    _flag, typ, lo, hi = _PARAM_SPEC[name]
    if isinstance(raw, bool):  # bool is an int subclass — reject explicitly.
        raise ConfigError(f"{name!r} must be a number, not a boolean")
    try:
        val = typ(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name!r} must be {typ.__name__}, got {raw!r}") from exc
    if not (lo <= val <= hi):
        raise ConfigError(f"{name!r}={val} out of range [{lo}, {hi}]")
    return val


def build_argv(task: str, config: dict[str, Any], run_dir: Path, results_path: Path) -> list[str]:
    """Build the subprocess argv for ``task`` from an allow-listed ``config``.

    Pure + side-effect free so it can be unit-tested directly. Every element
    is an explicit list entry — there is no shell interpolation anywhere, so
    a malicious value can at worst become a rejected/unknown CLI flag, never
    a shell command. Unknown config keys are rejected outright.
    """
    if task not in _TASKS:
        raise ConfigError(f"unknown task {task!r}; allowed: {sorted(_TASKS)}")
    if not isinstance(config, dict):
        raise ConfigError("config must be an object")

    spec = _TASKS[task]
    choices = spec.get("choices", ())
    trace_required = spec.get("trace_required", False)
    allowed = (set(spec["numeric"]) | set(spec["lists"]) | set(spec["flags"])
               | set(choices) | ({"trace"} if trace_required else set()))
    unknown = set(config) - allowed
    if unknown:
        raise ConfigError(
            f"task {task!r} rejects unknown config key(s): {sorted(unknown)}; "
            f"allowed: {sorted(allowed)}"
        )

    argv: list[str] = [sys.executable, str(BASE_DIR / spec["script"]), *spec["fixed"]]
    # Output convention differs per runner: reproduction/trace-gen write a
    # JSON via --run-dir/--out; phase1 consumes a cached trace and writes a
    # directory via --trace/--out-dir.
    if spec.get("output") == "out_dir":
        argv += ["--out-dir", str(run_dir)]
    elif spec.get("output") == "run_dir":
        # writes outputs + live progress into the run-dir (no separate --out).
        argv += ["--run-dir", str(run_dir)]
    else:
        argv += ["--run-dir", str(run_dir), "--out", str(results_path)]
    if trace_required:
        if "trace" not in config:
            raise ConfigError(f"task {task!r} requires config key 'trace'")
        argv += ["--trace", str(_resolve_trace(config["trace"]))]

    for name in spec["numeric"]:
        if name not in config:
            continue
        flag = _PARAM_SPEC[name][0]
        val = _coerce_numeric(name, config[name])
        argv += [flag, str(val)]

    for name in spec["lists"]:
        if name not in config:
            continue
        raw = config[name]
        if not isinstance(raw, list) or not raw:
            raise ConfigError(f"{name!r} must be a non-empty list of numbers")
        flag = _PARAM_SPEC[name][0]
        argv.append(flag)
        argv += [str(_coerce_numeric(name, item)) for item in raw]

    for name in spec["flags"]:
        if config.get(name):
            argv.append(_FLAG_SPEC[name])

    for name in choices:
        flag, allowed_vals = _CHOICE_SPEC[name]
        if name not in config:
            # ``subtask`` (→ --task) is mandatory for the trace-gen runner.
            raise ConfigError(
                f"task {task!r} requires config key {name!r} "
                f"(one of {list(allowed_vals)})"
            )
        val = config[name]
        if val not in allowed_vals:
            raise ConfigError(
                f"{name!r}={val!r} invalid; allowed: {list(allowed_vals)}"
            )
        argv += [flag, str(val)]

    return argv


# ---- Registry -------------------------------------------------------------


def _registry_path() -> Path:
    return RUNS_ROOT / "registry.json"


def load_registry() -> dict[str, dict[str, Any]]:
    """Return the id -> entry mapping (empty dict if none written yet)."""
    path = _registry_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_registry(reg: dict[str, dict[str, Any]]) -> None:
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    _registry_path().write_text(json.dumps(reg, indent=2, default=str), encoding="utf-8")


def _update_entry(run_id: str, **fields: Any) -> dict[str, Any] | None:
    with _LOCK:
        reg = load_registry()
        entry = reg.get(run_id)
        if entry is None:
            return None
        entry.update(fields)
        reg[run_id] = entry
        _save_registry(reg)
        return entry


# ---- Process liveness -----------------------------------------------------


def _pid_alive(pid: int | None) -> bool:
    """Best-effort check that ``pid`` names a live process (no psutil dep)."""
    if not pid:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        process_query = 0x1000  # PROCESS_QUERY_LIMITED_INFORMATION
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(process_query, False, int(pid))
        if not handle:
            return False
        try:
            code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            still_active = 259  # STILL_ACTIVE
            return code.value == still_active
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(int(pid), 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def refresh_status(run_id: str) -> dict[str, Any] | None:
    """Reconcile the registry's recorded status with the live process.

    Prefers the in-process ``Popen`` handle (gives us the real exit code);
    falls back to a PID-liveness probe for runs whose handle we lost (server
    restart). Persists any transition so ``GET /runs/<id>/status`` is honest.
    """
    with _LOCK:
        reg = load_registry()
        entry = reg.get(run_id)
        if entry is None:
            return None
        if entry.get("status") not in _ACTIVE_STATUSES:
            return entry  # terminal — nothing to reconcile.

        proc = _PROCS.get(run_id)
        if proc is not None:
            code = proc.poll()
            if code is None:
                return entry  # still running.
            entry["status"] = "done" if code == 0 else "error"
            entry["exit_code"] = int(code)
            entry["ended_utc"] = datetime.now(UTC).isoformat()
        elif not _pid_alive(entry.get("pid")):
            # Lost the handle and the PID is gone — mark terminal with an
            # unknown exit code rather than leaving the lock held forever.
            entry["status"] = "done"
            entry["exit_code"] = entry.get("exit_code")
            entry["ended_utc"] = datetime.now(UTC).isoformat()
        reg[run_id] = entry
        _save_registry(reg)
        return entry


def get_run(run_id: str) -> dict[str, Any] | None:
    """Return one run's (status-reconciled) registry entry, or ``None``."""
    if load_registry().get(run_id) is None:
        return None
    return refresh_status(run_id)


def list_runs() -> list[dict[str, Any]]:
    """Return every run's reconciled entry, newest first."""
    reg = load_registry()
    entries = [refresh_status(rid) or reg[rid] for rid in reg]
    entries.sort(key=lambda e: e.get("created_utc", ""), reverse=True)
    return entries


def active_run() -> dict[str, Any] | None:
    """Return the single running heavy job, if one currently holds the lock."""
    for run_id in load_registry():
        refreshed = refresh_status(run_id)
        if refreshed and refreshed.get("status") in _ACTIVE_STATUSES:
            return refreshed
    return None


# ---- Launch / stop --------------------------------------------------------


def launch_run(task: str, config: dict[str, Any]) -> dict[str, Any]:
    """Validate + spawn ``task`` as a managed subprocess and register it.

    Raises :class:`ConfigError` on a bad task/config (route → 400) and
    :class:`RuntimeError` if the single-job lock is held (route → 409) or the
    runner script is missing (route → 503). Never uses ``shell=True``.
    """
    with _LOCK:
        if active_run() is not None:
            raise RuntimeError("a heavy job is already running")

        run_id = f"{task}-{uuid4().hex[:8]}"
        run_dir = RUNS_ROOT / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # trace-gen writes an FID trace ``.npz`` rather than a results JSON;
        # send its ``--out`` to a deterministic ``trace.npz`` in the run-dir and
        # register that path so ``GET /runs/<id>/fid`` resolves it (§1 seam).
        if task == "trace-gen":
            trace_path: Path | None = run_dir / "trace.npz"
            results_path = trace_path
        elif task == "phase1":
            # phase1 writes ``summary.json`` (+ per-experiment JSONs + the
            # selection figure) into its ``--out-dir`` (= run_dir); it never
            # emits a single ``results.json``. Point results_path at the
            # summary so ``GET /runs/<id>/results`` surfaces the sweep.
            trace_path = None
            results_path = run_dir / "summary.json"
        elif task == "phase2":
            # phase2 writes ``phase2_summary.json`` (ranking + per-encoding
            # results) into its ``--run-dir``.
            trace_path = None
            results_path = run_dir / "phase2_summary.json"
        elif task == "memcap":
            trace_path = None
            results_path = run_dir / "memcap_summary.json"
        else:
            trace_path = None
            results_path = run_dir / "results.json"

        argv = build_argv(task, config, run_dir, results_path)

        script = Path(argv[1])
        if not script.exists():
            raise RuntimeError(f"runner script not found: {script}")

        # Empty events file up-front so ``GET /progress`` works immediately,
        # before the runner's ProgressLogger has written its first line.
        events = run_dir / "events.jsonl"
        if not events.exists():
            events.write_text("", encoding="utf-8")

        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"

        proc = subprocess.Popen(  # noqa: S603 - argv list, shell=False, no user shell.
            argv,
            cwd=str(BASE_DIR),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        _PROCS[run_id] = proc

        entry = {
            "id": run_id,
            "task": task,
            "run_dir": str(run_dir),
            "results_path": str(results_path),
            "pid": proc.pid,
            "status": "running",
            "exit_code": None,
            "config": config,
            "created_utc": datetime.now(UTC).isoformat(),
        }
        if trace_path is not None:
            entry["trace_path"] = str(trace_path)
        reg = load_registry()
        reg[run_id] = entry
        _save_registry(reg)
        return entry


def stop_run(run_id: str) -> bool:
    """Terminate the run's subprocess. Returns ``False`` if unknown."""
    with _LOCK:
        reg = load_registry()
        entry = reg.get(run_id)
        if entry is None:
            return False

        proc = _PROCS.get(run_id)
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        elif _pid_alive(entry.get("pid")):
            _terminate_pid(int(entry["pid"]))

        entry["status"] = "stopped"
        entry["ended_utc"] = datetime.now(UTC).isoformat()
        reg[run_id] = entry
        _save_registry(reg)
        return True


def _terminate_pid(pid: int) -> None:
    """Kill a bare PID we no longer hold a Popen handle for (best effort)."""
    if os.name == "nt":
        subprocess.run(  # noqa: S603,S607 - fixed argv, no shell.
            ["taskkill", "/F", "/PID", str(pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        try:
            os.kill(pid, 15)  # SIGTERM
        except (OSError, ProcessLookupError):
            pass


# ---- Trace resolution (for the FID endpoint) ------------------------------


def run_trace_names(entry: dict[str, Any]) -> list[tuple[str, Path]]:
    """A sweep run's saved waveform traces as ``[(label, Path)]`` from its
    ``run-dir/traces.json`` manifest (phase2/memcap write one per encoding).
    Empty for single-trace runs."""
    run_dir = entry.get("run_dir")
    if not run_dir:
        return []
    manifest = Path(run_dir) / "traces.json"
    if not manifest.exists():
        return []
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
        out: list[tuple[str, Path]] = []
        for t in data.get("traces", []):
            nm = t.get("name")
            if nm:
                out.append((str(t.get("fn") or nm), TRACES_ROOT / nm))
        return out
    except (json.JSONDecodeError, OSError):
        return []


def resolve_trace_path(entry: dict[str, Any]) -> Path | None:
    """Locate the trace ``.npz`` a run reads/writes (§1 schema).

    Resolution order: an explicit ``trace_path`` on the entry, a ``trace``
    name in its config (→ ``artifacts/traces/<name>.npz``), a saved-trace
    manifest (sweep runs), any ``.npz`` in the run-dir, then
    ``artifacts/traces/<task>.npz``. Returns the first that exists, else ``None``.
    """
    explicit = entry.get("trace_path")
    if explicit and Path(explicit).exists():
        return Path(explicit)

    cfg = entry.get("config") or {}
    name = cfg.get("trace")
    if name:
        candidate = TRACES_ROOT / f"{name}.npz"
        if candidate.exists():
            return candidate

    # Sweep runs (phase2/memcap) link their per-encoding traces via a manifest.
    for _label, path in run_trace_names(entry):
        if path.exists():
            return path

    run_dir = entry.get("run_dir")
    if run_dir:
        npzs = sorted(Path(run_dir).glob("*.npz"))
        if npzs:
            return npzs[0]

    task = entry.get("task")
    if task:
        candidate = TRACES_ROOT / f"{task}.npz"
        if candidate.exists():
            return candidate
    return None
