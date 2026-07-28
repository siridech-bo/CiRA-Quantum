"""Tests for the QRC Stage A control + inspection blueprint (``/api/qrc``).

Same isolation pattern as ``test_routes_qml.py``: a tmp SQLite DB + seeded
user for the auth-gated control endpoints. On top of that we point the
launcher's registry/trace roots at ``tmp_path`` and fabricate a fake run-dir,
registry, and a tiny trace ``.npz`` so the read-only endpoints can be
exercised without ever spawning a real (GPU) run.

Coverage:
* read-only endpoint shapes (``/runs``, ``/progress``, ``/results``,
  ``/status``, ``/fid``) against a fabricated run;
* the FID endpoint's server-side FFT payload + step-range guard;
* control endpoints reject unauthenticated calls (401);
* the single active heavy job lock (409);
* ``build_argv`` is allow-listed, range-checked, and shell-safe (a list, no
  shell metacharacters), never launching arbitrary args.
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def isolated_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db_path = tmp_path / "qrc.db"
    monkeypatch.setenv("SECRET_KEY", "qrc-test-secret")
    from app import config as config_module
    monkeypatch.setattr(config_module, "DATABASE_PATH", str(db_path))
    from app import models as models_module
    importlib.reload(models_module)
    monkeypatch.setattr(models_module, "DATABASE_PATH", str(db_path))
    models_module.init_db()
    regular = models_module.create_user("qrc_alice", "p4sswordpass")

    # Point the launcher's registry + trace roots at the tmp dir and start
    # from a clean in-process handle table.
    from app.qrc import launcher
    runs_root = tmp_path / "qrc_runs"
    traces_root = tmp_path / "traces"
    runs_root.mkdir(parents=True, exist_ok=True)
    traces_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(launcher, "RUNS_ROOT", runs_root)
    monkeypatch.setattr(launcher, "TRACES_ROOT", traces_root)
    launcher._PROCS.clear()

    from app import create_app
    app = create_app({"TESTING": True})
    return app, launcher, regular, runs_root, traces_root


def _login(client, username: str = "qrc_alice", password: str = "p4sswordpass"):
    r = client.post("/api/auth/login", json={"username": username, "password": password})
    assert r.status_code == 200, r.get_data(as_text=True)


def _write_registry(runs_root: Path, reg: dict) -> None:
    (runs_root / "registry.json").write_text(json.dumps(reg), encoding="utf-8")


def _fabricate_run(runs_root: Path, run_id: str = "narma-deadbeef", *, status: str = "done"):
    """Create a run-dir with events.jsonl + results.json and register it."""
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    events = [
        {"t": "2026-07-28T10:00:00", "elapsed_s": 0.0, "kind": "start",
         "message": "run started"},
        {"t": "2026-07-28T10:01:00", "elapsed_s": 60.0, "kind": "reservoir",
         "message": "reservoir pass", "phase": "reservoir", "step": 300, "total": 600,
         "eta_s": 60.0},
        {"t": "2026-07-28T10:02:00", "elapsed_s": 120.0, "kind": "order_done",
         "message": "NARMA10 done", "order": 10},
    ]
    (run_dir / "events.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8"
    )
    results = {"task": "narma", "qrc_narma": {"10": {"nmse_paper": 2.46e-5}}}
    (run_dir / "results.json").write_text(json.dumps(results), encoding="utf-8")

    entry = {
        "id": run_id,
        "task": "narma",
        "run_dir": str(run_dir),
        "results_path": str(run_dir / "results.json"),
        "pid": 12345,
        "status": status,
        "exit_code": 0 if status == "done" else None,
        "config": {"n_peaks": 8},
        "created_utc": "2026-07-28T10:00:00+00:00",
    }
    _write_registry(runs_root, {run_id: entry})
    return run_dir, entry


def _fake_trace(path: Path, n_steps: int = 4, fid_points: int = 64) -> None:
    """Write a minimal §1-schema trace .npz with a complex ``fids`` array."""
    rng = np.random.default_rng(0)
    fids = (rng.standard_normal((n_steps, fid_points))
            + 1j * rng.standard_normal((n_steps, fid_points))).astype(np.complex64)
    np.savez(path, fids=fids, fid_dwell=np.float64(3e-4), task="narma")


# ---- Read-only endpoints ---------------------------------------------------


def test_list_runs_public_shape(isolated_app):
    app, _launcher, _regular, runs_root, _traces = isolated_app
    _fabricate_run(runs_root)
    client = app.test_client()

    r = client.get("/api/qrc/runs")  # no auth — public
    assert r.status_code == 200
    body = r.get_json()
    assert isinstance(body, list) and len(body) == 1
    row = body[0]
    assert set(row) == {"id", "task", "status", "progress_pct", "eta_s", "created_utc"}
    assert row["id"] == "narma-deadbeef"
    assert row["task"] == "narma"
    assert row["progress_pct"] == 100.0  # terminal run


def test_progress_parses_events(isolated_app):
    app, _launcher, _regular, runs_root, _traces = isolated_app
    _fabricate_run(runs_root)
    client = app.test_client()

    r = client.get("/api/qrc/runs/narma-deadbeef/progress")
    assert r.status_code == 200
    body = r.get_json()
    assert set(body) == {"phase", "step", "total", "eta_s", "status", "events", "results"}
    assert body["phase"] == "reservoir"
    assert body["step"] == 300
    assert body["total"] == 600
    assert body["eta_s"] == 60.0
    assert len(body["events"]) == 3
    assert set(body["events"][0]) == {"t", "elapsed_s", "kind", "message"}
    assert body["results"]["qrc_narma"]["10"]["nmse_paper"] == pytest.approx(2.46e-5)


def test_progress_reads_durable_status_json(isolated_app):
    """Seam #1: step/total/ETA come from ProgressLogger's durable status.json
    (written by ``status()``, which never touches events.jsonl), and that
    snapshot takes precedence over any values on milestone events."""
    app, launcher, _regular, runs_root, _traces = isolated_app
    run_dir, entry = _fabricate_run(runs_root, status="running")
    # Only a start event (no step/total on it) — mirrors the real runner, which
    # emits step-level progress via status(), not events.
    (run_dir / "events.jsonl").write_text(
        json.dumps({"t": "2026-07-28T10:00:00", "elapsed_s": 0.0,
                    "kind": "start", "message": "run started"}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "status.json").write_text(
        json.dumps({"phase": "weather", "step": 512, "total": 1000,
                    "eta_s": 800.0, "elapsed_s": 42.0}),
        encoding="utf-8",
    )

    # Keep the run "running" (a dead PID would be reconciled to done → 100%).
    class _LiveProc:
        pid = 12345
        def poll(self):
            return None
    launcher._PROCS["narma-deadbeef"] = _LiveProc()

    client = app.test_client()
    body = client.get("/api/qrc/runs/narma-deadbeef/progress").get_json()
    assert body["phase"] == "weather"
    assert body["step"] == 512
    assert body["total"] == 1000
    assert body["eta_s"] == 800.0
    # runs-list progress_pct derives from the same durable snapshot.
    row = next(r for r in client.get("/api/qrc/runs").get_json()
               if r["id"] == "narma-deadbeef")
    assert row["progress_pct"] == 51.2
    assert row["eta_s"] == 800.0


def test_trace_gen_binary_results_path_no_500(isolated_app):
    """A trace-gen run registers a binary ``trace.npz`` as its output. The
    results loader must NOT try to decode it as UTF-8 text (regression: that
    raised UnicodeDecodeError -> HTTP 500 on /progress and /results)."""
    app, _launcher, _regular, runs_root, _traces = isolated_app
    run_dir, entry = _fabricate_run(runs_root, run_id="trace-gen-cafef00d", status="done")
    # Point results at a binary .npz (as the launcher does for trace-gen).
    trace = run_dir / "trace.npz"
    _fake_trace(trace)
    entry["results_path"] = str(trace)
    entry["task"] = "trace-gen"
    _write_registry(runs_root, {entry["id"]: entry})
    client = app.test_client()
    for ep in ("progress", "results"):
        r = client.get(f"/api/qrc/runs/{entry['id']}/{ep}")
        assert r.status_code == 200, f"/{ep} -> {r.status_code}"
    assert client.get(f"/api/qrc/runs/{entry['id']}/results").get_json() == {}


def test_progress_404_on_unknown(isolated_app):
    app, *_ = isolated_app
    client = app.test_client()
    assert client.get("/api/qrc/runs/nope/progress").status_code == 404


def test_results_endpoint(isolated_app):
    app, _launcher, _regular, runs_root, _traces = isolated_app
    _fabricate_run(runs_root)
    client = app.test_client()
    r = client.get("/api/qrc/runs/narma-deadbeef/results")
    assert r.status_code == 200
    assert r.get_json()["task"] == "narma"


def test_status_endpoint(isolated_app):
    app, _launcher, _regular, runs_root, _traces = isolated_app
    _fabricate_run(runs_root)
    client = app.test_client()
    r = client.get("/api/qrc/runs/narma-deadbeef/status")
    assert r.status_code == 200
    body = r.get_json()
    assert body == {"status": "done", "exit_code": 0}


def test_fid_endpoint_computes_fft(isolated_app):
    app, _launcher, _regular, runs_root, traces_root = isolated_app
    run_dir, entry = _fabricate_run(runs_root)
    trace_path = traces_root / "narma.npz"
    _fake_trace(trace_path, n_steps=4, fid_points=64)
    # Point the registry entry explicitly at the trace.
    entry["trace_path"] = str(trace_path)
    _write_registry(runs_root, {entry["id"]: entry})

    client = app.test_client()
    r = client.get("/api/qrc/runs/narma-deadbeef/fid?step=1")
    assert r.status_code == 200
    body = r.get_json()
    assert set(body) == {
        "step", "n_steps", "t", "real", "imag", "mag",
        "freq_hz", "spectrum_mag", "peaks_hz",
    }
    assert body["step"] == 1
    assert body["n_steps"] == 4
    # Time-domain arrays match the FID length; spectrum spans the full grid.
    assert len(body["t"]) == 64
    assert len(body["real"]) == 64
    assert len(body["imag"]) == 64
    assert len(body["mag"]) == 64
    assert len(body["freq_hz"]) == 64
    assert len(body["spectrum_mag"]) == 64
    assert isinstance(body["peaks_hz"], list)
    # dwell = 3e-4 s → t[1] == dwell.
    assert body["t"][1] == pytest.approx(3e-4)


def test_fid_step_out_of_range(isolated_app):
    app, _launcher, _regular, runs_root, traces_root = isolated_app
    _, entry = _fabricate_run(runs_root)
    trace_path = traces_root / "narma.npz"
    _fake_trace(trace_path, n_steps=4)
    entry["trace_path"] = str(trace_path)
    _write_registry(runs_root, {entry["id"]: entry})

    client = app.test_client()
    r = client.get("/api/qrc/runs/narma-deadbeef/fid?step=99")
    assert r.status_code == 400
    assert r.get_json()["code"] == "STEP_OUT_OF_RANGE"


def test_fid_missing_trace_404(isolated_app):
    app, _launcher, _regular, runs_root, _traces = isolated_app
    _fabricate_run(runs_root)  # no trace written anywhere
    client = app.test_client()
    r = client.get("/api/qrc/runs/narma-deadbeef/fid")
    assert r.status_code == 404
    assert r.get_json()["code"] == "TRACE_MISSING"


# ---- Embedding endpoint (Feature-Lab 2-D projection) -----------------------


def _fake_weather_trace(path: Path, n_steps: int = 12, fid_points: int = 64) -> None:
    """A §1 weather trace with the target + split the embedding colours by."""
    rng = np.random.default_rng(1)
    fids = (rng.standard_normal((n_steps, fid_points))
            + 1j * rng.standard_normal((n_steps, fid_points))).astype(np.complex64)
    weather_norm = rng.random((n_steps, 2))
    split = np.array([3, 6, 3])  # washout, train, test
    np.savez(path, fids=fids, fid_dwell=np.float64(3e-4), task="weather",
             weather_norm=weather_norm, split=split)


def test_embedding_pca_shape(isolated_app):
    app, _launcher, _regular, runs_root, traces_root = isolated_app
    _, entry = _fabricate_run(runs_root)
    trace_path = traces_root / "narma.npz"
    _fake_weather_trace(trace_path, n_steps=12, fid_points=64)
    entry["trace_path"] = str(trace_path)
    _write_registry(runs_root, {entry["id"]: entry})

    client = app.test_client()
    r = client.get("/api/qrc/runs/narma-deadbeef/embedding"
                   "?method=pca&feature=magnitude653&n_peaks=16")
    assert r.status_code == 200, r.get_data(as_text=True)
    body = r.get_json()
    assert body["method"] == "pca"
    assert body["feature"] == "magnitude653"
    assert body["n_points"] == 12
    assert len(body["points"]) == 12
    assert all(len(p) == 2 for p in body["points"])   # projected to 2-D
    assert len(body["color"]) == 12
    assert body["color_label"] == "temperature (norm)"
    # split tags follow the [3,6,3] washout/train/test layout.
    assert body["split"][0] == "washout" and body["split"][-1] == "test"


def test_embedding_bad_method_400(isolated_app):
    app, _launcher, _regular, runs_root, traces_root = isolated_app
    _, entry = _fabricate_run(runs_root)
    trace_path = traces_root / "narma.npz"
    _fake_weather_trace(trace_path)
    entry["trace_path"] = str(trace_path)
    _write_registry(runs_root, {entry["id"]: entry})
    client = app.test_client()
    r = client.get("/api/qrc/runs/narma-deadbeef/embedding?method=tsne")
    assert r.status_code == 400
    assert r.get_json()["code"] == "BAD_METHOD"


def test_embedding_missing_trace_404(isolated_app):
    app, _launcher, _regular, runs_root, _traces = isolated_app
    _fabricate_run(runs_root)  # no trace anywhere
    client = app.test_client()
    r = client.get("/api/qrc/runs/narma-deadbeef/embedding")
    assert r.status_code == 404
    assert r.get_json()["code"] == "TRACE_MISSING"


# ---- Control endpoints: auth gating ----------------------------------------


def test_control_requires_auth(isolated_app):
    app, *_ = isolated_app
    client = app.test_client()
    # No login → 401 on both mutating routes.
    r = client.post("/api/qrc/runs", json={"task": "narma", "config": {}})
    assert r.status_code == 401
    assert client.post("/api/qrc/runs/anything/stop").status_code == 401


def test_launch_rejects_bad_config(isolated_app):
    app, *_ = isolated_app
    client = app.test_client()
    _login(client)

    # Unknown task.
    r = client.post("/api/qrc/runs", json={"task": "bogus", "config": {}})
    assert r.status_code == 400
    assert r.get_json()["code"] == "BAD_CONFIG"

    # Out-of-range numeric param.
    r = client.post("/api/qrc/runs", json={"task": "narma", "config": {"n_peaks": 999999}})
    assert r.status_code == 400

    # Unknown config key.
    r = client.post("/api/qrc/runs", json={"task": "narma", "config": {"rm_rf": "/"}})
    assert r.status_code == 400


def test_single_job_lock_409(isolated_app):
    app, launcher, _regular, runs_root, _traces = isolated_app

    # Register a run that is "running" and back it with a fake live process
    # handle so the launcher's status refresh keeps it active.
    run_id = "narma-running1"
    (runs_root / run_id).mkdir(parents=True, exist_ok=True)
    _write_registry(runs_root, {run_id: {
        "id": run_id, "task": "narma", "run_dir": str(runs_root / run_id),
        "results_path": str(runs_root / run_id / "results.json"),
        "pid": 999999, "status": "running", "exit_code": None,
        "config": {}, "created_utc": "2026-07-28T09:00:00+00:00",
    }})

    class _LiveProc:
        pid = 999999

        def poll(self):
            return None  # still running

    launcher._PROCS[run_id] = _LiveProc()

    client = app.test_client()
    _login(client)
    r = client.post("/api/qrc/runs", json={"task": "weather", "config": {}})
    assert r.status_code == 409
    assert r.get_json()["code"] == "JOB_ACTIVE"


def test_launch_spawns_managed_subprocess(isolated_app, monkeypatch):
    app, launcher, _regular, runs_root, _traces = isolated_app

    captured: dict = {}

    class _FakePopen:
        def __init__(self, argv, **kwargs):
            captured["argv"] = argv
            captured["kwargs"] = kwargs
            self.pid = 4321
            self._alive = True

        def poll(self):
            return None if self._alive else 0

        def terminate(self):
            self._alive = False

        def wait(self, timeout=None):
            return 0

        def kill(self):
            self._alive = False

    monkeypatch.setattr(launcher.subprocess, "Popen", _FakePopen)

    client = app.test_client()
    _login(client)
    r = client.post("/api/qrc/runs", json={
        "task": "narma", "config": {"n_train": 50, "orders": [2, 10], "no_esn": True},
    })
    assert r.status_code == 201
    run_id = r.get_json()["id"]
    assert run_id.startswith("narma-")

    # Managed subprocess: argv is a list, shell was never used.
    argv = captured["argv"]
    assert isinstance(argv, list)
    assert captured["kwargs"].get("shell") in (None, False)
    assert "--task" in argv and "narma" in argv
    assert "--run-dir" in argv and "--out" in argv
    assert "--n-train" in argv and "50" in argv
    assert "--orders" in argv and "2" in argv and "10" in argv
    assert "--no-esn" in argv
    # No single element smuggles shell metacharacters.
    assert not any(ch in tok for tok in argv for ch in ";|&$`")

    # It is recorded in the registry as running.
    reg = launcher.load_registry()
    assert reg[run_id]["status"] == "running"
    assert reg[run_id]["pid"] == 4321

    # Stop terminates + flips status.
    r = client.post(f"/api/qrc/runs/{run_id}/stop")
    assert r.status_code == 200
    assert r.get_json() == {"ok": True}
    assert launcher.load_registry()[run_id]["status"] == "stopped"


def test_launch_missing_runner_script_maps_to_503(isolated_app, monkeypatch):
    """A missing runner script surfaces as a clean 503 (RUNNER_UNAVAILABLE),
    not a 500. We simulate the launcher raising the not-found RuntimeError so
    the test never spawns a process, regardless of which runner scripts
    happen to be present in the tree."""
    app, launcher, *_ = isolated_app

    def _boom(task, config):
        raise RuntimeError("runner script not found: scripts/qrc_phase1.py")
    monkeypatch.setattr(launcher, "launch_run", _boom)

    client = app.test_client()
    _login(client)
    r = client.post("/api/qrc/runs", json={"task": "phase1", "config": {}})
    assert r.status_code == 503
    assert r.get_json()["code"] == "RUNNER_UNAVAILABLE"


# ---- build_argv unit tests (shell-safety + allow-list) ---------------------


def test_build_argv_is_shell_safe_list(isolated_app):
    _app, launcher, *_ = isolated_app
    argv = launcher.build_argv(
        "weather",
        {"tau": 0.03, "horizons": [1, 45], "max_days": 500},
        Path("run_dir"),
        Path("run_dir/results.json"),
    )
    assert isinstance(argv, list)
    assert all(isinstance(tok, str) for tok in argv)
    assert argv[-6:] != []  # sanity
    assert "--task" in argv and "weather" in argv
    assert "--tau" in argv and "0.03" in argv
    assert "--horizons" in argv and "1" in argv and "45" in argv
    assert "--max-days" in argv and "500" in argv


def test_build_argv_rejects_unknown_and_out_of_range(isolated_app):
    _app, launcher, *_ = isolated_app
    with pytest.raises(launcher.ConfigError):
        launcher.build_argv("narma", {"evil": 1}, Path("d"), Path("d/r.json"))
    with pytest.raises(launcher.ConfigError):
        launcher.build_argv("narma", {"n_train": -5}, Path("d"), Path("d/r.json"))
    with pytest.raises(launcher.ConfigError):
        launcher.build_argv("nope", {}, Path("d"), Path("d/r.json"))
    # Boolean smuggled as a numeric is rejected.
    with pytest.raises(launcher.ConfigError):
        launcher.build_argv("narma", {"seed": True}, Path("d"), Path("d/r.json"))


def test_build_argv_trace_gen_maps_to_gen_traces_cli(isolated_app):
    """Seam #2: the trace-gen argv matches ``qrc_gen_traces.py``'s real CLI —
    ``--task`` (via the ``subtask`` choice) + a single ``--splits`` triple —
    and stays a shell-safe list."""
    _app, launcher, *_ = isolated_app
    argv = launcher.build_argv(
        "trace-gen",
        {"subtask": "weather", "fid_points": 256, "n_peaks": 128,
         "splits": [20, 60, 40], "horizons": [1, 45]},
        Path("run_dir"), Path("run_dir/trace.npz"),
    )
    assert isinstance(argv, list) and all(isinstance(t, str) for t in argv)
    assert "--task" in argv and "weather" in argv
    i = argv.index("--splits")
    assert argv[i + 1:i + 4] == ["20", "60", "40"]
    assert argv[argv.index("--out") + 1].endswith("trace.npz")
    assert not any(ch in tok for tok in argv for ch in ";|&$`")
    # subtask is mandatory (gen_traces requires --task) and value-checked.
    with pytest.raises(launcher.ConfigError):
        launcher.build_argv("trace-gen", {"fid_points": 256},
                            Path("d"), Path("d/t.npz"))
    with pytest.raises(launcher.ConfigError):
        launcher.build_argv("trace-gen", {"subtask": "bogus"},
                            Path("d"), Path("d/t.npz"))


def test_trace_gen_registers_trace_path_for_fid(isolated_app, monkeypatch):
    """Seam #2: a launched trace-gen run records the produced ``trace.npz`` path
    so ``resolve_trace_path`` (→ /fid) finds it deterministically."""
    _app, launcher, *_ = isolated_app

    class _FakePopen:
        def __init__(self, argv, **kw):
            self.pid = 777
        def poll(self):
            return None

    monkeypatch.setattr(launcher.subprocess, "Popen", _FakePopen)
    entry = launcher.launch_run("trace-gen", {"subtask": "narma", "fid_points": 64})
    run_dir = Path(entry["run_dir"])
    assert entry["trace_path"] == str(run_dir / "trace.npz")
    # Before the file exists, resolution yields None (endpoint → TRACE_MISSING).
    assert launcher.resolve_trace_path(entry) is None
    # Once the runner writes it, resolution is deterministic.
    np.savez(run_dir / "trace.npz", fids=np.zeros((2, 8), dtype=np.complex64),
             fid_dwell=np.float64(3e-4), task="narma")
    assert launcher.resolve_trace_path(entry) == run_dir / "trace.npz"
