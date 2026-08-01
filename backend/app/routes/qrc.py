"""QRC — Quantum Reservoir Computing control + inspection blueprint.

Serves the ``/api/qrc`` surface from ``docs/QRC/QRC_StageA_Contracts.md`` §3.
Coder D's Vue dashboard consumes it. Two tiers:

* **Read-only (unauthenticated)** — list runs, tail a run's live progress
  (parsed from its ``events.jsonl``), pull the results JSON, and compute a
  per-step FID spectrum server-side from the cached trace ``.npz`` (§1).
* **Control (auth-gated, mirrors how ``qml``/``solve`` gate mutating routes
  via ``@login_required``)** — launch a runner as a managed subprocess
  (single active heavy job, config allow-list), and stop it.

All heavy lifting (subprocess management, the registry, the config
allow-list) lives in ``app.qrc.launcher`` so this module stays a thin
translation layer between HTTP and that API.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from flask import Blueprint, jsonify, request

from app.auth import login_required
from app.qrc import launcher

qrc_bp = Blueprint("qrc", __name__)


# ---- Read-only: runs list --------------------------------------------------


def _progress_pct(entry: dict[str, Any]) -> float | None:
    """Best-effort completion percentage for the runs-list card.

    Terminal runs read 100; a running run's fraction comes from the latest
    ``step``/``total`` we can recover from its ``events.jsonl`` (see
    :func:`_parse_events`). ``None`` when we simply don't know yet.
    """
    status = entry.get("status")
    if status in ("done", "stopped", "error"):
        return 100.0
    snap = _parse_events(Path(entry["run_dir"]))
    step, total = snap.get("step"), snap.get("total")
    if step and total:
        return round(100.0 * step / total, 1)
    return None


@qrc_bp.route("/runs", methods=["GET"])
def qrc_list_runs():
    out = []
    for entry in launcher.list_runs():
        snap = _parse_events(Path(entry["run_dir"]))
        out.append({
            "id": entry["id"],
            "task": entry["task"],
            "status": entry.get("status"),
            "progress_pct": _progress_pct(entry),
            "eta_s": snap.get("eta_s"),
            "created_utc": entry.get("created_utc"),
        })
    return jsonify(out)


# ---- Read-only: progress ---------------------------------------------------


def _read_status(run_dir: Path) -> dict[str, Any]:
    """Read the durable ``status.json`` snapshot the ProgressLogger writes on
    every ``status()`` call, or ``{}``.

    The runner emits step-level progress + ETA through ``ProgressLogger.status``
    (not through ``events.jsonl``), so this file — not the event stream — is the
    authoritative source for the live ``phase``/``step``/``total``/``eta_s``.
    Malformed/absent → ``{}`` so the endpoint still answers from events.
    """
    path = run_dir / "status.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _parse_events(run_dir: Path) -> dict[str, Any]:
    """Parse a run-dir's ``events.jsonl`` into a progress snapshot.

    Each line is one ProgressLogger event ``{t, elapsed_s, kind, message,
    ...}``. We keep the raw event stream and surface the latest
    ``phase``/``step``/``total``/``eta_s``. Step-level progress is written
    durably to ``status.json`` by ``ProgressLogger.status`` (which does not
    touch ``events.jsonl``), so that snapshot is authoritative and takes
    precedence; any values attached to milestone events are the fallback.
    Malformed lines are skipped rather than 500-ing the whole endpoint.
    """
    events: list[dict[str, Any]] = []
    phase = step = total = eta_s = None
    events_file = run_dir / "events.jsonl"
    if events_file.exists():
        for line in events_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            events.append({
                "t": raw.get("t"),
                "elapsed_s": raw.get("elapsed_s"),
                "kind": raw.get("kind"),
                "message": raw.get("message"),
            })
            if raw.get("phase") is not None:
                phase = raw["phase"]
            if raw.get("step") is not None:
                step = raw["step"]
            if raw.get("total") is not None:
                total = raw["total"]
            if raw.get("eta_s") is not None:
                eta_s = raw["eta_s"]

    # The durable status.json snapshot (written by ProgressLogger.status) is
    # the authoritative source for live step/total/ETA — override with it.
    status = _read_status(run_dir)
    if status.get("phase") is not None:
        phase = status["phase"]
    if status.get("step") is not None:
        step = status["step"]
    if status.get("total") is not None:
        total = status["total"]
    if status.get("eta_s") is not None:
        eta_s = status["eta_s"]

    return {
        "phase": phase,
        "step": step,
        "total": total,
        "eta_s": eta_s,
        "events": events,
    }


def _load_results(entry: dict[str, Any]) -> dict[str, Any]:
    """Load a run's results JSON (checkpointed or final), or ``{}``."""
    results_path = entry.get("results_path")
    if not results_path:
        return {}
    path = Path(results_path)
    # A trace-gen run registers a binary ``trace.npz`` as its output — that is
    # not a results JSON, so never try to decode it as text (0xff → 500).
    if path.suffix.lower() != ".json" or not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


@qrc_bp.route("/runs/<run_id>/progress", methods=["GET"])
def qrc_run_progress(run_id: str):
    entry = launcher.get_run(run_id)
    if entry is None:
        return jsonify({"error": "Run not found"}), 404
    snap = _parse_events(Path(entry["run_dir"]))
    return jsonify({
        "phase": snap["phase"],
        "step": snap["step"],
        "total": snap["total"],
        "eta_s": snap["eta_s"],
        "status": entry.get("status"),
        "events": snap["events"],
        "results": _load_results(entry),
    })


# ---- Read-only: results ----------------------------------------------------


@qrc_bp.route("/runs/<run_id>/results", methods=["GET"])
def qrc_run_results(run_id: str):
    entry = launcher.get_run(run_id)
    if entry is None:
        return jsonify({"error": "Run not found"}), 404
    return jsonify(_load_results(entry))


# ---- Read-only: status -----------------------------------------------------


@qrc_bp.route("/runs/<run_id>/status", methods=["GET"])
def qrc_run_status(run_id: str):
    entry = launcher.get_run(run_id)
    if entry is None:
        return jsonify({"error": "Run not found"}), 404
    return jsonify({
        "status": entry.get("status"),
        "exit_code": entry.get("exit_code"),
    })


# ---- Read-only: FID spectrum (FFT computed server-side) --------------------


def _compute_fid(fids, fid_dwell: float, step: int, n_peaks: int) -> dict[str, Any]:
    """Time-domain + spectrum payload for one FID step (§1 / §3).

    ``fids`` is the ``[n_steps, fid_points]`` complex array from the trace.
    We return the raw complex FID (real/imag/mag) alongside its complex DFT
    (fft-shifted so frequencies run negative→positive, the NMR convention),
    plus the frequencies of the strongest spectral peaks.
    """
    import numpy as np

    signal = np.asarray(fids[step], dtype=np.complex128)
    n = signal.shape[0]
    t = (np.arange(n) * float(fid_dwell)).tolist()

    spectrum = np.fft.fftshift(np.fft.fft(signal))
    freq = np.fft.fftshift(np.fft.fftfreq(n, d=float(fid_dwell)))
    spectrum_mag = np.abs(spectrum)

    peaks_hz = _pick_peaks(freq, spectrum_mag, n_peaks)

    return {
        "t": t,
        "real": np.real(signal).tolist(),
        "imag": np.imag(signal).tolist(),
        "mag": np.abs(signal).tolist(),
        "freq_hz": freq.tolist(),
        "spectrum_mag": spectrum_mag.tolist(),
        "peaks_hz": peaks_hz,
    }


def _pick_peaks(freq, spectrum_mag, n_peaks: int) -> list[float]:
    """Return the frequencies of up to ``n_peaks`` strongest local maxima.

    A bin is a peak if it strictly exceeds both neighbours and sits above a
    small fraction of the global max (drops noise). Sorted by descending
    magnitude and capped — this is the server-side analogue of the 653-bin
    peak selection the feature builder does.
    """
    import numpy as np

    mag = np.asarray(spectrum_mag, dtype=np.float64)
    if mag.size < 3:
        return np.asarray(freq, dtype=np.float64).tolist()
    threshold = 0.01 * float(mag.max()) if mag.max() > 0 else 0.0
    interior = np.arange(1, mag.size - 1)
    is_peak = (
        (mag[interior] > mag[interior - 1])
        & (mag[interior] > mag[interior + 1])
        & (mag[interior] >= threshold)
    )
    idx = interior[is_peak]
    idx = idx[np.argsort(mag[idx])[::-1]][:n_peaks]
    return np.asarray(freq)[idx].tolist()


def _live_fid_source(entry: dict[str, Any]):
    """The LIVE streaming memmap a running reservoir is writing right now.

    ``StreamingTrace`` drops a ``live_fid.json`` in the run-dir pointing at its
    on-disk ``fids.dat`` memmap; this reads it so the FID/spectrum can be watched
    *as each step is computed*, before any ``.npz`` is saved. Only rows already
    computed (per the live ``status.json`` step) are exposed. Returns
    ``(mmap, fid_dwell, trace_name, trace_ref, n_available)`` or ``None``.
    """
    import numpy as np

    run_dir = entry.get("run_dir")
    if not run_dir:
        return None
    live = Path(run_dir) / "live_fid.json"
    if not live.exists():
        return None
    try:
        info = json.loads(live.read_text(encoding="utf-8"))
        fpath = Path(info["fids_path"])
        if not fpath.exists():
            return None
        n_steps, fp = int(info["n_steps"]), int(info["fid_points"])
        mm = np.memmap(str(fpath), dtype=np.complex64, mode="r", shape=(n_steps, fp))
        step_done = _read_status(Path(run_dir)).get("step") or 0
        n_avail = int(max(0, min(n_steps, int(step_done))))
        label = info.get("label") or "live"
        return mm, float(info.get("fid_dwell", 1.0)), f"live: {label}", str(fpath), n_avail
    except Exception:  # noqa: BLE001 - live view is best-effort
        return None


@qrc_bp.route("/runs/<run_id>/fid", methods=["GET"])
def qrc_run_fid(run_id: str):
    entry = launcher.get_run(run_id)
    if entry is None:
        return jsonify({"error": "Run not found"}), 404

    import numpy as np

    live = False
    trace_path = launcher.resolve_trace_path(entry)
    if trace_path is not None:
        with np.load(str(trace_path), allow_pickle=False) as npz:
            if "fids" not in npz:
                return jsonify({
                    "error": "Trace .npz is missing the 'fids' array.",
                    "code": "TRACE_MALFORMED",
                }), 422
            fids = np.asarray(npz["fids"])
            fid_dwell = float(npz["fid_dwell"]) if "fid_dwell" in npz else 1.0
        trace_name, trace_ref, n_avail = trace_path.name, str(trace_path), int(fids.shape[0])
    else:
        # Fall back to the live streaming memmap (run still computing).
        src = _live_fid_source(entry)
        if src is None:
            return jsonify({
                "error": "No trace .npz available for this run yet.",
                "code": "TRACE_MISSING",
            }), 404
        fids, fid_dwell, trace_name, trace_ref, n_avail = src
        live = True

    if n_avail <= 0:
        return jsonify({"error": "No FID steps computed yet.", "code": "NO_STEPS_YET"}), 404

    try:
        step = int(request.args.get("step", "0") or "0")
    except ValueError:
        return jsonify({"error": "step must be an integer", "code": "BAD_STEP"}), 400
    if not (0 <= step < n_avail):
        if live:
            # A running sweep advances through separate encodings, each with its
            # own fresh step range — clamp to the latest computed step rather
            # than erroring (a stale slider value must not break the live view).
            step = max(0, min(step, n_avail - 1))
        else:
            return jsonify({
                "error": f"step {step} out of range [0, {n_avail - 1}]",
                "code": "STEP_OUT_OF_RANGE",
            }), 400

    cfg = entry.get("config") or {}
    n_peaks = int(cfg.get("n_peaks") or 653)

    payload = _compute_fid(fids, fid_dwell, step, n_peaks)
    payload["step"] = step
    payload["n_steps"] = n_avail
    # Provenance: name the exact saved waveform (or live memmap) this came from.
    payload["trace_name"] = trace_name
    payload["trace_path"] = trace_ref
    payload["live"] = live
    return jsonify(payload)


# ---- Read-only: feature-space embedding (UMAP/PCA computed server-side) ----


def _trace_color(npz) -> tuple[list[float], str]:
    """A per-step scalar to colour the embedding scatter, + its label.

    Weather traces colour by normalized temperature (the primary target);
    NARMA traces colour by the driving input; otherwise fall back to the
    step index so the scatter still renders.
    """
    import numpy as np

    if "weather_norm" in npz:
        return np.asarray(npz["weather_norm"], dtype=float)[:, 0].tolist(), "temperature (norm)"
    if "narma_input" in npz:
        return np.asarray(npz["narma_input"], dtype=float).ravel().tolist(), "NARMA input"
    n = int(np.asarray(npz["fids"]).shape[0])
    return list(range(n)), "step index"


def _split_labels(npz, n_steps: int) -> list[str]:
    """Tag each step washout/train/test from the ``split`` triple (§1)."""
    import numpy as np

    try:
        washout, n_train, _ = (int(x) for x in np.asarray(npz["split"]).ravel()[:3])
    except (KeyError, ValueError):
        return ["all"] * n_steps
    labels = []
    for i in range(n_steps):
        if i < washout:
            labels.append("washout")
        elif i < washout + n_train:
            labels.append("train")
        else:
            labels.append("test")
    return labels


def _embed_2d(X, method: str, seed: int = 42):
    """Project ``X`` [n, d] to 2-D. ``pca`` (fast, always available) or ``umap``
    (nonlinear; needs the ``[featurelab]`` extra). Standardized first so no
    single high-variance feature dominates the projection."""
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    Xs = StandardScaler().fit_transform(np.asarray(X, dtype=float))
    if method == "umap":
        import umap  # guarded — raises ImportError if the extra isn't installed

        n_neighbors = int(min(15, max(2, Xs.shape[0] - 1)))
        reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=seed)
        return reducer.fit_transform(Xs)
    from sklearn.decomposition import PCA

    return PCA(n_components=2, random_state=seed).fit_transform(Xs)


@qrc_bp.route("/runs/<run_id>/embedding", methods=["GET"])
def qrc_run_embedding(run_id: str):
    """2-D projection of a run's reservoir feature vectors (Feature-Lab view).

    Query: ``method`` (``pca`` default | ``umap``), ``feature``
    (``multimodal`` default | ``phase`` | ``magnitude653``), ``n_peaks``.
    Builds the feature matrix from the cached trace, projects to 2-D, and
    returns points + a per-point colour (target) + split tag. This is the
    same server-side-compute pattern as ``/fid`` (no client-side heavy math).
    """
    entry = launcher.get_run(run_id)
    if entry is None:
        return jsonify({"error": "Run not found"}), 404

    trace_path = launcher.resolve_trace_path(entry)
    if trace_path is None:
        return jsonify({
            "error": "No trace .npz available for this run yet.",
            "code": "TRACE_MISSING",
        }), 404

    method = (request.args.get("method") or "pca").lower()
    if method not in ("pca", "umap"):
        return jsonify({"error": "method must be 'pca' or 'umap'", "code": "BAD_METHOD"}), 400
    feature = (request.args.get("feature") or "multimodal").lower()
    if feature not in ("magnitude653", "phase", "multimodal"):
        return jsonify({
            "error": "feature must be 'magnitude653', 'phase', or 'multimodal'",
            "code": "BAD_FEATURE",
        }), 400
    try:
        n_peaks = int(request.args.get("n_peaks", "653") or "653")
    except ValueError:
        return jsonify({"error": "n_peaks must be an integer", "code": "BAD_NPEAKS"}), 400

    import numpy as np

    from app.qrc.feature_methods import build_features

    with np.load(str(trace_path), allow_pickle=True) as z:
        npz = {k: z[k] for k in z.files}
    if "fids" not in npz:
        return jsonify({"error": "Trace .npz is missing 'fids'.", "code": "TRACE_MALFORMED"}), 422

    fids = np.asarray(npz["fids"])
    X, _names = build_features(fids, feature, n_peaks=n_peaks, select="first")

    try:
        coords = _embed_2d(X, method)
    except ImportError:
        return jsonify({
            "error": "UMAP unavailable — install the [featurelab] extra (umap-learn).",
            "code": "UMAP_UNAVAILABLE",
        }), 503

    color, color_label = _trace_color(npz)
    n_steps = int(fids.shape[0])
    return jsonify({
        "method": method,
        "feature": feature,
        "n_features_in": int(X.shape[1]),
        "n_points": n_steps,
        "points": np.asarray(coords, dtype=float).tolist(),
        "color": color[:n_steps],
        "color_label": color_label,
        "split": _split_labels(npz, n_steps),
        "trace_name": trace_path.name,
        "trace_path": str(trace_path),
    })


# ---- Control (auth-gated): launch / stop -----------------------------------


@qrc_bp.route("/runs", methods=["POST"])
@login_required
def qrc_launch_run():
    """Launch a runner as a managed subprocess.

    Body: ``{"task": "trace-gen|narma|weather|phase1", "config": {...}}``.
    Enforces a single active heavy job (409) and an allow-listed, numeric-
    range-checked config (400). Returns ``{"id": ...}`` on success.
    """
    payload = request.get_json(silent=True) or {}
    task = payload.get("task")
    config = payload.get("config") or {}
    if not isinstance(config, dict):
        return jsonify({"error": "config must be an object", "code": "BAD_CONFIG"}), 400

    try:
        entry = launcher.launch_run(task, config)
    except launcher.ConfigError as exc:
        return jsonify({"error": str(exc), "code": "BAD_CONFIG"}), 400
    except RuntimeError as exc:
        msg = str(exc)
        if "already running" in msg:
            return jsonify({"error": msg, "code": "JOB_ACTIVE"}), 409
        # Missing runner script (Coder A/B's runner not present yet).
        return jsonify({"error": msg, "code": "RUNNER_UNAVAILABLE"}), 503

    return jsonify({"id": entry["id"]}), 201


@qrc_bp.route("/runs/<run_id>/stop", methods=["POST"])
@login_required
def qrc_stop_run(run_id: str):
    if not launcher.stop_run(run_id):
        return jsonify({"error": "Run not found"}), 404
    return jsonify({"ok": True})
