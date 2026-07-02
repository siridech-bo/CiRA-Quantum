"""QML — Quantum Machine Learning blueprint.

QML-1 shipped the dataset gallery + jobs CRUD. QML-2 adds the live
training pipeline: a POST that creates a job + kicks off a background
trainer, and an SSE stream that mirrors the optimization side's
``/api/jobs/<id>/stream``.

Endpoints
~~~~~~~~~

* ``GET  /api/qml/health``                 liveness + capability flags
* ``GET  /api/qml/datasets``               public dataset gallery
* ``GET  /api/qml/datasets/<id>``          single dataset detail
* ``POST /api/qml/train``                  start a VQC training job
* ``GET  /api/qml/jobs``                   user's jobs, paginated
* ``GET  /api/qml/jobs/<id>``              one job's full state
* ``GET  /api/qml/jobs/<id>/stream``       live SSE training events
* ``DELETE /api/qml/jobs/<id>``            delete a job
"""

from __future__ import annotations

import json

from datetime import datetime

from flask import Blueprint, Response, jsonify, request

from app.auth import admin_required, get_current_user, login_required
from app.config import KEY_ENCRYPTION_SECRET
from app.crypto import decrypt_api_key
from app.models import (
    create_qml_job,
    create_qml_qpu_run,
    delete_qml_job,
    get_api_key_ciphertext,
    get_qml_job,
    get_qml_qpu_run,
    list_qml_jobs,
    list_qml_qpu_runs_for_job,
    update_qml_qpu_run,
)
from app.pipeline.events import get_event_bus
from app.qml import get_dataset, list_datasets
from app.qml import records as qml_records

qml_bp = Blueprint("qml", __name__)


# ---- Health ----------------------------------------------------------------


@qml_bp.route("/health", methods=["GET"])
def qml_health():
    """Tells the frontend which QML capabilities are available.

    ``pennylane`` / ``sklearn`` are the QML-2 training stack; until that
    ships the frontend uses these flags to grey out the Train button
    with a helpful tooltip rather than silently 500ing on submit.
    """
    have_pennylane = _can_import("pennylane")
    have_sklearn = _can_import("sklearn")
    have_qiskit_ibm = _can_import("qiskit_ibm_runtime")
    return jsonify({
        "status": "ok",
        "phase": "QML-1 foundation",
        "capabilities": {
            "pennylane": have_pennylane,
            "sklearn": have_sklearn,
            "qiskit_ibm_runtime": have_qiskit_ibm,
        },
    })


def _can_import(module: str) -> bool:
    try:
        __import__(module)
        return True
    except Exception:
        return False


# ---- Datasets (public) -----------------------------------------------------


@qml_bp.route("/datasets", methods=["GET"])
def qml_datasets():
    items = list_datasets()
    return jsonify({"datasets": items, "total": len(items)})


@qml_bp.route("/datasets/<dataset_id>", methods=["GET"])
def qml_dataset_detail(dataset_id: str):
    d = get_dataset(dataset_id)
    if d is None:
        return jsonify({"error": "Dataset not found"}), 404
    return jsonify(d)


@qml_bp.route("/datasets/<dataset_id>/preview", methods=["GET"])
def qml_dataset_preview(dataset_id: str):
    """A small scatter-plot-ready preview of a dataset.

    For inherently-2D datasets (Moons, Circles) we return the raw
    standard-scaled features. For higher-dimensional ones (Iris 4D,
    Wine 13D, Breast Cancer 30D, MNIST 64D) we project to 2 principal
    components so a student can still *see* the dataset before deciding
    to spend simulator time on it.

    Capped at 200 points per dataset to keep the payload small (~5 KB).
    Cached implicitly because the loader is deterministic.
    """
    d = get_dataset(dataset_id)
    if d is None:
        return jsonify({"error": "Dataset not found"}), 404
    if not _can_import("sklearn"):
        return jsonify({
            "error": "scikit-learn not installed on the server. "
                     "Install the [qml] extra to enable previews.",
            "code": "QML_STACK_MISSING",
        }), 503

    # Lazy import — keeps the route layer fast for clients that only
    # walk the gallery without expanding any dataset.
    from app.qml import data_loader

    split = data_loader.load(dataset_id, max_qubits=2, n_samples_cap=200)
    # Stitch train + test back together — the preview should show the
    # whole dataset, not just the train fold.
    import numpy as np
    X = np.vstack([split.X_train, split.X_test])
    y = np.concatenate([split.y_train, split.y_test])

    return jsonify({
        "id": dataset_id,
        "title": d["title"],
        "classes": split.classes,
        "feature_names": split.feature_names,  # may be ["PC1", "PC2"]
        "pca_applied": split.pca_applied,
        "n_points": int(X.shape[0]),
        # Round to 4 decimals so the JSON payload stays compact.
        "points": [
            {"x": round(float(X[i, 0]), 4),
             "y": round(float(X[i, 1]), 4),
             "label": int(y[i])}
            for i in range(X.shape[0])
        ],
        "notes": split.notes,
    })


# ---- Train (auth-gated) ----------------------------------------------------


@qml_bp.route("/train", methods=["POST"])
@login_required
def qml_train():
    """Create a QML training job and kick off the trainer thread.

    Returns immediately with ``{"job": {"id", "status": "queued", ...}}``;
    the client opens ``GET /api/qml/jobs/<id>/stream`` to watch live.
    """
    payload = request.get_json(silent=True) or {}
    dataset_id = payload.get("dataset_id")
    model = (payload.get("model") or "vqc").lower()
    if model != "vqc":
        return jsonify({
            "error": "Only the 'vqc' model is supported in QML-2.",
            "code": "UNSUPPORTED_MODEL",
        }), 400
    if not dataset_id or get_dataset(dataset_id) is None:
        return jsonify({
            "error": "Unknown dataset_id; pick one from /api/qml/datasets.",
            "code": "UNKNOWN_DATASET",
        }), 400

    # Validate that the training stack is actually installed before
    # creating a job row that would never make progress.
    if not _can_import("pennylane"):
        return jsonify({
            "error": "PennyLane is not installed on the server. "
                     "Run `pip install \".[qml]\"`.",
            "code": "QML_STACK_MISSING",
        }), 503

    user = get_current_user()
    hp = payload.get("hyperparameters") or {}
    if not isinstance(hp, dict):
        return jsonify({"error": "hyperparameters must be an object"}), 400

    job_id = create_qml_job(
        user["id"], dataset_id, model,
        hyperparameters=json.dumps(hp) if hp else None,
    )

    # Lazy-import the trainer so the route layer doesn't pull PennyLane
    # in for the dataset gallery / jobs CRUD paths. ``launch_training_job``
    # picks between the in-process thread and the Redis-RQ queue based
    # on the USE_REDIS_QUEUE env flag — the threaded path is the
    # default for fresh installs.
    from app.qml.trainer import launch_training_job
    launch_training_job(
        job_id=job_id,
        dataset_id=dataset_id,
        hyperparameters=hp,
        event_bus=get_event_bus(),
    )

    job = get_qml_job(job_id, user_id=user["id"])
    return jsonify({"job": job}), 201


# ---- Jobs (auth-gated) -----------------------------------------------------


@qml_bp.route("/jobs", methods=["GET"])
@login_required
def qml_jobs_list():
    user = get_current_user()
    page = int(request.args.get("page", "1") or "1")
    page_size = int(request.args.get("page_size", "20") or "20")
    return jsonify(list_qml_jobs(user["id"], page=page, page_size=page_size))


@qml_bp.route("/jobs/<job_id>", methods=["GET"])
@login_required
def qml_job_detail(job_id: str):
    user = get_current_user()
    job = get_qml_job(
        job_id,
        user_id=user["id"],
        is_admin=(user.get("role") == "admin"),
    )
    if job is None:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


@qml_bp.route("/jobs/<job_id>", methods=["DELETE"])
@login_required
def qml_job_delete(job_id: str):
    user = get_current_user()
    removed = delete_qml_job(
        job_id,
        user_id=user["id"],
        is_admin=(user.get("role") == "admin"),
    )
    if not removed:
        return jsonify({"error": "Job not found"}), 404
    return jsonify({"deleted": True})


# ---- SSE stream ------------------------------------------------------------


@qml_bp.route("/jobs/<job_id>/stream", methods=["GET"])
@login_required
def qml_job_stream(job_id: str):
    """Live training event stream for a QML job.

    Mirrors ``/api/jobs/<id>/stream`` on the optimization side. The
    trainer pushes ``loading`` / ``training`` / ``epoch`` / ``complete``
    / ``error`` events onto the shared bus; subscribers replay history
    + block for live ones.
    """
    user = get_current_user()
    job = get_qml_job(
        job_id,
        user_id=user["id"],
        is_admin=(user.get("role") == "admin"),
    )
    if job is None:
        return jsonify({"error": "Job not found"}), 404

    bus = get_event_bus()

    def generate():
        for event in bus.subscribe(job_id):
            yield "event: status\n"
            yield f"data: {json.dumps(event)}\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ---- QML-5: Real-QPU inference ---------------------------------------------


# Per-backend qubit budget for the Origin QPU inference path. Mirrors
# the optimization side (QAOACloudSampler / _REAL_HARDWARE_BACKENDS)
# so the two apps behave the same when a user's VQC width exceeds
# what the chosen chip can hold. Return a helpful error at submit
# time instead of letting pyqpanda3 emit a cryptic circuit-compile
# failure downstream.
_ORIGIN_QUBIT_CAPS: dict[str, int] = {
    "full_amplitude": 7,   # Origin's cloud simulator budget.
    "WK_C180": 12,
    "WK_C180_2": 12,
    "HanYuan_01": 12,
}


def _origin_qubit_cap(backend_name: str) -> int | None:
    """Return the qubit cap for the given Origin backend, or ``None``
    when the backend isn't in our whitelist (letting pyqpanda3 return
    its own error keeps forward-compat: new Origin chips work without
    a code change)."""
    return _ORIGIN_QUBIT_CAPS.get(backend_name)


def _load_test_split_from_metrics(metrics_json: str | None):
    """Reconstruct ``(X_test, y_test, weights, bias)`` from the parent
    job's persisted metrics. Returns ``None`` if the metrics can't be
    parsed or the fields aren't there.

    Prefers the ``test_split`` field (N-dim, added 2026-07-02) so
    real-QPU inference works for any qubit count up to the backend's
    cap. Falls back to ``scatter_points`` (2D only) for jobs trained
    before ``test_split`` was persisted — those will continue to work
    on the 2-qubit path.
    """
    if not metrics_json:
        return None
    try:
        m = json.loads(metrics_json)
    except Exception:
        return None
    weights = m.get("weights")
    bias = m.get("bias")
    if not weights or bias is None:
        return None

    import numpy as _np

    # ---- Preferred: N-dim test_split (works for any n_qubits) ----
    ts = m.get("test_split")
    if ts and ts.get("X_test") and ts.get("y_test"):
        X_test = _np.array(ts["X_test"], dtype=_np.float32)
        y_test = _np.array(ts["y_test"], dtype=_np.int64)
        return X_test, y_test, weights, float(bias)

    # ---- Fallback: 2D scatter_points (legacy jobs, 2-qubit only) ----
    scatter = m.get("scatter_points")
    if not scatter:
        return None
    test_pts = [p for p in scatter if p.get("split") == "test"]
    if not test_pts:
        return None
    X_test = _np.array([[p["x"], p["y"]] for p in test_pts], dtype=_np.float32)
    y_test = _np.array([int(p["label"]) for p in test_pts], dtype=_np.int64)
    return X_test, y_test, weights, float(bias)


@qml_bp.route("/jobs/<job_id>/qpu/ibmq", methods=["POST"])
@login_required
def qml_submit_qpu_ibmq(job_id: str):
    """Submit a trained VQC for real-QPU inference on IBM Quantum.

    Body: ``{"shots": int = 1024, "backend_name": str | None = null}``.
    Reuses the user's stored ``ibm_quantum`` BYOK key. Returns the
    queued ``qml_qpu_runs`` row immediately; the frontend polls via
    ``POST /api/qml/qpu-runs/<run_id>/refresh`` until the job is terminal.
    """
    user = get_current_user()
    parent = get_qml_job(job_id, user_id=user["id"])
    if parent is None:
        return jsonify({"error": "Parent job not found"}), 404
    if parent["status"] != "complete":
        return jsonify({
            "error": "Train the VQC first — real-QPU inference reuses the trained weights.",
            "code": "PARENT_NOT_COMPLETE",
        }), 400

    if not _can_import("qiskit_ibm_runtime"):
        return jsonify({
            "error": "qiskit-ibm-runtime is not installed. "
                     "Run `pip install \".[ibm-quantum]\"`.",
            "code": "IBM_STACK_MISSING",
        }), 503

    payload = request.get_json(silent=True) or {}
    shots = int(payload.get("shots") or 1024)
    backend_name = payload.get("backend_name") or None

    # Pull the user's IBM Quantum BYOK token.
    cipher = get_api_key_ciphertext(user["id"], "ibm_quantum")
    if cipher is None:
        return jsonify({
            "error": "No IBM Quantum API key on file. Add one in API Keys "
                     "(provider: ibm_quantum) before launching a real-QPU run.",
            "code": "BYOK_MISSING",
        }), 400
    try:
        api_key = decrypt_api_key(cipher, KEY_ENCRYPTION_SECRET).strip()
    except Exception:
        return jsonify({"error": "Failed to decrypt stored IBM Quantum key."}), 500

    split = _load_test_split_from_metrics(parent.get("metrics"))
    if split is None:
        return jsonify({
            "error": (
                "This job's metrics don't include a persisted test "
                "split. Jobs trained before 2026-07-02 only saved the "
                "2D scatter used for the decision-boundary plot; "
                "re-train to enable real-QPU inference."
            ),
            "code": "NO_TEST_SPLIT",
        }), 400
    X_test, y_test, weights, bias = split

    # Create the run row first so we have a stable id even if submission errors.
    run_id = create_qml_qpu_run(
        qml_job_id=job_id,
        user_id=user["id"],
        provider="ibmq",
        shots=shots,
        backend_name=backend_name,
    )
    update_qml_qpu_run(run_id, status="submitting")

    # Submit. Wrapped so we capture any submission error in the row.
    try:
        from app.qml.qpu_ibmq import submit_inference
        envelope = submit_inference(
            api_key=api_key,
            weights=weights,
            bias=bias,
            X_test=X_test,
            backend_name=backend_name,
            shots=shots,
        )
    except Exception as exc:
        update_qml_qpu_run(
            run_id,
            status="error",
            error=f"{type(exc).__name__}: {exc}",
            completed_at=datetime.utcnow().isoformat(),
        )
        run = get_qml_qpu_run(run_id, user_id=user["id"])
        return jsonify({"qpu_run": run, "error": str(exc)}), 502

    update_qml_qpu_run(
        run_id,
        status="submitted",
        cloud_job_id=envelope.get("cloud_job_id"),
        backend_name=envelope.get("backend_name"),
    )
    run = get_qml_qpu_run(run_id, user_id=user["id"])
    return jsonify({"qpu_run": run}), 201


@qml_bp.route("/jobs/<job_id>/qpu/originqc", methods=["POST"])
@login_required
def qml_submit_qpu_originqc(job_id: str):
    """Submit a trained VQC for real-QPU inference on Origin Quantum.

    Asymmetric vs IBM: Origin doesn't batch PUBs, so we evaluate exactly
    one representative test point per submission. Body:
    ``{"shots": int = 2048, "backend_name": str = "full_amplitude",
       "sample_index": int | null = null}``.
    Reuses the user's stored ``originqc`` BYOK key.
    """
    user = get_current_user()
    parent = get_qml_job(job_id, user_id=user["id"])
    if parent is None:
        return jsonify({"error": "Parent job not found"}), 404
    if parent["status"] != "complete":
        return jsonify({
            "error": "Train the VQC first — real-QPU inference reuses the trained weights.",
            "code": "PARENT_NOT_COMPLETE",
        }), 400

    if not _can_import("pyqpanda3"):
        return jsonify({
            "error": "pyqpanda3 is not installed. "
                     "Run `pip install \".[quantum]\"`.",
            "code": "ORIGIN_STACK_MISSING",
        }), 503

    payload = request.get_json(silent=True) or {}
    shots = int(payload.get("shots") or 2048)
    backend_name = (payload.get("backend_name") or "full_amplitude").strip()
    sample_index_raw = payload.get("sample_index")
    sample_index = int(sample_index_raw) if sample_index_raw is not None else None

    cipher = get_api_key_ciphertext(user["id"], "originqc")
    if cipher is None:
        return jsonify({
            "error": "No Origin Quantum API key on file. Add one in API Keys "
                     "(provider: originqc) before launching a real-QPU run.",
            "code": "BYOK_MISSING",
        }), 400
    try:
        api_key = decrypt_api_key(cipher, KEY_ENCRYPTION_SECRET).strip()
    except Exception:
        return jsonify({"error": "Failed to decrypt stored Origin Quantum key."}), 500

    split = _load_test_split_from_metrics(parent.get("metrics"))
    if split is None:
        return jsonify({
            "error": (
                "This job's metrics don't include a persisted test "
                "split. Jobs trained before 2026-07-02 only saved the "
                "2D scatter used for the decision-boundary plot; "
                "re-train to enable real-QPU inference."
            ),
            "code": "NO_TEST_SPLIT",
        }), 400
    X_test, y_test, weights, bias = split

    # Qubit-cap preflight for known Origin backends. Origin's
    # ``full_amplitude`` simulator holds 7 qubits, and the Wukong /
    # HanYuan chips hold 12. Trying to submit a wider VQC produces
    # a cryptic pyqpanda3 error after the shots have already been
    # billed on some plans; catching it here saves the user a bad
    # experience and (potentially) real money.
    n_features = int(X_test.shape[1])
    cap = _origin_qubit_cap(backend_name)
    if cap is not None and n_features > cap:
        return jsonify({
            "error": (
                f"This VQC uses {n_features} qubits, but backend "
                f"{backend_name!r} only supports up to {cap}. Either "
                f"pick a wider backend (Wukong=12) or re-train with "
                f"fewer features (Settings → n_qubits)."
            ),
            "code": "QUBIT_CAP_EXCEEDED",
        }), 400

    run_id = create_qml_qpu_run(
        qml_job_id=job_id,
        user_id=user["id"],
        provider="originqc",
        shots=shots,
        backend_name=backend_name,
    )
    update_qml_qpu_run(run_id, status="submitting")

    try:
        from app.qml.qpu_originqc import submit_inference as origin_submit
        envelope = origin_submit(
            api_key=api_key,
            weights=weights,
            bias=bias,
            X_test=X_test,
            y_test=y_test,
            sample_index=sample_index,
            backend_name=backend_name,
            shots=shots,
        )
    except Exception as exc:
        update_qml_qpu_run(
            run_id,
            status="error",
            error=f"{type(exc).__name__}: {exc}",
            completed_at=datetime.utcnow().isoformat(),
        )
        run = get_qml_qpu_run(run_id, user_id=user["id"])
        return jsonify({"qpu_run": run, "error": str(exc)}), 502

    # Stash the per-run state the Origin materializer needs (sample
    # index + true label) so refresh() can reconstruct the prediction
    # without re-loading the parent job's metrics.
    submission_ctx = json.dumps({
        "sample_index": envelope["sample_index"],
        "sample_x": envelope["sample_x"],
        "sample_true_label": envelope["sample_true_label"],
    })
    update_qml_qpu_run(
        run_id,
        status="submitted",
        cloud_job_id=envelope.get("cloud_job_id"),
        backend_name=envelope.get("backend_name"),
        submission_context=submission_ctx,
    )
    run = get_qml_qpu_run(run_id, user_id=user["id"])
    return jsonify({"qpu_run": run}), 201


@qml_bp.route("/jobs/<job_id>/qpu", methods=["GET"])
@login_required
def qml_list_qpu_runs(job_id: str):
    user = get_current_user()
    parent = get_qml_job(job_id, user_id=user["id"])
    if parent is None:
        return jsonify({"error": "Parent job not found"}), 404
    runs = list_qml_qpu_runs_for_job(job_id)
    return jsonify({"qpu_runs": runs, "total": len(runs)})


@qml_bp.route("/qpu-runs/<run_id>", methods=["GET"])
@login_required
def qml_get_qpu_run(run_id: str):
    user = get_current_user()
    run = get_qml_qpu_run(run_id, user_id=user["id"])
    if run is None:
        return jsonify({"error": "QPU run not found"}), 404
    return jsonify(run)


@qml_bp.route("/qpu-runs/<run_id>/refresh", methods=["POST"])
@login_required
def qml_refresh_qpu_run(run_id: str):
    """Hit IBM, update the local row, return the latest state.

    Frontend polls this every ~5 s while the run is non-terminal. Idempotent
    — calling it on an already-complete row is a no-op that just returns
    the persisted result.
    """
    user = get_current_user()
    run = get_qml_qpu_run(run_id, user_id=user["id"])
    if run is None:
        return jsonify({"error": "QPU run not found"}), 404

    if run["status"] in {"complete", "error"}:
        return jsonify({"qpu_run": run})  # terminal — return as-is

    cloud_job_id = run.get("cloud_job_id")
    if not cloud_job_id:
        return jsonify({"qpu_run": run})  # not yet submitted

    # Reload the parent's test split so we can materialize.
    parent = get_qml_job(run["qml_job_id"], user_id=user["id"])
    if parent is None:
        return jsonify({"error": "Parent job missing"}), 500
    split = _load_test_split_from_metrics(parent.get("metrics"))
    if split is None:
        return jsonify({"error": "Parent metrics unavailable"}), 500
    X_test, y_test, weights, bias = split

    provider = run.get("provider") or "ibmq"
    provider_key = "ibm_quantum" if provider == "ibmq" else "originqc"
    cipher = get_api_key_ciphertext(user["id"], provider_key)
    if cipher is None:
        return jsonify({
            "error": f"{provider_key} key was removed mid-run",
        }), 400
    api_key = decrypt_api_key(cipher, KEY_ENCRYPTION_SECRET).strip()

    try:
        if provider == "ibmq":
            from app.qml.qpu_ibmq import try_materialize as ibm_try
            outcome = ibm_try(
                api_key=api_key,
                cloud_job_id=cloud_job_id,
                weights=weights,
                bias=bias,
                X_test=X_test,
                y_test=y_test,
                shots=int(run["shots"]),
            )
        elif provider == "originqc":
            from app.qml.qpu_originqc import try_materialize as origin_try
            try:
                ctx = json.loads(run.get("submission_context") or "{}")
            except Exception:
                ctx = {}
            outcome = origin_try(
                api_key=api_key,
                cloud_job_id=cloud_job_id,
                weights=weights,
                bias=bias,
                sample_index=int(ctx.get("sample_index", 0)),
                sample_true_label=int(ctx.get("sample_true_label", 0)),
            )
        else:
            outcome = {
                "terminal": True, "status": "error",
                "error": f"Unknown provider: {provider}",
            }
    except Exception as exc:
        outcome = {
            "terminal": True,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }

    if not outcome.get("terminal"):
        update_qml_qpu_run(
            run_id,
            status=outcome.get("status", "running"),
            live_status=outcome.get("live_status"),
            queue_position=outcome.get("queue_position"),
        )
    elif outcome.get("status") == "complete":
        # Wall time from row creation — IBM/Origin don't surface
        # per-step times consistently across versions.
        try:
            created = datetime.fromisoformat(run["created_at"])
            wall_ms = int((datetime.utcnow() - created).total_seconds() * 1000)
        except Exception:
            wall_ms = None

        m = outcome.get("metrics") or {}
        if provider == "ibmq":
            metrics_payload = {
                "test_accuracy": m.get("test_accuracy"),
                "confusion_matrix": m.get("confusion_matrix"),
                "predictions": m.get("predictions"),
                "probabilities": m.get("probabilities"),
                "backend_name": run.get("backend_name"),
                "shots": int(run["shots"]),
                "is_real_hardware": True,
                "mode": "batch",
            }
        else:  # originqc — single-point payload
            metrics_payload = {
                "sample_index": m.get("sample_index"),
                "sample_true_label": m.get("sample_true_label"),
                "predicted_label": m.get("predicted_label"),
                "prob_class1": m.get("prob_class1"),
                "p_zero_on_q0": m.get("p_zero_on_q0"),
                "expect_z": m.get("expect_z"),
                "correct": m.get("correct"),
                "backend_name": run.get("backend_name"),
                "shots": int(run["shots"]),
                "is_real_hardware": (run.get("backend_name") or "") in (
                    "WK_C180", "HanYuan_01",
                ),
                "mode": "single_point",
            }
        update_qml_qpu_run(
            run_id,
            status="complete",
            metrics=json.dumps(metrics_payload),
            wall_time_ms=wall_ms,
            completed_at=datetime.utcnow().isoformat(),
        )
    else:
        update_qml_qpu_run(
            run_id,
            status="error",
            error=outcome.get("error") or "Unknown cloud error",
            completed_at=datetime.utcnow().isoformat(),
        )

    return jsonify({"qpu_run": get_qml_qpu_run(run_id, user_id=user["id"])})


# ---- QML-7: Public benchmark archive ---------------------------------------


@qml_bp.route("/benchmarks", methods=["GET"])
def qml_benchmarks_list():
    """Public list of every archived training run.

    Supports ``?dataset_id=<id>`` and ``?model=<id>`` filters; defaults
    to newest-first by ``completed_at``. Mirrors the optimization
    side's ``/api/benchmarks`` list endpoint.
    """
    dataset_filter = (request.args.get("dataset_id") or "").strip() or None
    model_filter = (request.args.get("model") or "").strip() or None

    summaries: list[dict] = []
    by_dataset: dict[str, int] = {}
    by_model: dict[str, int] = {}
    for rec in qml_records.iter_records():
        by_dataset[rec.dataset_id] = by_dataset.get(rec.dataset_id, 0) + 1
        by_model[rec.model] = by_model.get(rec.model, 0) + 1
        if dataset_filter and rec.dataset_id != dataset_filter:
            continue
        if model_filter and rec.model != model_filter:
            continue
        summaries.append(qml_records.summarize(rec))

    summaries.sort(key=lambda r: r.get("completed_at", ""), reverse=True)
    return jsonify({
        "records": summaries,
        "total": len(summaries),
        "facets": {
            "datasets": by_dataset,
            "models": by_model,
        },
        "filters": {
            "dataset_id": dataset_filter,
            "model": model_filter,
        },
    })


@qml_bp.route("/benchmarks/<record_id>", methods=["GET"])
def qml_benchmark_detail(record_id: str):
    rec = qml_records.load_record(record_id)
    if rec is None:
        return jsonify({"error": "Record not found"}), 404
    return jsonify(rec.to_dict())


@qml_bp.route("/benchmarks/<record_id>/cite", methods=["GET"])
def qml_benchmark_cite(record_id: str):
    """BibTeX citation for ``record_id``. Returns ``text/plain``."""
    rec = qml_records.load_record(record_id)
    if rec is None:
        return jsonify({"error": "Record not found"}), 404
    return Response(qml_records.bibtex_entry(rec), mimetype="text/plain")


@qml_bp.route("/benchmarks/archive/<job_id>", methods=["POST"])
@admin_required
def qml_benchmark_archive(job_id: str):
    """Snapshot a completed ``qml_jobs`` row into the public archive.

    Admin-only — same gating policy as the optimization benchmark
    archive. Body: ``{"notes": "<optional text>"}``. Returns the
    written record's ``record_id`` + relative archive path.
    """
    parent = get_qml_job(job_id, is_admin=True)
    if parent is None:
        return jsonify({"error": "Job not found"}), 404
    if parent.get("status") != "complete":
        return jsonify({
            "error": "Only completed runs can be archived.",
            "code": "PARENT_NOT_COMPLETE",
        }), 400

    user = get_current_user()
    payload = request.get_json(silent=True) or {}
    notes = (payload.get("notes") or "").strip()

    # Pull display name of the original job's contributor (not the
    # archiving admin). The archive credits the runner, not the
    # gatekeeper.
    from app.models import get_user_by_id
    contributor = get_user_by_id(parent["user_id"])
    contributor_name = (
        contributor.get("display_name") if contributor else None
    ) or (user.get("display_name") if user else "anonymous")

    qpu_run_rows = list_qml_qpu_runs_for_job(job_id)

    record = qml_records.build_record_from_job(
        job_row=parent,
        contributor_display_name=contributor_name,
        qpu_run_rows=qpu_run_rows,
        notes=notes,
    )
    path = qml_records.write_record(record)
    return jsonify({
        "record_id": record.record_id,
        "archive_path": str(path),
        "summary": qml_records.summarize(record),
    }), 201


@qml_bp.route("/benchmarks/<record_id>", methods=["DELETE"])
@admin_required
def qml_benchmark_delete(record_id: str):
    """Admin-only takedown. Use sparingly — records are meant to be
    append-only for a reproducibility audit trail. Documented as an
    operational hatch for bad submissions / leaked tokens."""
    if not qml_records.delete_record(record_id):
        return jsonify({"error": "Record not found"}), 404
    return jsonify({"deleted": True, "record_id": record_id})
