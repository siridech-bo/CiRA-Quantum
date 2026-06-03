"""Trainer orchestrator — bridges the route layer to the VQC.

What this module owns:

* Load the dataset via :mod:`app.qml.data_loader`.
* Apply the user's hyperparameters with sensible defaults + caps.
* Hand the split to :func:`app.qml.vqc.train_vqc`, emitting an event
  per epoch onto the shared :class:`app.pipeline.events.EventBus`.
* Update the ``qml_jobs`` row at start, mid-training (training_history
  JSON), and on completion.

Why this is its own module: the route handler shouldn't know about
PennyLane, NumPy, or the dataset shape. It posts the job, hands off
to a thread that runs ``run_training_job(...)``, and gets out of the
way. Mirrors the optimization side's :class:`app.pipeline.orchestrator.Orchestrator`.
"""

from __future__ import annotations

import json
import threading
import traceback
from datetime import datetime
from typing import Any

from app.models import update_qml_job
from app.pipeline.events import EventBus
from app.qml import data_loader
from app.qml.baselines import serialize as serialize_baselines
from app.qml.baselines import train_baselines
from app.qml.vqc import VQCConfig, train_vqc


def _build_circuit_info_dict(cfg: VQCConfig) -> dict[str, Any]:
    """Build the same shape as :class:`vqc.CircuitInfo` but as a JSON-
    serialisable dict, so the route can emit it on the SSE stream
    *before* training starts — students see the circuit immediately.
    """
    return {
        "backend_name": "PennyLane default.qubit",
        "backend_kind": "statevector",
        "is_real_hardware": False,
        "n_qubits": cfg.n_qubits,
        "n_layers": cfg.n_layers,
        "n_trainable_params": cfg.n_layers * cfg.n_qubits + 1,
        "encoding": "AngleEmbedding (RX rotations, one per input feature)",
        "entangler": "BasicEntanglerLayers (RY rotations + ring CNOT)",
        "measurement": "PauliZ expectation on qubit 0",
        "shots": None,
    }

# Hard caps. Anything above ``MAX_QUBITS`` blows the statevector
# simulator. ``MAX_EPOCHS`` keeps the simulator-bound circuit honest;
# even on toy datasets a 200-epoch run pushes into minutes.
MAX_QUBITS = 10
MAX_EPOCHS = 200


def _coerce_config(payload: dict[str, Any], dataset_n_features: int) -> VQCConfig:
    """Build a VQCConfig from the user's POST body, with caps."""
    n_qubits = int(payload.get("n_qubits") or dataset_n_features)
    n_qubits = max(1, min(n_qubits, MAX_QUBITS, dataset_n_features))
    n_layers = max(1, min(int(payload.get("n_layers") or 2), 6))
    n_epochs = max(1, min(int(payload.get("n_epochs") or 30), MAX_EPOCHS))
    batch_size = max(1, min(int(payload.get("batch_size") or 16), 64))
    learning_rate = float(payload.get("learning_rate") or 0.05)
    seed = int(payload.get("seed") or 42)
    return VQCConfig(
        n_qubits=n_qubits,
        n_layers=n_layers,
        n_epochs=n_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
    )


def run_training_job(
    *,
    job_id: str,
    dataset_id: str,
    hyperparameters: dict[str, Any],
    event_bus: EventBus,
) -> None:
    """Synchronous training run. Called from a daemon thread by the route.

    Every status change is mirrored into both the DB (so the detail
    page survives a page refresh mid-training) and the event bus (so
    the open SSE stream picks it up live).
    """
    try:
        event_bus.emit(job_id, "loading", message="Loading dataset…")

        # Cap features so the statevector simulator stays sane.
        requested_n_qubits = hyperparameters.get("n_qubits")
        max_qubits_hint = (
            int(requested_n_qubits)
            if requested_n_qubits is not None
            else MAX_QUBITS
        )
        split = data_loader.load(
            dataset_id,
            max_qubits=min(max_qubits_hint, MAX_QUBITS),
            n_samples_cap=hyperparameters.get("n_samples_cap"),
        )

        cfg = _coerce_config(hyperparameters, split.n_features)
        circuit_info = _build_circuit_info_dict(cfg)
        update_qml_job(
            job_id,
            status="training",
            hyperparameters=json.dumps({
                "n_qubits": cfg.n_qubits,
                "n_layers": cfg.n_layers,
                "n_epochs": cfg.n_epochs,
                "batch_size": cfg.batch_size,
                "learning_rate": cfg.learning_rate,
                "seed": cfg.seed,
                "pca_applied": split.pca_applied,
                "n_samples_train": int(split.X_train.shape[0]),
                "n_samples_test": int(split.X_test.shape[0]),
                "feature_names": split.feature_names,
                "classes": split.classes,
                "circuit_info": circuit_info,
            }),
        )
        event_bus.emit(
            job_id,
            "training",
            n_qubits=cfg.n_qubits,
            n_layers=cfg.n_layers,
            n_epochs=cfg.n_epochs,
            n_samples_train=int(split.X_train.shape[0]),
            n_samples_test=int(split.X_test.shape[0]),
            pca_applied=split.pca_applied,
            notes=split.notes,
            feature_names=split.feature_names,
            classes=split.classes,
            circuit_info=circuit_info,
        )

        # Running history list so the DB row reflects partial progress
        # even if the process restarts mid-training.
        live_history: list[dict[str, float]] = []

        def on_epoch(epoch_idx: int, metrics: dict[str, float]) -> None:
            live_history.append(metrics)
            event_bus.emit(
                job_id,
                "epoch",
                epoch=metrics["epoch"],
                total_epochs=cfg.n_epochs,
                loss=metrics["loss"],
                train_accuracy=metrics["train_accuracy"],
                test_accuracy=metrics["test_accuracy"],
            )
            # Persist partial progress every 5 epochs to keep DB writes
            # bounded — full history at completion below.
            if (epoch_idx + 1) % 5 == 0 or (epoch_idx + 1) == cfg.n_epochs:
                update_qml_job(
                    job_id,
                    training_history=json.dumps(live_history),
                )

        # Decision-boundary live updates. Only meaningful for n_qubits=2
        # (the input space the VQC actually sees). We push a fresh grid
        # snapshot every 3 epochs while training, so the student watches
        # the boundary curve form — the centerpiece pedagogical signal.
        def on_decision_grid(epoch_idx: int, grid: dict[str, Any]) -> None:
            event_bus.emit(
                job_id,
                "decision_grid",
                epoch=epoch_idx + 1,
                total_epochs=cfg.n_epochs,
                grid=grid,
            )

        result = train_vqc(
            split.X_train, split.y_train,
            split.X_test, split.y_test,
            config=cfg,
            on_epoch=on_epoch,
            notes=split.notes,
            decision_grid_every=3 if cfg.n_qubits == 2 else 0,
            decision_grid_resolution=20,
            on_decision_grid=on_decision_grid,
        )
        # Pull circuit_info off the result (dataclass), fall back to the
        # pre-training shape if anything was missed.
        if result.circuit_info is not None:
            ci = result.circuit_info
            result_circuit_info = {
                "backend_name": ci.backend_name,
                "backend_kind": ci.backend_kind,
                "is_real_hardware": ci.is_real_hardware,
                "n_qubits": ci.n_qubits,
                "n_layers": ci.n_layers,
                "n_trainable_params": ci.n_trainable_params,
                "encoding": ci.encoding,
                "entangler": ci.entangler,
                "measurement": ci.measurement,
                "shots": ci.shots,
            }
        else:
            result_circuit_info = circuit_info

        # QML-3 — classical baselines on the SAME train/test split. Run
        # after the VQC so the loss curve has already streamed and the
        # comparison table is the *last* thing the student sees. The four
        # baselines together take well under a second on the curated
        # datasets, so we don't bother fanning them out across threads.
        event_bus.emit(
            job_id,
            "baselines_training",
            message=("Training classical baselines on the same split "
                     "(Logistic Regression, SVM-RBF, Random Forest, MLP)…"),
        )
        baseline_seed = int(hyperparameters.get("seed") or 42)
        baselines = train_baselines(
            split.X_train, split.y_train,
            split.X_test, split.y_test,
            seed=baseline_seed,
        )
        baselines_serialized = serialize_baselines(baselines)

        # Serialize the training scatter (train + test points) when we
        # have a 2D input space — paired with the decision grid these
        # let the detail page render the boundary heatmap with the
        # actual samples overlaid, no extra preview round-trip.
        scatter_points: list[dict[str, Any]] | None = None
        if cfg.n_qubits == 2 and split.X_train.shape[1] == 2:
            scatter_points = []
            for i in range(split.X_train.shape[0]):
                scatter_points.append({
                    "x": round(float(split.X_train[i, 0]), 4),
                    "y": round(float(split.X_train[i, 1]), 4),
                    "label": int(split.y_train[i]),
                    "split": "train",
                })
            for i in range(split.X_test.shape[0]):
                scatter_points.append({
                    "x": round(float(split.X_test[i, 0]), 4),
                    "y": round(float(split.X_test[i, 1]), 4),
                    "label": int(split.y_test[i]),
                    "split": "test",
                })

        result_dict = {
            "final_train_accuracy": result.final_train_accuracy,
            "final_test_accuracy": result.final_test_accuracy,
            "final_loss": result.final_loss,
            "confusion_matrix": result.confusion_matrix,
            "weights": result.weights,
            "bias": result.bias,
            "n_qubits": result.n_qubits,
            "pca_applied": split.pca_applied,
            "classes": split.classes,
            "feature_names": split.feature_names,
            "notes": result.notes,
            "train_time_ms": result.train_time_ms,
            "circuit_info": result_circuit_info,
            "baselines": baselines_serialized,
            "decision_grid": result.final_decision_grid,
            "scatter_points": scatter_points,
        }
        update_qml_job(
            job_id,
            status="complete",
            metrics=json.dumps(result_dict),
            training_history=json.dumps(result.history),
            train_time_ms=int(result.train_time_ms),
            completed_at=datetime.utcnow().isoformat(),
        )
        event_bus.emit(job_id, "complete", **result_dict)

    except Exception as exc:  # broad on purpose — orchestrator owns failure
        message = f"{type(exc).__name__}: {exc}"
        tb = traceback.format_exc()
        update_qml_job(
            job_id,
            status="error",
            error=message,
            completed_at=datetime.utcnow().isoformat(),
        )
        event_bus.emit(job_id, "error", message=message, traceback=tb)


def launch_training_thread(
    *,
    job_id: str,
    dataset_id: str,
    hyperparameters: dict[str, Any],
    event_bus: EventBus,
) -> threading.Thread:
    """Fire-and-forget daemon thread. The route returns immediately so
    the client can open the SSE stream and watch training live."""
    t = threading.Thread(
        target=run_training_job,
        kwargs={
            "job_id": job_id,
            "dataset_id": dataset_id,
            "hyperparameters": hyperparameters,
            "event_bus": event_bus,
        },
        daemon=True,
        name=f"qml-train-{job_id[:8]}",
    )
    t.start()
    return t


# ---- QML-8: env-driven launcher selection ---------------------------------


def _use_redis_queue() -> bool:
    """Same env flag the optimization side uses, so a single
    ``USE_REDIS_QUEUE=1`` toggles both apps simultaneously."""
    import os
    return (os.environ.get("USE_REDIS_QUEUE") or "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def launch_training_job(
    *,
    job_id: str,
    dataset_id: str,
    hyperparameters: dict[str, Any],
    event_bus: EventBus,
) -> str:
    """Single entry the route layer calls. Returns a short tag of
    which execution path was taken (``"thread"`` or ``"queue"``) so
    the route can log it. Picks RQ over a thread when
    ``USE_REDIS_QUEUE=1`` is set in the environment.

    The RQ path requires a running ``rq`` worker subscribed to the
    same queue. Falls back to the threaded path if Redis init fails
    so a misconfigured env doesn't break the user-visible path.
    """
    if _use_redis_queue():
        try:
            import os
            import redis
            from rq import Queue

            url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
            queue_name = os.environ.get(
                "QML_QUEUE_NAME",
                os.environ.get("QUEUE_NAME", "cira-quantum-solves"),
            )
            conn = redis.from_url(url)
            queue = Queue(queue_name, connection=conn, default_timeout=3600)

            from app.qml.worker_entry import run_qml_training_job
            queue.enqueue(
                run_qml_training_job,
                kwargs={
                    "job_id": job_id,
                    "dataset_id": dataset_id,
                    "hyperparameters": hyperparameters,
                },
                job_id=f"qml-{job_id}",
                job_timeout=3600,
            )
            return "queue"
        except Exception:  # noqa: BLE001 — log + fall back
            import logging
            logging.getLogger(__name__).exception(
                "RQ enqueue for QML failed; falling back to threaded path."
            )

    launch_training_thread(
        job_id=job_id,
        dataset_id=dataset_id,
        hyperparameters=hyperparameters,
        event_bus=event_bus,
    )
    return "thread"
