"""Variational Quantum Classifier — PennyLane statevector training.

The model in one sentence: an ``AngleEmbedding`` (one feature per qubit
as an RX rotation), followed by ``n_layers`` of ``BasicEntanglerLayers``
(parameterized RY rotations + ring CNOTs), measured as the expectation
of ``PauliZ`` on the first qubit. A trainable bias is added to that
expectation, then a sigmoid turns it into a class probability.

Trained with the Adam optimizer on binary cross-entropy. After each
epoch, the trainer emits an event with the running loss + train/test
accuracy — the frontend renders these as a live loss curve.

Why this architecture:

* ``AngleEmbedding`` is the simplest ansatz that teaches the
  "one-feature-one-qubit" rule. More-expressive encodings (amplitude,
  IQP) are deferred to QML-4 / QML-5 once students grok this one.
* ``BasicEntanglerLayers`` is PennyLane's stock entangler block. Single-
  parameter rotations per qubit + ring of CNOTs. Trainable count is
  small (``n_layers × n_qubits``), so Adam converges in 30–80 epochs
  even on the harder datasets.
* PauliZ⟨0⟩ ∈ [-1, 1] → bias + sigmoid → probability. A single output
  qubit is sufficient for binary classification and keeps the gradient
  expressions short. Multi-class generalization (one-hot via N output
  qubits) is straightforward but deferred to QML-3 / QML-4.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class VQCConfig:
    """Hyperparameters surfaced to the user."""

    n_qubits: int = 2          # set to dataset.n_features at call time
    n_layers: int = 2
    n_epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 0.05
    seed: int = 42


@dataclass
class CircuitInfo:
    """Description of the *actual* quantum circuit + the backend that
    ran it. Surfaced verbatim to the frontend so a student can see
    the gates, wires, and execution mode — not just the loss curve.

    ``backend_kind`` is the educationally-important field: ``statevector``
    is what we run on locally (exact, no shot noise, no queue). When a
    QML-5 / QML-6 cloud run lands, this flips to ``qpu`` and the frontend
    surfaces shots + queue position instead.
    """

    backend_name: str
    backend_kind: str  # 'statevector' | 'shot_simulator' | 'qpu'
    is_real_hardware: bool
    n_qubits: int
    n_layers: int
    n_trainable_params: int
    encoding: str         # human-readable, e.g. "AngleEmbedding (RX)"
    entangler: str        # e.g. "BasicEntanglerLayers (RY + ring CNOT)"
    measurement: str      # e.g. "PauliZ on qubit 0"
    shots: int | None = None  # None for statevector


@dataclass
class TrainResult:
    """What the trainer returns and what the route stores in qml_jobs."""

    final_train_accuracy: float
    final_test_accuracy: float
    final_loss: float
    history: list[dict[str, float]]
    weights: list[list[float]]  # serialisable; shape (n_layers, n_qubits)
    bias: float
    confusion_matrix: list[list[int]]  # 2x2: [[TN, FP], [FN, TP]]
    pca_applied: bool
    n_qubits: int
    train_time_ms: int
    circuit_info: CircuitInfo | None = None
    notes: list[str] = field(default_factory=list)
    # Final-epoch decision-boundary snapshot for 2-qubit jobs. ``None``
    # for higher-dim, where there is no 2D space to plot the boundary in.
    final_decision_grid: dict | None = None


# Type alias for the per-epoch callback. The trainer fires this after
# every full pass through the data so the route can emit an SSE event.
ProgressFn = Callable[[int, dict[str, float]], None]


def _build_circuit(n_qubits: int, n_layers: int):
    """Lazy-build the qnode. Imported here so ``app.qml`` doesn't pull
    PennyLane in just to register the dataset gallery."""
    import pennylane as qml

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="autograd")
    def circuit(x, weights):
        qml.AngleEmbedding(x, wires=range(n_qubits), rotation="X")
        qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
        return qml.expval(qml.PauliZ(0))

    return circuit


def _model_predict_proba(circuit, X, weights, bias):
    """σ(<Z₀> + bias). Returns float probabilities in [0, 1]."""
    raw = np.array([float(circuit(x, weights)) for x in X])
    return 1.0 / (1.0 + np.exp(-(raw + bias)))


def _binary_cross_entropy(probs: np.ndarray, y: np.ndarray) -> float:
    eps = 1e-9
    p = np.clip(probs, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _accuracy(probs: np.ndarray, y: np.ndarray) -> float:
    preds = (probs >= 0.5).astype(np.int64)
    return float(np.mean(preds == y))


def _confusion(probs: np.ndarray, y: np.ndarray) -> list[list[int]]:
    preds = (probs >= 0.5).astype(np.int64)
    tn = int(np.sum((preds == 0) & (y == 0)))
    fp = int(np.sum((preds == 1) & (y == 0)))
    fn = int(np.sum((preds == 0) & (y == 1)))
    tp = int(np.sum((preds == 1) & (y == 1)))
    return [[tn, fp], [fn, tp]]


def compute_decision_grid(
    circuit,
    weights,
    bias,
    X_train: np.ndarray,
    *,
    resolution: int = 20,
    pad: float = 0.5,
) -> dict:
    """Evaluate the VQC on a grid spanning the training data, for the
    decision-boundary visualization.

    Only meaningful when ``X_train`` is 2-dimensional (n_qubits == 2).
    Returns a flat ``probabilities`` array (length ``resolution²``)
    in row-major order plus the axis bounds so the frontend can render
    a heatmap behind the scatter plot.

    ``pad`` extends the grid slightly past the data range so the
    boundary lines don't clip at the edges of the visible points.
    """
    if X_train.ndim != 2 or X_train.shape[1] != 2:
        raise ValueError("decision grid only works in 2D input space")
    x_min, x_max = float(X_train[:, 0].min()) - pad, float(X_train[:, 0].max()) + pad
    y_min, y_max = float(X_train[:, 1].min()) - pad, float(X_train[:, 1].max()) + pad
    xs = np.linspace(x_min, x_max, resolution)
    ys = np.linspace(y_min, y_max, resolution)
    probs: list[float] = []
    bias_f = float(bias)
    # Row-major: outer loop over rows (y), inner over columns (x).
    for y in ys:
        for x in xs:
            raw = float(circuit(np.array([x, y], dtype=np.float32), weights))
            probs.append(1.0 / (1.0 + np.exp(-(raw + bias_f))))
    return {
        "resolution": resolution,
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        # Round to 4 decimals to keep SSE payload small (~3 KB at 20x20).
        "probabilities": [round(p, 4) for p in probs],
    }


def train_vqc(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    config: VQCConfig,
    on_epoch: ProgressFn | None = None,
    notes: list[str] | None = None,
    decision_grid_every: int = 0,
    decision_grid_resolution: int = 20,
    on_decision_grid=None,
) -> TrainResult:
    """Fit a VQC on ``(X_train, y_train)`` and evaluate on the held-out set.

    Calls ``on_epoch(epoch_index, metrics_dict)`` after each epoch so
    the route layer can emit an SSE event. Setting it to ``None`` (the
    test path) runs the trainer silently.
    """
    import pennylane as qml
    from pennylane import numpy as pnp

    n_qubits = config.n_qubits
    n_layers = config.n_layers
    circuit = _build_circuit(n_qubits, n_layers)

    # PennyLane numpy wrapper so AdamOptimizer can take gradients.
    rng = np.random.default_rng(config.seed)
    init_weights = rng.uniform(-0.5, 0.5, size=(n_layers, n_qubits)).astype(np.float64)
    weights = pnp.array(init_weights, requires_grad=True)
    bias = pnp.array(0.0, requires_grad=True)

    optimizer = qml.AdamOptimizer(stepsize=config.learning_rate)

    def loss_fn(weights, bias, X_batch, y_batch):
        # Manual sum so autograd traces it correctly. Vectorising over
        # X_batch isn't trivial here because each call goes through the
        # qnode — but the simulator is fast enough that the loop is
        # acceptable for the curated 100–500-sample datasets.
        eps = 1e-9
        total = 0.0
        for x, y in zip(X_batch, y_batch):
            raw = circuit(x, weights)
            prob = 1.0 / (1.0 + pnp.exp(-(raw + bias)))
            prob = pnp.clip(prob, eps, 1 - eps)
            total = total + (-(y * pnp.log(prob) + (1 - y) * pnp.log(1 - prob)))
        return total / len(X_batch)

    history: list[dict[str, float]] = []
    start = time.perf_counter()
    n = X_train.shape[0]

    for epoch in range(config.n_epochs):
        # Shuffle indices once per epoch for SGD-style minibatching.
        perm = rng.permutation(n)
        for start_i in range(0, n, config.batch_size):
            batch_idx = perm[start_i:start_i + config.batch_size]
            X_b = X_train[batch_idx]
            y_b = y_train[batch_idx].astype(np.float64)
            (weights, bias, _, _), _ = optimizer.step_and_cost(
                loss_fn, weights, bias, X_b, y_b
            )

        # Eval on the whole train + test sets so the curve is smooth.
        train_probs = _model_predict_proba(circuit, X_train, weights, float(bias))
        test_probs = _model_predict_proba(circuit, X_test, weights, float(bias))
        epoch_metrics = {
            "epoch": epoch + 1,
            "loss": _binary_cross_entropy(train_probs, y_train),
            "train_accuracy": _accuracy(train_probs, y_train),
            "test_accuracy": _accuracy(test_probs, y_test),
        }
        history.append(epoch_metrics)
        if on_epoch is not None:
            on_epoch(epoch, epoch_metrics)

        # Optional decision-boundary snapshot. Only meaningful for
        # n_qubits=2; skipped silently otherwise so this stays a no-op
        # for higher-dim runs.
        if (
            n_qubits == 2
            and decision_grid_every > 0
            and on_decision_grid is not None
            and X_train.shape[1] == 2
            and ((epoch + 1) % decision_grid_every == 0
                 or (epoch + 1) == config.n_epochs)
        ):
            grid = compute_decision_grid(
                circuit, weights, float(bias),
                X_train, resolution=decision_grid_resolution,
            )
            on_decision_grid(epoch, grid)

    final_train_probs = _model_predict_proba(circuit, X_train, weights, float(bias))
    final_test_probs = _model_predict_proba(circuit, X_test, weights, float(bias))
    train_time_ms = int((time.perf_counter() - start) * 1000)

    circuit_info = CircuitInfo(
        backend_name="PennyLane default.qubit",
        backend_kind="statevector",
        is_real_hardware=False,
        n_qubits=n_qubits,
        n_layers=n_layers,
        # +1 for the trainable scalar bias added after the PauliZ readout.
        n_trainable_params=n_layers * n_qubits + 1,
        encoding="AngleEmbedding (RX rotations, one per input feature)",
        entangler="BasicEntanglerLayers (RY rotations + ring CNOT)",
        measurement="PauliZ expectation on qubit 0",
        shots=None,  # statevector = exact, no shots
    )

    # Final decision-grid snapshot for the persisted metrics. Always
    # computed for 2-qubit jobs (regardless of decision_grid_every) so
    # the detail page can show the final boundary even when the user
    # opens it after training finished.
    final_grid: dict | None = None
    if n_qubits == 2 and X_train.shape[1] == 2:
        final_grid = compute_decision_grid(
            circuit, weights, float(bias),
            X_train, resolution=decision_grid_resolution,
        )

    return TrainResult(
        final_train_accuracy=_accuracy(final_train_probs, y_train),
        final_test_accuracy=_accuracy(final_test_probs, y_test),
        final_loss=_binary_cross_entropy(final_train_probs, y_train),
        history=history,
        weights=np.asarray(weights).tolist(),
        bias=float(bias),
        confusion_matrix=_confusion(final_test_probs, y_test),
        pca_applied=False,  # caller patches this from the DatasetSplit
        n_qubits=n_qubits,
        train_time_ms=train_time_ms,
        circuit_info=circuit_info,
        notes=list(notes or []),
        final_decision_grid=final_grid,
    )
