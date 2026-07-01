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
from typing import Any, Literal

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
    # Ansatz selection. ``basic_ry`` is the original educational path
    # (BasicEntanglerLayers: scalar RY per qubit + ring CNOTs, trained
    # with Adam). ``su2_brick`` is the research-tier alternate where
    # each per-qubit block is a full SU(2) gate and updates use the
    # Riemannian (Stiefel-manifold) projection + polar retraction from
    # Guo & Yang 2025, arXiv:2501.07387v2. The educational lesson runs
    # on ``basic_ry``; ``su2_brick`` exists for research comparisons.
    ansatz: Literal["basic_ry", "su2_brick"] = "basic_ry"
    # Optimizer choice for the ``su2_brick`` path only. ``sgd`` is the
    # vanilla Riemannian gradient descent in Guo & Yang's setup; ``adam``
    # adds a per-gate first-moment buffer + scalar second moment with
    # retraction-compatible vector transport (Becigneul & Ganea 2018).
    # Ignored for ``basic_ry`` (which uses PennyLane's AdamOptimizer
    # internally). Default ``sgd`` preserves the existing head-to-head
    # numbers; flip to ``adam`` for an apples-to-apples comparison
    # against basic_ry's Adam path.
    momentum: Literal["sgd", "adam"] = "sgd"


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
    # ``su2_brick`` ansatz only. Shape (n_layers, n_qubits, 2, 2, 2) —
    # the trailing axis is [real, imag] for each complex entry. ``None``
    # for the default ``basic_ry`` path, where ``weights`` carries the
    # scalar RY angles instead. Kept separate so the existing ``weights``
    # contract (n_layers × n_qubits scalars) isn't broken for callers
    # that consume it directly.
    su2_unitaries: list | None = None


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
    if config.ansatz == "su2_brick":
        return _train_su2_brick(
            X_train, y_train, X_test, y_test,
            config=config,
            on_epoch=on_epoch,
            notes=notes,
            decision_grid_every=decision_grid_every,
            decision_grid_resolution=decision_grid_resolution,
            on_decision_grid=on_decision_grid,
        )

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


def _serialize_unitaries(U: np.ndarray) -> list:
    """Flatten complex (n_layers, n_qubits, 2, 2) into JSON-safe nested
    lists of [real, imag] pairs. Shape on the wire is
    (n_layers, n_qubits, 2, 2, 2) with the trailing axis = [re, im].
    """
    out: list = []
    n_layers, n_qubits, _, _ = U.shape
    for layer in range(n_layers):
        layer_block: list = []
        for q in range(n_qubits):
            mat: list = []
            for i in range(2):
                row: list = []
                for j in range(2):
                    z = U[layer, q, i, j]
                    row.append([float(z.real), float(z.imag)])
                mat.append(row)
            layer_block.append(mat)
        out.append(layer_block)
    return out


def _train_su2_brick(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    config: VQCConfig,
    on_epoch: ProgressFn | None,
    notes: list[str] | None,
    decision_grid_every: int,
    decision_grid_resolution: int,
    on_decision_grid,
) -> TrainResult:
    """SU(2) brick-wall VQC trained with Riemannian gradient descent.

    Per-qubit blocks are full ``QubitUnitary`` 2×2 gates (3 real DOF
    each, parameterized as the matrix itself). After every minibatch
    step the Euclidean gradient dL/dU is projected onto the tangent
    space of U(2) at the current iterate and the update is mapped
    back to the manifold via a polar retraction. The bias scalar is
    updated with vanilla SGD because it lives in Euclidean space.

    Following Guo & Yang 2025 (arXiv:2501.07387v2), eq. (4). The
    pure-numpy primitives live in ``app.qml.ansatz_su2`` and have their
    own unit tests; this function wires them into the PennyLane
    AngleEmbedding → entangler → PauliZ readout pipeline that the rest
    of the QML module is built around.

    NOT a drop-in replacement for the default ``basic_ry`` path — the
    weights serialization shape differs (see ``TrainResult.su2_unitaries``)
    and there's no Adam momentum on the bias. Use this for research
    comparisons, not for the introductory lesson.
    """
    import pennylane as qml
    from pennylane import numpy as pnp

    from app.qml.ansatz_su2 import (
        init_near_identity,
        retract_polar,
        riemannian_adam_step,
        riemannian_grad,
    )

    n_qubits = config.n_qubits
    n_layers = config.n_layers
    lr = config.learning_rate
    rng = np.random.default_rng(config.seed)

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="autograd")
    def circuit(x, unitaries):
        qml.AngleEmbedding(x, wires=range(n_qubits), rotation="X")
        for layer in range(n_layers):
            for q in range(n_qubits):
                qml.QubitUnitary(unitaries[layer, q], wires=[q])
            # Ring-CNOT entangler (matches BasicEntanglerLayers' topology
            # so the two ansätze are comparable depth-for-depth).
            for q in range(n_qubits):
                qml.CNOT(wires=[q, (q + 1) % n_qubits])
        return qml.expval(qml.PauliZ(0))

    # Initialize near identity. See init_near_identity for the rationale —
    # Haar-random init triggers barren-plateau-flavored gradient signal.
    U_state = np.empty((n_layers, n_qubits, 2, 2), dtype=np.complex128)
    for layer in range(n_layers):
        for q in range(n_qubits):
            U_state[layer, q] = init_near_identity(rng, scale=0.1)
    bias_val = 0.0

    # Adam state — only used when config.momentum == "adam". One first-
    # moment buffer per gate, scalar second moment per gate, single step
    # counter. Initializing m to zeros is the standard Adam convention
    # and is also a no-op tangent vector (so the t=1 transport is free).
    use_adam = config.momentum == "adam"
    adam_m = np.zeros_like(U_state) if use_adam else None
    adam_v = np.zeros((n_layers, n_qubits)) if use_adam else None
    adam_t = 0

    def loss_fn(unitaries, bias, X_batch, y_batch):
        eps = 1e-9
        total = 0.0
        for x, y in zip(X_batch, y_batch):
            raw = circuit(x, unitaries)
            prob = 1.0 / (1.0 + pnp.exp(-(raw + bias)))
            prob = pnp.clip(prob, eps, 1 - eps)
            total = total + (-(y * pnp.log(prob) + (1 - y) * pnp.log(1 - prob)))
        return total / len(X_batch)

    grad_fn = qml.grad(loss_fn, argnums=[0, 1])

    history: list[dict[str, float]] = []
    start = time.perf_counter()
    n = X_train.shape[0]

    for epoch in range(config.n_epochs):
        perm = rng.permutation(n)
        for start_i in range(0, n, config.batch_size):
            batch_idx = perm[start_i:start_i + config.batch_size]
            X_b = X_train[batch_idx]
            y_b = y_train[batch_idx].astype(np.float64)

            # PennyLane needs fresh pnp-wrapped inputs each step so the
            # autograd tape is rebuilt — re-wrapping after the manifold
            # update is the cleanest way to do that.
            U_pnp = pnp.array(U_state, requires_grad=True)
            b_pnp = pnp.array(bias_val, requires_grad=True)
            g_U, g_b = grad_fn(U_pnp, b_pnp, X_b, y_b)

            # Riemannian update on each per-qubit gate.
            #
            # PennyLane's autograd returns the "Wirtinger-conjugate"
            # gradient G for real-valued loss over complex U: i.e. the
            # tangent of f along an arbitrary dU is ``Re tr(G^T · dU)``.
            # The Euclidean steepest-ascent direction in the Frobenius
            # real inner product ``<A,B>_R = Re tr(A† · B)`` is therefore
            # ``G.conj()``. Skipping this conjugation does *gradient
            # ascent* on the loss — caught by the smoke test before the
            # formal suite was written.
            if use_adam:
                adam_t += 1
                for layer in range(n_layers):
                    for q in range(n_qubits):
                        U = U_state[layer, q]
                        G = np.asarray(g_U[layer, q]).conj()
                        U_new, m_new, v_new = riemannian_adam_step(
                            U, G,
                            m=adam_m[layer, q],
                            v=float(adam_v[layer, q]),
                            t=adam_t,
                            lr=lr,
                        )
                        U_state[layer, q] = U_new
                        adam_m[layer, q] = m_new
                        adam_v[layer, q] = v_new
            else:
                for layer in range(n_layers):
                    for q in range(n_qubits):
                        U = U_state[layer, q]
                        G = np.asarray(g_U[layer, q]).conj()
                        T = riemannian_grad(U, G)
                        U_state[layer, q] = retract_polar(U, lr * T)
            # Euclidean SGD on the scalar bias. Adam on the bias is
            # left as future work — empirically the bias converges
            # fast and isn't the bottleneck on the losses we see.
            bias_val = bias_val - lr * float(g_b)

        # Whole-set eval so the curve matches what basic_ry plots.
        train_probs = _model_predict_proba(circuit, X_train, U_state, bias_val)
        test_probs = _model_predict_proba(circuit, X_test, U_state, bias_val)
        epoch_metrics = {
            "epoch": epoch + 1,
            "loss": _binary_cross_entropy(train_probs, y_train),
            "train_accuracy": _accuracy(train_probs, y_train),
            "test_accuracy": _accuracy(test_probs, y_test),
        }
        history.append(epoch_metrics)
        if on_epoch is not None:
            on_epoch(epoch, epoch_metrics)

        if (
            n_qubits == 2
            and decision_grid_every > 0
            and on_decision_grid is not None
            and X_train.shape[1] == 2
            and ((epoch + 1) % decision_grid_every == 0
                 or (epoch + 1) == config.n_epochs)
        ):
            grid = compute_decision_grid(
                circuit, U_state, bias_val,
                X_train, resolution=decision_grid_resolution,
            )
            on_decision_grid(epoch, grid)

    final_train_probs = _model_predict_proba(circuit, X_train, U_state, bias_val)
    final_test_probs = _model_predict_proba(circuit, X_test, U_state, bias_val)
    train_time_ms = int((time.perf_counter() - start) * 1000)

    circuit_info = CircuitInfo(
        backend_name="PennyLane default.qubit",
        backend_kind="statevector",
        is_real_hardware=False,
        n_qubits=n_qubits,
        n_layers=n_layers,
        # Three real DOF per SU(2) gate (the matrix has 8 reals but
        # 5 are eaten by unitarity + global phase), plus the bias scalar.
        n_trainable_params=3 * n_layers * n_qubits + 1,
        encoding="AngleEmbedding (RX rotations, one per input feature)",
        entangler="SU(2) brick-wall (Riemannian, ring CNOT)",
        measurement="PauliZ expectation on qubit 0",
        shots=None,
    )

    final_grid: dict | None = None
    if n_qubits == 2 and X_train.shape[1] == 2:
        final_grid = compute_decision_grid(
            circuit, U_state, bias_val,
            X_train, resolution=decision_grid_resolution,
        )

    return TrainResult(
        final_train_accuracy=_accuracy(final_train_probs, y_train),
        final_test_accuracy=_accuracy(final_test_probs, y_test),
        final_loss=_binary_cross_entropy(final_train_probs, y_train),
        history=history,
        # `weights` is the basic_ry contract — kept empty here so any
        # caller iterating the field sees an unambiguous "no scalar
        # angles" rather than mismatched shape.
        weights=[],
        bias=float(bias_val),
        confusion_matrix=_confusion(final_test_probs, y_test),
        pca_applied=False,
        n_qubits=n_qubits,
        train_time_ms=train_time_ms,
        circuit_info=circuit_info,
        notes=list(notes or []),
        final_decision_grid=final_grid,
        su2_unitaries=_serialize_unitaries(U_state),
    )
