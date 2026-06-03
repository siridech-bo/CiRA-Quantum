"""Dataset loader for the QML training pipeline.

Every dataset in :mod:`app.qml.datasets` resolves through this module to
a uniform ``DatasetSplit`` shape:

* ``X_train``, ``X_test`` — float32 arrays, standard-scaled.
* ``y_train``, ``y_test`` — int64 arrays in ``{0, 1}`` (every dataset is
  binary in this pipeline; multiclass datasets like Iris are restricted
  to the first two classes).
* ``feature_names`` — human-readable column names (for the
  decision-boundary plot's axis labels in QML-4).

Why standard-scale + PCA-cap to ``n_features ≤ max_qubits``:

PennyLane's ``AngleEmbedding`` encodes one feature per qubit. We never
want to train a 30-qubit statevector circuit for breast-cancer — the
statevector blows up at ~25 qubits. The loader takes an optional
``max_qubits`` cap and, when the dataset exceeds it, applies PCA to
project down. The trainer surfaces this so students see it happened.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DatasetSplit:
    """Loader output. Every field is small enough to serialize for tests."""

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    classes: list[str]
    n_features: int
    pca_applied: bool
    notes: list[str]


# Random seed used by every split. Kept low and constant so two students
# running the same dataset get the same train/test partition — important
# for reproducible educational comparisons (TA: "your accuracy should
# match mine for the same hyperparameters").
_RANDOM_STATE = 42
_TEST_SIZE = 0.25


def load(
    dataset_id: str,
    *,
    max_qubits: int | None = None,
    n_samples_cap: int | None = None,
) -> DatasetSplit:
    """Load and prep a registered dataset.

    Parameters
    ----------
    dataset_id : str
        One of the IDs in :mod:`app.qml.datasets`.
    max_qubits : int | None
        Cap the number of features (= number of qubits in
        ``AngleEmbedding``). If the raw dataset has more features, PCA
        is applied and ``pca_applied`` is set to ``True`` in the result.
        ``None`` means no cap.
    n_samples_cap : int | None
        Subsample the dataset to this many points before splitting.
        Helps keep training times reasonable on the simulator. ``None``
        means use the full dataset.
    """
    from sklearn.decomposition import PCA
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X, y, feature_names, classes = _raw_load(dataset_id)
    notes: list[str] = []

    # Optional subsample. We do this before the split so the train/test
    # ratio stays the same.
    if n_samples_cap is not None and X.shape[0] > n_samples_cap:
        rng = np.random.default_rng(_RANDOM_STATE)
        idx = rng.choice(X.shape[0], size=n_samples_cap, replace=False)
        X, y = X[idx], y[idx]
        notes.append(f"Subsampled to {n_samples_cap} points.")

    # Standardize. Necessary so ``AngleEmbedding`` rotations stay in a
    # reasonable range and gradients are well-conditioned.
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    pca_applied = False
    if max_qubits is not None and X.shape[1] > max_qubits:
        pca = PCA(n_components=max_qubits, random_state=_RANDOM_STATE)
        X = pca.fit_transform(X).astype(np.float32)
        feature_names = [f"PC{i + 1}" for i in range(max_qubits)]
        pca_applied = True
        notes.append(
            f"PCA projected to {max_qubits} components "
            f"(retained variance: {sum(pca.explained_variance_ratio_):.1%})."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y.astype(np.int64),
        test_size=_TEST_SIZE,
        random_state=_RANDOM_STATE,
        stratify=y,
    )

    return DatasetSplit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
        classes=classes,
        n_features=X.shape[1],
        pca_applied=pca_applied,
        notes=notes,
    )


# ---- Raw loaders -----------------------------------------------------------


def _raw_load(dataset_id: str) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Return (X, y, feature_names, class_names) for ``dataset_id``.

    Every dataset in this module is restricted to two classes. Multiclass
    sources (Iris, Wine) are filtered to the first two classes the source
    file exposes.
    """
    loader = _LOADERS.get(dataset_id)
    if loader is None:
        raise ValueError(f"Unknown QML dataset: {dataset_id}")
    return loader()


def _load_moons() -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    from sklearn.datasets import make_moons
    X, y = make_moons(n_samples=200, noise=0.15, random_state=_RANDOM_STATE)
    return X, y, ["x", "y"], ["inner", "outer"]


def _load_circles() -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    from sklearn.datasets import make_circles
    X, y = make_circles(n_samples=200, noise=0.08, factor=0.4, random_state=_RANDOM_STATE)
    return X, y, ["x", "y"], ["inner", "outer"]


def _load_iris() -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    from sklearn.datasets import load_iris
    raw = load_iris()
    mask = raw.target < 2  # Setosa (0) vs Versicolor (1) — linearly separable.
    return (
        raw.data[mask],
        raw.target[mask],
        list(raw.feature_names),
        ["setosa", "versicolor"],
    )


def _load_wine() -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    from sklearn.datasets import load_wine
    raw = load_wine()
    mask = raw.target < 2
    return (
        raw.data[mask],
        raw.target[mask],
        list(raw.feature_names),
        [raw.target_names[0], raw.target_names[1]],
    )


def _load_mnist_0v1() -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    from sklearn.datasets import load_digits
    raw = load_digits()
    mask = (raw.target == 0) | (raw.target == 1)
    X = raw.data[mask]
    y = raw.target[mask]
    feature_names = [f"px{i}" for i in range(X.shape[1])]
    return X, y, feature_names, ["digit_0", "digit_1"]


def _load_breast_cancer() -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    from sklearn.datasets import load_breast_cancer
    raw = load_breast_cancer()
    return (
        raw.data,
        raw.target,
        list(raw.feature_names),
        [raw.target_names[0], raw.target_names[1]],
    )


_LOADERS: dict[str, Any] = {
    "moons": _load_moons,
    "circles": _load_circles,
    "iris": _load_iris,
    "wine": _load_wine,
    "mnist_0v1": _load_mnist_0v1,
    "breast_cancer": _load_breast_cancer,
}
