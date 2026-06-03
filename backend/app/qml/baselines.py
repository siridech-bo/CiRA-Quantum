"""Classical-baseline trainers for the QML pipeline.

Why this module exists:

The pedagogical point of a VQC isn't that it's the best classifier —
it's that it's a *different kind of* classifier, with a 5-parameter
hypothesis class on toy datasets where sklearn baselines effortlessly
saturate. To make that comparison honest, we run four canonical
classical models on the *same* train/test split the VQC saw and
surface the results side-by-side:

* **Logistic Regression** — the linear baseline. Beats the VQC on
  linearly-separable datasets (Iris setosa-vs-versicolor) and loses on
  the curved ones (Moons, Circles). Teaches "when is a quantum kernel
  even useful?"
* **SVM-RBF** — the kernel baseline. The RBF kernel is the classical
  analogue of a feature map; it's what a quantum kernel needs to beat.
* **Random Forest** — the ensemble baseline. Robust, no hyperparameter
  tuning, sets a high accuracy bar on the harder real-world datasets
  (Wine, Breast Cancer).
* **MLP** — the neural baseline. A small 2-hidden-layer net is the
  closest classical analogue to a VQC: parametric, gradient-trained,
  non-convex loss. The fairest head-to-head.

Each baseline is tiny on these datasets (< 1 s wall time), so we run all
four sequentially after the VQC completes. The educational payoff is
worth the extra few seconds.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class BaselineResult:
    """One classical baseline's numbers — same shape used in QML-7's archive."""

    name: str           # short id, e.g. "logreg"
    title: str          # display name, e.g. "Logistic Regression"
    library: str        # "scikit-learn"
    version: str        # "1.4.x" — for reproducibility
    family: str         # "linear" | "kernel" | "ensemble" | "neural"
    train_accuracy: float
    test_accuracy: float
    train_time_ms: int
    confusion_matrix: list[list[int]]  # 2x2: [[TN, FP], [FN, TP]]
    notes: str          # one-line "what this baseline brings"


def _confusion(preds: np.ndarray, y: np.ndarray) -> list[list[int]]:
    tn = int(np.sum((preds == 0) & (y == 0)))
    fp = int(np.sum((preds == 1) & (y == 0)))
    fn = int(np.sum((preds == 0) & (y == 1)))
    tp = int(np.sum((preds == 1) & (y == 1)))
    return [[tn, fp], [fn, tp]]


def _fit_and_score(
    estimator: Any,
    name: str,
    title: str,
    family: str,
    notes: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> BaselineResult:
    """Fit one sklearn estimator + record metrics in a uniform shape."""
    import sklearn
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=FutureWarning)
        start = time.perf_counter()
        estimator.fit(X_train, y_train)
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        preds_train = estimator.predict(X_train)
        preds_test = estimator.predict(X_test)

    return BaselineResult(
        name=name,
        title=title,
        library="scikit-learn",
        version=sklearn.__version__,
        family=family,
        train_accuracy=float(np.mean(preds_train == y_train)),
        test_accuracy=float(np.mean(preds_test == y_test)),
        train_time_ms=elapsed_ms,
        confusion_matrix=_confusion(preds_test, y_test),
        notes=notes,
    )


def train_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    seed: int = 42,
) -> list[BaselineResult]:
    """Run all four classical baselines on the given split, in order.

    The seed is plumbed into every estimator that takes one (LogReg's
    solver, RandomForest's bootstrap, MLP's weight init) so two runs
    return identical numbers — same reproducibility contract as the
    VQC.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC

    out: list[BaselineResult] = []

    out.append(_fit_and_score(
        LogisticRegression(max_iter=2000, random_state=seed),
        name="logreg",
        title="Logistic Regression",
        family="linear",
        notes="Linear decision boundary; the floor every other model has to beat.",
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
    ))

    out.append(_fit_and_score(
        SVC(kernel="rbf", gamma="scale", random_state=seed),
        name="svm_rbf",
        title="SVM (RBF kernel)",
        family="kernel",
        notes=("Classical kernel method. What a quantum kernel needs to beat "
               "before the quantum advantage discussion can start."),
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
    ))

    out.append(_fit_and_score(
        RandomForestClassifier(
            n_estimators=100, max_depth=None, random_state=seed, n_jobs=1,
        ),
        name="random_forest",
        title="Random Forest",
        family="ensemble",
        notes=("Tree-bagging ensemble. Strong off-the-shelf baseline on tabular "
               "datasets; sets the bar on Wine + Breast Cancer."),
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
    ))

    # MLP with two hidden layers — closest classical analogue to a VQC
    # (parametric, gradient-trained, non-convex loss).
    out.append(_fit_and_score(
        MLPClassifier(
            hidden_layer_sizes=(16, 8),
            activation="relu",
            solver="adam",
            max_iter=500,
            random_state=seed,
        ),
        name="mlp",
        title="MLP (16-8 ReLU)",
        family="neural",
        notes=("Small feed-forward net. Parametric + gradient-trained, like the "
               "VQC — the fairest head-to-head."),
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
    ))

    return out


def serialize(results: list[BaselineResult]) -> list[dict[str, Any]]:
    """Dataclass → plain dict for JSON persistence + SSE emission."""
    return [
        {
            "name": r.name,
            "title": r.title,
            "library": r.library,
            "version": r.version,
            "family": r.family,
            "train_accuracy": r.train_accuracy,
            "test_accuracy": r.test_accuracy,
            "train_time_ms": r.train_time_ms,
            "confusion_matrix": r.confusion_matrix,
            "notes": r.notes,
        }
        for r in results
    ]
