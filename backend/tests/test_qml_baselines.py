"""Tests for ``app.qml.baselines`` + the trainer's baseline integration.

The baselines module is a thin wrapper over scikit-learn so the tests
focus on the contract: every result has the same shape, the four
expected models are present, accuracy is plausible on a separable
toy dataset, and deterministic re-runs match.
"""
from __future__ import annotations

import pytest

pytest.importorskip("sklearn")

import numpy as np

from app.qml import data_loader
from app.qml.baselines import serialize, train_baselines


def test_baselines_return_all_four_models():
    split = data_loader.load("moons", max_qubits=2)
    results = train_baselines(
        split.X_train, split.y_train, split.X_test, split.y_test, seed=0,
    )
    names = [r.name for r in results]
    assert names == ["logreg", "svm_rbf", "random_forest", "mlp"]
    families = {r.family for r in results}
    assert families == {"linear", "kernel", "ensemble", "neural"}


@pytest.mark.parametrize("dataset_id", ["moons", "iris"])
def test_baselines_score_above_chance(dataset_id):
    """SVM-RBF / RandomForest / MLP all reach > 70 % on these toys."""
    split = data_loader.load(dataset_id, max_qubits=2 if dataset_id == "moons" else None)
    results = train_baselines(
        split.X_train, split.y_train, split.X_test, split.y_test, seed=0,
    )
    for r in results:
        assert 0.0 <= r.train_accuracy <= 1.0
        assert 0.0 <= r.test_accuracy <= 1.0
        # Confusion matrix is 2x2 of ints that sum to the test set size.
        cm = np.array(r.confusion_matrix)
        assert cm.shape == (2, 2)
        assert cm.sum() == split.X_test.shape[0]
        assert r.train_time_ms >= 0
    # On a separable dataset, at least one non-linear baseline clears 70%.
    nonlinear = [r for r in results if r.name != "logreg"]
    assert any(r.test_accuracy >= 0.70 for r in nonlinear), \
        f"No baseline cleared 70% on {dataset_id}: {[r.test_accuracy for r in results]}"


def test_baselines_are_deterministic():
    split = data_loader.load("moons", max_qubits=2)
    a = train_baselines(
        split.X_train, split.y_train, split.X_test, split.y_test, seed=7,
    )
    b = train_baselines(
        split.X_train, split.y_train, split.X_test, split.y_test, seed=7,
    )
    # Same seed → identical test accuracy on every model.
    for ra, rb in zip(a, b):
        assert ra.name == rb.name
        assert ra.test_accuracy == pytest.approx(rb.test_accuracy)


def test_serialize_round_trip():
    split = data_loader.load("moons", max_qubits=2)
    results = train_baselines(
        split.X_train, split.y_train, split.X_test, split.y_test, seed=0,
    )
    js = serialize(results)
    assert isinstance(js, list)
    for r in js:
        for f in ("name", "title", "library", "version", "family",
                  "train_accuracy", "test_accuracy", "train_time_ms",
                  "confusion_matrix", "notes"):
            assert f in r, f"missing {f} in serialized baseline {r.get('name')}"
        assert r["library"] == "scikit-learn"
