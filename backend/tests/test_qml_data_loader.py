"""Tests for ``app.qml.data_loader``.

Cheap-ish (Moons / Circles / Iris are tiny, ~100–200 samples) so we run
all 6 datasets through the loader. The breast-cancer test also verifies
the PCA cap path.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.qml import data_loader

# Skip the entire module if scikit-learn isn't on the path. CI environments
# without the [qml] extra installed should still pass the suite.
pytest.importorskip("sklearn")


@pytest.mark.parametrize("dataset_id,expected_features", [
    ("moons", 2),
    ("circles", 2),
    ("iris", 4),
    ("wine", 13),
    ("mnist_0v1", 64),
    ("breast_cancer", 30),
])
def test_loader_returns_well_shaped_split(dataset_id, expected_features):
    split = data_loader.load(dataset_id)
    # Train + test arrays line up with their labels.
    assert split.X_train.shape[1] == expected_features
    assert split.X_train.shape[0] == split.y_train.shape[0]
    assert split.X_test.shape[0] == split.y_test.shape[0]
    # Binary classification — labels are exactly {0, 1}.
    assert set(np.unique(split.y_train)).issubset({0, 1})
    assert set(np.unique(split.y_test)).issubset({0, 1})
    # Standard-scaled: mean close to 0, std close to 1 over the union.
    X = np.vstack([split.X_train, split.X_test])
    assert abs(float(X.mean())) < 1.0
    # Two classes by name.
    assert len(split.classes) == 2
    # Features named.
    assert len(split.feature_names) == expected_features
    # No PCA when no cap.
    assert split.pca_applied is False


def test_loader_pca_cap_applies():
    split = data_loader.load("breast_cancer", max_qubits=4)
    # Original is 30 features, requested 4 — PCA must have reduced it.
    assert split.pca_applied is True
    assert split.X_train.shape[1] == 4
    assert split.X_test.shape[1] == 4
    assert split.feature_names == ["PC1", "PC2", "PC3", "PC4"]
    # The notes string mentions the retained variance ratio.
    assert any("variance" in n for n in split.notes)


def test_loader_unknown_id_raises():
    with pytest.raises(ValueError, match="Unknown QML dataset"):
        data_loader.load("not-a-real-dataset")


def test_loader_n_samples_cap_subsamples():
    split = data_loader.load("breast_cancer", n_samples_cap=80)
    total = split.X_train.shape[0] + split.X_test.shape[0]
    assert total == 80
    assert any("Subsampled" in n for n in split.notes)


def test_loader_is_deterministic():
    """Two calls return identical splits — important for the educational
    'your accuracy should match mine' invariant."""
    a = data_loader.load("moons")
    b = data_loader.load("moons")
    np.testing.assert_array_equal(a.X_train, b.X_train)
    np.testing.assert_array_equal(a.y_train, b.y_train)
