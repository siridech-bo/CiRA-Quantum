"""QML dataset registry.

Each entry is a small, self-describing record. The blueprint exposes
them as JSON so the frontend Dataset Gallery can render cards without
the backend doing any sklearn imports — the heavy data-loading lives in
QML-2 once a model actually needs to train against the dataset.

Six datasets ship in QML-1 (per the planning decision):

* Iris            — 4 features, 3 classes (we'll restrict to 2 for VQC)
* Moons           — 2 features, 2 classes, non-linear
* MNIST 0v1       — 64-feature digits (sklearn's 8x8), binary 0-vs-1
* Wine            — 13 features, 3 classes (binary on classes 0 vs 1)
* Circles         — 2 features, 2 classes, concentric (kernel showcase)
* Breast Cancer   — 30 features, 2 classes, real-world medical data
"""

from __future__ import annotations

from typing import Any

# Difficulty buckets keep the gallery sorted in a way that maps to
# "what's a sensible first solve". "easy" = 2D, separable enough for the
# minimal 2-qubit VQC; "medium" = needs more qubits or sharper kernels;
# "hard" = realistic dimensionality where the classical baseline tends
# to win and the educational point is the comparison.
_DATASETS: list[dict[str, Any]] = [
    {
        "id": "moons",
        "title": "Two Moons",
        "category": "synthetic",
        "difficulty": "easy",
        "n_features": 2,
        "n_classes": 2,
        "n_samples": 200,
        "summary": (
            "Two interleaving half-circles. Classic non-linear benchmark "
            "— the line of best separation is curved, so a linear classifier "
            "fails and a kernel or VQC succeeds."
        ),
        "source": "sklearn.datasets.make_moons",
    },
    {
        "id": "circles",
        "title": "Concentric Circles",
        "category": "synthetic",
        "difficulty": "easy",
        "n_features": 2,
        "n_classes": 2,
        "n_samples": 200,
        "summary": (
            "Inner ring vs outer ring. No linear classifier can separate "
            "them; a quantum feature map that lifts the data into a higher-"
            "dimensional Hilbert space can."
        ),
        "source": "sklearn.datasets.make_circles",
    },
    {
        "id": "iris",
        "title": "Iris (Setosa vs Versicolor)",
        "category": "real",
        "difficulty": "easy",
        "n_features": 4,
        "n_classes": 2,
        "n_samples": 100,
        "summary": (
            "Fisher's 1936 iris dataset, restricted to the two species "
            "that are linearly separable. The textbook first-classifier "
            "problem — useful as a sanity check that a VQC is wired right."
        ),
        "source": "sklearn.datasets.load_iris",
    },
    {
        "id": "wine",
        "title": "Wine (classes 0 vs 1)",
        "category": "real",
        "difficulty": "medium",
        "n_features": 13,
        "n_classes": 2,
        "n_samples": 130,
        "summary": (
            "Chemical analysis of three wine cultivars; we restrict to the "
            "first two for a binary task. 13 features test whether the VQC "
            "can compress useful structure into a small qubit count."
        ),
        "source": "sklearn.datasets.load_wine",
    },
    {
        "id": "mnist_0v1",
        "title": "MNIST 0 vs 1",
        "category": "real",
        "difficulty": "medium",
        "n_features": 64,
        "n_classes": 2,
        "n_samples": 360,
        "summary": (
            "Handwritten digits from sklearn's 8×8 corpus, binary 0-vs-1. "
            "Demonstrates that the VQC can survive PCA-reduced image data "
            "without exploding qubit count."
        ),
        "source": "sklearn.datasets.load_digits",
    },
    {
        "id": "breast_cancer",
        "title": "Breast Cancer (WDBC)",
        "category": "real",
        "difficulty": "hard",
        "n_features": 30,
        "n_classes": 2,
        "n_samples": 569,
        "summary": (
            "Wisconsin diagnostic breast cancer — 30 cell-nucleus measurements, "
            "benign vs malignant. The honest test: a real medical dataset "
            "where classical baselines (SVM, MLP) set a high bar."
        ),
        "source": "sklearn.datasets.load_breast_cancer",
    },
]

_BY_ID = {d["id"]: d for d in _DATASETS}


def list_datasets() -> list[dict[str, Any]]:
    """All registered datasets, in display order."""
    return [dict(d) for d in _DATASETS]


def get_dataset(dataset_id: str) -> dict[str, Any] | None:
    d = _BY_ID.get(dataset_id)
    return dict(d) if d else None
