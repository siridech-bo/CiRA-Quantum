"""QML (Quantum Machine Learning) sister app — package root.

This package houses the QML scaffolding that ships with QML-1 (Foundation)
and grows through QML-2 (local VQC) → QML-8 (polish). It deliberately
mirrors the optimization side's layout — a ``datasets`` registry plays
the same role here that ``templates`` and ``benchmarking.registry`` play
in the optimization app.

QML-1 is intentionally minimal: schema + dataset registry + blueprint
shell. The actual PennyLane training pipeline lands in QML-2.
"""

from app.qml.datasets import get_dataset, list_datasets

__all__ = ["get_dataset", "list_datasets"]
