"""Tests for the ansatz field flowing from POST payload → VQCConfig →
persisted hyperparameters → repro_hash.

The trainer itself is tested in test_qml_trainer.py; this file pins the
glue around the new ``ansatz`` field so changing the default or the
allowed-values list fails loud.
"""
from __future__ import annotations

import pytest

pytest.importorskip("pennylane")  # _coerce_config imports VQCConfig
pytest.importorskip("sklearn")

from app.qml.records import compute_repro_hash
from app.qml.trainer import _build_circuit_info_dict, _coerce_config


def test_coerce_config_defaults_to_basic_ry():
    cfg = _coerce_config({}, dataset_n_features=2)
    assert cfg.ansatz == "basic_ry"


def test_coerce_config_accepts_su2_brick():
    cfg = _coerce_config({"ansatz": "su2_brick"}, dataset_n_features=4)
    assert cfg.ansatz == "su2_brick"


def test_coerce_config_unknown_ansatz_falls_back_silently():
    """Unknown ansatz values fall back to the default rather than raising,
    so a typo in the payload doesn't surface as a deep 4xx through the
    job pipeline. The route layer can layer stricter validation on top
    if needed; this is the safety net."""
    cfg = _coerce_config({"ansatz": "garbage"}, dataset_n_features=2)
    assert cfg.ansatz == "basic_ry"
    cfg2 = _coerce_config({"ansatz": ""}, dataset_n_features=2)
    assert cfg2.ansatz == "basic_ry"


def test_circuit_info_dict_branches_on_ansatz():
    """The pre-training circuit_info shown on the SSE stream must reflect
    the right ansatz — entangler label + trainable-param count differ."""
    from app.qml.vqc import VQCConfig

    basic = _build_circuit_info_dict(VQCConfig(
        n_qubits=2, n_layers=3, ansatz="basic_ry",
    ))
    su2 = _build_circuit_info_dict(VQCConfig(
        n_qubits=2, n_layers=3, ansatz="su2_brick",
    ))
    # basic_ry: 1 angle per (layer, qubit) + 1 bias.
    assert basic["n_trainable_params"] == 3 * 2 + 1
    assert "BasicEntanglerLayers" in basic["entangler"]
    # su2_brick: 3 DOF per gate + 1 bias.
    assert su2["n_trainable_params"] == 3 * 3 * 2 + 1
    assert "SU(2)" in su2["entangler"]


def test_repro_hash_differs_between_ansatze():
    """Reproducibility-by-default contract: two jobs with identical
    hyperparameters except for the ansatz must yield distinct repro
    hashes so they're not confusable in the record archive.
    """
    code_version = "abc123"
    dataset_id = "moons"
    model = "vqc"
    base = {
        "n_qubits": 2, "n_layers": 2, "n_epochs": 10,
        "batch_size": 32, "learning_rate": 0.1, "seed": 42,
    }
    h_basic = compute_repro_hash(
        code_version, dataset_id, model, {**base, "ansatz": "basic_ry"}
    )
    h_su2 = compute_repro_hash(
        code_version, dataset_id, model, {**base, "ansatz": "su2_brick"}
    )
    assert h_basic != h_su2, (
        "ansatz must be part of the repro hash; otherwise two runs with "
        "different ansätze would collide in the archive"
    )


def test_strip_weights_drops_su2_unitaries():
    """SU(2) unitaries are model state, not citation-worthy data — they
    follow the same archive-drop rule as ``weights``."""
    from app.qml.records import _strip_weights

    metrics = {
        "final_test_accuracy": 0.9,
        "weights": [[0.1, 0.2]],
        "su2_unitaries": [[[[[1.0, 0.0], [0.0, 0.0]],
                            [[0.0, 0.0], [1.0, 0.0]]]]],
        "scatter_points": [{"x": 0, "y": 0, "label": 0, "split": "train"}],
        "decision_grid": {"resolution": 20},
    }
    out = _strip_weights(metrics)
    assert "su2_unitaries" not in out
    assert "weights" not in out
    assert "scatter_points" not in out
    assert "decision_grid" not in out
    # Real metrics survive.
    assert out["final_test_accuracy"] == 0.9
