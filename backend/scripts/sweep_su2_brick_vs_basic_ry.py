"""SU(2) brick-wall vs basic_ry — controlled head-to-head across datasets.

Generated for the research note at
``docs/research_notes/2501.07387_guo_yang_circuit_compilation.md``.

The first head-to-head was only on 2-qubit toy datasets (``moons``,
``circles``). Before claiming the SU(2) Riemannian ansatz is a real
research-tier upgrade we need to see whether the win generalizes to:

* higher input dimensionality (PCA'd onto 4 qubits)
* real-world data (Iris, Wine, Breast Cancer)
* convergence rate, not just final accuracy

Runs both ``basic_ry`` and ``su2_brick`` on each dataset at fixed seed
and matched hyperparameters, then writes the comparison as a CSV the
research note can paste in verbatim.

Run with:

    cd backend
    python scripts/sweep_su2_brick_vs_basic_ry.py
"""
from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path

from app.qml import data_loader
from app.qml.vqc import VQCConfig, TrainResult, train_vqc


@dataclass(frozen=True)
class SweepCell:
    dataset: str
    n_qubits: int
    n_layers: int
    n_epochs: int
    # LR is tuned per (ansatz, optimizer) combo. Adam-on-angles, vanilla
    # Riemannian SGD, and Riemannian Adam all sit at different
    # natural-step-size scales. Values picked from the manual tuning
    # in the research note's "implementation log" section.
    lr_basic: float       # basic_ry (PennyLane AdamOptimizer over RY angles)
    lr_su2_sgd: float     # su2_brick + Riemannian SGD
    lr_su2_adam: float    # su2_brick + Riemannian Adam (Becigneul-Ganea)


# Sweep matrix. Keep it tight — every cell is two full VQC training
# runs at PennyLane's default.qubit statevector cost.
SWEEP: list[SweepCell] = [
    SweepCell("moons",         n_qubits=2, n_layers=2, n_epochs=15,
              lr_basic=0.20, lr_su2_sgd=0.10, lr_su2_adam=0.03),
    SweepCell("circles",       n_qubits=2, n_layers=2, n_epochs=15,
              lr_basic=0.20, lr_su2_sgd=0.10, lr_su2_adam=0.03),
    SweepCell("iris",          n_qubits=4, n_layers=2, n_epochs=15,
              lr_basic=0.20, lr_su2_sgd=0.10, lr_su2_adam=0.03),
    SweepCell("wine",          n_qubits=4, n_layers=2, n_epochs=15,
              lr_basic=0.20, lr_su2_sgd=0.10, lr_su2_adam=0.03),
    SweepCell("breast_cancer", n_qubits=4, n_layers=2, n_epochs=15,
              lr_basic=0.20, lr_su2_sgd=0.10, lr_su2_adam=0.03),
]

OUT_CSV = Path(__file__).resolve().parents[2] / "docs" / "research_notes" / "sweep_su2_brick_results.csv"


def _epochs_to_train_acc(result: TrainResult, threshold: float) -> int | None:
    """First epoch (1-indexed) at which train accuracy ≥ threshold.
    Returns None if never reached. Useful for convergence-rate comparison.
    """
    for h in result.history:
        if h["train_accuracy"] >= threshold:
            return int(h["epoch"])
    return None


def run_one(cell: SweepCell, ansatz: str, lr: float, momentum: str) -> dict[str, object]:
    split = data_loader.load(cell.dataset, max_qubits=cell.n_qubits)
    cfg = VQCConfig(
        n_qubits=split.n_features,
        n_layers=cell.n_layers,
        n_epochs=cell.n_epochs,
        batch_size=32,
        learning_rate=lr,
        seed=42,
        ansatz=ansatz,
        momentum=momentum,
    )
    t0 = time.perf_counter()
    result = train_vqc(
        split.X_train, split.y_train,
        split.X_test, split.y_test,
        config=cfg,
    )
    wall = time.perf_counter() - t0
    return {
        "dataset": cell.dataset,
        "n_qubits": split.n_features,
        "n_layers": cell.n_layers,
        "n_epochs": cell.n_epochs,
        "ansatz": ansatz,
        "momentum": momentum,
        "lr": lr,
        "pca_applied": split.pca_applied,
        "n_params": result.circuit_info.n_trainable_params,
        "final_loss": round(result.final_loss, 4),
        "final_train_acc": round(result.final_train_accuracy, 3),
        "final_test_acc": round(result.final_test_accuracy, 3),
        "wall_seconds": round(wall, 1),
        "epochs_to_train_0.80": _epochs_to_train_acc(result, 0.80),
        "epochs_to_train_0.90": _epochs_to_train_acc(result, 0.90),
    }


def main() -> None:
    rows: list[dict[str, object]] = []
    combos = [
        ("basic_ry",  "sgd",  None),   # uses PennyLane AdamOptimizer internally
        ("su2_brick", "sgd",  None),
        ("su2_brick", "adam", None),
    ]
    for cell in SWEEP:
        print(f"\n=== {cell.dataset} (qubits={cell.n_qubits}, layers={cell.n_layers}) ===")
        for ansatz, momentum, _ in combos:
            if ansatz == "basic_ry":
                lr = cell.lr_basic
            elif momentum == "sgd":
                lr = cell.lr_su2_sgd
            else:
                lr = cell.lr_su2_adam
            label = f"{ansatz}/{momentum}"
            print(f"  {label:18s} lr={lr} ...", end=" ", flush=True)
            row = run_one(cell, ansatz, lr, momentum)
            rows.append(row)
            print(
                f"test={row['final_test_acc']:.3f} "
                f"train={row['final_train_acc']:.3f} "
                f"loss={row['final_loss']:.4f} "
                f"wall={row['wall_seconds']}s "
                f"e@0.8={row['epochs_to_train_0.80']} "
                f"e@0.9={row['epochs_to_train_0.90']}"
            )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {OUT_CSV.relative_to(OUT_CSV.parents[2])}")

    # Markdown table for pasting into the research note.
    print("\n--- markdown summary ---")
    print("| dataset | qubits | ansatz/opt | n_params | test acc | train acc | final loss | wall | e@0.80 | e@0.90 |")
    print("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        combo = f"{r['ansatz']}/{r['momentum']}"
        print(
            f"| {r['dataset']} | {r['n_qubits']} | {combo} | {r['n_params']} | "
            f"{r['final_test_acc']:.3f} | {r['final_train_acc']:.3f} | {r['final_loss']:.4f} | "
            f"{r['wall_seconds']}s | {r['epochs_to_train_0.80']} | {r['epochs_to_train_0.90']} |"
        )


if __name__ == "__main__":
    main()
