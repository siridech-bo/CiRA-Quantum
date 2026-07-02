"""Number Partitioning — natural unconstrained binary encoding.

Given a list of positive numbers, split into two groups so the
absolute difference of the two group sums is minimized. Encoded as:

    minimize (S - 2·Σ n_i · x_i)²   where S = Σ n_i, x_i ∈ {0, 1}

Expanded:
    linear[x_i]  =  4·n_i·(n_i - S)
    quadratic[x_i·x_j] (i<j)  =  8·n_i·n_j
    offset       =  S²

The offset lets the perfect-partition optimum (or the closest achievable
under parity) evaluate to 0 (or 1, 4, 9, ... for odd-S instances).

This is the formulator that led to the whole hardcoded thread — Claude
consistently emitted the right structure (5 vars, 0 constraints, offset
present) but got the coefficients wrong by ~20-30%, producing an
optimum of 5 instead of 1 on the [4, 3, 2, 3, 1] test case
(2026-07-02 investigation).
"""

from __future__ import annotations

from typing import Any


def formulate_number_partitioning(numbers: list[float]) -> dict[str, Any]:
    """Emit a valid ``cqm_v1`` document for Number Partitioning on
    ``numbers``. All coefficients are computed exactly; no LLM in the
    loop. Includes ``expected_optimum`` computed by brute-force so the
    validator's Layer A oracle has ground truth to compare against.

    Parameters
    ----------
    numbers
        List of positive numbers to partition. Zeros and negatives are
        rejected — Number Partitioning is defined for positive integers,
        and mixing signs produces a different problem shape.

    Returns
    -------
    dict
        A ``cqm_v1`` JSON dict ready to feed to ``compile_cqm_json``.
    """
    if not numbers or len(numbers) < 2:
        raise ValueError(
            f"Number Partitioning requires at least 2 numbers; got {len(numbers)}",
        )
    for i, n in enumerate(numbers):
        if not isinstance(n, (int, float)) or n <= 0:
            raise ValueError(
                f"Number at index {i} must be a positive number; got {n!r}",
            )

    n_count = len(numbers)
    total = sum(numbers)

    variables = [
        {
            "name": f"x_{i}",
            "type": "binary",
            "description": (
                f"Number {i} (value {numbers[i]}): 0 = group A, 1 = group B"
            ),
        }
        for i in range(n_count)
    ]

    # Linear: 4·n_i·(n_i - S) for each i
    linear = {
        f"x_{i}": 4.0 * numbers[i] * (numbers[i] - total)
        for i in range(n_count)
    }

    # Quadratic: 8·n_i·n_j for each unordered pair i < j
    quadratic = {}
    for i in range(n_count):
        for j in range(i + 1, n_count):
            key = f"x_{i}*x_{j}"
            quadratic[key] = 8.0 * numbers[i] * numbers[j]

    offset = total * total

    expected_optimum = _brute_force_min_imbalance_squared(numbers)

    return {
        "version": "1",
        "description": (
            f"Number Partitioning on {list(numbers)} "
            f"(S={total}). Natural unconstrained binary encoding: each "
            "x_i marks the side (0 = group A, 1 = group B). Squared "
            f"imbalance shifted by S² = {offset:g} via objective.offset "
            "so the perfect-partition optimum reads as 0 (or the odd-S "
            "residual as the smallest odd square)."
        ),
        "variables": variables,
        "objective": {
            "sense": "minimize",
            "linear": linear,
            "quadratic": quadratic,
            "offset": float(offset),
        },
        "constraints": [],
        "test_instance": {
            "description": (
                f"S = {total}. Optimum squared imbalance = "
                f"{expected_optimum:g}."
            ),
            "expected_optimum": float(expected_optimum),
        },
    }


def _brute_force_min_imbalance_squared(numbers: list[float]) -> float:
    """Return the minimum (SumA - SumB)² over all 2^N partitions.
    Only used at formulator time — the resulting scalar is embedded in
    ``test_instance.expected_optimum`` and never recomputed at solve
    time. Exponential complexity is acceptable for instances up to
    ~20 numbers, comfortably above the qpu_ready-template range.
    """
    total = sum(numbers)
    n = len(numbers)
    # For each subset mask, imbalance = total - 2·sum(subset)
    best = float("inf")
    for mask in range(1 << n):
        subset_sum = 0.0
        for i in range(n):
            if mask & (1 << i):
                subset_sum += numbers[i]
        imbalance = total - 2 * subset_sum
        val = imbalance * imbalance
        if val < best:
            best = val
            if best == 0:
                return 0.0
    return best
