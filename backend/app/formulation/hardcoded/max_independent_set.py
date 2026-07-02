"""Maximum Independent Set — penalty-encoded natural form.

Given an undirected graph, pick a maximum-cardinality set of nodes with
no two chosen nodes sharing an edge. Encoded as:

    maximize  Σ x_i  -  λ · Σ_{(i,j) ∈ E} x_i·x_j    x_i ∈ {0, 1}

Expanded:
    linear[x_i]  =  1
    quadratic[x_i·x_j] (i<j, edge)  =  -λ

The penalty λ must exceed the marginal benefit of any single node so
that any edge-violation strictly worsens the objective versus a valid
subset. ``λ = node_count + 1`` is a defensive default (safe for any
graph density, at the cost of a slightly harder QAOA landscape); the
caller can lower it toward 2 for sparse graphs to sharpen the
optimization gap. Passing ``penalty`` explicitly bypasses the default.

Zero hard constraints — the whole point is a pure QUBO/BQM shape so the
lowered circuit stays at N qubits for QAOA/annealer tiers.
"""

from __future__ import annotations

from typing import Any


def formulate_max_independent_set(
    node_count: int,
    edges: list[list[int]],
    penalty: float | None = None,
) -> dict[str, Any]:
    """Emit a valid ``cqm_v1`` document for MIS on
    ``(node_count, edges)``. All coefficients computed exactly.

    Parameters
    ----------
    node_count
        Number of nodes; variables named ``x_0 .. x_{node_count-1}``.
    edges
        Iterable of ``[u, v]`` pairs. Duplicates and self-loops dropped.
    penalty
        Positive coefficient scaling the per-edge quadratic penalty.
        Default: ``node_count + 1`` — the smallest guaranteed-safe value
        for arbitrary graphs.

    Returns
    -------
    dict
        A ``cqm_v1`` JSON dict ready to feed to ``compile_cqm_json``.
    """
    if node_count < 2:
        raise ValueError(
            f"MIS requires at least 2 nodes; got {node_count}",
        )
    if not isinstance(edges, list):
        raise ValueError(
            f"edges must be a list; got {type(edges).__name__}",
        )

    lam = float(penalty) if penalty is not None else float(node_count + 1)
    if lam <= 0:
        raise ValueError(
            f"penalty must be positive; got {lam}",
        )

    edge_set: set[tuple[int, int]] = set()
    for e in edges:
        if not isinstance(e, (list, tuple)) or len(e) != 2:
            raise ValueError(
                f"Edge must be [u, v]; got {e!r}",
            )
        u, v = int(e[0]), int(e[1])
        if u == v:
            continue
        if not (0 <= u < node_count) or not (0 <= v < node_count):
            raise ValueError(
                f"Edge {(u, v)} references node outside [0, {node_count})",
            )
        edge_set.add((u, v) if u < v else (v, u))

    variables = [
        {
            "name": f"x_{i}",
            "type": "binary",
            "description": f"Node {i} selected into the independent set (0/1)",
        }
        for i in range(node_count)
    ]

    linear = {f"x_{i}": 1.0 for i in range(node_count)}
    quadratic = {
        f"x_{u}*x_{v}": -lam for (u, v) in edge_set
    }

    expected_optimum = _brute_force_max_independent_set(node_count, edge_set)

    return {
        "version": "1",
        "description": (
            f"Max Independent Set on a {node_count}-node graph with "
            f"{len(edge_set)} edges. Penalty encoding: each x_i is 1 iff "
            f"node i is selected; per-edge penalty λ = {lam:g} punishes "
            "any chosen pair sharing an edge. No hard constraints — "
            "pure QUBO shape for QAOA tiers."
        ),
        "variables": variables,
        "objective": {
            "sense": "maximize",
            "linear": linear,
            "quadratic": quadratic,
        },
        "constraints": [],
        "test_instance": {
            "description": (
                f"Optimum independent-set size = {expected_optimum}."
            ),
            "expected_optimum": float(expected_optimum),
        },
    }


def _brute_force_max_independent_set(
    node_count: int,
    edge_set: set[tuple[int, int]],
) -> int:
    """Return the size of the maximum independent set by scanning all
    2^N subsets. Formulator-time only; never invoked in the solve
    loop."""
    best = 0
    for mask in range(1 << node_count):
        valid = True
        for (u, v) in edge_set:
            if ((mask >> u) & 1) and ((mask >> v) & 1):
                valid = False
                break
        if valid:
            size = bin(mask).count("1")
            if size > best:
                best = size
    return best
