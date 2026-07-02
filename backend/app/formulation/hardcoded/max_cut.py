"""Max-Cut — natural unconstrained binary encoding.

Given an undirected graph, split the nodes into two groups to maximize
the number (or weight) of edges that cross between them. Encoded as:

    maximize  Σ_{(i,j) ∈ E} (x_i + x_j - 2·x_i·x_j)   x_i ∈ {0, 1}

Expanded:
    linear[x_i]   =  deg(i)     (weighted: sum of incident edge weights)
    quadratic[x_i·x_j] (unordered)  =  -2 · w(i,j)

Zero constraints — that's the whole point of picking this encoding for
QAOA/annealer tiers. Anything with indicator variables or edge-slack
inflates the qubit count and gets skipped by the small-QPU tiers.

The formulator is deliberately opinionated: multi-edges collapse into a
single quadratic term (edge weights add), self-loops are silently
dropped (they contribute a constant that dimod handles via the linear
diagonal but is irrelevant to any optimum), and node indices are
normalized to the ``[0, node_count)`` range so the classifier can pass
raw edge lists without pre-processing.
"""

from __future__ import annotations

from typing import Any


def formulate_max_cut(
    node_count: int,
    edges: list[list[int] | list[float]],
) -> dict[str, Any]:
    """Emit a valid ``cqm_v1`` document for Max-Cut on the graph
    ``(node_count, edges)``. All coefficients are computed exactly.

    Parameters
    ----------
    node_count
        Number of nodes; variables are named ``x_0 .. x_{node_count-1}``.
    edges
        Iterable of ``[u, v]`` or ``[u, v, weight]`` triples. Unweighted
        edges default to weight 1. Self-loops (``u == v``) are dropped.
        Duplicate edges have their weights summed.

    Returns
    -------
    dict
        A ``cqm_v1`` JSON dict ready to feed to ``compile_cqm_json``.
    """
    if node_count < 2:
        raise ValueError(
            f"Max-Cut requires at least 2 nodes; got {node_count}",
        )
    if not isinstance(edges, list):
        raise ValueError(
            f"edges must be a list; got {type(edges).__name__}",
        )

    # Accumulate edge weights into a canonical (u, v) with u < v key so
    # duplicates and (u,v)/(v,u) both collapse. Self-loops dropped.
    weights: dict[tuple[int, int], float] = {}
    for e in edges:
        if not isinstance(e, (list, tuple)) or len(e) not in (2, 3):
            raise ValueError(
                f"Edge must be [u, v] or [u, v, weight]; got {e!r}",
            )
        u, v = int(e[0]), int(e[1])
        w = float(e[2]) if len(e) == 3 else 1.0
        if u == v:
            continue
        if not (0 <= u < node_count) or not (0 <= v < node_count):
            raise ValueError(
                f"Edge {(u, v)} references node outside [0, {node_count})",
            )
        key = (u, v) if u < v else (v, u)
        weights[key] = weights.get(key, 0.0) + w

    variables = [
        {
            "name": f"x_{i}",
            "type": "binary",
            "description": f"Node {i} group assignment: 0 = group A, 1 = group B",
        }
        for i in range(node_count)
    ]

    # linear[x_i] = weighted degree
    linear: dict[str, float] = {f"x_{i}": 0.0 for i in range(node_count)}
    for (u, v), w in weights.items():
        linear[f"x_{u}"] += w
        linear[f"x_{v}"] += w

    # quadratic[x_i * x_j] = -2 * w(i,j)  (canonical i < j key form)
    quadratic = {
        f"x_{u}*x_{v}": -2.0 * w for (u, v), w in weights.items()
    }

    expected_optimum = _brute_force_max_cut(node_count, weights)

    return {
        "version": "1",
        "description": (
            f"Max-Cut on a {node_count}-node graph with {len(weights)} "
            "edges. Natural unconstrained binary encoding: each x_i "
            "marks the side of the cut. Per-edge contribution "
            "(x_i + x_j - 2·x_i·x_j) equals 1 iff endpoints differ."
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
                f"Optimum cut weight = {expected_optimum:g}."
            ),
            "expected_optimum": float(expected_optimum),
        },
    }


def _brute_force_max_cut(
    node_count: int,
    weights: dict[tuple[int, int], float],
) -> float:
    """Return the maximum cut weight by exhausting all 2^N assignments.
    Only used at formulator time to embed ``expected_optimum``; never
    invoked in the solve path. Exponential in ``node_count`` but the
    qpu_ready templates cap at ~10 nodes."""
    best = 0.0
    for mask in range(1 << node_count):
        cut = 0.0
        for (u, v), w in weights.items():
            bu = (mask >> u) & 1
            bv = (mask >> v) & 1
            if bu != bv:
                cut += w
        if cut > best:
            best = cut
    return best
