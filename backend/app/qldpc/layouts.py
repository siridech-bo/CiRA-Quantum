"""Tanner-graph 2D layouts + routing metrics (Sprint 2).

Sprint 1 ships the Tanner graph as a node/edge list via
``serializers.code_to_tanner_payload``. Sprint 2 lays positions on
top so the frontend has something to render, and computes the
routing-quality metrics QEC architects need to answer "can this code
be laid out on a real chip?"

All layouts are computed via ``networkx`` (already installed in
Sprint 1). We deliberately do **not** invoke ``qldpc``'s native
position helpers — only ``BBCode`` exposes them and their API is
brittle to call generically. networkx covers all four families with
one code path.
"""

from __future__ import annotations

from typing import Any

import networkx as nx


# The layout strategies surfaced to the frontend's strategy selector.
# Ordered so the most-readable-for-newcomers default is first.
SUPPORTED_STRATEGIES = ("bipartite", "kamada_kawai", "spring", "circular")


def compute_layout(
    graph: nx.Graph,
    strategy: str,
    *,
    seed: int = 42,
) -> dict[str, tuple[float, float]]:
    """Return ``{node_id: (x, y)}`` with coordinates normalized to ``[0, 1]²``.

    Strategies:

    * ``bipartite``   — data nodes on the left column, checks on the right.
                         Most readable for understanding Tanner structure.
    * ``kamada_kawai`` — energy-minimization. Often reveals lattice structure
                         for topological codes. Deterministic, O(n²).
    * ``spring``       — Fruchterman-Reingold force-directed. Deterministic
                         with a fixed seed.
    * ``circular``     — all nodes on a circle. Useful for finite-rate codes
                         where the bipartite split is large and lopsided.

    Raises :class:`ValueError` on an unknown strategy so the route can
    return a clear 400 to the frontend.
    """
    if strategy not in SUPPORTED_STRATEGIES:
        raise ValueError(
            f"Unknown layout strategy '{strategy}'. "
            f"Supported: {', '.join(SUPPORTED_STRATEGIES)}"
        )

    if len(graph.nodes) == 0:
        return {}

    if strategy == "bipartite":
        # Split nodes by Sprint 1's ``type`` attribute on each node.
        data_nodes = [n for n, attrs in graph.nodes(data=True) if attrs.get("type") == "data"]
        # Fall back to any-set-of-nodes if attribute is missing (shouldn't
        # happen with our serializer but keeps the layout robust).
        if not data_nodes:
            data_nodes = list(graph.nodes)[: len(graph.nodes) // 2 or 1]
        raw = nx.bipartite_layout(graph, nodes=data_nodes, align="horizontal")
    elif strategy == "kamada_kawai":
        # On disconnected graphs networkx raises; fall back to spring.
        try:
            raw = nx.kamada_kawai_layout(graph)
        except (nx.NetworkXError, Exception):
            raw = nx.spring_layout(graph, seed=seed, iterations=50)
    elif strategy == "spring":
        raw = nx.spring_layout(graph, seed=seed, iterations=50)
    else:  # circular
        raw = nx.circular_layout(graph)

    return _normalize_to_unit_square(raw)


def _normalize_to_unit_square(
    raw: dict[Any, Any],
) -> dict[str, tuple[float, float]]:
    """Map arbitrary 2D coordinates to ``[0, 1]²`` with a small margin."""
    if not raw:
        return {}
    xs = [float(p[0]) for p in raw.values()]
    ys = [float(p[1]) for p in raw.values()]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    xspan = max(xmax - xmin, 1e-9)
    yspan = max(ymax - ymin, 1e-9)
    margin = 0.05
    scale = 1.0 - 2 * margin
    return {
        str(node): (
            margin + (float(p[0]) - xmin) / xspan * scale,
            margin + (float(p[1]) - ymin) / yspan * scale,
        )
        for node, p in raw.items()
    }


def compute_routing_metrics(
    graph: nx.Graph,
    positions: dict[str, tuple[float, float]],
) -> dict[str, Any]:
    """Compute routing-quality metrics over a fixed layout.

    All distances are Euclidean in the normalized ``[0, 1]²`` space.
    ``edge_crossings`` is a brute-force segment-intersection count
    (O(E²)) — fine at our sizes (Surface = 670 edges → ~225K pair
    checks, <1 s on commodity hardware).
    """
    if len(graph.edges) == 0:
        return {
            "num_nodes": int(len(graph.nodes)),
            "num_edges": 0,
            "avg_edge_length": 0.0,
            "min_edge_length": 0.0,
            "max_edge_length": 0.0,
            "p95_edge_length": 0.0,
            "edge_crossings": 0,
        }

    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    lengths: list[float] = []
    for u, v in graph.edges():
        pu = positions.get(str(u))
        pv = positions.get(str(v))
        if pu is None or pv is None:
            continue
        segments.append((pu, pv))
        dx = pv[0] - pu[0]
        dy = pv[1] - pu[1]
        lengths.append((dx * dx + dy * dy) ** 0.5)

    lengths_sorted = sorted(lengths)

    def _percentile(sorted_vals: list[float], pct: float) -> float:
        if not sorted_vals:
            return 0.0
        k = max(0, min(len(sorted_vals) - 1, int(pct * (len(sorted_vals) - 1))))
        return sorted_vals[k]

    return {
        "num_nodes": int(len(graph.nodes)),
        "num_edges": int(len(graph.edges)),
        "avg_edge_length": float(sum(lengths) / len(lengths)),
        "min_edge_length": float(min(lengths)),
        "max_edge_length": float(max(lengths)),
        "p95_edge_length": float(_percentile(lengths_sorted, 0.95)),
        "edge_crossings": int(_count_crossings(segments)),
    }


def _count_crossings(
    segments: list[tuple[tuple[float, float], tuple[float, float]]],
) -> int:
    """Count pairs of line segments that strictly cross.

    Endpoints touching don't count as crossings (which would falsely
    flag every check node sharing a data qubit). The orientation test
    follows the standard CLRS / Bentley-Ottmann formulation.
    """
    count = 0
    n = len(segments)
    for i in range(n):
        ax, ay = segments[i][0]
        bx, by = segments[i][1]
        for j in range(i + 1, n):
            cx, cy = segments[j][0]
            dx, dy = segments[j][1]
            # Skip pairs that share an endpoint — common in graphs where
            # multiple edges hit the same check node; not a "crossing"
            # for routing purposes.
            if (ax, ay) in {(cx, cy), (dx, dy)} or (bx, by) in {(cx, cy), (dx, dy)}:
                continue
            if _strict_segment_cross(ax, ay, bx, by, cx, cy, dx, dy):
                count += 1
    return count


def _orient(ax: float, ay: float, bx: float, by: float, cx: float, cy: float) -> float:
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def _strict_segment_cross(
    ax: float, ay: float, bx: float, by: float,
    cx: float, cy: float, dx: float, dy: float,
) -> bool:
    o1 = _orient(ax, ay, bx, by, cx, cy)
    o2 = _orient(ax, ay, bx, by, dx, dy)
    o3 = _orient(cx, cy, dx, dy, ax, ay)
    o4 = _orient(cx, cy, dx, dy, bx, by)
    # Strict: both pairs of orientations must have opposite signs.
    return (o1 * o2 < 0) and (o3 * o4 < 0)


def layout_payload(graph: nx.Graph, strategy: str) -> dict[str, Any]:
    """Bundle positions + metrics for one Tanner subgraph."""
    positions = compute_layout(graph, strategy)
    metrics = compute_routing_metrics(graph, positions)
    return {
        "positions": {k: [float(v[0]), float(v[1])] for k, v in positions.items()},
        "metrics": metrics,
    }


def build_nx_graph(subgraph_payload: dict[str, Any]) -> nx.Graph:
    """Rebuild a networkx graph from Sprint 1's JSON node/edge payload.

    The serializer emits ``{"nodes": [{"id", "type"}, ...],
    "edges": [{"source", "target"}, ...]}``. We use an undirected
    Graph for layout/metrics — direction has no meaning here.
    """
    g = nx.Graph()
    for node in subgraph_payload.get("nodes", []):
        g.add_node(node["id"], type=node.get("type", "data"))
    for edge in subgraph_payload.get("edges", []):
        g.add_edge(edge["source"], edge["target"])
    return g
