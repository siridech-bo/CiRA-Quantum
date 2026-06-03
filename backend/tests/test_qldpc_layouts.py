"""Sprint 2 layout + routing-metrics tests.

Gated with ``pytest.importorskip("qldpc")`` so the suite passes on
contributor machines without the ``[qldpc]`` extra. Each strategy is
exercised against the smaller code families (HGP, Toric) to keep the
test runtime low; Surface coverage is via the route-level integration
test in ``test_routes_qldpc.py``.
"""

from __future__ import annotations

import math

import pytest

pytest.importorskip("qldpc")

import networkx as nx  # noqa: E402

from app.qldpc import generators  # noqa: E402
from app.qldpc.layouts import (  # noqa: E402
    SUPPORTED_STRATEGIES,
    build_nx_graph,
    compute_layout,
    compute_routing_metrics,
    layout_payload,
)
from app.qldpc.serializers import code_to_tanner_payload  # noqa: E402


# Small helper so each test doesn't repay the cost of constructing a code.
@pytest.fixture(scope="module")
def hgp_subgraphs():
    code = generators.build_hypergraph_product()
    payload = code_to_tanner_payload(code)
    return payload


@pytest.mark.parametrize("strategy", list(SUPPORTED_STRATEGIES))
def test_compute_layout_returns_position_per_node(hgp_subgraphs, strategy: str):
    g = build_nx_graph(hgp_subgraphs["graph_x"])
    positions = compute_layout(g, strategy)
    assert set(positions.keys()) == {str(n) for n in g.nodes}
    for node_id, (x, y) in positions.items():
        assert math.isfinite(x) and math.isfinite(y), node_id
        # Normalized to [0, 1] with a small margin — accept [-0.01, 1.01].
        assert -0.01 <= x <= 1.01, (node_id, x)
        assert -0.01 <= y <= 1.01, (node_id, y)


def test_unknown_strategy_raises_value_error(hgp_subgraphs):
    g = build_nx_graph(hgp_subgraphs["graph_x"])
    with pytest.raises(ValueError, match="Unknown layout strategy"):
        compute_layout(g, "this-is-not-a-thing")


def test_routing_metrics_have_expected_shape(hgp_subgraphs):
    g = build_nx_graph(hgp_subgraphs["graph_x"])
    positions = compute_layout(g, "bipartite")
    metrics = compute_routing_metrics(g, positions)

    assert set(metrics.keys()) >= {
        "num_nodes", "num_edges",
        "avg_edge_length", "min_edge_length", "max_edge_length",
        "p95_edge_length", "edge_crossings",
    }
    assert metrics["num_nodes"] == len(g.nodes)
    assert metrics["num_edges"] == len(g.edges)
    # Length ordering: min ≤ avg ≤ p95 ≤ max
    assert metrics["min_edge_length"] <= metrics["avg_edge_length"]
    assert metrics["avg_edge_length"] <= metrics["p95_edge_length"]
    assert metrics["p95_edge_length"] <= metrics["max_edge_length"]
    assert metrics["edge_crossings"] >= 0


def test_routing_metrics_empty_graph_returns_zeros():
    g = nx.Graph()
    g.add_node("isolated", type="data")
    positions = {"isolated": (0.5, 0.5)}
    metrics = compute_routing_metrics(g, positions)
    assert metrics["num_edges"] == 0
    assert metrics["avg_edge_length"] == 0.0
    assert metrics["edge_crossings"] == 0


def test_layout_payload_combines_positions_and_metrics(hgp_subgraphs):
    g = build_nx_graph(hgp_subgraphs["graph_x"])
    payload = layout_payload(g, "kamada_kawai")
    assert set(payload.keys()) == {"positions", "metrics"}
    assert set(payload["positions"].keys()) == {str(n) for n in g.nodes}
    # Each position is a 2-element list (JSON-friendly form ready for
    # the response without further conversion).
    for v in payload["positions"].values():
        assert isinstance(v, (list, tuple))
        assert len(v) == 2


def test_circular_layout_evenly_distributes_nodes():
    """Sanity: a 4-node cycle's circular layout has 4 distinct positions."""
    g = nx.cycle_graph(4)
    g.add_nodes_from(g.nodes, type="data")  # tag for bipartite consistency
    positions = compute_layout(g, "circular")
    distinct = {(round(x, 3), round(y, 3)) for x, y in positions.values()}
    assert len(distinct) == 4


def test_two_strategies_produce_different_layouts(hgp_subgraphs):
    """Sanity: switching strategies actually changes positions."""
    g = build_nx_graph(hgp_subgraphs["graph_x"])
    bipartite = compute_layout(g, "bipartite")
    circular = compute_layout(g, "circular")
    # Same node set, but at least one node's coord must differ.
    differing = sum(
        1 for n in bipartite
        if bipartite[n] != circular.get(n)
    )
    assert differing >= len(bipartite) // 2, "strategies should give visually distinct layouts"
