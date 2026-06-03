"""JSON serialization helpers for qLDPC code objects (Sprint 1).

The ``qldpc`` library returns ``galois.FieldArray`` matrices and
``networkx.DiGraph`` Tanner graphs. Both need shaping into plain JSON
before they cross the HTTP boundary. These helpers are pure functions
so they're easy to unit-test independently of Flask.

All matrix operations are over GF(2): the field is binary, and
``a @ b`` followed by ``% 2`` gives the expected mod-2 product.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np


def _disable_gap_prompt() -> None:
    """Suppress qldpc's interactive 'do you have GAP?' prompt.

    ``qldpc.external.gap.is_installed`` blocks on ``input()`` when GAP
    isn't on PATH (the upstream lib was written for notebook use). In a
    Flask worker that ``input()`` call hangs the request thread forever.
    We pre-empt the prompt by replacing the function with one that just
    returns ``False`` — equivalent to typing "N" at the prompt, which
    is the correct answer for a server with no GAP. Idempotent + cheap.
    """
    try:
        import qldpc.external.gap as _gap
        _gap.is_installed = lambda: False
    except Exception:
        # If qldpc isn't installed at all the routes 503 long before
        # we get here; nothing to do.
        pass


def matrix_to_int_lists(matrix: Any) -> list[list[int]]:
    """Convert a numpy/galois 2-D array to plain ``list[list[int]]``."""
    arr = np.asarray(matrix, dtype=int)
    return arr.tolist()


def _nonzero_count(arr: np.ndarray) -> int:
    return int(np.count_nonzero(arr))


def code_to_matrix_payload(code: Any) -> dict[str, Any]:
    """Return ``H_X, H_Z`` plus shape metadata as a JSON-ready dict.

    Output schema (matches what the frontend ``QldpcMatrixPayload`` type
    expects):

    .. code-block:: json

        {
          "matrix_x": [[0,1,0,...], ...],
          "matrix_z": [[1,0,1,...], ...],
          "n": <int>,           // physical qubits = #columns
          "k": <int>,           // logical qubits = code.dimension
          "num_checks_x": <int>,
          "num_checks_z": <int>,
          "nonzeros_x": <int>,
          "nonzeros_z": <int>
        }
    """
    mx = np.asarray(code.matrix_x, dtype=int)
    mz = np.asarray(code.matrix_z, dtype=int)
    return {
        "matrix_x": mx.tolist(),
        "matrix_z": mz.tolist(),
        "n": int(code.num_qudits),
        "k": int(code.dimension),
        "num_checks_x": int(mx.shape[0]),
        "num_checks_z": int(mz.shape[0]),
        "nonzeros_x": _nonzero_count(mx),
        "nonzeros_z": _nonzero_count(mz),
    }


def verify_css_commutativity(code: Any) -> dict[str, Any]:
    """Verify ``H_X · H_Zᵀ ≡ 0 (mod 2)`` — the CSS commutativity condition.

    A correct CSS code has every X-stabilizer commuting with every
    Z-stabilizer. Returns the residual non-zero count (0 means the
    code is well-formed; anything else is a bug in the construction
    or a non-CSS code mistakenly being treated as one).
    """
    mx = np.asarray(code.matrix_x, dtype=int)
    mz = np.asarray(code.matrix_z, dtype=int)
    product = (mx @ mz.T) % 2
    residual = _nonzero_count(product)
    return {
        "commutes": residual == 0,
        "residual_nonzero_count": residual,
        "product_shape": list(product.shape),
    }


def compute_distance(code: Any, *, exact: bool = False) -> dict[str, Any]:
    """Compute the code distance with timing + mode metadata.

    Defaults to ``bound=True`` (a fast upper bound). Pass ``exact=True``
    to invoke ``get_distance_exact()`` — slow for HGP/BB but tractable
    for Surface/Toric at our default sizes.

    Returns ``{distance, mode, time_ms}`` ready to JSON-encode.
    """
    _disable_gap_prompt()
    started = time.perf_counter()
    if exact:
        d = code.get_distance_exact()
        mode = "exact"
    else:
        d = code.get_distance(bound=True)
        mode = "upper_bound"
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    # qldpc occasionally returns inf for "no bound found" — coerce so
    # the JSON encoder stays valid.
    distance: int | float | None
    if d == float("inf"):
        distance = None
    elif isinstance(d, float) and d.is_integer():
        distance = int(d)
    else:
        distance = int(d)
    return {
        "distance": distance,
        "mode": mode,
        "time_ms": elapsed_ms,
    }


def code_to_tanner_payload(code: Any) -> dict[str, Any]:
    """Return Tanner subgraph node/edge data ready for Sprint 2 viz.

    The Tanner graph of a CSS code is bipartite: data qubits on one
    side, check operators on the other. The ``qldpc`` lib exposes two
    subgraphs — one per Pauli type. We export both so the Sprint 2
    viewer can render them as overlaid bipartite layouts.

    Output schema:

    .. code-block:: json

        {
          "graph_x": { "nodes": [{"id": str, "type": "data"|"check"}], "edges": [{"source": str, "target": str}] },
          "graph_z": { "nodes": [...], "edges": [...] },
          "node_count_x": <int>,
          "node_count_z": <int>
        }

    Node IDs are stringified so the frontend can use them as keys
    without worrying about networkx's heterogeneous node types.
    """
    return {
        "graph_x": _digraph_to_payload(code.graph_x),
        "graph_z": _digraph_to_payload(code.graph_z),
        "node_count_x": int(len(code.graph_x.nodes)),
        "node_count_z": int(len(code.graph_z.nodes)),
    }


def _digraph_to_payload(graph: Any) -> dict[str, list[dict[str, str]]]:
    nodes: list[dict[str, str]] = []
    for n, attrs in graph.nodes(data=True):
        # ``qldpc`` tags nodes with a ``Node`` object (kind + index).
        # We flatten that into ``{id, type}`` for the frontend.
        node_obj = attrs.get("node") if isinstance(attrs, dict) else None
        if node_obj is not None:
            kind = "check" if getattr(node_obj, "is_data", None) is False else "data"
        else:
            # Fallback heuristic: data nodes are positive integers; the
            # qldpc Tanner subgraphs sometimes use string keys for checks.
            kind = "data" if isinstance(n, (int, np.integer)) and int(n) >= 0 else "check"
        nodes.append({"id": str(n), "type": kind})
    edges = [{"source": str(u), "target": str(v)} for u, v in graph.edges()]
    return {"nodes": nodes, "edges": edges}
