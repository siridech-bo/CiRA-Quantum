"""Sprint 1 generator + serializer tests.

These are gated with ``pytest.importorskip("qldpc")`` so the suite
passes on contributor machines that haven't installed the ``[qldpc]``
extra. On the production host (where the live `/health` flag is True)
they validate that each factory builds a well-formed CSS code and the
serializers emit JSON-ready shapes.
"""

from __future__ import annotations

import numpy as np
import pytest

# Skip the entire module if qldpc isn't installed.
pytest.importorskip("qldpc")

from app.qldpc import generators, serializers
from app.qldpc.code_families import (
    get_code_family_live,
    invalidate_live_cache,
    list_code_families_live,
)


@pytest.fixture(autouse=True)
def _reset_cache():
    invalidate_live_cache()
    yield
    invalidate_live_cache()


# ---- Generator factories --------------------------------------------------


@pytest.mark.parametrize("family_id", ["surface", "toric", "hypergraph_product", "bicycle"])
def test_factory_builds_well_formed_css_code(family_id: str):
    factory = generators.get_factory(family_id)
    assert factory is not None
    code = factory()

    # CSS shape: every matrix row is over GF(2); commutativity holds.
    mx = np.asarray(code.matrix_x, dtype=int)
    mz = np.asarray(code.matrix_z, dtype=int)
    assert mx.ndim == 2 and mz.ndim == 2
    assert mx.shape[1] == mz.shape[1] == code.num_qudits
    product = (mx @ mz.T) % 2
    assert np.all(product == 0), f"{family_id} violates H_X H_Z^T = 0"

    # n, k must be positive and consistent with matrix shapes.
    assert int(code.num_qudits) > 0
    assert int(code.dimension) > 0


def test_factory_unknown_id_returns_none():
    assert generators.get_factory("does-not-exist") is None


# ---- Serializers ----------------------------------------------------------


def test_matrix_payload_shape():
    code = generators.build_surface()
    payload = serializers.code_to_matrix_payload(code)
    assert set(payload.keys()) >= {
        "matrix_x", "matrix_z", "n", "k",
        "num_checks_x", "num_checks_z", "nonzeros_x", "nonzeros_z",
    }
    # Surface(13) rotated: 84 checks × 169 data qubits.
    assert payload["n"] == 169
    assert payload["k"] == 1
    assert len(payload["matrix_x"]) == payload["num_checks_x"]
    assert len(payload["matrix_x"][0]) == payload["n"]
    # Every entry must be 0 or 1.
    flat = [v for row in payload["matrix_x"] for v in row]
    assert set(flat) <= {0, 1}


def test_css_check_passes_for_surface():
    code = generators.build_surface()
    check = serializers.verify_css_commutativity(code)
    assert check["commutes"] is True
    assert check["residual_nonzero_count"] == 0


def test_compute_distance_returns_int_with_mode_and_time():
    code = generators.build_surface()
    payload = serializers.compute_distance(code, exact=False)
    assert payload["mode"] == "upper_bound"
    assert isinstance(payload["distance"], int)
    assert payload["distance"] > 0
    assert payload["time_ms"] >= 0
    # Surface(13) rotated: distance is exactly 13.
    assert payload["distance"] == 13


def test_compute_distance_exact_for_surface():
    code = generators.build_surface()
    payload = serializers.compute_distance(code, exact=True)
    assert payload["mode"] == "exact"
    assert payload["distance"] == 13


def test_tanner_payload_has_expected_node_counts():
    code = generators.build_surface()
    payload = serializers.code_to_tanner_payload(code)
    assert "graph_x" in payload and "graph_z" in payload
    # Tanner subgraphs of a CSS code: each contains data qubits + the
    # appropriate-type check nodes.
    assert payload["node_count_x"] > 0
    assert payload["node_count_z"] > 0
    # Every node has a type field that's one of two values.
    for graph_key in ("graph_x", "graph_z"):
        for node in payload[graph_key]["nodes"]:
            assert node["type"] in {"data", "check"}, node
        # Every edge has source + target.
        for edge in payload[graph_key]["edges"]:
            assert "source" in edge and "target" in edge


# ---- Live overlay on code_families ----------------------------------------


def test_list_code_families_live_marks_each_entry_live():
    items = list_code_families_live()
    assert len(items) >= 4
    for item in items:
        assert item["live"] is True
        assert isinstance(item["n"], int)
        assert isinstance(item["k"], int)
        # d may be an int or None (in case the lib failed to bound it).
        assert item["d"] is None or isinstance(item["d"], int)


def test_get_code_family_live_returns_live_for_surface():
    fam = get_code_family_live("surface")
    assert fam is not None
    assert fam["live"] is True
    # Live numbers should match the well-known Surface(13) values.
    assert fam["n"] == 169
    assert fam["k"] == 1
    assert fam["d"] == 13


def test_get_code_family_live_unknown_returns_none():
    assert get_code_family_live("does-not-exist") is None
