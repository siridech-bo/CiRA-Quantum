"""Shared fixtures for backend tests.

Provides:
  * BQM loaders + small hand-crafted BQMs (Phase 1, GPU SA).
  * CQM-JSON loaders + canonical instance fixtures (Phase 2, validation).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import dimod
import pytest

INSTANCES_DIR = Path(__file__).parent / "instances"


def load_bqm_from_json(path: Path) -> dimod.BinaryQuadraticModel:
    """Load a BQM from a simple JSON serialization.

    Format::

        {
          "vartype": "BINARY" | "SPIN",
          "linear":    {"<var>": <bias>, ...},
          "quadratic": {"<u>,<v>": <bias>, ...},
          "offset":    <float>
        }

    Variable labels are stored as strings; they are converted to ``int`` if
    they parse cleanly, otherwise left as strings.
    """
    with open(path) as f:
        data = json.load(f)

    vartype = dimod.SPIN if data["vartype"] == "SPIN" else dimod.BINARY

    def _key(s: str):
        try:
            return int(s)
        except ValueError:
            return s

    linear = {_key(k): float(v) for k, v in data["linear"].items()}

    quadratic: dict[tuple, float] = {}
    for k, v in data["quadratic"].items():
        u, w = k.split(",")
        quadratic[(_key(u), _key(w))] = float(v)

    offset = float(data.get("offset", 0.0))

    return dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype)


# ----- Hand-crafted BQMs -----


@pytest.fixture
def two_var_bqm() -> dimod.BinaryQuadraticModel:
    """The exact instance the spec calls out:

    BQM = {0: -1, 1: 1, (0, 1): 2}, BINARY
        - (0, 0):  0
        - (0, 1):  1
        - (1, 0): -1   <- optimum
        - (1, 1):  2
    """
    return dimod.BinaryQuadraticModel(
        {0: -1.0, 1: 1.0},
        {(0, 1): 2.0},
        0.0,
        dimod.BINARY,
    )


def _random_bqm(n: int, seed: int, vartype=dimod.BINARY) -> dimod.BinaryQuadraticModel:
    """Deterministic dense random BQM."""
    rng = random.Random(seed)
    linear = {i: rng.uniform(-2.0, 2.0) for i in range(n)}
    quadratic = {}
    for i in range(n):
        for j in range(i + 1, n):
            quadratic[(i, j)] = rng.uniform(-2.0, 2.0)
    return dimod.BinaryQuadraticModel(linear, quadratic, 0.0, vartype)


@pytest.fixture
def random_5var_bqm() -> dimod.BinaryQuadraticModel:
    return _random_bqm(5, seed=7)


@pytest.fixture
def random_10var_bqm() -> dimod.BinaryQuadraticModel:
    return _random_bqm(10, seed=11)


# ----- BQMs loaded from JSON instance files -----


@pytest.fixture
def tiny_5var_bqm() -> dimod.BinaryQuadraticModel:
    return load_bqm_from_json(INSTANCES_DIR / "tiny_5var.json")


@pytest.fixture
def medium_100var_bqm() -> dimod.BinaryQuadraticModel:
    return load_bqm_from_json(INSTANCES_DIR / "medium_100var.json")


@pytest.fixture
def benchmark_1000var_bqm() -> dimod.BinaryQuadraticModel:
    return load_bqm_from_json(INSTANCES_DIR / "benchmark_1000var.json")


@pytest.fixture
def empty_bqm() -> dimod.BinaryQuadraticModel:
    return dimod.BinaryQuadraticModel.empty(dimod.BINARY)


# ----- Phase 2 — CQM JSON helpers and fixtures -----


def load_cqm_json(path: Path) -> dict:
    """Load a CQM JSON file (cqm_v1 schema) as a plain Python dict.

    The compiler under test consumes this dict; the loader exists only to
    keep tests independent of the compiler's filesystem expectations.
    """
    with open(path) as f:
        return json.load(f)


@pytest.fixture
def knapsack_5item_cqm_json() -> dict:
    return load_cqm_json(INSTANCES_DIR / "knapsack_5item.json")


@pytest.fixture
def setcover_4item_cqm_json() -> dict:
    return load_cqm_json(INSTANCES_DIR / "setcover_4item.json")


@pytest.fixture
def maxcut_6node_cqm_json() -> dict:
    return load_cqm_json(INSTANCES_DIR / "maxcut_6node.json")


@pytest.fixture
def graphcoloring_4node_cqm_json() -> dict:
    return load_cqm_json(INSTANCES_DIR / "graphcoloring_4node.json")


@pytest.fixture
def jss_3job_3machine_cqm_json() -> dict:
    return load_cqm_json(INSTANCES_DIR / "jss_3job_3machine.json")


@pytest.fixture
def jss_2job_2machine_cqm_json() -> dict:
    return load_cqm_json(INSTANCES_DIR / "jss_2job_2machine.json")


@pytest.fixture
def minimal_cqm_json() -> dict:
    """Smallest valid cqm_v1 document: one binary variable, no constraints."""
    return {
        "version": "1",
        "variables": [
            {"name": "x", "type": "binary", "description": "A solitary binary"},
        ],
        "objective": {
            "sense": "minimize",
            "linear": {"x": -1.0},
            "quadratic": {},
        },
        "constraints": [],
        "test_instance": {
            "description": "Trivial: setting x=1 minimizes -x.",
            "expected_optimum": -1.0,
        },
    }
