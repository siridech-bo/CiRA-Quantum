"""Phase 2 — CQM-JSON compiler tests.

The compiler converts the cqm_v1 JSON schema into a
``dimod.ConstrainedQuadraticModel`` plus a variable registry and the
original objective sense (used downstream by the validation harness
to convert energies into user-facing units).
"""

from __future__ import annotations

import dimod
import pytest

from app.optimization.compiler import compile_cqm_json


def test_compile_minimal_cqm(minimal_cqm_json):
    cqm, registry, sense = compile_cqm_json(minimal_cqm_json)
    assert isinstance(cqm, dimod.ConstrainedQuadraticModel)
    assert sense == "minimize"
    assert list(cqm.variables) == ["x"]
    assert cqm.vartype("x") == dimod.BINARY
    assert registry["x"] == "A solitary binary"
    assert len(cqm.constraints) == 0


def test_compile_with_binary_and_integer_vars():
    cqm_json = {
        "version": "1",
        "variables": [
            {"name": "x", "type": "binary", "description": "A binary"},
            {
                "name": "y",
                "type": "integer",
                "lower_bound": 0,
                "upper_bound": 10,
                "description": "An integer in [0,10]",
            },
        ],
        "objective": {"sense": "minimize", "linear": {"x": 1, "y": 1}, "quadratic": {}},
        "constraints": [],
    }
    cqm, _registry, _sense = compile_cqm_json(cqm_json)
    assert cqm.vartype("x") == dimod.BINARY
    assert cqm.vartype("y") == dimod.INTEGER
    assert int(cqm.lower_bound("y")) == 0
    assert int(cqm.upper_bound("y")) == 10


def test_compile_with_inequality_constraints(setcover_4item_cqm_json, knapsack_5item_cqm_json):
    # Set cover: four `inequality_ge` constraints.
    cqm_set, _, _ = compile_cqm_json(setcover_4item_cqm_json)
    senses_set = {label: c.sense.name for label, c in cqm_set.constraints.items()}
    assert all(s == "Ge" for s in senses_set.values()), senses_set

    # Knapsack: one `inequality_le` constraint.
    cqm_kp, _, sense_kp = compile_cqm_json(knapsack_5item_cqm_json)
    assert sense_kp == "maximize"
    assert "capacity" in cqm_kp.constraints
    assert cqm_kp.constraints["capacity"].sense.name == "Le"


def test_compile_rejects_invalid_schema():
    # Wrong version
    with pytest.raises(ValueError, match=r"version"):
        compile_cqm_json({"version": "0"})

    # Unknown variable type
    with pytest.raises(ValueError, match=r"variable type"):
        compile_cqm_json({
            "version": "1",
            "variables": [{"name": "x", "type": "complex"}],
            "objective": {"sense": "minimize", "linear": {}, "quadratic": {}},
            "constraints": [],
        })

    # Unknown constraint type
    with pytest.raises(ValueError, match=r"constraint type"):
        compile_cqm_json({
            "version": "1",
            "variables": [{"name": "x", "type": "binary"}],
            "objective": {"sense": "minimize", "linear": {}, "quadratic": {}},
            "constraints": [{
                "label": "c", "type": "approximately",
                "linear": {"x": 1}, "quadratic": {}, "rhs": 0,
            }],
        })

    # Unknown sense
    with pytest.raises(ValueError, match=r"sense"):
        compile_cqm_json({
            "version": "1",
            "variables": [{"name": "x", "type": "binary"}],
            "objective": {"sense": "satisfice", "linear": {}, "quadratic": {}},
            "constraints": [],
        })

    # Reference to undeclared variable in objective
    with pytest.raises(ValueError, match=r"undeclared"):
        compile_cqm_json({
            "version": "1",
            "variables": [{"name": "x", "type": "binary"}],
            "objective": {"sense": "minimize", "linear": {"y": 1}, "quadratic": {}},
            "constraints": [],
        })


def test_variable_registry_preserves_descriptions(knapsack_5item_cqm_json):
    _cqm, registry, _sense = compile_cqm_json(knapsack_5item_cqm_json)
    assert registry["x_0"].startswith("Item 0 selected")
    assert registry["x_4"].startswith("Item 4 selected")
    # Every variable in the JSON appears in the registry
    declared = {v["name"] for v in knapsack_5item_cqm_json["variables"]}
    assert set(registry.keys()) == declared
