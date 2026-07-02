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


def test_objective_offset_is_applied_and_maximize_flips_sign():
    """The optional ``objective.offset`` field lets natural-form
    encodings whose quadratic body is shifted (e.g. number
    partitioning's squared-imbalance form has an implicit -S² constant)
    report the user-facing optimum as 0. Behavior:

    - Absent = no-op (offset stays at 0).
    - Present under ``sense="minimize"`` = added directly.
    - Present under ``sense="maximize"`` = sign-flipped along with the
      linear/quadratic coefficients, so the semantics ("this value is
      the constant contribution to the objective in user units")
      survive the internal minimize-negation convention.

    Regression guard for the number-partitioning few-shot anchor —
    without offset support the natural form would report -S² at the
    perfect-partition optimum and the validator's Layer A would treat
    it as a mismatch against the template's expected_optimum=0.
    """
    # Absent = no-op.
    cqm, _reg, _sense = compile_cqm_json({
        "version": "1",
        "variables": [{"name": "x", "type": "binary"}],
        "objective": {"sense": "minimize", "linear": {"x": 1}, "quadratic": {}},
        "constraints": [],
    })
    assert cqm.objective.offset == 0.0

    # Minimize + offset = added directly.
    cqm, _reg, _sense = compile_cqm_json({
        "version": "1",
        "variables": [{"name": "x", "type": "binary"}],
        "objective": {
            "sense": "minimize",
            "linear": {"x": -1},
            "quadratic": {},
            "offset": 42,
        },
        "constraints": [],
    })
    assert cqm.objective.offset == 42.0
    # Optimum at x=1: -1 + 42 = 41
    assert cqm.objective.energy({"x": 1}) == 41

    # Maximize + offset = sign-flipped (matches the linear/quadratic
    # convention so the internal minimize form is a faithful negation
    # of the user-facing maximize form).
    cqm, _reg, _sense = compile_cqm_json({
        "version": "1",
        "variables": [{"name": "x", "type": "binary"}],
        "objective": {
            "sense": "maximize",
            "linear": {"x": 3},
            "quadratic": {},
            "offset": 5,
        },
        "constraints": [],
    })
    # Internally negated: linear becomes -3, offset becomes -5. So at
    # x=1 the internal (minimize) energy is -3 + -5 = -8. The
    # user-facing max value at x=1 is 3 + 5 = 8 = -(-8).
    assert cqm.objective.offset == -5.0
    assert cqm.objective.energy({"x": 1}) == -8


def test_natural_form_number_partitioning_hits_zero_at_perfect_split():
    """The Number Partitioning few-shot example in ``examples.json``
    encodes the [3, 1, 2, 4] instance in the natural unconstrained form
    with ``offset=S²=100`` so both perfect-partition assignments
    ({3,2} vs {1,4} and its bit-flipped mirror) evaluate to 0. This
    test compiles the example straight out of the file and evaluates
    it — proving the anchor's math is correct end-to-end so a
    contributor who tweaks the example without knowing the S²
    accounting doesn't silently break the few-shot."""
    import json
    from pathlib import Path

    examples_path = (
        Path(__file__).parent.parent
        / "app" / "formulation" / "prompts" / "examples.json"
    )
    data = json.loads(examples_path.read_text(encoding="utf-8"))
    # Find the Number Partitioning example by its problem statement.
    np_example = next(
        e for e in data["examples"]
        if "positive numbers" in e["problem"].lower()
    )
    cqm, _reg, sense = compile_cqm_json(np_example["cqm_json"])
    assert sense == "minimize"
    assert len(cqm.variables) == 4      # exactly one qubit per number
    assert len(cqm.constraints) == 0    # unconstrained — the anchor's whole point

    # Both perfect partitions evaluate to 0.
    assert cqm.objective.energy({"x_0": 0, "x_1": 1, "x_2": 0, "x_3": 1}) == 0
    assert cqm.objective.energy({"x_0": 1, "x_1": 0, "x_2": 1, "x_3": 0}) == 0
    # Trivial all-in-one-group assignments evaluate to S² = 100.
    assert cqm.objective.energy({"x_0": 0, "x_1": 0, "x_2": 0, "x_3": 0}) == 100
    assert cqm.objective.energy({"x_0": 1, "x_1": 1, "x_2": 1, "x_3": 1}) == 100
