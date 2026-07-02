"""Hardcoded formulator round-trip tests.

Each family emits a ``cqm_v1`` dict; the round-trip is:

    formulate → compile_cqm_json → energy at brute-force optimum
                                 ≈ test_instance.expected_optimum

For the natural unconstrained families (max_cut, MIS, number
partitioning), the compiled CQM has zero constraints and the energy at
the ground-truth assignment is directly comparable to the expected
optimum (with sign flip for maximize senses).

For portfolio selection the compiled CQM carries the budget as a real
constraint, and the objective energy at the winning portfolio is
compared against the exact brute-force ``expected_optimum``.

These tests are the whole reason the hardcoded package exists — they
prove the LLM was drifting on arithmetic, not structure. The LLM path's
Number Partitioning few-shot got the variable count / offset structure
right but the coefficients drifted 17–32% (see docstrings in
number_partitioning.py). The tests here are exact.
"""

from __future__ import annotations

import pytest

from app.formulation.hardcoded import (
    HardcodedFormulationError,
    formulate,
    list_families,
    parameter_schema,
)
from app.formulation.hardcoded.max_cut import formulate_max_cut
from app.formulation.hardcoded.max_independent_set import (
    formulate_max_independent_set,
)
from app.formulation.hardcoded.number_partitioning import (
    formulate_number_partitioning,
)
from app.formulation.hardcoded.portfolio_selection import (
    formulate_portfolio_selection,
)
from app.optimization.compiler import compile_cqm_json


# ---------------------------------------------------------------------
# Registry surface
# ---------------------------------------------------------------------


def test_list_families_contains_all_four():
    families = list_families()
    assert set(families) == {
        "max_cut",
        "number_partitioning",
        "max_independent_set",
        "portfolio_selection",
    }


def test_parameter_schema_for_each_family_has_required_keys():
    for family in list_families():
        schema = parameter_schema(family)
        assert schema["type"] == "object"
        assert "required" in schema
        assert isinstance(schema["required"], list)
        assert len(schema["required"]) >= 1


def test_registry_rejects_unknown_family():
    with pytest.raises(HardcodedFormulationError, match="Unknown"):
        formulate("traveling_salesman", {})


def test_registry_rejects_missing_required_params():
    with pytest.raises(HardcodedFormulationError, match="missing required"):
        formulate("max_cut", {"node_count": 4})  # missing edges


# ---------------------------------------------------------------------
# Number partitioning — the family that motivated the whole thread
# ---------------------------------------------------------------------


def test_number_partitioning_the_test_case_that_broke_the_llm():
    """[4, 3, 2, 3, 1] — Claude consistently emitted the right structure
    (5 vars, 0 constraints, offset present) but got the coefficients
    wrong by ~20-30%, producing an optimum of 5 instead of 1. The
    hardcoded formulator gets it exactly right."""
    cqm_json = formulate_number_partitioning([4, 3, 2, 3, 1])

    # Structure: 5 binary vars, 0 constraints, offset = S² = 169
    assert len(cqm_json["variables"]) == 5
    assert cqm_json["constraints"] == []
    assert cqm_json["objective"]["offset"] == 169.0
    assert cqm_json["objective"]["sense"] == "minimize"

    # S = 13 → best partition is {4,3} vs {2,3,1} (imbalance = 1)
    assert cqm_json["test_instance"]["expected_optimum"] == 1.0

    # Round-trip: compile and evaluate at the ground-truth optimum
    cqm, _reg, sense = compile_cqm_json(cqm_json)
    assert sense == "minimize"
    # {4,3} in group A (x=0), {2,3,1} in group B (x=1)
    winner = {"x_0": 0, "x_1": 0, "x_2": 1, "x_3": 1, "x_4": 1}
    assert cqm.objective.energy(winner) == 1.0


def test_number_partitioning_perfect_split_hits_zero():
    """[3, 1, 2, 4] with S=10 has a perfect partition {3,2} vs {1,4}
    (imbalance = 0), and the offset makes the compiled energy == 0."""
    cqm_json = formulate_number_partitioning([3, 1, 2, 4])
    assert cqm_json["test_instance"]["expected_optimum"] == 0.0
    cqm, _reg, _sense = compile_cqm_json(cqm_json)
    # {3,2} in A, {1,4} in B
    assert cqm.objective.energy({"x_0": 0, "x_1": 1, "x_2": 0, "x_3": 1}) == 0
    # Bit-flipped mirror also 0
    assert cqm.objective.energy({"x_0": 1, "x_1": 0, "x_2": 1, "x_3": 0}) == 0


def test_number_partitioning_rejects_negative_or_zero():
    with pytest.raises(ValueError, match="positive"):
        formulate_number_partitioning([3, 0, 2])
    with pytest.raises(ValueError, match="positive"):
        formulate_number_partitioning([3, -1, 2])


def test_number_partitioning_rejects_singleton():
    with pytest.raises(ValueError, match="at least 2"):
        formulate_number_partitioning([5])


# ---------------------------------------------------------------------
# Max-Cut
# ---------------------------------------------------------------------


def test_max_cut_square_graph_optimum_is_four():
    """4-node cycle graph — bipartite, so all 4 edges can be cut."""
    cqm_json = formulate_max_cut(
        node_count=4,
        edges=[[0, 1], [1, 2], [2, 3], [0, 3]],
    )
    assert cqm_json["objective"]["sense"] == "maximize"
    assert cqm_json["test_instance"]["expected_optimum"] == 4.0
    # Linear coeff = degree (each node has degree 2 in a cycle)
    assert cqm_json["objective"]["linear"] == {
        "x_0": 2.0, "x_1": 2.0, "x_2": 2.0, "x_3": 2.0,
    }
    # Quadratic = -2 per edge
    for key, coef in cqm_json["objective"]["quadratic"].items():
        assert coef == -2.0, key

    # Round-trip: alternating assignment cuts every edge
    cqm, _reg, sense = compile_cqm_json(cqm_json)
    assert sense == "maximize"
    winner = {"x_0": 0, "x_1": 1, "x_2": 0, "x_3": 1}
    # Under maximize, internal minimization form negates coeffs, so
    # the user-facing cut value is -(internal_energy).
    assert -cqm.objective.energy(winner) == 4.0


def test_max_cut_weighted_graph_uses_weight():
    """Weighted edges — linear coeff = weighted degree."""
    cqm_json = formulate_max_cut(
        node_count=3,
        edges=[[0, 1, 2.5], [1, 2, 1.5]],
    )
    # Node 1 sits between both edges, degree 4.0 (2.5 + 1.5)
    assert cqm_json["objective"]["linear"]["x_0"] == 2.5
    assert cqm_json["objective"]["linear"]["x_1"] == 4.0
    assert cqm_json["objective"]["linear"]["x_2"] == 1.5
    # Optimum: split node 1 alone → cuts both edges (total weight 4.0)
    assert cqm_json["test_instance"]["expected_optimum"] == 4.0


def test_max_cut_collapses_duplicate_edges():
    """Duplicate edges collapse via weight summation, not distinct
    quadratic terms."""
    cqm_json = formulate_max_cut(
        node_count=2, edges=[[0, 1], [0, 1], [1, 0]]
    )
    # Only one unordered edge key present in quadratic dict
    assert len(cqm_json["objective"]["quadratic"]) == 1
    # But its weight is 3 (three copies) → coef = -2·3 = -6
    key = next(iter(cqm_json["objective"]["quadratic"]))
    assert cqm_json["objective"]["quadratic"][key] == -6.0


def test_max_cut_drops_self_loops():
    cqm_json = formulate_max_cut(node_count=3, edges=[[0, 0], [1, 2]])
    # Only the non-self-loop edge survives
    assert len(cqm_json["objective"]["quadratic"]) == 1


def test_max_cut_rejects_out_of_range_edge():
    with pytest.raises(ValueError, match="outside"):
        formulate_max_cut(node_count=3, edges=[[0, 5]])


# ---------------------------------------------------------------------
# Max Independent Set
# ---------------------------------------------------------------------


def test_max_independent_set_triangle_has_optimum_one():
    """Triangle graph: any two nodes share an edge, so the MIS is a
    single node."""
    cqm_json = formulate_max_independent_set(
        node_count=3,
        edges=[[0, 1], [1, 2], [0, 2]],
    )
    assert cqm_json["test_instance"]["expected_optimum"] == 1.0

    # Default penalty = N + 1 = 4
    for coef in cqm_json["objective"]["quadratic"].values():
        assert coef == -4.0
    # Linear = +1 for every node
    for coef in cqm_json["objective"]["linear"].values():
        assert coef == 1.0

    cqm, _reg, sense = compile_cqm_json(cqm_json)
    assert sense == "maximize"
    # Picking any single node: obj = 1, no edge penalty
    assert -cqm.objective.energy({"x_0": 1, "x_1": 0, "x_2": 0}) == 1.0
    # Picking two adjacent nodes: obj = 2 - 4 = -2
    assert -cqm.objective.energy({"x_0": 1, "x_1": 1, "x_2": 0}) == -2.0


def test_max_independent_set_bipartite_optimum_is_larger_side():
    """K_{2,3} bipartite graph — MIS = 3 (the larger side)."""
    cqm_json = formulate_max_independent_set(
        node_count=5,
        edges=[[0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4]],
    )
    assert cqm_json["test_instance"]["expected_optimum"] == 3.0

    cqm, _reg, _sense = compile_cqm_json(cqm_json)
    # {2, 3, 4} is the max independent set → objective = 3
    winner = {"x_0": 0, "x_1": 0, "x_2": 1, "x_3": 1, "x_4": 1}
    assert -cqm.objective.energy(winner) == 3.0


def test_max_independent_set_custom_penalty():
    cqm_json = formulate_max_independent_set(
        node_count=3, edges=[[0, 1]], penalty=2.5,
    )
    key = next(iter(cqm_json["objective"]["quadratic"]))
    assert cqm_json["objective"]["quadratic"][key] == -2.5


def test_max_independent_set_rejects_bad_penalty():
    with pytest.raises(ValueError, match="positive"):
        formulate_max_independent_set(node_count=3, edges=[[0, 1]], penalty=0)


# ---------------------------------------------------------------------
# Portfolio selection
# ---------------------------------------------------------------------


def test_portfolio_selection_picks_best_return_when_lambda_zero():
    """With zero risk aversion the problem reduces to picking top-K
    returns."""
    cqm_json = formulate_portfolio_selection(
        returns=[3.0, 1.0, 5.0, 2.0],
        covariance=[[1.0] * 4 for _ in range(4)],
        max_assets=2,
        risk_aversion=0.0,
    )
    # Top-2 returns: assets 2 (5.0) and 0 (3.0) → objective = 8
    assert cqm_json["test_instance"]["expected_optimum"] == 8.0
    assert cqm_json["objective"]["sense"] == "maximize"
    # Zero-lambda: quadratic is entirely empty (no risk term)
    assert cqm_json["objective"]["quadratic"] == {}
    # Budget constraint present
    assert len(cqm_json["constraints"]) == 1
    assert cqm_json["constraints"][0]["type"] == "inequality_le"
    assert cqm_json["constraints"][0]["rhs"] == 2.0


def test_portfolio_selection_folds_diagonal_into_linear():
    """The variance diagonal x_i² = x_i for binaries, so Σ_{i,i} must
    appear in the linear coefficient (as -λ·Σ_{i,i}), not in
    quadratic."""
    cqm_json = formulate_portfolio_selection(
        returns=[10.0, 10.0],
        covariance=[[4.0, 0.0], [0.0, 4.0]],
        max_assets=2,
        risk_aversion=1.0,
    )
    # μ_i - λ · Σ_{i,i} = 10 - 4 = 6
    assert cqm_json["objective"]["linear"] == {"x_0": 6.0, "x_1": 6.0}
    # Off-diagonal is zero, so no quadratic term at all
    assert cqm_json["objective"]["quadratic"] == {}


def test_portfolio_selection_symmetrizes_covariance():
    """Upper-triangle-only inputs shouldn't silently drop the
    off-diagonal covariance."""
    cqm_json = formulate_portfolio_selection(
        returns=[1.0, 1.0],
        covariance=[[1.0, 2.0], [0.0, 1.0]],  # asymmetric
        max_assets=2,
        risk_aversion=1.0,
    )
    # After symmetrization Σ_{0,1} = (2 + 0)/2 = 1.0, so quadratic
    # coefficient = -2 · 1 · 1 = -2
    assert cqm_json["objective"]["quadratic"]["x_0*x_1"] == -2.0


def test_portfolio_selection_rejects_bad_budget():
    with pytest.raises(ValueError, match="max_assets"):
        formulate_portfolio_selection(
            returns=[1.0, 2.0],
            covariance=[[1.0, 0.0], [0.0, 1.0]],
            max_assets=0,
        )


def test_portfolio_selection_rejects_bad_covariance_shape():
    with pytest.raises(ValueError, match="matrix"):
        formulate_portfolio_selection(
            returns=[1.0, 2.0],
            covariance=[[1.0]],  # wrong shape
            max_assets=1,
        )


# ---------------------------------------------------------------------
# Registry dispatch — end-to-end
# ---------------------------------------------------------------------


def test_registry_dispatches_to_number_partitioning():
    cqm_json = formulate("number_partitioning", {"numbers": [3, 1, 2, 4]})
    assert cqm_json["test_instance"]["expected_optimum"] == 0.0


def test_registry_dispatches_to_max_cut():
    cqm_json = formulate(
        "max_cut", {"node_count": 3, "edges": [[0, 1], [1, 2]]}
    )
    # Path graph 0-1-2: MC = 2 (assign x_1 alone)
    assert cqm_json["test_instance"]["expected_optimum"] == 2.0


def test_registry_dispatches_to_max_independent_set():
    cqm_json = formulate(
        "max_independent_set", {"node_count": 3, "edges": [[0, 1], [1, 2]]}
    )
    # Path graph 0-1-2: MIS = 2 (nodes 0 and 2)
    assert cqm_json["test_instance"]["expected_optimum"] == 2.0


def test_registry_dispatches_to_portfolio_selection():
    cqm_json = formulate(
        "portfolio_selection",
        {
            "returns": [1.0, 2.0, 3.0],
            "covariance": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "max_assets": 2,
            "risk_aversion": 0.0,
        },
    )
    # Pick top-2 (returns 3 and 2), lambda=0 so no variance penalty → 5
    assert cqm_json["test_instance"]["expected_optimum"] == 5.0
