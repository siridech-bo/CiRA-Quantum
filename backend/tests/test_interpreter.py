"""Phase 4 — SampleSet → human-readable interpreter tests."""

from __future__ import annotations

import dimod

from app.optimization.interpreter import interpret_solution


def _knapsack_cqm() -> tuple[dimod.ConstrainedQuadraticModel, dict, str]:
    """Build a tiny knapsack CQM for interpreter tests."""
    cqm = dimod.ConstrainedQuadraticModel()
    for i in range(3):
        cqm.add_variable("BINARY", f"x_{i}")
    cqm.objective.add_linear("x_0", -3)
    cqm.objective.add_linear("x_1", -4)
    cqm.objective.add_linear("x_2", -5)
    registry = {
        "x_0": "Item 0 selected (weight 2, value 3)",
        "x_1": "Item 1 selected (weight 3, value 4)",
        "x_2": "Item 2 selected (weight 4, value 5)",
    }
    return cqm, registry, "maximize"


def test_interpret_solution_emits_selected_binaries(_=None):
    cqm, registry, sense = _knapsack_cqm()
    sample = {"x_0": 1, "x_1": 0, "x_2": 1}
    text = interpret_solution(sample, registry, cqm, sense=sense)
    assert "Item 0" in text
    assert "Item 2" in text
    # Item 1 was not selected — it should not appear in the "selected" block.
    assert "Item 1" not in text or "not selected" in text.lower()


def test_interpret_solution_handles_integer_variables():
    cqm = dimod.ConstrainedQuadraticModel()
    cqm.add_variable("INTEGER", "makespan", lower_bound=0, upper_bound=20)
    cqm.add_variable("INTEGER", "start_1", lower_bound=0, upper_bound=10)
    cqm.objective.add_linear("makespan", 1)
    registry = {
        "makespan": "Time at which the last job finishes",
        "start_1": "Start time of job 1",
    }
    sample = {"makespan": 7, "start_1": 0}
    text = interpret_solution(sample, registry, cqm, sense="minimize")
    assert "makespan" in text.lower()
    assert "7" in text
    assert "start_1" in text or "Start time of job 1" in text


def test_interpret_solution_quotes_objective_value():
    """The objective value (in user units) should be reported."""
    cqm, registry, sense = _knapsack_cqm()
    # x_0=1, x_1=0, x_2=1 → value = 3 + 5 = 8 (user-facing for maximize)
    sample = {"x_0": 1, "x_1": 0, "x_2": 1}
    text = interpret_solution(sample, registry, cqm, sense=sense)
    # The interpreter should mention the objective value somewhere.
    assert "8" in text


def test_interpret_solution_handles_unknown_variables_gracefully():
    """A variable in the sample with no registry entry should still appear
    (just without the human description) rather than crash the formatter."""
    cqm = dimod.ConstrainedQuadraticModel()
    cqm.add_variable("BINARY", "x_0")
    cqm.objective.add_linear("x_0", -1)
    sample = {"x_0": 1}
    text = interpret_solution(sample, {}, cqm, sense="minimize")  # empty registry
    assert "x_0" in text


def test_interpret_solution_short_circuits_on_empty_sample():
    cqm = dimod.ConstrainedQuadraticModel()
    text = interpret_solution({}, {}, cqm, sense="minimize")
    assert "no" in text.lower() or "empty" in text.lower()


def test_interpret_solution_marks_infeasible_solutions():
    """If a sample violates a constraint, the interpreter should say so
    rather than silently report it as the answer."""
    cqm = dimod.ConstrainedQuadraticModel()
    cqm.add_variable("BINARY", "x_0")
    cqm.add_variable("BINARY", "x_1")
    cqm.objective.add_linear("x_0", -1)
    cqm.objective.add_linear("x_1", -1)
    # Constraint: x_0 + x_1 <= 1 (can pick at most one)
    qm = dimod.QuadraticModel()
    qm.add_variable("BINARY", "x_0")
    qm.add_variable("BINARY", "x_1")
    qm.add_linear("x_0", 1)
    qm.add_linear("x_1", 1)
    cqm.add_constraint(qm, sense="<=", rhs=1, label="pick_one")

    # Infeasible sample: both selected.
    sample = {"x_0": 1, "x_1": 1}
    text = interpret_solution(sample, {"x_0": "item 0", "x_1": "item 1"}, cqm, sense="maximize")
    assert "infeasible" in text.lower() or "violate" in text.lower() or "violation" in text.lower()
