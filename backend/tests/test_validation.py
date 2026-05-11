"""Phase 2 — three-layer validation harness tests.

Layer A (oracle agreement): ``ExactCQMSolver`` on a small CQM matches
expected_optimum.
Layer B (solver agreement): CPU SA and GPU SA agree on best feasible
energy within tolerance.
Layer C (constraint coverage): every constraint sees both satisfying
and violating assignments under random sampling.
"""

from __future__ import annotations

import pytest
import torch

from app.optimization.compiler import compile_cqm_json
from app.optimization.validation import ValidationReport, validate_cqm

GPU_AVAILABLE = torch.cuda.is_available()


# ---- Layer A ----

def test_layer_a_oracle_agreement_passes_for_correct_cqm(minimal_cqm_json):
    cqm, _registry, sense = compile_cqm_json(minimal_cqm_json)
    # minimal CQM minimizes -x → optimum at x=1, energy=-1 (in user units).
    expected = minimal_cqm_json["test_instance"]["expected_optimum"]
    report = validate_cqm(cqm, expected_optimum=expected, sense=sense, skip_layer_b=True)
    assert report.oracle_agreement, report
    assert report.energy_oracle == pytest.approx(-1.0, abs=1e-6)


def test_layer_a_oracle_agreement_fails_for_wrong_objective_sign(minimal_cqm_json):
    # Flip sense: the original CQM minimizes -x (optimum at x=1, value -1).
    # With sense flipped to "maximize", the compiler negates again so the
    # CQM internally minimizes +x; ExactCQMSolver picks x=0 and the
    # user-facing maximize-sense reading is 0, *not* the original optimum -1.
    flipped = dict(minimal_cqm_json)
    flipped["objective"] = dict(minimal_cqm_json["objective"])
    flipped["objective"]["sense"] = "maximize"
    cqm, _registry, sense = compile_cqm_json(flipped)
    report = validate_cqm(
        cqm,
        expected_optimum=-1.0,         # the original (now-stale) optimum
        sense=sense,
        skip_layer_b=True,
    )
    assert not report.oracle_agreement, report
    assert report.energy_oracle == pytest.approx(0.0, abs=1e-6)
    assert not report.passed


# ---- Layer B ----

@pytest.mark.skipif(not GPU_AVAILABLE, reason="Layer B uses GPU SA")
def test_layer_b_solver_agreement(setcover_4item_cqm_json):
    cqm, _registry, sense = compile_cqm_json(setcover_4item_cqm_json)
    expected = setcover_4item_cqm_json["test_instance"]["expected_optimum"]
    report = validate_cqm(cqm, expected_optimum=expected, sense=sense)
    assert report.solver_agreement, report
    # Both samplers should land on the user-facing optimum (cost=2).
    assert report.energy_cpu_sa == pytest.approx(2.0, abs=1e-3)
    assert report.energy_gpu_sa == pytest.approx(2.0, abs=1e-3)


# ---- Layer C ----

def test_layer_c_detects_inactive_constraint():
    """A constraint that no random assignment can violate (`x <= 100` over
    binary `x`) must be flagged as not-active by Layer C."""
    cqm_json = {
        "version": "1",
        "variables": [{"name": "x", "type": "binary", "description": "binary"}],
        "objective": {"sense": "minimize", "linear": {"x": 1}, "quadratic": {}},
        "constraints": [
            {
                "label": "vacuous_le",
                "type": "inequality_le",
                "linear": {"x": 1}, "quadratic": {},
                "rhs": 100,
            }
        ],
    }
    cqm, _registry, sense = compile_cqm_json(cqm_json)
    report = validate_cqm(cqm, sense=sense, skip_layer_b=True)
    assert report.constraints_active.get("vacuous_le") is False, report.constraints_active
    assert any("vacuous_le" in str(w).lower() for w in report.warnings)


def test_layer_c_detects_missing_constraint(knapsack_5item_cqm_json):
    """If the capacity constraint is removed entirely, Layer C reports nothing
    wrong (it can't see what isn't there) — but Layer A's oracle catches the
    discrepancy: the unconstrained knapsack picks every item.

    This pins Layer C's known limitation: it audits *what is present*, not
    what *should be present*. The combination of A + C is what catches both
    shapes of formulation error.
    """
    no_cap = dict(knapsack_5item_cqm_json)
    no_cap["constraints"] = []  # remove capacity constraint
    cqm, _registry, sense = compile_cqm_json(no_cap)
    expected = knapsack_5item_cqm_json["test_instance"]["expected_optimum"]
    report = validate_cqm(cqm, expected_optimum=expected, sense=sense, skip_layer_b=True)

    # Layer C: trivially passes (no constraints exist to mark inactive)
    assert report.constraints_active == {}
    # Layer A: catches the broken formulation — without capacity, the bag
    # holds all five items for value 30, not the intended 26.
    assert not report.oracle_agreement
    assert report.energy_oracle == pytest.approx(30.0, abs=1e-6)
    assert not report.passed


# ---- Aggregation ----

def test_validation_report_aggregates_correctly(setcover_4item_cqm_json):
    cqm, _registry, sense = compile_cqm_json(setcover_4item_cqm_json)
    expected = setcover_4item_cqm_json["test_instance"]["expected_optimum"]
    report = validate_cqm(cqm, expected_optimum=expected, sense=sense, skip_layer_b=True)

    assert isinstance(report, ValidationReport)
    expected_passed = (
        report.oracle_agreement
        and report.solver_agreement
        and all(report.constraints_active.values())
    )
    assert report.passed == expected_passed
