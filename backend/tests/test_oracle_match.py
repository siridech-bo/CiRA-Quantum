"""Phase 2 — gold-standard oracle-match tests.

These run the full validation harness on each canonical instance and
assert that ``ExactCQMSolver`` (Layer A) — when it can run — agrees with
the ``expected_optimum`` shipped in the JSON, and that the validation
report comes out green overall. The JSS-3x3 instance has 19 variables,
above the default oracle ceiling of 12, so its Layer A is recorded as
*skipped* (vacuous pass) and only Layers B + C carry weight there.
"""

from __future__ import annotations

import pytest
import torch

from app.optimization.compiler import compile_cqm_json
from app.optimization.validation import validate_cqm

GPU_AVAILABLE = torch.cuda.is_available()


def _validate_from_json(cqm_json: dict, **kwargs):
    cqm, _registry, sense = compile_cqm_json(cqm_json)
    expected = cqm_json["test_instance"]["expected_optimum"]
    return validate_cqm(cqm, expected_optimum=expected, sense=sense, **kwargs), cqm


@pytest.mark.skipif(not GPU_AVAILABLE, reason="oracle-match Layer B requires GPU SA")
def test_knapsack_5item_matches_oracle(knapsack_5item_cqm_json):
    report, _cqm = _validate_from_json(knapsack_5item_cqm_json)
    assert not report.oracle_skipped
    assert report.oracle_agreement, report.warnings
    assert report.energy_oracle == pytest.approx(26.0, abs=1e-6)
    assert report.passed, report.warnings


@pytest.mark.skipif(not GPU_AVAILABLE, reason="oracle-match Layer B requires GPU SA")
def test_setcover_4item_matches_oracle(setcover_4item_cqm_json):
    report, _cqm = _validate_from_json(setcover_4item_cqm_json)
    assert not report.oracle_skipped
    assert report.oracle_agreement
    assert report.energy_oracle == pytest.approx(2.0, abs=1e-6)
    assert report.passed, report.warnings


@pytest.mark.skipif(not GPU_AVAILABLE, reason="oracle-match Layer B requires GPU SA")
def test_jss_3x3_matches_brute_force(jss_2job_2machine_cqm_json, jss_3job_3machine_cqm_json):
    """The brute-force-match arm uses the 2-job × 2-machine instance
    (7 variables, brute-forceable). The full 3-job × 3-machine encoding
    has 19 variables — too many for ``ExactCQMSolver`` (it would need to
    allocate a 115 TiB cartesian product) — and its big-M disjunctive
    constraints are too tight for vanilla SA to find feasible samples;
    we run the validation harness on it solely to verify it gracefully
    skips Layer A and that Layer C's coverage report is sane.
    See DECISIONS.md for the full rationale.
    """
    # Brute-force match on the small instance.
    report_2x2, _ = _validate_from_json(jss_2job_2machine_cqm_json)
    assert not report_2x2.oracle_skipped
    assert report_2x2.oracle_agreement, report_2x2.warnings
    assert report_2x2.energy_oracle == pytest.approx(4.0, abs=1e-6)
    assert report_2x2.passed, report_2x2.warnings

    # Sanity-check on the 3x3 instance: oracle is skipped, every constraint
    # in the encoding still flips between satisfied and violated under
    # random sampling (Layer C is meaningful even when Layer A can't run).
    cqm_3x3, _registry, sense = compile_cqm_json(jss_3job_3machine_cqm_json)
    report_3x3 = validate_cqm(cqm_3x3, sense=sense, skip_layer_b=True)
    assert report_3x3.oracle_skipped
    assert all(report_3x3.constraints_active.values()), report_3x3.warnings


@pytest.mark.skipif(not GPU_AVAILABLE, reason="oracle-match Layer B requires GPU SA")
def test_maxcut_6node_matches_oracle(maxcut_6node_cqm_json):
    report, _cqm = _validate_from_json(maxcut_6node_cqm_json)
    assert not report.oracle_skipped
    assert report.oracle_agreement
    assert report.energy_oracle == pytest.approx(7.0, abs=1e-6)
    assert report.passed, report.warnings
