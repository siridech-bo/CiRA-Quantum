"""Phase 3 — FormulationProvider base + cqm_v1 schema validation tests."""

from __future__ import annotations

import json

import pytest

from app.formulation.base import (
    FormulationError,
    FormulationProvider,
    FormulationResult,
    extract_json_object,
    validate_cqm_json,
)

# ---- Schema validation ----


def test_validate_against_schema_accepts_valid_json(knapsack_5item_cqm_json):
    # The Phase-2 fixtures all conform to cqm_v1; the validator must
    # agree, so the formulation layer doesn't reject what the compiler
    # already happily compiles.
    validate_cqm_json(knapsack_5item_cqm_json)


def test_validate_rejects_missing_required_fields():
    with pytest.raises(FormulationError, match=r"version"):
        validate_cqm_json({"variables": [], "objective": {}, "constraints": []})

    with pytest.raises(FormulationError, match=r"objective"):
        validate_cqm_json({
            "version": "1",
            "variables": [{"name": "x", "type": "binary"}],
            "constraints": [],
        })


def test_validate_rejects_five_deliberately_malformed_documents():
    """Phase 3 DoD bullet: schema validation rejects 5 deliberately-malformed
    CQM JSONs. Each one is a different *shape* of failure."""
    bad: list[dict] = [
        # 1. unsupported version
        {"version": "0", "variables": [{"name": "x", "type": "binary"}],
         "objective": {"sense": "minimize", "linear": {}, "quadratic": {}},
         "constraints": []},
        # 2. unknown variable type
        {"version": "1", "variables": [{"name": "x", "type": "ternary"}],
         "objective": {"sense": "minimize", "linear": {}, "quadratic": {}},
         "constraints": []},
        # 3. integer variable missing bounds
        {"version": "1",
         "variables": [{"name": "y", "type": "integer"}],
         "objective": {"sense": "minimize", "linear": {"y": 1}, "quadratic": {}},
         "constraints": []},
        # 4. unknown objective sense
        {"version": "1",
         "variables": [{"name": "x", "type": "binary"}],
         "objective": {"sense": "satisfice", "linear": {}, "quadratic": {}},
         "constraints": []},
        # 5. unknown constraint type
        {"version": "1",
         "variables": [{"name": "x", "type": "binary"}],
         "objective": {"sense": "minimize", "linear": {}, "quadratic": {}},
         "constraints": [{"label": "c", "type": "approximately",
                          "linear": {"x": 1}, "quadratic": {}, "rhs": 0}]},
    ]
    for doc in bad:
        with pytest.raises(FormulationError):
            validate_cqm_json(doc)


# ---- Dataclass ----


def test_formulation_result_dataclass():
    r = FormulationResult(
        cqm_json={"version": "1", "variables": [], "objective": {}, "constraints": []},
        variable_registry={"x_0": "Item 0"},
        raw_llm_output='{"version": "1"}',
        tokens_used=120,
        model="claude-test",
    )
    assert r.cqm_json["version"] == "1"
    assert r.variable_registry["x_0"] == "Item 0"
    assert r.tokens_used == 120
    assert r.model == "claude-test"
    assert r.raw_llm_output.startswith("{")


# ---- Provider ABC contract ----


def test_formulation_provider_is_abstract():
    """``FormulationProvider`` cannot be instantiated directly — concrete
    providers must implement ``formulate`` and ``estimate_cost``."""
    with pytest.raises(TypeError):
        FormulationProvider()  # type: ignore[abstract]


# ---- JSON-from-text extraction (the key resilience helper) ----


def test_extract_json_object_strips_prose_and_fences():
    """LLMs sometimes wrap JSON in chatter or in ```json fences. The
    extractor must recover the JSON regardless and return the parsed
    object, not the raw text."""
    cases = [
        '{"version": "1"}',
        'Here you go:\n```json\n{"version": "1"}\n```',
        'Some preamble.\n{"version": "1"}\nTrailing notes.',
        '```\n{"version": "1"}\n```',
    ]
    for text in cases:
        assert extract_json_object(text) == {"version": "1"}, f"failed on: {text!r}"


def test_extract_json_object_raises_on_no_json():
    with pytest.raises(FormulationError, match=r"no JSON"):
        extract_json_object("I'm sorry, I can only help with optimization problems.")


def test_extract_json_object_raises_on_unparseable_json():
    with pytest.raises(FormulationError, match=r"invalid JSON|parse"):
        extract_json_object('Here is the answer:\n{not valid json,,,}')


# ---- Round-trip with a real example ----


def test_real_phase2_fixtures_round_trip(setcover_4item_cqm_json):
    """A FormulationResult holding a valid Phase-2 fixture should
    serialize and deserialize without losing structure — the dataclass
    and the schema agree on the same canonical shape."""
    validate_cqm_json(setcover_4item_cqm_json)
    serialized = json.dumps(setcover_4item_cqm_json, sort_keys=True)
    reloaded = json.loads(serialized)
    validate_cqm_json(reloaded)
    assert reloaded == setcover_4item_cqm_json
