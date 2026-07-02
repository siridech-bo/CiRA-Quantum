"""Classifier prompt + parser tests.

The parser is pure — no network, no LLM — so we can exercise every
edge case here without touching Anthropic/OpenAI. The Claude and
OpenAI provider tests cover the HTTP wiring separately.
"""

from __future__ import annotations

import json

import pytest

from app.formulation.base import ClassificationResult, FormulationError
from app.formulation.classifier_prompt import (
    build_classifier_system_prompt,
    build_classifier_user_message,
    parse_classifier_response,
)


def test_system_prompt_lists_every_registered_family():
    prompt = build_classifier_system_prompt()
    for family in ("max_cut", "number_partitioning", "max_independent_set", "portfolio_selection"):
        assert f"### {family}" in prompt


def test_system_prompt_stays_under_2k_tokens_worth_of_chars():
    """Loose budget check — the classifier prompt runs on every
    request when routing is on; we don't want it ballooning silently
    when a new family lands."""
    prompt = build_classifier_system_prompt()
    # ~4 chars/token → 2000 tokens ≈ 8000 chars
    assert len(prompt) < 8000, f"prompt is {len(prompt)} chars, tighten it"


def test_user_message_embeds_problem_verbatim():
    msg = build_classifier_user_message("I have [4, 3, 2, 3, 1]")
    assert "[4, 3, 2, 3, 1]" in msg


def test_parse_confident_number_partitioning_match():
    text = json.dumps({
        "family": "number_partitioning",
        "parameters": {"numbers": [4, 3, 2, 3, 1]},
        "confidence": 0.95,
        "reasoning": "positive integers, two-group split",
    })
    result = parse_classifier_response(text, tokens_used=200, model="claude")
    assert result.family == "number_partitioning"
    assert result.parameters == {"numbers": [4, 3, 2, 3, 1]}
    assert result.confidence == 0.95
    assert result.is_confident is True


def test_parse_low_confidence_still_returns_result():
    """A confident non-match should still round-trip — the
    orchestrator distinguishes 'classifier said no' from 'classifier
    failed'. Only the parse-error path returns None."""
    text = json.dumps({
        "family": "",
        "parameters": {},
        "confidence": 0.9,
        "reasoning": "problem has real-valued objective; no family fits",
    })
    result = parse_classifier_response(text, tokens_used=150, model="gpt-5-mini")
    assert result.family == ""
    assert result.is_confident is False  # empty family → not confident


def test_parse_wraps_llm_prose_around_json():
    """Same JSON extractor as the formulation path handles fenced
    blocks and prose."""
    text = (
        "Here's the classification:\n"
        "```json\n"
        + json.dumps({
            "family": "max_cut",
            "parameters": {"node_count": 4, "edges": [[0, 1], [1, 2]]},
            "confidence": 0.88,
            "reasoning": "graph split objective",
        })
        + "\n```\nHope that helps!"
    )
    result = parse_classifier_response(text, tokens_used=100, model="claude")
    assert result.family == "max_cut"
    assert result.confidence == 0.88


def test_parse_clamps_out_of_range_confidence():
    """Models occasionally emit 1.5 or -0.1 under temperature pressure —
    clamp instead of raise so an otherwise-good result isn't
    discarded."""
    text = json.dumps({
        "family": "max_cut",
        "parameters": {"node_count": 3, "edges": [[0, 1]]},
        "confidence": 1.5,
        "reasoning": "very confident",
    })
    result = parse_classifier_response(text, tokens_used=100, model="claude")
    assert result.confidence == 1.0


def test_parse_downgrades_unknown_family_to_no_match():
    """Model hallucinating a family name shouldn't crash — audit trail
    keeps the original name in reasoning, family becomes empty."""
    text = json.dumps({
        "family": "traveling_salesman",  # not registered
        "parameters": {"cities": [[0, 0], [1, 1]]},
        "confidence": 0.9,
        "reasoning": "TSP problem",
    })
    result = parse_classifier_response(text, tokens_used=100, model="claude")
    assert result.family == ""
    assert result.confidence == 0.0
    assert "traveling_salesman" in result.reasoning


def test_parse_rejects_non_dict_parameters():
    text = json.dumps({
        "family": "max_cut",
        "parameters": "not a dict",
        "confidence": 0.9,
        "reasoning": "",
    })
    with pytest.raises(FormulationError, match="parameters"):
        parse_classifier_response(text, tokens_used=100, model="claude")


def test_parse_rejects_non_numeric_confidence():
    text = json.dumps({
        "family": "max_cut",
        "parameters": {"node_count": 3, "edges": []},
        "confidence": "high",
        "reasoning": "",
    })
    with pytest.raises(FormulationError, match="confidence"):
        parse_classifier_response(text, tokens_used=100, model="claude")


def test_parse_rejects_completely_unparseable_json():
    with pytest.raises(FormulationError):
        parse_classifier_response(
            "sorry, I don't know", tokens_used=50, model="claude"
        )


def test_is_confident_gate_uses_family_and_threshold():
    r = ClassificationResult(
        family="max_cut", parameters={}, confidence=0.86,
    )
    assert r.is_confident is True
    r = ClassificationResult(
        family="max_cut", parameters={}, confidence=0.84,
    )
    assert r.is_confident is False
    r = ClassificationResult(
        family="", parameters={}, confidence=0.99,
    )
    assert r.is_confident is False  # no family means no match, ignore confidence
