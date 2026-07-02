"""Stage-1 routing tests — classifier → hardcoded vs LLM fallback.

The orchestrator's ``_formulate_with_routing`` helper decides on each
request whether to skip the LLM CQM-emission step and use a
deterministic hardcoded formulator instead. This module covers every
branch of that decision:

  * Confident classification of a known family → hardcoded route,
    coefficients come from the formulator (not the classifier).
  * Low-confidence classification → LLM fallback, but the classifier's
    result is preserved in the audit trail.
  * Classifier returns None (network failure) → LLM fallback with an
    ``llm_no_classifier`` audit route.
  * Provider without ``classify_problem`` attribute (test stub, legacy
    integration) → LLM fallback, same audit route.
  * Hardcoded formulator raises ``HardcodedFormulationError`` after a
    confident classification → LLM fallback (routing bug shouldn't
    surface as a user error).
  * ``ENABLE_HARDCODED_ROUTING=0`` → LLM path unconditionally, no
    classifier call.
"""

from __future__ import annotations

import asyncio

import pytest

from app.formulation.base import ClassificationResult, FormulationResult
from app.pipeline.orchestrator import Orchestrator


# --- Stubs -------------------------------------------------------------


def _fake_np_cqm() -> dict:
    """A stand-in CQM for the LLM fallback — the orchestrator only
    cares that ``formulate`` returns a valid FormulationResult; the
    routing tests inspect the metadata, not the CQM."""
    return {
        "version": "1",
        "variables": [{"name": "x", "type": "binary", "description": "x"}],
        "objective": {"sense": "minimize", "linear": {"x": 1}, "quadratic": {}},
        "constraints": [],
    }


class _StubProviderWithClassifier:
    """Configurable stub. ``classification`` controls what
    ``classify_problem`` returns; ``formulate_calls`` counts LLM
    fallback usage."""

    name = "stub-with-classifier"

    def __init__(
        self,
        *,
        classification: ClassificationResult | None,
        classification_raises: bool = False,
    ):
        self._classification = classification
        self._classification_raises = classification_raises
        self.classify_calls = 0
        self.formulate_calls = 0

    async def classify_problem(self, problem_statement, api_key, timeout=30):
        self.classify_calls += 1
        if self._classification_raises:
            raise RuntimeError("classifier explosion")
        return self._classification

    async def formulate(self, problem_statement, api_key, timeout=60):
        self.formulate_calls += 1
        return FormulationResult(
            cqm_json=_fake_np_cqm(),
            variable_registry={"x": "x"},
            raw_llm_output="{}",
            tokens_used=100,
            model="stub",
        )

    def estimate_cost(self, problem_statement):
        return 0.0


class _StubProviderWithoutClassifier:
    """Legacy-provider stub — no ``classify_problem`` attribute."""

    name = "stub-legacy"

    def __init__(self):
        self.formulate_calls = 0

    async def formulate(self, problem_statement, api_key, timeout=60):
        self.formulate_calls += 1
        return FormulationResult(
            cqm_json=_fake_np_cqm(),
            variable_registry={"x": "x"},
            raw_llm_output="{}",
            tokens_used=100,
            model="stub-legacy",
        )

    def estimate_cost(self, problem_statement):
        return 0.0


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro) if asyncio.get_event_loop().is_running() else asyncio.run(coro)


# --- Tests -------------------------------------------------------------


def test_confident_classification_takes_hardcoded_route():
    """Classifier confidently matches number_partitioning with valid
    params → the hardcoded formulator emits the CQM; provider.formulate
    is NEVER called."""
    stub = _StubProviderWithClassifier(
        classification=ClassificationResult(
            family="number_partitioning",
            parameters={"numbers": [4, 3, 2, 3, 1]},
            confidence=0.95,
            reasoning="clean number partition",
            tokens_used=250,
            model="claude",
        ),
    )
    orch = Orchestrator(sampler=None, provider_resolver=lambda _: stub)

    result, route = asyncio.run(
        orch._formulate_with_routing(
            provider=stub,
            problem_statement="split [4, 3, 2, 3, 1]",
            api_key="k",
        )
    )
    assert route["route"] == "hardcoded"
    assert route["family"] == "number_partitioning"
    assert route["confidence"] == 0.95
    assert route["classifier_tokens"] == 250
    # CQM must come from the deterministic formulator: 5 vars + offset 169
    assert len(result.cqm_json["variables"]) == 5
    assert result.cqm_json["objective"]["offset"] == 169.0
    # LLM CQM emission was skipped
    assert stub.formulate_calls == 0


def test_low_confidence_falls_back_to_llm():
    stub = _StubProviderWithClassifier(
        classification=ClassificationResult(
            family="max_cut",
            parameters={"node_count": 4, "edges": [[0, 1]]},
            confidence=0.4,  # below the 0.85 threshold
            reasoning="unclear if the user wants Max-Cut or a bipartite check",
            tokens_used=180,
            model="claude",
        ),
    )
    orch = Orchestrator(sampler=None, provider_resolver=lambda _: stub)

    result, route = asyncio.run(
        orch._formulate_with_routing(
            provider=stub, problem_statement="p", api_key="k",
        )
    )
    assert route["route"] == "llm"
    # Audit trail preserves what the classifier said
    assert route["family"] == "max_cut"
    assert route["confidence"] == 0.4
    assert route["reasoning"].startswith("unclear")
    assert stub.formulate_calls == 1
    # LLM produced the CQM
    assert result.cqm_json["variables"][0]["name"] == "x"


def test_classifier_returns_none_uses_llm_no_classifier_route():
    stub = _StubProviderWithClassifier(classification=None)
    orch = Orchestrator(sampler=None, provider_resolver=lambda _: stub)

    _result, route = asyncio.run(
        orch._formulate_with_routing(
            provider=stub, problem_statement="p", api_key="k",
        )
    )
    assert route["route"] == "llm_no_classifier"
    assert route["classifier_tokens"] == 0
    assert stub.formulate_calls == 1


def test_provider_without_classifier_attribute_uses_llm_no_classifier_route():
    stub = _StubProviderWithoutClassifier()
    orch = Orchestrator(sampler=None, provider_resolver=lambda _: stub)

    _result, route = asyncio.run(
        orch._formulate_with_routing(
            provider=stub, problem_statement="p", api_key="k",
        )
    )
    assert route["route"] == "llm_no_classifier"
    assert "does not implement" in route["reasoning"]
    assert stub.formulate_calls == 1


def test_confident_but_hardcoded_rejects_params_falls_back_to_llm():
    """A confident classifier calling out a family with parameters the
    hardcoded formulator's semantic check rejects (negative numbers for
    number_partitioning). Routing must NOT surface the
    HardcodedFormulationError to the user — it falls back to the LLM
    path instead."""
    stub = _StubProviderWithClassifier(
        classification=ClassificationResult(
            family="number_partitioning",
            parameters={"numbers": [4, -3, 2]},  # invalid: negative
            confidence=0.95,
            reasoning="looks like partition to me",
            tokens_used=250,
            model="claude",
        ),
    )
    orch = Orchestrator(sampler=None, provider_resolver=lambda _: stub)

    _result, route = asyncio.run(
        orch._formulate_with_routing(
            provider=stub, problem_statement="p", api_key="k",
        )
    )
    assert route["route"] == "llm"
    # Audit trail keeps the classifier's original decision — the
    # fallback was a routing-side recovery, not a classifier mistake.
    assert route["family"] == "number_partitioning"
    assert stub.formulate_calls == 1


def test_feature_flag_disables_routing(monkeypatch):
    """``ENABLE_HARDCODED_ROUTING=0`` → LLM path, classifier not
    even called."""
    monkeypatch.setenv("ENABLE_HARDCODED_ROUTING", "0")
    stub = _StubProviderWithClassifier(
        classification=ClassificationResult(
            family="number_partitioning",
            parameters={"numbers": [1, 2, 3]},
            confidence=0.99,
            reasoning="obviously partition",
        ),
    )
    orch = Orchestrator(sampler=None, provider_resolver=lambda _: stub)

    _result, route = asyncio.run(
        orch._formulate_with_routing(
            provider=stub, problem_statement="p", api_key="k",
        )
    )
    assert route["route"] == "llm"
    assert route["reasoning"] == "routing disabled via ENABLE_HARDCODED_ROUTING=0"
    assert stub.classify_calls == 0
    assert stub.formulate_calls == 1


def test_empty_family_from_classifier_falls_back_to_llm():
    """Classifier confidently says 'no match' (empty family) → LLM
    path. This is distinct from the None case: the classifier ran, it
    just didn't recognize the problem."""
    stub = _StubProviderWithClassifier(
        classification=ClassificationResult(
            family="",
            parameters={},
            confidence=0.95,
            reasoning="freeform problem with unusual constraints",
        ),
    )
    orch = Orchestrator(sampler=None, provider_resolver=lambda _: stub)

    _result, route = asyncio.run(
        orch._formulate_with_routing(
            provider=stub, problem_statement="p", api_key="k",
        )
    )
    assert route["route"] == "llm"
    assert route["family"] == ""
    assert stub.formulate_calls == 1
