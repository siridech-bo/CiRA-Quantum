"""Phase 3 — Claude formulation provider tests (httpx-mocked)."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from app.formulation import FormulationError
from app.formulation.claude import CLAUDE_API_URL, ClaudeFormulationProvider

FIXTURE = Path(__file__).parent / "fixtures" / "mock_claude_response.json"


@pytest.fixture
def claude_response() -> dict:
    return json.loads(FIXTURE.read_text())


async def test_claude_provider_calls_correct_endpoint(httpx_mock, claude_response):
    httpx_mock.add_response(url=CLAUDE_API_URL, method="POST", json=claude_response)
    provider = ClaudeFormulationProvider()
    await provider.formulate("knapsack with 5 items", api_key="sk-ant-test", timeout=10)
    request = httpx_mock.get_request()
    assert request.url == CLAUDE_API_URL
    assert request.method == "POST"


async def test_claude_provider_extracts_json_from_response(httpx_mock, claude_response):
    httpx_mock.add_response(url=CLAUDE_API_URL, method="POST", json=claude_response)
    provider = ClaudeFormulationProvider()
    result = await provider.formulate("knapsack", api_key="sk-ant-test", timeout=10)
    assert result.cqm_json["version"] == "1"
    assert result.cqm_json["test_instance"]["expected_optimum"] == 26
    assert result.tokens_used == 4321 + 478
    assert result.model.startswith("claude-")
    assert "x_0" in result.variable_registry


async def test_claude_provider_raises_on_invalid_json(httpx_mock):
    bad = {
        "id": "msg_x", "type": "message", "role": "assistant",
        "model": "claude-test",
        "content": [{"type": "text", "text": "I cannot help with that."}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    httpx_mock.add_response(url=CLAUDE_API_URL, method="POST", json=bad)
    provider = ClaudeFormulationProvider()
    with pytest.raises(FormulationError, match=r"no JSON|cqm_v1"):
        await provider.formulate("write a poem", api_key="sk-ant-test", timeout=10)


async def test_claude_provider_raises_on_schema_violation(httpx_mock):
    """Even valid JSON fails fast if it doesn't conform to cqm_v1 — the
    pipeline shouldn't compile-then-discover a missing required field."""
    bad_doc = '{"version": "1", "variables": []}'  # missing objective + constraints
    bad_response = {
        "id": "msg_y", "type": "message", "role": "assistant",
        "model": "claude-test",
        "content": [{"type": "text", "text": bad_doc}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    httpx_mock.add_response(url=CLAUDE_API_URL, method="POST", json=bad_response)
    provider = ClaudeFormulationProvider()
    with pytest.raises(FormulationError, match=r"cqm_v1"):
        await provider.formulate("...", api_key="sk-ant-test", timeout=10)


async def test_claude_provider_passes_api_key_in_header(httpx_mock, claude_response):
    httpx_mock.add_response(url=CLAUDE_API_URL, method="POST", json=claude_response)
    provider = ClaudeFormulationProvider()
    await provider.formulate("knapsack", api_key="sk-ant-test-headercheck", timeout=10)
    request = httpx_mock.get_request()
    assert request.headers.get("x-api-key") == "sk-ant-test-headercheck"
    assert request.headers.get("anthropic-version") is not None


async def test_claude_provider_translates_http_error(httpx_mock):
    httpx_mock.add_response(url=CLAUDE_API_URL, method="POST", status_code=401,
                            json={"error": {"message": "invalid_api_key"}})
    provider = ClaudeFormulationProvider()
    with pytest.raises(FormulationError, match=r"401|HTTP"):
        await provider.formulate("...", api_key="bad-key", timeout=10)


def test_estimate_cost():
    provider = ClaudeFormulationProvider()
    short = provider.estimate_cost("Pack items.")
    long = provider.estimate_cost("Pack items. " * 1000)
    assert short > 0.0
    assert long > short, "longer prompt must cost more"
    assert short < 1.0, "single-prompt formulation should not cost a dollar"


async def test_claude_provider_does_not_log_api_key(httpx_mock, claude_response, caplog):
    """An API key must never end up in the application's log stream —
    the BYOK promise depends on it."""
    import logging

    caplog.set_level(logging.DEBUG)
    httpx_mock.add_response(url=CLAUDE_API_URL, method="POST", json=claude_response)
    provider = ClaudeFormulationProvider()
    await provider.formulate("knapsack", api_key="sk-ant-secret-do-not-log", timeout=10)
    body = "\n".join(record.getMessage() for record in caplog.records)
    assert "sk-ant-secret-do-not-log" not in body, body


async def test_claude_provider_handles_network_error(httpx_mock):
    """A transport failure must surface as FormulationError — the
    pipeline orchestrator shouldn't see a raw httpx exception."""
    httpx_mock.add_exception(httpx.ConnectError("simulated DNS failure"))
    provider = ClaudeFormulationProvider()
    with pytest.raises(FormulationError, match=r"network|connect|DNS|simulated"):
        await provider.formulate("knapsack", api_key="sk-ant-test", timeout=5)
