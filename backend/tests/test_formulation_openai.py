"""Phase 3 — OpenAI formulation provider tests (httpx-mocked)."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from app.formulation import FormulationError
from app.formulation.openai import OPENAI_API_URL, OpenAIFormulationProvider

FIXTURE = Path(__file__).parent / "fixtures" / "mock_openai_response.json"


@pytest.fixture
def openai_response() -> dict:
    return json.loads(FIXTURE.read_text())


async def test_openai_provider_calls_correct_endpoint(httpx_mock, openai_response):
    httpx_mock.add_response(url=OPENAI_API_URL, method="POST", json=openai_response)
    provider = OpenAIFormulationProvider()
    await provider.formulate("knapsack with 5 items", api_key="sk-test", timeout=10)
    request = httpx_mock.get_request()
    assert request.url == OPENAI_API_URL
    assert request.method == "POST"


async def test_openai_provider_extracts_json_from_response(httpx_mock, openai_response):
    httpx_mock.add_response(url=OPENAI_API_URL, method="POST", json=openai_response)
    provider = OpenAIFormulationProvider()
    result = await provider.formulate("knapsack", api_key="sk-test", timeout=10)
    assert result.cqm_json["test_instance"]["expected_optimum"] == 26
    assert result.tokens_used == 5501
    assert result.model.startswith("gpt-")


async def test_openai_provider_raises_on_invalid_json(httpx_mock):
    bad = {
        "id": "chatcmpl-x", "object": "chat.completion", "created": 0, "model": "gpt-5-mini",
        "choices": [{"index": 0,
                     "message": {"role": "assistant",
                                 "content": "I refuse to formulate that."},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    httpx_mock.add_response(url=OPENAI_API_URL, method="POST", json=bad)
    provider = OpenAIFormulationProvider()
    with pytest.raises(FormulationError, match=r"no JSON|cqm_v1"):
        await provider.formulate("write me a poem", api_key="sk-test", timeout=10)


async def test_openai_provider_passes_api_key_in_bearer_header(httpx_mock, openai_response):
    httpx_mock.add_response(url=OPENAI_API_URL, method="POST", json=openai_response)
    provider = OpenAIFormulationProvider()
    await provider.formulate("knapsack", api_key="sk-openai-secret", timeout=10)
    auth = httpx_mock.get_request().headers.get("authorization", "")
    assert auth == "Bearer sk-openai-secret"


async def test_openai_provider_translates_http_error(httpx_mock):
    httpx_mock.add_response(url=OPENAI_API_URL, method="POST", status_code=429,
                            json={"error": {"message": "rate_limited"}})
    provider = OpenAIFormulationProvider()
    with pytest.raises(FormulationError, match=r"429|HTTP|rate"):
        await provider.formulate("...", api_key="sk-test", timeout=10)


def test_estimate_cost():
    provider = OpenAIFormulationProvider()
    short = provider.estimate_cost("Pack items.")
    long = provider.estimate_cost("Pack items. " * 1000)
    assert short > 0.0
    assert long > short
    assert short < 1.0


async def test_openai_provider_does_not_log_api_key(httpx_mock, openai_response, caplog):
    import logging

    caplog.set_level(logging.DEBUG)
    httpx_mock.add_response(url=OPENAI_API_URL, method="POST", json=openai_response)
    provider = OpenAIFormulationProvider()
    await provider.formulate("knapsack", api_key="sk-openai-no-log-please", timeout=10)
    body = "\n".join(record.getMessage() for record in caplog.records)
    assert "sk-openai-no-log-please" not in body, body


async def test_openai_provider_handles_network_error(httpx_mock):
    httpx_mock.add_exception(httpx.ConnectError("simulated DNS failure"))
    provider = OpenAIFormulationProvider()
    with pytest.raises(FormulationError, match=r"network|connect|DNS|simulated"):
        await provider.formulate("knapsack", api_key="sk-test", timeout=5)
