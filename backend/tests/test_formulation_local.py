"""Phase 3 — local-LLM (Ollama) formulation provider tests."""

from __future__ import annotations

import json

import httpx
import pytest

from app.formulation import FormulationError
from app.formulation.local import LocalFormulationProvider

VALID_CQM_TEXT = json.dumps({
    "version": "1",
    "variables": [
        {"name": "x", "type": "binary", "description": "a binary"},
    ],
    "objective": {"sense": "minimize", "linear": {"x": -1}, "quadratic": {}},
    "constraints": [],
    "test_instance": {"description": "Trivial.", "expected_optimum": -1},
})


async def test_local_provider_calls_ollama_endpoint(httpx_mock):
    provider = LocalFormulationProvider(endpoint="http://example-host:11434")
    httpx_mock.add_response(
        url="http://example-host:11434/api/chat",
        method="POST",
        json={"message": {"role": "assistant", "content": VALID_CQM_TEXT},
              "done": True,
              "prompt_eval_count": 100, "eval_count": 50,
              "model": "llama3:8b"},
    )
    result = await provider.formulate("trivial", api_key="", timeout=10)
    request = httpx_mock.get_request()
    assert request.url == "http://example-host:11434/api/chat"
    # The local provider doesn't need an api_key — it should accept "".
    assert result.cqm_json["version"] == "1"
    assert result.tokens_used == 150


async def test_local_provider_handles_streaming_response(httpx_mock):
    """Ollama streams NDJSON chunks; the provider must concatenate the
    ``message.content`` across all chunks before parsing JSON."""
    provider = LocalFormulationProvider(
        endpoint="http://example-host:11434", stream=True
    )
    # Split the JSON string into 4 roughly-equal chunks to simulate streaming.
    parts = [VALID_CQM_TEXT[i:i+50] for i in range(0, len(VALID_CQM_TEXT), 50)]
    chunks = [
        {"message": {"role": "assistant", "content": p}, "done": False, "model": "llama3:8b"}
        for p in parts
    ]
    chunks.append({
        "message": {"role": "assistant", "content": ""},
        "done": True,
        "prompt_eval_count": 100, "eval_count": 50, "model": "llama3:8b",
    })
    body = "\n".join(json.dumps(c) for c in chunks).encode()
    httpx_mock.add_response(
        url="http://example-host:11434/api/chat",
        method="POST",
        content=body,
        headers={"content-type": "application/x-ndjson"},
    )
    result = await provider.formulate("trivial", api_key="", timeout=10)
    assert result.cqm_json["version"] == "1"
    assert result.tokens_used == 150


async def test_local_provider_raises_on_invalid_json(httpx_mock):
    provider = LocalFormulationProvider(endpoint="http://example-host:11434")
    httpx_mock.add_response(
        url="http://example-host:11434/api/chat",
        method="POST",
        json={"message": {"role": "assistant", "content": "I cannot help."},
              "done": True, "model": "llama3:8b"},
    )
    with pytest.raises(FormulationError, match=r"no JSON|cqm_v1"):
        await provider.formulate("write a poem", api_key="", timeout=10)


async def test_local_provider_translates_http_error(httpx_mock):
    provider = LocalFormulationProvider(endpoint="http://example-host:11434")
    httpx_mock.add_response(
        url="http://example-host:11434/api/chat",
        method="POST",
        status_code=500,
        text="model not loaded",
    )
    with pytest.raises(FormulationError, match=r"500|HTTP"):
        await provider.formulate("...", api_key="", timeout=5)


async def test_local_provider_handles_network_error(httpx_mock):
    httpx_mock.add_exception(httpx.ConnectError("ollama not running"))
    provider = LocalFormulationProvider(endpoint="http://example-host:11434")
    with pytest.raises(FormulationError, match=r"network|connect|ollama"):
        await provider.formulate("...", api_key="", timeout=5)


def test_local_estimate_cost_is_zero():
    """Local inference is free — no metering, no charges."""
    provider = LocalFormulationProvider()
    assert provider.estimate_cost("anything goes here") == 0.0


def test_local_provider_endpoint_from_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LOCAL_LLM_ENDPOINT", "http://other-host:9999")
    p = LocalFormulationProvider()
    assert p.endpoint == "http://other-host:9999"


def test_local_provider_model_from_env(monkeypatch: pytest.MonkeyPatch):
    """Operators dial reliability vs latency by setting LOCAL_LLM_MODEL —
    e.g. ``LOCAL_LLM_MODEL=qwen3:14b`` for the integration gate, default
    qwen3:8b for interactive Modules use."""
    monkeypatch.setenv("LOCAL_LLM_MODEL", "qwen3:14b")
    p = LocalFormulationProvider()
    assert p.model == "qwen3:14b"


async def test_local_provider_env_overrides_at_call_time(httpx_mock, monkeypatch):
    """Mid-process env changes propagate without re-importing the
    module — the registered singleton picks up new endpoints/models
    on the next ``formulate`` call."""
    p = LocalFormulationProvider(endpoint="http://stale-host:11434", model="qwen3:8b")
    monkeypatch.setenv("LOCAL_LLM_ENDPOINT", "http://fresh-host:11434")
    monkeypatch.setenv("LOCAL_LLM_MODEL", "qwen3:14b")
    httpx_mock.add_response(
        url="http://fresh-host:11434/api/chat",
        method="POST",
        json={"message": {"role": "assistant", "content": VALID_CQM_TEXT},
              "done": True, "prompt_eval_count": 10, "eval_count": 5,
              "model": "qwen3:14b"},
    )
    await p.formulate("trivial", api_key="", timeout=10)
    request = httpx_mock.get_request()
    assert request.url == "http://fresh-host:11434/api/chat"
    body = json.loads(request.content)
    assert body["model"] == "qwen3:14b"
