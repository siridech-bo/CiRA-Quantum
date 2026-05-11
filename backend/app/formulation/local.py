"""Local-LLM formulation provider.

Targets the Ollama HTTP API (``POST /api/chat``) running on the same
host or any ``LOCAL_LLM_ENDPOINT``-resolvable address. The provider
supports both single-response and streamed (NDJSON) replies — Ollama
defaults to streaming, so we tolerate both shapes without a
configuration step.

Cost
----
Local inference is metered by the operator's own electricity bill, not
by us. ``estimate_cost`` returns 0 — the platform's cost-preview UI
should show "free (local)" for this provider.

API-key handling
----------------
Ollama does not authenticate. The ``api_key`` parameter is ignored by
this provider. We accept it (with no value required) for interface
parity with the cloud providers, so the pipeline orchestrator doesn't
have to special-case the call site.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import httpx

from app.formulation.base import (
    FormulationError,
    FormulationProvider,
    FormulationResult,
    extract_json_object,
    register_provider,
    validate_cqm_json,
)

logger = logging.getLogger(__name__)

DEFAULT_ENDPOINT = "http://localhost:11434"
# Default chosen empirically: qwen3:8b produces schema-valid CQMs for
# JSS-class prompts in ~60 s on a 16 GB consumer GPU; llama3.1:8b times
# out at 300 s on the same prompt. Both fit comfortably alongside the
# Phase 1 GPU SA solver. See DECISIONS.md for the comparison and the
# README for the honest tier of "what the local provider can / can't do."
#
# Operators who care about reliability over latency can dial up to
# `qwen3:14b` via the LOCAL_LLM_MODEL env var. On the same JSS-3-job-2-machine
# prompt, qwen3:14b lands the correct optimum on 2/2 trials (vs 1/2 for the
# 8B), at 7-8× the latency (~8 min vs ~90 s per call).
DEFAULT_MODEL = "qwen3:8b"


def _load_prompts_dir() -> Path:
    return Path(__file__).parent / "prompts"


def _system_prompt() -> str:
    return (_load_prompts_dir() / "system.txt").read_text(encoding="utf-8")


def _few_shot_messages() -> list[dict]:
    examples = json.loads(
        (_load_prompts_dir() / "examples.json").read_text(encoding="utf-8")
    )["examples"]
    msgs: list[dict] = []
    for ex in examples:
        msgs.append({"role": "user", "content": ex["problem"]})
        msgs.append({"role": "assistant", "content": json.dumps(ex["cqm_json"])})
    return msgs


class LocalFormulationProvider(FormulationProvider):
    name = "local"

    def __init__(
        self,
        endpoint: str | None = None,
        model: str | None = None,
        stream: bool = False,
    ):
        # Read endpoint and model from env on each instance so
        # monkeypatched env vars in tests take effect; the registry's
        # singleton happens *before* env tweaks. ``LOCAL_LLM_MODEL``
        # lets operators dial up (e.g. to ``qwen3:14b``) without code
        # changes — see the README's "Local LLM — known limitations"
        # section for the reliability/latency trade-off.
        self.endpoint = (
            endpoint
            if endpoint is not None
            else os.environ.get("LOCAL_LLM_ENDPOINT", DEFAULT_ENDPOINT)
        )
        self.model = (
            model
            if model is not None
            else os.environ.get("LOCAL_LLM_MODEL", DEFAULT_MODEL)
        )
        self.stream = stream

    async def formulate(
        self,
        problem_statement: str,
        api_key: str,
        timeout: int = 60,
    ) -> FormulationResult:
        # api_key is intentionally unused — accepted for interface parity.
        del api_key

        # Re-read env on each call so the registered singleton tracks
        # operator config changes without restart. (LOCAL_LLM_ENDPOINT
        # and LOCAL_LLM_MODEL are documented in the README.)
        endpoint = os.environ.get("LOCAL_LLM_ENDPOINT", self.endpoint)
        model = os.environ.get("LOCAL_LLM_MODEL", self.model)
        url = endpoint.rstrip("/") + "/api/chat"

        messages: list[dict] = [{"role": "system", "content": _system_prompt()}]
        messages.extend(_few_shot_messages())
        messages.append({"role": "user", "content": problem_statement})

        payload = {
            "model": model,
            "messages": messages,
            "stream": self.stream,
            "format": "json",
        }

        logger.debug("local.formulate model=%s url=%s stream=%s", model, url, self.stream)

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, json=payload)
        except httpx.HTTPError as e:
            # Some httpx exception classes have an empty `str(e)`; fall back
            # to the class name so the surfaced FormulationError is debuggable.
            detail = str(e) or type(e).__name__
            raise FormulationError(
                f"network error calling local LLM (Ollama) at {url}: {detail}"
            ) from e

        if response.status_code != 200:
            raise FormulationError(
                f"Local LLM HTTP {response.status_code}: {response.text[:300]}"
            )

        text, prompt_tokens, completion_tokens, model = self._parse_body(response)

        cqm_json = extract_json_object(text)
        validate_cqm_json(cqm_json)

        registry = {
            v["name"]: v.get("description", "") for v in cqm_json.get("variables", [])
        }
        return FormulationResult(
            cqm_json=cqm_json,
            variable_registry=registry,
            raw_llm_output=text,
            tokens_used=prompt_tokens + completion_tokens,
            model=model,
            extras={"prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens},
        )

    @staticmethod
    def _parse_body(response: httpx.Response) -> tuple[str, int, int, str]:
        """Handle both single-JSON and NDJSON-streamed Ollama responses.

        Returns ``(content_text, prompt_tokens, completion_tokens, model)``.
        Streamed bodies have one JSON object per line, each with a
        partial ``message.content``; the final chunk has ``done=True``
        and the token counts.
        """
        body_bytes = response.content
        text = body_bytes.decode("utf-8")
        # Heuristic: if there's a newline and at least two parseable JSON
        # objects, treat the body as NDJSON. Otherwise parse as a single
        # JSON object. (Both shapes can arrive over the same endpoint
        # depending on the ``stream`` flag.)
        if "\n" in text.strip() and text.count("\n") >= 1:
            chunks: list[dict] = []
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    chunks.append(json.loads(line))
                except json.JSONDecodeError:
                    # Tolerate a stray non-JSON line (some proxies inject
                    # keepalive comments) — skip and keep going.
                    continue
            if not chunks:
                raise FormulationError("Local LLM returned an empty stream")
            content = "".join(
                (c.get("message", {}) or {}).get("content", "") for c in chunks
            )
            terminal = chunks[-1]
        else:
            try:
                terminal = json.loads(text)
            except json.JSONDecodeError as e:
                raise FormulationError(
                    f"Local LLM returned unparseable response: {e}"
                ) from e
            content = (terminal.get("message", {}) or {}).get("content", "")

        prompt_tokens = int(terminal.get("prompt_eval_count", 0) or 0)
        completion_tokens = int(terminal.get("eval_count", 0) or 0)
        model = terminal.get("model", "")
        return content, prompt_tokens, completion_tokens, model

    def estimate_cost(self, problem_statement: str) -> float:
        # Local inference is free — operator's electricity is not platform cost.
        del problem_statement
        return 0.0


register_provider(LocalFormulationProvider())
