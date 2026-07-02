"""OpenAI (Chat Completions API) formulation provider.

Mirror of ``claude.py`` with three differences:

  * Endpoint is ``POST /v1/chat/completions`` instead of ``/v1/messages``.
  * Auth header is ``Authorization: Bearer …`` instead of ``x-api-key``.
  * The system prompt is the first message in ``messages`` (role
    ``system``); Anthropic carries it as a top-level ``system`` field.

We use ``httpx.AsyncClient`` for the same testability reason as Claude.
"""

from __future__ import annotations

import json
import logging
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

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

DEFAULT_MODEL = "gpt-5-mini"
DEFAULT_MAX_TOKENS = 8192

# GPT-5-mini list pricing (current at Phase 3 ship; conservative round numbers).
_INPUT_USD_PER_TOKEN = 0.25 / 1_000_000
_OUTPUT_USD_PER_TOKEN = 2.0 / 1_000_000
_FIXED_PROMPT_TOKENS = 5500
_CHARS_PER_TOKEN = 4.0


def _load_prompts_dir() -> Path:
    return Path(__file__).parent / "prompts"


def _system_prompt() -> str:
    return (_load_prompts_dir() / "system.txt").read_text(encoding="utf-8")


def _few_shot_messages() -> list[dict]:
    """Convert ``examples.json`` into OpenAI-style alternating
    user/assistant messages."""
    examples = json.loads(
        (_load_prompts_dir() / "examples.json").read_text(encoding="utf-8")
    )["examples"]
    msgs: list[dict] = []
    for ex in examples:
        msgs.append({"role": "user", "content": ex["problem"]})
        msgs.append({"role": "assistant", "content": json.dumps(ex["cqm_json"])})
    return msgs


class OpenAIFormulationProvider(FormulationProvider):
    name = "openai"

    def __init__(self, model: str = DEFAULT_MODEL, max_tokens: int = DEFAULT_MAX_TOKENS):
        self.model = model
        self.max_tokens = max_tokens

    async def formulate(
        self,
        problem_statement: str,
        api_key: str,
        timeout: int = 60,
    ) -> FormulationResult:
        if not api_key:
            raise FormulationError("OpenAI provider requires a non-empty api_key")

        messages: list[dict] = [{"role": "system", "content": _system_prompt()}]
        messages.extend(_few_shot_messages())
        messages.append({"role": "user", "content": problem_statement})

        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages,
            # response_format is optional — not all OpenAI endpoints honor
            # it, and the schema validator catches bad outputs anyway.
            "response_format": {"type": "json_object"},
        }
        headers = {
            "authorization": f"Bearer {api_key}",
            "content-type": "application/json",
        }

        logger.debug("openai.formulate model=%s url=%s", self.model, OPENAI_API_URL)

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(OPENAI_API_URL, json=payload, headers=headers)
        except httpx.HTTPError as e:
            detail = str(e) or type(e).__name__
            raise FormulationError(f"network error calling OpenAI: {detail}") from e

        if response.status_code != 200:
            try:
                detail = response.json()
            except Exception:
                detail = response.text[:300]
            raise FormulationError(
                f"OpenAI HTTP {response.status_code}: {detail}"
            )

        body = response.json()
        choices = body.get("choices", [])
        if not choices:
            raise FormulationError("OpenAI returned no choices")
        text = choices[0].get("message", {}).get("content", "") or ""

        cqm_json = extract_json_object(text)
        validate_cqm_json(cqm_json)

        usage = body.get("usage", {}) or {}
        tokens_used = int(usage.get("total_tokens", 0)) or (
            int(usage.get("prompt_tokens", 0)) + int(usage.get("completion_tokens", 0))
        )
        model = body.get("model", self.model)

        registry = {
            v["name"]: v.get("description", "") for v in cqm_json.get("variables", [])
        }
        return FormulationResult(
            cqm_json=cqm_json,
            variable_registry=registry,
            raw_llm_output=text,
            tokens_used=tokens_used,
            model=model,
            extras={"prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens")},
        )

    async def summarize_solution(
        self,
        problem_statement: str,
        interpreted_solution: str,
        api_key: str,
        timeout: int = 30,
    ) -> str | None:
        """Best-effort plain-English rewrite. Returns ``None`` on any
        failure so the solve still completes without a summary."""
        if not api_key:
            return None

        system = (
            "You translate optimization-solver output into a plain-English "
            "answer that matches the user's original question. Rules:\n"
            "1. Account for EVERY variable/item/entity the solver "
            "reported. Never drop items to save space.\n"
            "2. Use the user's own vocabulary (nodes/items/groups/bins/etc.).\n"
            "3. Show concrete values: which item goes where, the totals, "
            "and the final objective value.\n"
            "4. Ignore auxiliary/slack variables (sum_A, sum_B, y_bin, "
            "absolute-difference indicators). Report only what the user "
            "asked about.\n"
            "5. Start with the answer directly. No preamble.\n"
            "6. 2-4 short sentences. Use braces {a, b, c} for lists."
        )
        user = (
            f"USER'S QUESTION:\n{problem_statement}\n\n"
            f"SOLVER'S RAW OUTPUT:\n{interpreted_solution}\n\n"
            "Rewrite as a plain-English answer following the rules."
        )
        payload = {
            "model": self.model,
            "max_tokens": 400,
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        headers = {
            "authorization": f"Bearer {api_key}",
            "content-type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    OPENAI_API_URL, json=payload, headers=headers,
                )
        except httpx.HTTPError:
            return None

        if response.status_code != 200:
            return None

        body = response.json()
        choices = body.get("choices", [])
        if not choices:
            return None
        text = (choices[0].get("message", {}).get("content", "") or "").strip()
        return text or None

    def estimate_cost(self, problem_statement: str) -> float:
        prompt_tokens = _FIXED_PROMPT_TOKENS + max(1, len(problem_statement) / _CHARS_PER_TOKEN)
        completion_tokens = min(self.max_tokens, max(800, prompt_tokens * 0.25))
        return float(
            prompt_tokens * _INPUT_USD_PER_TOKEN
            + completion_tokens * _OUTPUT_USD_PER_TOKEN
        )


register_provider(OpenAIFormulationProvider())
