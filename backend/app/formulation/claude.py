"""Claude (Anthropic Messages API) formulation provider.

Uses ``httpx.AsyncClient`` directly — not the ``anthropic`` SDK — so
tests can mock the HTTP layer with ``pytest-httpx`` without relying
on the SDK's internal transport. Production behavior is identical.

Cost model
----------
Sonnet pricing (current at Phase 3 ship): $3 per 1M input tokens,
$15 per 1M output tokens. ``estimate_cost`` is a heuristic: ~0.25
tokens per character, plus the system prompt + few-shot examples
(~5500 tokens of fixed overhead), times the per-token rates. It is
deliberately *conservative* (over-estimates) so the UI's pre-submit
warning never under-promises.
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

CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_API_VERSION = "2023-06-01"

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 8192

# Sonnet 4.6 list pricing (USD per token).
_INPUT_USD_PER_TOKEN = 3.0 / 1_000_000
_OUTPUT_USD_PER_TOKEN = 15.0 / 1_000_000
# Approximate fixed prompt overhead in tokens (system prompt + 3 worked examples).
_FIXED_PROMPT_TOKENS = 5500
_CHARS_PER_TOKEN = 4.0  # ~0.25 tokens per char


def _load_prompts_dir() -> Path:
    return Path(__file__).parent / "prompts"


def _system_prompt() -> str:
    return (_load_prompts_dir() / "system.txt").read_text(encoding="utf-8")


def _few_shot_messages() -> list[dict]:
    """Convert ``examples.json`` into Anthropic-style alternating
    user/assistant turns."""
    examples = json.loads(
        (_load_prompts_dir() / "examples.json").read_text(encoding="utf-8")
    )["examples"]
    msgs: list[dict] = []
    for ex in examples:
        msgs.append({"role": "user", "content": ex["problem"]})
        msgs.append({"role": "assistant", "content": json.dumps(ex["cqm_json"])})
    return msgs


class ClaudeFormulationProvider(FormulationProvider):
    name = "claude"

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
            raise FormulationError("Claude provider requires a non-empty api_key")

        messages = _few_shot_messages()
        messages.append({"role": "user", "content": problem_statement})

        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": _system_prompt(),
            "messages": messages,
        }
        headers = {
            "x-api-key": api_key,
            "anthropic-version": CLAUDE_API_VERSION,
            "content-type": "application/json",
        }

        # Log the request *without* the api key. The headers dict is not
        # logged; only the url + model end up in the record.
        logger.debug("claude.formulate model=%s url=%s", self.model, CLAUDE_API_URL)

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(CLAUDE_API_URL, json=payload, headers=headers)
        except httpx.HTTPError as e:
            detail = str(e) or type(e).__name__
            raise FormulationError(f"network error calling Claude: {detail}") from e

        if response.status_code != 200:
            # Surface the status code but *not* the api key.
            try:
                detail = response.json()
            except Exception:
                detail = response.text[:300]
            raise FormulationError(
                f"Claude HTTP {response.status_code}: {detail}"
            )

        body = response.json()
        text = _join_content_text(body.get("content", []))
        try:
            cqm_json = extract_json_object(text)
        except FormulationError:
            raise
        validate_cqm_json(cqm_json)

        usage = body.get("usage", {}) or {}
        tokens_used = int(usage.get("input_tokens", 0)) + int(usage.get("output_tokens", 0))
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
            extras={"input_tokens": usage.get("input_tokens"),
                    "output_tokens": usage.get("output_tokens")},
        )

    def estimate_cost(self, problem_statement: str) -> float:
        prompt_tokens = _FIXED_PROMPT_TOKENS + max(1, len(problem_statement) / _CHARS_PER_TOKEN)
        # Assume completions are ~25% the size of the prompt budget,
        # capped by max_tokens. This is the part that dominates cost.
        completion_tokens = min(self.max_tokens, max(800, prompt_tokens * 0.25))
        cost = (
            prompt_tokens * _INPUT_USD_PER_TOKEN
            + completion_tokens * _OUTPUT_USD_PER_TOKEN
        )
        return float(cost)


def _join_content_text(content: list) -> str:
    """Anthropic's ``content`` is a list of typed blocks. We concatenate
    every ``text`` block's body in order; non-text blocks (tool use,
    images) are ignored — formulation always asks for text."""
    out = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            out.append(block.get("text", ""))
    return "\n".join(out)


# Auto-register on import.
register_provider(ClaudeFormulationProvider())
