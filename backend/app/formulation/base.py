"""Formulation provider abstract base + shared helpers.

Every concrete provider (Claude, OpenAI, local LLM) implements the same
async ``formulate`` contract:

    result = await provider.formulate(problem_statement, api_key)

and produces a ``FormulationResult`` whose ``cqm_json`` field has been
validated against ``schemas/cqm_v1.json``. The downstream pipeline
(Phase 4) feeds that JSON into ``compile_cqm_json`` and then into
``validate_cqm`` — so a CQM that left this layer is guaranteed to be
schema-valid, but not yet guaranteed to be *correct*. Validation is the
next stage's job.

LLMs sometimes wrap JSON in commentary or fenced code blocks. The
``extract_json_object`` helper is shared across providers so each
implementation only owns its HTTP-and-headers contract, not the
fragile job of parsing whatever the model decided to emit.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jsonschema

# ---- Errors ----


class FormulationError(Exception):
    """Anything that goes wrong during formulation: HTTP failure, schema
    violation, refused-to-formulate response, malformed JSON. The Phase
    4 pipeline catches this and marks the job as ``error``."""


# ---- Result dataclass ----


@dataclass
class FormulationResult:
    """Output of one ``formulate`` call.

    ``cqm_json`` has been validated against ``cqm_v1`` before this
    object is returned; ``variable_registry`` is the var-name → human
    description map the LLM included in its response (the compiler
    will recompute its own registry, but having it here lets the
    pipeline surface descriptions to the user even if compilation
    fails).
    """

    cqm_json: dict
    variable_registry: dict[str, str]
    raw_llm_output: str
    tokens_used: int
    model: str
    extras: dict[str, Any] = field(default_factory=dict)


# ---- Provider ABC ----


class FormulationProvider(ABC):
    """Abstract LLM formulation provider.

    Subclasses set ``name`` as a class attribute and implement
    ``formulate`` (async, network-bound) and ``estimate_cost``
    (synchronous, token-count-based). The ABC owns the registry so
    ``app.formulation.get_provider("claude")`` resolves consistently
    no matter where the import happens.
    """

    name: str = ""

    @abstractmethod
    async def formulate(
        self,
        problem_statement: str,
        api_key: str,
        timeout: int = 60,
    ) -> FormulationResult:
        """Convert a natural-language problem statement to a validated
        cqm_v1 JSON document. Raise ``FormulationError`` on any failure
        — the caller is the pipeline orchestrator and treats every
        exception path the same way."""

    @abstractmethod
    def estimate_cost(self, problem_statement: str) -> float:
        """USD estimate based on token count, for the UI's pre-submit
        cost preview. Should not make network calls."""


# ---- Provider registry ----

_PROVIDERS: dict[str, FormulationProvider] = {}


def register_provider(provider: FormulationProvider) -> None:
    if not provider.name:
        raise ValueError("provider must declare a non-empty name")
    if provider.name in _PROVIDERS:
        raise ValueError(f"provider {provider.name!r} is already registered")
    _PROVIDERS[provider.name] = provider


def get_provider(name: str) -> FormulationProvider:
    """Resolve a registered provider. Auto-imports built-in providers
    on first call so callers don't have to remember which module to
    touch."""
    if not _PROVIDERS:
        _bootstrap_builtins()
    if name not in _PROVIDERS:
        raise KeyError(
            f"Unknown formulation provider: {name!r}. Known: {sorted(_PROVIDERS)}"
        )
    return _PROVIDERS[name]


def list_providers() -> list[str]:
    if not _PROVIDERS:
        _bootstrap_builtins()
    return sorted(_PROVIDERS)


def _bootstrap_builtins() -> None:
    """Lazy import of the three built-in providers. Done lazily so that
    ``import app.formulation`` doesn't pull in the HTTP client just to
    instantiate the registry — and so a missing optional dep on one
    provider doesn't poison the others."""
    # Importing each module triggers its module-level register_provider
    # call. We swallow ImportError so a missing extra doesn't kill the
    # whole layer.
    for modname in ("claude", "openai", "local"):
        try:
            __import__(f"app.formulation.{modname}")
        except ImportError:
            pass


# ---- JSON Schema validator ----


_SCHEMA_PATH = Path(__file__).parent / "schemas" / "cqm_v1.json"
_SCHEMA: dict | None = None


def _load_schema() -> dict:
    global _SCHEMA
    if _SCHEMA is None:
        with open(_SCHEMA_PATH) as f:
            _SCHEMA = json.load(f)
    return _SCHEMA


def validate_cqm_json(doc: dict) -> None:
    """Validate ``doc`` against the cqm_v1 schema. Raises
    ``FormulationError`` (with the JSON-Schema validator's message
    preserved) on failure. Returns ``None`` on success — the caller
    already has the document.
    """
    try:
        jsonschema.validate(doc, _load_schema())
    except jsonschema.ValidationError as e:
        path = "/".join(str(p) for p in e.absolute_path) or "<root>"
        raise FormulationError(
            f"cqm_v1 validation failed at /{path}: {e.message}"
        ) from e


# ---- LLM-output JSON extraction ----


# A non-greedy match for the *first* balanced-looking JSON object. The
# extractor below iteratively trims the candidate until ``json.loads``
# accepts it, so this regex only has to find a starting point.
_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json_object(text: str) -> dict:
    """Pull the first parseable JSON object out of LLM output.

    Tolerates three common LLM tics:
      1. Plain JSON with no wrapper.
      2. JSON inside a ```json …``` (or plain ```) fenced block.
      3. JSON inside prose ("Here you go: {…}\nLet me know if…").

    Returns the parsed ``dict``. Raises ``FormulationError`` on no JSON
    found or on unparseable content (with a useful suffix of the input
    so the caller can debug the model's actual output).
    """
    if not isinstance(text, str):
        raise FormulationError(f"expected string, got {type(text).__name__}")

    # 1. Strip ```json fences (and bare ``` fences) first; whatever's
    # inside is treated as the candidate.
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        candidate = fence_match.group(1)
    else:
        m = _JSON_BLOCK_RE.search(text)
        if m is None:
            raise FormulationError(
                "no JSON object found in LLM output: "
                + text[:200].replace("\n", " ")
            )
        candidate = m.group(0)

    # 2. Try parsing. If it fails, peel braces off the end (in case of
    # trailing commentary jammed inside) until parse succeeds or the
    # candidate is exhausted.
    last_err: Exception | None = None
    while candidate:
        try:
            obj = json.loads(candidate)
            if not isinstance(obj, dict):
                raise FormulationError("LLM returned a JSON value that isn't an object")
            return obj
        except json.JSONDecodeError as e:
            last_err = e
            # Trim back to the last preceding '}' and retry.
            cut = candidate.rfind("}", 0, -1)
            if cut == -1:
                break
            candidate = candidate[: cut + 1]

    raise FormulationError(
        f"invalid JSON in LLM output (parse error: {last_err}). "
        f"first 200 chars: {text[:200].replace(chr(10), ' ')}"
    )
