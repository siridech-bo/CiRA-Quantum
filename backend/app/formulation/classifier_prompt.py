"""Shared classifier prompt + response parser.

Both Claude and OpenAI providers use these helpers to build the
classification system prompt and to parse the model's JSON response
into a ``ClassificationResult``. Keeping this in one module means:

  1. Adding a new hardcoded family updates the classifier's known set
     via ``hardcoded.list_families()`` — no per-provider copy to keep
     in sync.
  2. Bug fixes to the prompt land once (e.g. tightening the
     "confidence" bar or adding a family-specific extraction rule).
  3. Tests can hit the parser directly without spinning up a provider.

The prompt is deliberately compact — under 1500 tokens including all
four family schemas — because classification runs before formulation
on every request when the routing feature flag is on, and we don't
want the extra latency dominating.
"""

from __future__ import annotations

import json
from typing import Any

from app.formulation.base import ClassificationResult, FormulationError
from app.formulation.hardcoded import list_families, parameter_schema


def build_classifier_system_prompt() -> str:
    """Assemble the classifier's system prompt.

    Lists the known hardcoded families with their JSON-schema-like
    parameter shapes and tells the model exactly what JSON envelope to
    return. Deliberately terse — models over-classify when the prompt
    is chatty, and false positives here mean routing a freeform problem
    into a rigid formulator.
    """
    family_blocks: list[str] = []
    for family in list_families():
        schema = parameter_schema(family)
        family_blocks.append(
            f"### {family}\nSchema:\n"
            f"{json.dumps(schema, indent=2)}"
        )
    families_doc = "\n\n".join(family_blocks)

    return (
        "You classify optimization problem statements. Given a "
        "natural-language problem, decide if it matches one of the "
        "known families below. If it does, extract the exact numeric "
        "parameters the schema requires. If it doesn't (or you're "
        "unsure), say so — a false match ships wrong math to the user, "
        "so err toward NOT matching.\n\n"
        "## Known families\n\n"
        f"{families_doc}\n\n"
        "## Output format\n\n"
        "Return ONLY a JSON object with these keys:\n"
        "  - family: one of the family names above, or \"\" for no match\n"
        "  - parameters: the extracted parameters matching the family's "
        "schema (empty dict if family is \"\")\n"
        "  - confidence: a float in [0, 1] — your confidence the "
        "problem is a genuine instance of this family with these "
        "parameters\n"
        "  - reasoning: one short sentence explaining your choice\n\n"
        "## Rules\n\n"
        "1. Parameters must be literal values from the problem "
        "statement, not paraphrases. If the user says [4, 3, 2, 3, 1], "
        "return exactly that list — not a rounded or reordered version.\n"
        "2. If the problem includes constraints or objectives the "
        "family doesn't support (weighted edges when the schema says "
        "unweighted, side-constraints beyond the family's built-in "
        "shape, etc.), set family=\"\" and explain in reasoning.\n"
        "3. Use confidence < 0.5 for weak guesses. Use > 0.85 only when "
        "the problem is unambiguously an instance of the family AND you "
        "have all required parameters.\n"
        "4. Never emit code, prose, or fenced blocks — only the JSON "
        "object. No commentary before or after."
    )


def build_classifier_user_message(problem_statement: str) -> str:
    return (
        f"PROBLEM:\n{problem_statement}\n\n"
        "Classify per the rules and return the JSON object."
    )


def parse_classifier_response(
    text: str,
    tokens_used: int,
    model: str,
) -> ClassificationResult:
    """Parse a classifier response.

    Returns a ``ClassificationResult`` on success. Raises
    ``FormulationError`` on any problem (bad JSON, missing keys,
    unknown family). The Claude/OpenAI provider wrappers turn a raise
    into a ``None`` return so the orchestrator falls back cleanly.
    """
    from app.formulation.base import extract_json_object  # avoid cycle at import
    try:
        parsed = extract_json_object(text)
    except FormulationError:
        raise
    family = parsed.get("family", "")
    if not isinstance(family, str):
        raise FormulationError(
            f"classifier returned non-string family: {family!r}"
        )
    parameters = parsed.get("parameters", {}) or {}
    if not isinstance(parameters, dict):
        raise FormulationError(
            f"classifier returned non-dict parameters: {parameters!r}"
        )
    try:
        confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        raise FormulationError(
            f"classifier returned non-numeric confidence: {parsed.get('confidence')!r}"
        )
    reasoning = str(parsed.get("reasoning", "") or "")

    # Bounds-check confidence rather than trusting the model. Models
    # sometimes emit 1.5 or -0.1 under pressure; clamp instead of raise
    # so an otherwise-good classification doesn't get discarded.
    confidence = max(0.0, min(1.0, confidence))

    # An unknown family name is not automatically fatal — the orch
    # treats "" as "no match" and routes to LLM. But if the model
    # invented a family name, we downgrade to "" with a note in
    # reasoning; that way the audit trail preserves what happened.
    if family and family not in list_families():
        reasoning = f"unknown family {family!r} - falling back. {reasoning}"
        family = ""
        parameters = {}
        confidence = 0.0

    return ClassificationResult(
        family=family,
        parameters=parameters,
        confidence=confidence,
        reasoning=reasoning,
        raw_llm_output=text,
        tokens_used=tokens_used,
        model=model,
    )
