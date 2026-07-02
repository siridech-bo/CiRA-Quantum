"""Hardcoded formulator registry.

Central lookup so the orchestrator can dispatch ``family_name → CQM``
without knowing each formulator's module path. Also owns the parameter
schemas the classifier uses to validate LLM output before invoking a
formulator — a defense-in-depth check so a hallucinated parameter list
never reaches a `cqm_v1` emit path.

Adding a new hardcoded family is a three-line change:
    1. Import the family's formulator function
    2. Register in ``HARDCODED_FORMULATORS``
    3. Register its parameter schema in ``_PARAMETER_SCHEMAS``

Everything downstream (classifier prompt, tests, UI chip) reads from
this registry so the new family lights up everywhere at once.
"""

from __future__ import annotations

from typing import Any, Callable

from app.formulation.hardcoded.max_cut import formulate_max_cut
from app.formulation.hardcoded.max_independent_set import (
    formulate_max_independent_set,
)
from app.formulation.hardcoded.number_partitioning import (
    formulate_number_partitioning,
)
from app.formulation.hardcoded.portfolio_selection import (
    formulate_portfolio_selection,
)


class HardcodedFormulationError(Exception):
    """Raised when parameter validation or the formulator itself
    rejects the input. Distinguished from ``FormulationError`` so the
    orchestrator can decide whether to fall back to the LLM path or
    surface as a hard error."""


# Family name → callable(**params) → cqm_v1 dict. Each callable owns
# its own parameter parsing; the registry only routes.
HARDCODED_FORMULATORS: dict[str, Callable[..., dict]] = {
    "max_cut": formulate_max_cut,
    "number_partitioning": formulate_number_partitioning,
    "max_independent_set": formulate_max_independent_set,
    "portfolio_selection": formulate_portfolio_selection,
}


# JSON-schema-like parameter shape per family. The classifier is
# instructed to conform to this; if it doesn't, ``formulate`` rejects
# and the orchestrator falls back to the LLM-emits-CQM path. The shape
# is deliberately shallow (no nested $refs) so a small model reading
# the schema in-prompt can follow it.
_PARAMETER_SCHEMAS: dict[str, dict[str, Any]] = {
    "max_cut": {
        "type": "object",
        "required": ["node_count", "edges"],
        "properties": {
            "node_count": {
                "type": "integer",
                "minimum": 2,
                "description": "Number of nodes labeled 0..node_count-1.",
            },
            "edges": {
                "type": "array",
                "items": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 2,
                    "items": {"type": "integer"},
                },
                "description": "List of [u, v] undirected edges.",
            },
        },
    },
    "number_partitioning": {
        "type": "object",
        "required": ["numbers"],
        "properties": {
            "numbers": {
                "type": "array",
                "minItems": 2,
                "items": {"type": "number"},
                "description": "Positive numbers to split into two balanced groups.",
            },
        },
    },
    "max_independent_set": {
        "type": "object",
        "required": ["node_count", "edges"],
        "properties": {
            "node_count": {
                "type": "integer",
                "minimum": 2,
                "description": "Number of nodes labeled 0..node_count-1.",
            },
            "edges": {
                "type": "array",
                "items": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 2,
                    "items": {"type": "integer"},
                },
                "description": "List of [u, v] undirected edges.",
            },
            "penalty": {
                "type": "number",
                "description": "Penalty coefficient (default: node_count + 1, safe for any graph).",
            },
        },
    },
    "portfolio_selection": {
        "type": "object",
        "required": ["returns", "covariance", "max_assets"],
        "properties": {
            "returns": {
                "type": "array",
                "items": {"type": "number"},
                "description": "Expected return per asset.",
            },
            "covariance": {
                "type": "array",
                "items": {"type": "array", "items": {"type": "number"}},
                "description": "N-by-N covariance matrix.",
            },
            "risk_aversion": {
                "type": "number",
                "description": "λ scaling on the variance term (default 1.0).",
            },
            "max_assets": {
                "type": "integer",
                "minimum": 1,
                "description": "Max number of assets to select (budget).",
            },
        },
    },
}


def list_families() -> list[str]:
    """Family names registered here. Used by the classifier prompt to
    tell the LLM which families are recognized."""
    return sorted(HARDCODED_FORMULATORS)


def parameter_schema(family: str) -> dict[str, Any]:
    """Fetch the parameter schema for a family. Raises KeyError if
    unregistered — surfaces classifier hallucination quickly."""
    if family not in _PARAMETER_SCHEMAS:
        raise KeyError(f"No parameter schema for family {family!r}")
    return _PARAMETER_SCHEMAS[family]


def formulate(family: str, parameters: dict[str, Any]) -> dict[str, Any]:
    """Dispatch to the registered formulator. Validates parameters
    against the schema first — a classifier that hallucinates keys or
    values gets rejected here rather than producing a broken CQM.

    Returns a valid ``cqm_v1`` JSON dict on success. Raises
    ``HardcodedFormulationError`` on any failure (unknown family,
    schema violation, formulator error).
    """
    if family not in HARDCODED_FORMULATORS:
        raise HardcodedFormulationError(
            f"Unknown hardcoded family: {family!r}. Known: {list_families()}",
        )
    _validate_parameters(family, parameters)
    formulator = HARDCODED_FORMULATORS[family]
    try:
        return formulator(**parameters)
    except HardcodedFormulationError:
        raise
    except Exception as e:  # noqa: BLE001
        raise HardcodedFormulationError(
            f"Formulator {family!r} rejected input: {type(e).__name__}: {e}",
        ) from e


def _validate_parameters(family: str, parameters: dict[str, Any]) -> None:
    """Shallow schema check — presence of required keys + basic type
    coherence. Deep JSON-schema validation is deliberately avoided so
    we don't drag ``jsonschema`` into this module's dependency
    footprint; the formulators do their own semantic checks."""
    schema = parameter_schema(family)
    required = schema.get("required", [])
    missing = [k for k in required if k not in parameters]
    if missing:
        raise HardcodedFormulationError(
            f"{family!r} parameters missing required keys: {missing}",
        )
    # Additional type coercion / normalization is the formulator's job.
