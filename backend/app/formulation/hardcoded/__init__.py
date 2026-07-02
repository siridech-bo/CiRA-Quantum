"""Hardcoded deterministic formulators for canonical optimization problems.

Each formulator takes structured parameters (numbers / edges / matrices)
and returns a mathematically-exact ``cqm_v1`` JSON document without any
LLM involvement. Coefficient-level accuracy is 100% by construction —
the failure mode we hit with LLM-emitted CQMs (Max-Cut degree miscounts,
Number Partitioning coefficient drift) simply cannot occur here.

The orchestrator's stage 1 tries an LLM classifier first: if the
classifier recognizes the problem family with high confidence and
extracts clean parameters, we route to a formulator here and skip the
LLM CQM-emission path entirely. Everything else falls back to the
legacy LLM-emits-CQM behavior — freeform problems remain supported.

Registered families and their ``cqm_v1`` shapes are documented in the
individual modules.
"""

from app.formulation.hardcoded.registry import (
    HARDCODED_FORMULATORS,
    HardcodedFormulationError,
    formulate,
    list_families,
    parameter_schema,
)

__all__ = [
    "HARDCODED_FORMULATORS",
    "HardcodedFormulationError",
    "formulate",
    "list_families",
    "parameter_schema",
]
