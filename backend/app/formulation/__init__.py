"""LLM formulation provider layer (Phase 3).

Public surface:

    FormulationProvider, FormulationResult, FormulationError
    validate_cqm_json, extract_json_object
    get_provider, list_providers

Each concrete provider lives in its own module — ``claude.py``,
``openai.py``, ``local.py``. ``get_provider("claude")`` returns the
matching provider instance.
"""

from __future__ import annotations

from app.formulation.base import (
    FormulationError,
    FormulationProvider,
    FormulationResult,
    extract_json_object,
    get_provider,
    list_providers,
    register_provider,
    validate_cqm_json,
)

__all__ = [
    "FormulationError",
    "FormulationProvider",
    "FormulationResult",
    "extract_json_object",
    "get_provider",
    "list_providers",
    "register_provider",
    "validate_cqm_json",
]
