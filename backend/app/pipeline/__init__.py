"""Solve pipeline (Phase 4).

Public surface::

    EventBus           — per-job thread-safe event queue + replay
    Orchestrator       — five-stage async pipeline (formulate → compile →
                         validate → solve → interpret)
    PipelineError      — raised on any stage failure
"""

from __future__ import annotations

from app.pipeline.events import EventBus, get_event_bus
from app.pipeline.orchestrator import Orchestrator, PipelineError

__all__ = ["EventBus", "Orchestrator", "PipelineError", "get_event_bus"]
