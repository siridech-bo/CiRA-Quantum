"""Solver tier registry (Phase 2 v2).

Each registered solver is uniquely identified by ``name``. The registry
is the source of truth for what solvers exist on this build, what
parameters they accept, and which Python class actually implements them.

The registry is a process-global singleton — appropriate for the v1/v2
deployment model (single Flask process, single solver host). When the
platform fans out into worker processes (Phase 6) each worker bootstraps
its own copy on import.
"""

from __future__ import annotations

import importlib
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SolverIdentity:
    """Identity card for a registered solver tier.

    Frozen so registered identities can be hashed, copied freely into
    ``RunRecord`` objects, and compared by value.
    """

    name: str
    version: str
    source: str
    hardware: str | None
    parameter_schema: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# Module-level singleton. Registered entries map name → (identity, sampler_class).
_REGISTRY: dict[str, tuple[SolverIdentity, type]] = {}
_BOOTSTRAPPED = False


def register_solver(identity: SolverIdentity, sampler_cls: type) -> None:
    """Register ``sampler_cls`` under ``identity.name``. Raises ``ValueError``
    if a solver with that name is already registered."""
    if identity.name in _REGISTRY:
        existing, _ = _REGISTRY[identity.name]
        raise ValueError(
            f"Solver {identity.name!r} is already registered "
            f"(version {existing.version}, source {existing.source})"
        )
    _REGISTRY[identity.name] = (identity, sampler_cls)


def get_solver(name: str) -> tuple[SolverIdentity, type]:
    """Look up a registered solver. Raises ``KeyError`` on unknown names."""
    if name not in _REGISTRY:
        raise KeyError(f"No registered solver named {name!r}")
    return _REGISTRY[name]


def list_solvers() -> list[SolverIdentity]:
    """Return a deterministic, alphabetical list of registered identities."""
    return [identity for identity, _cls in sorted(_REGISTRY.values(), key=lambda x: x[0].name)]


# ---- Bootstrap of the three baseline tiers ----


def _gpu_hardware_id() -> str | None:
    """Best-effort device label for the GPU SA registration; ``None`` when
    CUDA isn't available so the GPU tier is registered as headless."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        name = torch.cuda.get_device_name(0)
        cc = torch.cuda.get_device_capability(0)
        return f"cuda:0:{name.lower().replace(' ', '_')}:sm_{cc[0]}{cc[1]}"
    except Exception:
        return None


def _package_version(modname: str, fallback: str = "unknown") -> str:
    try:
        mod = importlib.import_module(modname)
    except ImportError:
        return fallback
    return getattr(mod, "__version__", fallback)


def bootstrap_default_solvers() -> None:
    """Register the three baseline tiers (``exact_cqm``, ``cpu_sa_neal``,
    ``gpu_sa`` when CUDA is up). Idempotent: a second call is a no-op."""
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return
    _BOOTSTRAPPED = True

    # Schemas live next to this module.
    schemas_dir = Path(__file__).parent / "schemas"

    # The bootstrap is forgiving: a missing optional dependency just skips
    # that tier. The registry is still useful with partial coverage.
    try:
        import dimod

        register_solver(
            SolverIdentity(
                name="exact_cqm",
                version=_package_version("dimod"),
                source="dimod",
                hardware=None,
                parameter_schema=_load_schema(schemas_dir / "exact_cqm_params.json"),
            ),
            dimod.ExactCQMSolver,
        )
    except Exception:  # pragma: no cover — only fires on broken installs
        logger.exception("Failed to register ExactCQMSolver")

    try:
        from dwave.samplers import SimulatedAnnealingSampler

        register_solver(
            SolverIdentity(
                name="cpu_sa_neal",
                version=_package_version("dwave.samplers", fallback=_package_version("dwave_samplers")),
                source="dwave-samplers",
                hardware="cpu",
                parameter_schema=_load_schema(schemas_dir / "cpu_sa_params.json"),
            ),
            SimulatedAnnealingSampler,
        )
    except Exception:  # pragma: no cover
        logger.exception("Failed to register dwave-samplers SimulatedAnnealingSampler")

    gpu_hw = _gpu_hardware_id()
    if gpu_hw is not None:
        try:
            from app.optimization.gpu_sa import GPUSimulatedAnnealingSampler

            register_solver(
                SolverIdentity(
                    name="gpu_sa",
                    version="0.1.0",  # tracked alongside the cira-quantum codebase
                    source="cira-quantum",
                    hardware=gpu_hw,
                    parameter_schema=_load_schema(schemas_dir / "gpu_sa_params.json"),
                ),
                GPUSimulatedAnnealingSampler,
            )
        except Exception:  # pragma: no cover — surfaces real CUDA failures during dev
            logger.exception("Failed to register GPUSimulatedAnnealingSampler")


def _load_schema(path: Path) -> dict:
    """Load an inline parameter-schema if present, else return an empty dict.

    Per-solver parameter schemas are optional — the v2 spec only requires
    that ``SolverIdentity.parameter_schema`` *exists*; specific schemas
    will be added when contribution review (Phase 10) needs them.
    """
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)
