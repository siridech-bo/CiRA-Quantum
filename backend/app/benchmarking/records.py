"""Run records (Phase 2 v2).

Every solver invocation that we want to *cite, replay, or compare* later
goes through ``record_run``. Each call produces a ``RunRecord`` written
to ``benchmarks/archive/<record_id>.json`` and (optionally) a gzipped
``dimod.SampleSet`` next to it.

Reproducibility model
---------------------
The ``repro_hash`` field is a deterministic SHA-256 (truncated to 16 hex
chars) over four canonical inputs:

  1. ``code_version``      — git SHA at run time, or
                             ``"<package_version>+dev"`` outside a tree.
  2. ``instance_id``       — the registered instance's stable ID.
  3. ``solver``            — the solver identity dict.
  4. ``parameters``        — solver kwargs, JSON-canonicalized.

Two records with the same hash on the same code_version are expected to
produce identical results modulo seeded randomness — that is what
``replay_record`` verifies.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import platform
import secrets
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import dimod

from app.benchmarking.registry import SolverIdentity, get_solver

# ---- Archive root ----

# benchmarks/archive/ at the repo root; allow override for tests.
_DEFAULT_ARCHIVE = Path(__file__).resolve().parents[2].parent / "benchmarks" / "archive"


def _archive_root() -> Path:
    override = os.environ.get("CIRA_BENCH_ARCHIVE")
    if override:
        return Path(override)
    return _DEFAULT_ARCHIVE


def archive_path(record_id: str, *, suffix: str = ".json") -> Path:
    """Resolve the on-disk path for ``record_id``'s record (or sample-set)."""
    return _archive_root() / f"{record_id}{suffix}"


# ---- Identifiers and provenance ----


def _now() -> datetime:
    return datetime.now(tz=UTC)


def _new_record_id() -> str:
    """Sortable timestamp + random tail. ``20260506T143501.123456Z_a3c81d``-style.
    Microsecond resolution keeps record IDs strictly increasing even when
    a suite runs many records inside the same wall-clock second."""
    now = _now()
    ts = now.strftime("%Y%m%dT%H%M%S") + f".{now.microsecond:06d}Z"
    return f"{ts}_{secrets.token_hex(3)}"


def _detect_code_version() -> str:
    """Best-effort ``git rev-parse HEAD`` from the package's parent dir,
    falling back to ``"<package>+dev"`` if no git tree is reachable."""
    package_root = Path(__file__).resolve().parents[2]
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=package_root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if sha.returncode == 0 and sha.stdout.strip():
            return sha.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "0.1.0+dev"


def _detect_hardware_id(solver_hardware: str | None) -> str:
    """Concrete-hardware identifier string for the run record. More
    specific than ``solver_hardware`` (which is a class hint)."""
    if solver_hardware and solver_hardware.startswith("cuda"):
        try:
            import torch

            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                cc = torch.cuda.get_device_capability(0)
                cuda_v = torch.version.cuda or "unknown"
                return f"cuda:{name}|cc{cc[0]}.{cc[1]}|cuda_runtime={cuda_v}"
        except ImportError:
            pass
        return solver_hardware
    if solver_hardware == "cpu" or solver_hardware is None:
        return f"cpu:{platform.processor() or platform.machine()}|{platform.system()}"
    return solver_hardware


def _canonical_parameters(params: dict) -> str:
    """JSON-canonicalize ``params`` for deterministic hashing.

    Sorts keys, fixes float formatting via ``json.dumps`` defaults, and
    rejects unhashable values (callables, tensors) so a typo won't make
    the hash silently change run-over-run.
    """
    def _scrub(v: Any) -> Any:
        if isinstance(v, dict):
            return {k: _scrub(v[k]) for k in sorted(v)}
        if isinstance(v, list | tuple):
            return [_scrub(x) for x in v]
        if isinstance(v, str | int | float | bool) or v is None:
            return v
        raise TypeError(
            f"parameter value of type {type(v).__name__} is not JSON-serializable; "
            "RunRecord.parameters must be a plain JSON-compatible dict"
        )

    scrubbed = _scrub(params or {})
    return json.dumps(scrubbed, sort_keys=True, separators=(",", ":"))


_INIT_ONLY_KWARGS_BY_SOLVER: dict[str, set[str]] = {
    "gpu_sa": {"kernel", "device"},
    "cpsat": {"num_workers"},
    "highs": {"presolve"},
    # Phase 9A — QAOA hyperparameters are constructor-time only.
    "qaoa_sim": {"layer", "optimizer", "top_k", "max_qubits"},
    # Phase 9B — cloud QAOA. Phase 5D moved this to BYOK: ``api_key`` is
    # a real constructor parameter now (each user passes their own
    # Origin Quantum credential). ``api_key`` is listed here so it gets
    # routed into ``__init__``, but ``record_run`` scrubs it from the
    # archived parameters dict so the credential never lands in a
    # RunRecord JSON.
    "qaoa_originqc": {
        "api_key",
        "backend_name",
        "url",
        "layer",
        "shots",
        "max_qubits",
        "max_submissions",
        "top_k",
        "train_optimizer",
    },
    # Phase 9C — quantum-inspired classical tiers.
    "parallel_tempering": {"num_replicas", "beta_range"},
    "simulated_bifurcation": {"mode"},
}


def _split_parameters(solver_name: str, params: dict) -> tuple[dict, dict]:
    """Split a flat parameters dict into ``(init_kwargs, sample_kwargs)``.

    The mapping of which keys belong to ``__init__`` vs ``sample()`` is
    per-solver and currently hardcoded for the three baseline tiers; if
    Phase 10's contribution pipeline brings in more solvers with init
    kwargs, this becomes a registry-time declaration.
    """
    init_keys = _INIT_ONLY_KWARGS_BY_SOLVER.get(solver_name, set())
    init_kwargs = {k: v for k, v in (params or {}).items() if k in init_keys}
    sample_kwargs = {k: v for k, v in (params or {}).items() if k not in init_keys}
    return init_kwargs, sample_kwargs


def compute_repro_hash(
    code_version: str,
    instance_id: str,
    solver: SolverIdentity,
    parameters: dict,
) -> str:
    payload = "\n".join([
        f"code_version={code_version}",
        f"instance_id={instance_id}",
        f"solver={json.dumps(solver.to_dict(), sort_keys=True, separators=(',', ':'))}",
        f"parameters={_canonical_parameters(parameters)}",
    ])
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ---- The RunRecord dataclass ----


@dataclass
class RunRecord:
    record_id: str
    code_version: str
    solver: SolverIdentity
    parameters: dict
    instance_id: str
    hardware_id: str
    started_at: datetime
    completed_at: datetime
    repro_hash: str
    results: dict
    sample_set_path: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["solver"] = self.solver.to_dict()
        d["started_at"] = self.started_at.isoformat()
        d["completed_at"] = self.completed_at.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> RunRecord:
        return cls(
            record_id=d["record_id"],
            code_version=d["code_version"],
            solver=SolverIdentity(**d["solver"]),
            parameters=d["parameters"],
            instance_id=d["instance_id"],
            hardware_id=d["hardware_id"],
            started_at=datetime.fromisoformat(d["started_at"]),
            completed_at=datetime.fromisoformat(d["completed_at"]),
            repro_hash=d["repro_hash"],
            results=d["results"],
            sample_set_path=d.get("sample_set_path"),
            warnings=d.get("warnings", []),
        )


# ---- Sample-set archival ----


def _archive_sample_set(record_id: str, sampleset: dimod.SampleSet) -> str:
    """Write ``sampleset.to_serializable()`` as gzipped JSON. Returns the
    archive-relative filename so it can be re-loaded by ``record_id``."""
    archive_root = _archive_root()
    archive_root.mkdir(parents=True, exist_ok=True)
    rel = f"{record_id}_samples.json.gz"
    path = archive_root / rel
    payload = json.dumps(sampleset.to_serializable()).encode()
    with gzip.open(path, "wb") as f:
        f.write(payload)
    return rel


def load_archived_sample_set(record_id_or_relpath: str) -> dimod.SampleSet:
    """Inverse of ``_archive_sample_set``; accepts either a record_id or
    a relative archive path (e.g. ``'<id>_samples.json.gz'``)."""
    archive_root = _archive_root()
    if record_id_or_relpath.endswith(".json.gz"):
        path = archive_root / record_id_or_relpath
    else:
        path = archive_root / f"{record_id_or_relpath}_samples.json.gz"
    with gzip.open(path, "rb") as f:
        data = json.loads(f.read().decode())
    return dimod.SampleSet.from_serializable(data)


# ---- record_run / load / replay ----


def record_run(
    *,
    solver_name: str,
    instance_id: str,
    bqm: dimod.BinaryQuadraticModel,
    parameters: dict,
    archive_samples: bool = True,
    cqm: dimod.ConstrainedQuadraticModel | None = None,
    sense: str = "minimize",
    expected_optimum: float | None = None,
) -> RunRecord:
    """Run the named solver on ``bqm`` with ``parameters``, persist a
    ``RunRecord`` to ``benchmarks/archive/``, and return it.

    Parameters
    ----------
    solver_name
        Registered solver name; resolved through
        :func:`app.benchmarking.registry.get_solver`.
    instance_id
        Stable identifier for the instance (used in the ``repro_hash``).
    bqm
        The model to sample. Solvers that natively consume CQMs receive
        the optional ``cqm`` argument instead and ignore ``bqm``.
    parameters
        The kwargs to pass to ``sampler.sample(...)``. Must be plain JSON
        types — see :func:`_canonical_parameters`.
    archive_samples
        If True, write the SampleSet alongside the record. Default True.
    cqm
        Used for the ``ExactCQMSolver`` path (which can't sample a BQM)
        and to compute feasibility / user-facing energies after sampling.
    sense
        ``"minimize"`` or ``"maximize"`` — used to convert the best
        feasible CQM-internal energy into the user-facing units that the
        Benchmark dashboard ultimately displays.
    expected_optimum
        Known optimum (or best-known value) in user-facing units, taken
        from the instance manifest. When supplied, the record's
        ``results`` dict gains an honest convergence flag
        (``converged_to_expected``) and the absolute gap to it. When
        ``None`` (instance has no ground truth), both fields are ``None``
        and the dashboard is expected to render this as "heuristic
        estimate, no ground truth available."
    """
    identity, sampler_cls = get_solver(solver_name)

    # Some samplers split their kwargs between __init__ and sample(). Most
    # `dimod.Sampler` implementations accept everything at sample(), but the
    # GPU SA built in Phase 1 takes ``kernel`` (and ``device``) at construction
    # time. We split the parameter dict here rather than asking each caller
    # to know the per-solver convention; the full dict is still recorded in
    # the RunRecord for reproducibility.
    init_kwargs, sample_kwargs = _split_parameters(solver_name, parameters)
    sampler = sampler_cls(**init_kwargs)

    # CQM-native solvers consume the CQM directly (no BQM lowering / penalty).
    # The Phase-2 baseline is ``dimod.ExactCQMSolver``; Phase 8 adds the
    # classical tiers (CP-SAT, HiGHS) via the same path. New CQM-native
    # adapters opt in by setting the class attribute ``_CQM_NATIVE = True``.
    is_cqm_solver = (
        sampler_cls.__name__ == "ExactCQMSolver"
        or getattr(sampler_cls, "_CQM_NATIVE", False)
    )

    started = _now()
    if is_cqm_solver:
        if cqm is None:
            raise ValueError(
                f"{sampler_cls.__name__} requires the cqm argument (CQM-native solver)"
            )
        sampleset = sampler.sample_cqm(cqm, **sample_kwargs)
    else:
        sampleset = sampler.sample(bqm, **sample_kwargs)
    completed = _now()

    elapsed_ms = (completed - started).total_seconds() * 1000.0
    results = _summarize(sampleset, cqm=cqm, sense=sense, expected_optimum=expected_optimum)
    results["elapsed_ms"] = elapsed_ms

    record_id = _new_record_id()
    code_version = _detect_code_version()
    hardware_id = _detect_hardware_id(identity.hardware)

    # Scrub credential-style kwargs before they touch the persisted
    # RunRecord. ``api_key`` (qaoa_originqc BYOK) and any future
    # auth-bearing param goes through this list — the archived JSON
    # must NEVER contain a plaintext key. The repro hash uses the
    # scrubbed params too so two reproductions with the same kwargs
    # but different keys still match.
    archived_parameters = {
        k: v for k, v in (parameters or {}).items() if k not in _SECRET_PARAM_KEYS
    }
    repro_hash = compute_repro_hash(code_version, instance_id, identity, archived_parameters)

    sample_set_path = _archive_sample_set(record_id, sampleset) if archive_samples else None

    record = RunRecord(
        record_id=record_id,
        code_version=code_version,
        solver=identity,
        parameters=archived_parameters,
        instance_id=instance_id,
        hardware_id=hardware_id,
        started_at=started,
        completed_at=completed,
        repro_hash=repro_hash,
        results=results,
        sample_set_path=sample_set_path,
    )
    _write_record(record)
    return record


# Parameter keys that carry credentials. Stripped before persistence
# AND before the repro hash so credential rotation doesn't break
# reproducibility checks.
_SECRET_PARAM_KEYS: frozenset[str] = frozenset({"api_key"})


def _summarize(
    sampleset: dimod.SampleSet,
    *,
    cqm: dimod.ConstrainedQuadraticModel | None,
    sense: str,
    expected_optimum: float | None,
) -> dict:
    """Distill a SampleSet into a small, JSON-friendly summary dict.

    The honesty fields ``converged_to_expected`` and ``gap_to_expected``
    are populated when an ``expected_optimum`` is supplied — that is the
    Benchmark dashboard's "did this run actually find the known optimum,
    or is this a heuristic estimate?" signal. When the instance has no
    ground truth (``expected_optimum=None``), both fields are ``None`` and
    the dashboard is expected to surface that as such, *not* claim
    convergence.
    """
    n_samples = int(len(sampleset))
    best_energy: float | None = None
    best_user: float | None = None
    n_feasible: int | None = None

    if n_samples == 0:
        return {
            "best_energy": None,
            "best_user_energy": None,
            "num_samples": 0,
            "num_feasible": None,
            "expected_optimum": expected_optimum,
            "gap_to_expected": None,
            "converged_to_expected": None,
            "extras": {},
        }

    if cqm is not None and "is_feasible" in sampleset.record.dtype.names:
        feasible = sampleset.filter(lambda d: d.is_feasible)
        n_feasible = int(len(feasible))
        if n_feasible > 0:
            best_energy = float(feasible.first.energy)
    else:
        best_energy = float(sampleset.first.energy)

    if best_energy is not None:
        best_user = -best_energy if sense == "maximize" else best_energy

    gap, converged = _convergence_check(best_user, expected_optimum)

    return {
        "best_energy": best_energy,
        "best_user_energy": best_user,
        "num_samples": n_samples,
        "num_feasible": n_feasible,
        "expected_optimum": expected_optimum,
        "gap_to_expected": gap,
        "converged_to_expected": converged,
        "extras": {},
    }


def _convergence_check(
    best_user: float | None, expected: float | None
) -> tuple[float | None, bool | None]:
    """Compute (gap_to_expected, converged_to_expected) honestly.

    Returns ``(None, None)`` when either value is missing — the dashboard
    must read "no ground truth" rather than infer convergence. When both
    are present, the gap is absolute (``|best - expected|``) and the
    convergence flag is true iff that gap is within ``max(1e-3 * |expected|, 1e-9)``
    — tight enough to flag any real heuristic miss on integer-valued
    objectives, loose enough to absorb float rounding from the BQM
    lowering on continuous objectives.
    """
    if best_user is None or expected is None:
        return None, None
    gap = abs(float(best_user) - float(expected))
    tol = max(1e-3 * abs(float(expected)), 1e-9)
    return gap, gap <= tol


def _write_record(record: RunRecord) -> None:
    archive_root = _archive_root()
    archive_root.mkdir(parents=True, exist_ok=True)
    path = archive_path(record.record_id)
    with open(path, "w") as f:
        json.dump(record.to_dict(), f, indent=2)


def load_record(record_id: str) -> RunRecord:
    path = archive_path(record_id)
    with open(path) as f:
        return RunRecord.from_dict(json.load(f))


@dataclass
class ReplayResult:
    """Outcome of a replay: which fields agreed, which drifted."""
    original: RunRecord
    replayed: RunRecord
    agree: bool
    notes: list[str] = field(default_factory=list)


def replay_record(
    record_id: str,
    *,
    bqm: dimod.BinaryQuadraticModel,
    cqm: dimod.ConstrainedQuadraticModel | None = None,
    sense: str = "minimize",
) -> ReplayResult:
    """Re-execute the run identified by ``record_id`` against the supplied
    BQM/CQM and compare the new record to the archived one.

    Returns a :class:`ReplayResult` describing the comparison. ``agree``
    is true iff (a) the same code_version is currently in effect, (b) the
    repro_hash recomputes to the same value, and (c) the best energy of
    the new run matches the original within 1e-9 (seeded determinism).
    Two records that disagree on (a) trigger a notes entry but do *not*
    fail outright: a different code_version producing different energies
    is the *expected* "spec drift" signal, not a bug.
    """
    original = load_record(record_id)
    replayed = record_run(
        solver_name=original.solver.name,
        instance_id=original.instance_id,
        bqm=bqm,
        parameters=dict(original.parameters),
        archive_samples=False,
        cqm=cqm,
        sense=sense,
    )

    notes: list[str] = []
    if replayed.code_version != original.code_version:
        notes.append(
            f"code_version drift: original={original.code_version!r}, "
            f"replay={replayed.code_version!r}"
        )
    if replayed.repro_hash != original.repro_hash:
        notes.append(
            f"repro_hash drift: original={original.repro_hash!r}, "
            f"replay={replayed.repro_hash!r}"
        )

    e1 = original.results.get("best_energy")
    e2 = replayed.results.get("best_energy")
    if e1 is None or e2 is None:
        energies_agree = e1 is None and e2 is None
    else:
        energies_agree = abs(e1 - e2) <= 1e-9

    agree = (
        replayed.code_version == original.code_version
        and replayed.repro_hash == original.repro_hash
        and energies_agree
    )

    return ReplayResult(original=original, replayed=replayed, agree=agree, notes=notes)
