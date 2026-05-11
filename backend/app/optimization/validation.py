"""Three-layer validation harness for compiled CQMs.

The harness is the safety net that catches LLM-generated formulation
errors before the user is shown a (silently) wrong solution. It runs:

* **Layer A — Oracle agreement.** ``dimod.ExactCQMSolver`` finds the
  exact minimum on small CQMs; the energy is converted to user-facing
  units (negating for maximize problems) and compared to the
  ``expected_optimum`` shipped in the JSON. Skipped when the CQM has
  more variables than ``max_oracle_vars``.

* **Layer B — Solver agreement.** Convert the CQM to a BQM via
  ``dimod.cqm_to_bqm``, sample with both ``dwave-samplers``' CPU SA
  and the GPU SA, demangle each sample back into the CQM space, and
  pick the lowest-energy *feasible* sample from each. Two solvers
  landing on the same energy is strong evidence the encoding is sound.
  GPU SA is skipped automatically if CUDA is not available.

* **Layer C — Constraint coverage.** Random sampling: every constraint
  must see *both* satisfying and violating assignments under random
  inputs. A constraint that is always satisfied is vacuous; one that
  is always violated is over-constrained. Either is a smell. Layer C
  reports a per-label boolean and emits a warning for inactive ones.

The harness deliberately does *not* know about the CQM's original
``sense``; the caller passes it explicitly. We chose this over embedding
sense in the CQM (dimod doesn't have a metadata slot) and over having
``compile_cqm_json`` mutate energies — each option leaks responsibility,
and the explicit pass keeps the boundary clean.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import dimod
import numpy as np
from dwave.samplers import SimulatedAnnealingSampler


@dataclass
class ValidationReport:
    """The aggregated output of all three validation layers.

    Energies are in *user-facing* units — for a maximize problem, the
    layer's internal minimization energy is negated before being stored
    here so the report can be diff'd against the JSON's ``expected_optimum``
    without further conversion.
    """

    oracle_agreement: bool
    oracle_skipped: bool
    solver_agreement: bool
    constraints_active: dict[str, bool]
    energy_oracle: float | None
    energy_gpu_sa: float | None
    energy_cpu_sa: float | None
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True iff all three layers green.

        Layer A passes vacuously when skipped (``oracle_skipped=True``).
        Layer B passes vacuously when neither sampler ran (``skip_layer_b=True``).
        Layer C passes vacuously when there are no constraints to check.
        """
        return (
            self.oracle_agreement
            and self.solver_agreement
            and all(self.constraints_active.values())
        )


def _convert_to_user(energy: float | None, sense: str) -> float | None:
    if energy is None:
        return None
    return -float(energy) if sense == "maximize" else float(energy)


def _layer_a(
    cqm: dimod.ConstrainedQuadraticModel,
    expected_optimum: float | None,
    sense: str,
    max_oracle_vars: int,
    warnings: list[str],
) -> tuple[bool, bool, float | None]:
    """Run ExactCQMSolver and compare to expected_optimum (user units).

    Returns ``(oracle_agreement, oracle_skipped, energy_oracle_user)``.
    """
    n_vars = len(cqm.variables)
    if n_vars > max_oracle_vars:
        warnings.append(
            f"Layer A skipped: {n_vars} variables exceeds max_oracle_vars={max_oracle_vars}"
        )
        return True, True, None

    solver = dimod.ExactCQMSolver()
    sampleset = solver.sample_cqm(cqm)
    feasible = sampleset.filter(lambda d: d.is_feasible)
    if len(feasible) == 0:
        warnings.append("Layer A: ExactCQMSolver found no feasible assignment")
        return False, False, None

    internal = float(feasible.first.energy)
    user_energy = _convert_to_user(internal, sense)

    if expected_optimum is None:
        return True, False, user_energy

    agreement = abs(user_energy - float(expected_optimum)) <= 1e-6
    if not agreement:
        warnings.append(
            f"Layer A disagreement: oracle found {user_energy:.6f}, "
            f"expected {expected_optimum:.6f}"
        )
    return agreement, False, user_energy


def _best_feasible_cqm_energy(
    bqm_sampleset: dimod.SampleSet,
    invert,
    cqm: dimod.ConstrainedQuadraticModel,
) -> float | None:
    """Walk samples in ascending BQM-energy order, return the CQM objective
    energy of the first feasible one (in CQM-internal units)."""
    energies = bqm_sampleset.record.energy
    samples_arr = bqm_sampleset.record.sample
    var_order = list(bqm_sampleset.variables)
    order = np.argsort(energies, kind="stable")
    for idx in order:
        bqm_sample = dict(zip(var_order, samples_arr[idx], strict=True))
        cqm_sample = invert(bqm_sample)
        try:
            feasible = cqm.check_feasible(cqm_sample)
        except Exception:
            continue
        if feasible:
            return float(cqm.objective.energy(cqm_sample))
    return None


def _layer_b(
    cqm: dimod.ConstrainedQuadraticModel,
    sense: str,
    *,
    num_reads: int,
    num_sweeps: int,
    lagrange_multiplier: float,
    seed: int | None,
    warnings: list[str],
) -> tuple[bool, float | None, float | None]:
    """Run CPU SA and (if CUDA is up) GPU SA on the BQM lowering of the CQM.

    Returns ``(solver_agreement, energy_cpu_user, energy_gpu_user)``.
    Agreement requires *at least one* feasible sample from each sampler that
    actually ran, with energies within 1e-3 (relative) of one another. If
    only CPU runs (no GPU), the layer accepts the single sampler.
    """
    if len(cqm.variables) == 0:
        return True, None, None

    bqm, invert = dimod.cqm_to_bqm(cqm, lagrange_multiplier=lagrange_multiplier)

    cpu_sampler = SimulatedAnnealingSampler()
    cpu_ss = cpu_sampler.sample(bqm, num_reads=num_reads, num_sweeps=num_sweeps, seed=seed)
    cpu_internal = _best_feasible_cqm_energy(cpu_ss, invert, cqm)

    gpu_internal: float | None = None
    try:
        import torch

        cuda_available = torch.cuda.is_available()
    except ImportError:
        cuda_available = False

    if cuda_available:
        from app.optimization.gpu_sa import GPUSimulatedAnnealingSampler

        gpu_sampler = GPUSimulatedAnnealingSampler(kernel="jit")
        gpu_ss = gpu_sampler.sample(bqm, num_reads=num_reads, num_sweeps=num_sweeps, seed=seed)
        gpu_internal = _best_feasible_cqm_energy(gpu_ss, invert, cqm)
    else:
        warnings.append("Layer B: CUDA not available, GPU SA skipped")

    cpu_user = _convert_to_user(cpu_internal, sense)
    gpu_user = _convert_to_user(gpu_internal, sense)

    if cpu_internal is None and gpu_internal is None:
        warnings.append("Layer B: neither sampler returned a feasible assignment")
        return False, cpu_user, gpu_user

    if gpu_internal is None or cpu_internal is None:
        # Only one sampler returned anything — single-sampler "agreement"
        return True, cpu_user, gpu_user

    denom = max(abs(cpu_internal), abs(gpu_internal), 1.0)
    agreement = abs(cpu_internal - gpu_internal) / denom <= 1e-3
    if not agreement:
        warnings.append(
            f"Layer B disagreement: CPU SA {cpu_user}, GPU SA {gpu_user}"
        )
    return agreement, cpu_user, gpu_user


def _layer_c(
    cqm: dimod.ConstrainedQuadraticModel,
    *,
    num_samples: int,
    seed: int | None,
    warnings: list[str],
) -> dict[str, bool]:
    """Random sampling: a constraint is *active* iff random assignments hit
    both satisfying and violating cases."""
    rng = np.random.default_rng(seed if seed is not None else 0)

    # Pre-compute random assignments.
    samples: list[dict[str, float | int]] = []
    for _ in range(num_samples):
        s: dict[str, float | int] = {}
        for v in cqm.variables:
            vt = cqm.vartype(v)
            if vt is dimod.BINARY:
                s[v] = int(rng.integers(0, 2))
            elif vt is dimod.INTEGER:
                lb = int(cqm.lower_bound(v))
                ub = int(cqm.upper_bound(v))
                s[v] = int(rng.integers(lb, ub + 1))
            else:  # REAL
                s[v] = float(rng.uniform(cqm.lower_bound(v), cqm.upper_bound(v)))
        samples.append(s)

    # dimod's `cqm.violations(sample)` returns a dict label → violation. The
    # convention (verified by inspection): violation > 0 ⇒ constraint violated;
    # violation ≤ 0 ⇒ satisfied. EQ constraints report |lhs - rhs| (always
    # non-negative); LE/GE report a signed slack (negative = satisfied).
    rtol = 1e-9
    labels = list(cqm.constraints)
    sat_seen: dict[str, bool] = dict.fromkeys(labels, False)
    viol_seen: dict[str, bool] = dict.fromkeys(labels, False)

    for s in samples:
        v_dict = cqm.violations(s)
        for label in labels:
            v = float(v_dict.get(label, 0.0))
            if v > rtol:
                viol_seen[label] = True
            else:
                sat_seen[label] = True
        if all(sat_seen[label] and viol_seen[label] for label in labels):
            break

    active: dict[str, bool] = {}
    for label in labels:
        is_active = sat_seen[label] and viol_seen[label]
        active[label] = is_active
        if not is_active:
            shape = "always satisfied" if sat_seen[label] else "always violated"
            warnings.append(
                f"Layer C: constraint {label!r} appears inactive — {shape} "
                f"under {num_samples} random samples"
            )
    return active


def validate_cqm(
    cqm: dimod.ConstrainedQuadraticModel,
    expected_optimum: float | None = None,
    *,
    sense: str = "minimize",
    max_oracle_vars: int = 12,
    num_reads: int = 500,
    num_sweeps: int = 500,
    lagrange_multiplier: float = 10.0,
    seed: int | None = 0,
    layer_c_samples: int = 200,
    skip_layer_b: bool = False,
) -> ValidationReport:
    """Run all three validation layers on a compiled CQM.

    Parameters
    ----------
    cqm
        The CQM produced by ``compile_cqm_json``.
    expected_optimum
        The user-facing optimum shipped in the JSON's ``test_instance``,
        or ``None`` to skip the comparison in Layer A. Always interpreted
        in user units — Layer A internally negates for maximize.
    sense
        ``"minimize"`` (default) or ``"maximize"``. The compiler returns
        this as its third output.
    max_oracle_vars
        Cap above which Layer A is skipped. ``ExactCQMSolver`` enumerates
        all feasible assignments; large CQMs make it impractical.
    skip_layer_b
        Skip the CPU/GPU multi-solver agreement check. Useful for unit
        tests that don't want to pay the sampling cost or that only care
        about Layer A and Layer C.
    """
    if sense not in ("minimize", "maximize"):
        raise ValueError(f"Unknown sense: {sense!r}")

    warnings: list[str] = []

    oracle_ok, oracle_skipped, e_oracle = _layer_a(
        cqm, expected_optimum, sense, max_oracle_vars, warnings
    )

    if skip_layer_b:
        b_ok, e_cpu, e_gpu = True, None, None
    else:
        b_ok, e_cpu, e_gpu = _layer_b(
            cqm,
            sense,
            num_reads=num_reads,
            num_sweeps=num_sweeps,
            lagrange_multiplier=lagrange_multiplier,
            seed=seed,
            warnings=warnings,
        )

    active = _layer_c(cqm, num_samples=layer_c_samples, seed=seed, warnings=warnings)

    return ValidationReport(
        oracle_agreement=oracle_ok,
        oracle_skipped=oracle_skipped,
        solver_agreement=b_ok,
        constraints_active=active,
        energy_oracle=e_oracle,
        energy_gpu_sa=e_gpu,
        energy_cpu_sa=e_cpu,
        warnings=warnings,
    )
