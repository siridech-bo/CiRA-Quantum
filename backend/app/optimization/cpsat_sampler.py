"""OR-Tools CP-SAT adapter exposed as a ``dimod``-compatible sampler.

Phase 8 — Classical Solver Tiers. CP-SAT is the SOTA constraint-programming
solver for scheduling, routing, and other combinatorial problems with
linear constraints over integer/boolean variables. We register it
alongside ``cpu_sa_neal``, ``gpu_sa``, and ``exact_cqm`` so the public
Benchmark dashboard can plot QUBO solvers against the actual classical
SOTA — not against each other.

Design notes:

* **CQM-native.** Unlike GPU SA, CP-SAT consumes the CQM directly. The
  lowering to BQM with Lagrange penalties (Phase 2) is a *concession* a
  QUBO solver makes; CP-SAT has no need for it. The class declares
  ``_CQM_NATIVE = True`` so ``records.record_run`` dispatches to
  ``sample_cqm(cqm)``, not ``sample(bqm)``.

* **Real-valued objective coefficients.** CP-SAT is an integer solver,
  so non-integer coefficients are scaled by a large constant
  (``_OBJ_SCALE``) before being added to the model, and the objective
  value is divided back down on the way out. This is exact when the
  inputs are rationals with denominators ≤ ``_OBJ_SCALE``; the
  validation harness's tolerance (1e-3) absorbs the residual on
  irrationals.

* **Quadratic objective.** CP-SAT models quadratic terms by introducing
  an auxiliary product variable per pair (with a ``AddMultiplicationEquality``
  constraint). The instances in the Phase-2 small suites all have
  bounded integer variables, so the product is finite-domain.

* **Real (continuous) variables are rejected.** CP-SAT is CP — discrete
  only. If a CQM has REAL variables, this sampler raises. None of the
  Phase-2 instance suites use REAL variables (everything is binary or
  integer), so this is fine in practice.
"""

from __future__ import annotations

import time
from typing import Any

import dimod
from ortools.sat.python import cp_model

# Scale factor for floating-point objective coefficients. CP-SAT optimizes
# integer objectives; we scale floats up by this and divide back down.
_OBJ_SCALE = 1_000_000

# Default wall-clock cap per call. Phase-2 small suites typically finish
# CP-SAT in well under a second; this is a safety net for larger inputs.
_DEFAULT_TIME_LIMIT_S = 60.0


def _scale_int(coeff: float) -> int:
    """Scale a float coefficient to CP-SAT's integer domain."""
    return int(round(coeff * _OBJ_SCALE))


def _is_integral(coeff: float, tol: float = 1e-9) -> bool:
    return abs(coeff - round(coeff)) < tol


class CPSATSampler(dimod.Sampler):
    """OR-Tools CP-SAT wrapped as a ``dimod.Sampler``.

    The sampler reads CQM directly (CQM-native). It exposes a single
    "sample" — CP-SAT returns the optimal solution it found, not a
    distribution — and therefore best fits with ``num_reads = 1``
    semantically. We honor a ``num_reads`` parameter for API symmetry
    with the SA-class solvers but it is ignored by the underlying solver.
    """

    _CQM_NATIVE = True

    def __init__(self, num_workers: int = 4):
        if num_workers < 1:
            raise ValueError("num_workers must be >= 1")
        self._num_workers = num_workers

    # ----- dimod.Sampler interface -----

    @property
    def parameters(self) -> dict:
        return {
            "time_limit": [],
            "num_workers": [],
            "seed": [],
        }

    @property
    def properties(self) -> dict:
        return {
            "ortools_version": _ortools_version(),
            "supports_quadratic_objective": True,
            "supports_real_variables": False,
        }

    def sample(  # pragma: no cover — CQM-native solvers should not hit this
        self,
        bqm: dimod.BinaryQuadraticModel,
        **kwargs: Any,
    ) -> dimod.SampleSet:
        raise NotImplementedError(
            "CPSATSampler is CQM-native; call sample_cqm(cqm), not sample(bqm). "
            "The Phase-2 records.record_run dispatcher should select the CQM path "
            "automatically via the _CQM_NATIVE class attribute."
        )

    def sample_cqm(
        self,
        cqm: dimod.ConstrainedQuadraticModel,
        time_limit: float = _DEFAULT_TIME_LIMIT_S,
        num_workers: int | None = None,
        seed: int | None = None,
    ) -> dimod.SampleSet:
        """Solve the CQM with CP-SAT and return a single best sample.

        Returns a ``dimod.SampleSet`` of length 1, with energies expressed
        in the same scale as ``cqm.objective`` (the scaling-back is done
        here so downstream summarization sees real-valued energy).
        """
        model = cp_model.CpModel()

        # ---- Variable translation ----
        variables = list(cqm.variables)
        cp_vars: dict[Any, cp_model.IntVar] = {}
        for v in variables:
            vt = cqm.vartype(v)
            if vt is dimod.BINARY:
                cp_vars[v] = model.NewBoolVar(str(v))
            elif vt is dimod.INTEGER:
                lb = int(cqm.lower_bound(v))
                ub = int(cqm.upper_bound(v))
                cp_vars[v] = model.NewIntVar(lb, ub, str(v))
            elif vt is dimod.SPIN:
                # SPIN ∈ {-1, +1} → represent as a bool b and map -1+2b.
                b = model.NewBoolVar(str(v))
                # Store the bool; we'll handle the affine mapping at
                # term-translation time by tracking the per-variable
                # "linear template" (offset, multiplier).
                cp_vars[v] = b
            else:
                raise ValueError(
                    f"CPSATSampler does not support REAL variables (variable {v!r})"
                )

        def lin_term(v: Any, coeff: float) -> tuple[cp_model.IntVar, int, int]:
            """Return (CP var, coefficient, constant offset) for one linear
            term ``coeff * v``. The constant offset arises when v is a SPIN
            variable and we substitute -1+2b."""
            cv = cp_vars[v]
            if cqm.vartype(v) is dimod.SPIN:
                # coeff * (-1 + 2b)  →  (2*coeff)*b  +  (-coeff)
                return cv, _scale_int(2 * coeff), _scale_int(-coeff)
            return cv, _scale_int(coeff), 0

        # ---- Constraints ----
        for cname, constraint in cqm.constraints.items():
            lhs = constraint.lhs
            sense = constraint.sense
            rhs = float(constraint.rhs)

            # Aggregate linear terms (and SPIN-substitution offsets).
            scaled_terms: list[tuple[cp_model.IntVar, int]] = []
            const_accum = 0
            for v, c in lhs.linear.items():
                cv, scaled_c, offset = lin_term(v, float(c))
                if scaled_c != 0:
                    scaled_terms.append((cv, scaled_c))
                const_accum += offset

            # Quadratic terms inside a constraint → product var per pair.
            if hasattr(lhs, "quadratic"):
                for (u, w), c in lhs.quadratic.items():
                    scaled_c = _scale_int(float(c))
                    if scaled_c == 0:
                        continue
                    if cqm.vartype(u) is dimod.SPIN or cqm.vartype(w) is dimod.SPIN:
                        raise ValueError(
                            "Quadratic terms with SPIN variables not yet supported "
                            f"in CPSATSampler (constraint {cname!r})"
                        )
                    lb_p = (
                        int(cqm.lower_bound(u)) * int(cqm.lower_bound(w))
                        if u != w
                        else int(cqm.lower_bound(u)) ** 2
                    )
                    ub_p = (
                        int(cqm.upper_bound(u)) * int(cqm.upper_bound(w))
                        if u != w
                        else int(cqm.upper_bound(u)) ** 2
                    )
                    lo, hi = min(lb_p, ub_p), max(lb_p, ub_p)
                    prod = model.NewIntVar(lo, hi, f"_q_{u}_{w}_{cname}")
                    model.AddMultiplicationEquality(prod, [cp_vars[u], cp_vars[w]])
                    scaled_terms.append((prod, scaled_c))

            # RHS in scaled units.
            rhs_scaled = _scale_int(rhs) - const_accum

            expr = sum(scaled_c * cv for cv, scaled_c in scaled_terms)
            if sense is dimod.constrained.Sense.Le:
                model.Add(expr <= rhs_scaled)
            elif sense is dimod.constrained.Sense.Ge:
                model.Add(expr >= rhs_scaled)
            elif sense is dimod.constrained.Sense.Eq:
                model.Add(expr == rhs_scaled)
            else:  # pragma: no cover
                raise ValueError(f"unknown sense {sense!r} on constraint {cname!r}")

        # ---- Objective ----
        obj = cqm.objective
        obj_terms: list[tuple[cp_model.IntVar, int]] = []
        obj_const = _scale_int(float(obj.offset))
        for v, c in obj.linear.items():
            cv, scaled_c, offset = lin_term(v, float(c))
            if scaled_c != 0:
                obj_terms.append((cv, scaled_c))
            obj_const += offset
        for (u, w), c in obj.quadratic.items():
            scaled_c = _scale_int(float(c))
            if scaled_c == 0:
                continue
            if cqm.vartype(u) is dimod.SPIN or cqm.vartype(w) is dimod.SPIN:
                raise ValueError(
                    "Quadratic objective terms with SPIN variables not supported"
                )
            lb_p = (
                int(cqm.lower_bound(u)) * int(cqm.lower_bound(w))
                if u != w
                else int(cqm.lower_bound(u)) ** 2
            )
            ub_p = (
                int(cqm.upper_bound(u)) * int(cqm.upper_bound(w))
                if u != w
                else int(cqm.upper_bound(u)) ** 2
            )
            lo, hi = min(lb_p, ub_p), max(lb_p, ub_p)
            prod = model.NewIntVar(lo, hi, f"_obj_q_{u}_{w}")
            model.AddMultiplicationEquality(prod, [cp_vars[u], cp_vars[w]])
            obj_terms.append((prod, scaled_c))

        objective_expr = sum(scaled_c * cv for cv, scaled_c in obj_terms) + obj_const
        # All CQM objectives in this codebase are encoded as "minimize"
        # (the orchestrator sign-flips maximize problems before solving).
        model.Minimize(objective_expr)

        # ---- Solve ----
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = float(time_limit)
        solver.parameters.num_search_workers = int(num_workers or self._num_workers)
        if seed is not None:
            solver.parameters.random_seed = int(seed)

        started = time.perf_counter()
        status = solver.Solve(model)
        elapsed = time.perf_counter() - started

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            # No feasible solution found — return an empty sample set.
            # The Phase-2 summarizer treats this as 0 feasible / infeasible.
            return dimod.SampleSet.from_samples_cqm(
                [],
                cqm=cqm,
                aggregate_samples=False,
            )

        # ---- Decode ----
        sample: dict[Any, int] = {}
        for v in variables:
            cv = cp_vars[v]
            val = solver.Value(cv)
            if cqm.vartype(v) is dimod.SPIN:
                sample[v] = -1 + 2 * val
            else:
                sample[v] = int(val)

        # CP-SAT's ObjectiveValue is already in the scaled domain (since
        # we minimized the scaled expression). Divide back to get the
        # user-units objective.
        sample_set = dimod.SampleSet.from_samples_cqm(
            sample,
            cqm=cqm,
            aggregate_samples=False,
        )
        # The energy from from_samples_cqm is recomputed from the CQM
        # itself, so we don't need to back-scale anything — it lands in
        # the right units automatically.
        # Record the solver-reported wall time on the record as an extra.
        sample_set.info.setdefault("cpsat_status", solver.StatusName(status))
        sample_set.info.setdefault("cpsat_wall_time_s", elapsed)
        sample_set.info.setdefault("cpsat_objective_scaled", solver.ObjectiveValue())
        return sample_set


def _ortools_version() -> str:
    try:
        import ortools

        return getattr(ortools, "__version__", "unknown")
    except ImportError:  # pragma: no cover
        return "missing"
