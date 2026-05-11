"""HiGHS adapter exposed as a ``dimod``-compatible sampler.

Phase 8 — Classical Solver Tiers. HiGHS is the SOTA open-source MIP /
LP solver. We register it alongside CP-SAT so the Benchmark dashboard
can plot the right baseline for MIP-shaped problems (knapsack,
set-cover, portfolio) — CP-SAT is strongest on combinatorial / scheduling
problems; HiGHS is strongest on linear-integer / mixed-integer.

Design notes:

* **CQM-native.** Same dispatch as CP-SAT — declares ``_CQM_NATIVE = True``
  so ``records.record_run`` calls ``sample_cqm(cqm)``.
* **Linear-objective only.** HiGHS is a MIP solver and only handles
  linear objectives. CQMs with quadratic objective terms are rejected
  here — those go to CP-SAT (which can handle them by introducing
  product variables) or to GPU SA (which lowers to BQM). The Phase-2
  small suites' instances *as-formulated* (knapsack/setcover/maxcut/
  graph-coloring) have linear objectives once the LLM picks the standard
  MIP encoding; the rare quadratic-objective formulation falls back to
  the QUBO-class solvers.
* **Real (continuous) variables OK.** Unlike CP-SAT, HiGHS supports
  REAL variables natively. The Phase-2 suites don't use them, but the
  Phase-5C user-facing solve path eventually will, and we don't want
  to artificially restrict the adapter.
* **No SPIN handling.** SPIN variables (±1) would need affine
  substitution like in the CP-SAT adapter. None of the Phase-2 / 5B
  suites use them; deferring to a follow-up if needed.
"""

from __future__ import annotations

import time
from typing import Any

import dimod
import highspy
import numpy as np

# Default wall-clock cap per call.
_DEFAULT_TIME_LIMIT_S = 60.0
# HiGHS uses ±1e30 as ±infinity.
_HIGHS_INF = 1e30


class HiGHSSampler(dimod.Sampler):
    """HiGHS wrapped as a ``dimod.Sampler``. CQM-native, linear-objective only."""

    _CQM_NATIVE = True

    def __init__(self, presolve: bool = True):
        self._presolve = bool(presolve)

    # ----- dimod.Sampler interface -----

    @property
    def parameters(self) -> dict:
        return {
            "time_limit": [],
            "presolve": [],
            "seed": [],
        }

    @property
    def properties(self) -> dict:
        return {
            "highspy_available": True,
            "supports_quadratic_objective": False,
            "supports_real_variables": True,
        }

    def sample(  # pragma: no cover
        self,
        bqm: dimod.BinaryQuadraticModel,
        **kwargs: Any,
    ) -> dimod.SampleSet:
        raise NotImplementedError(
            "HiGHSSampler is CQM-native; call sample_cqm(cqm), not sample(bqm)."
        )

    def sample_cqm(
        self,
        cqm: dimod.ConstrainedQuadraticModel,
        time_limit: float = _DEFAULT_TIME_LIMIT_S,
        presolve: bool | None = None,
        seed: int | None = None,
    ) -> dimod.SampleSet:
        # ---- Validate scope ----
        if any(cqm.vartype(v) is dimod.SPIN for v in cqm.variables):
            raise ValueError("HiGHSSampler does not support SPIN variables yet")
        if cqm.objective.quadratic:
            raise ValueError(
                "HiGHSSampler does not support quadratic objectives; "
                "use CPSATSampler or a QUBO-class sampler for that."
            )
        for cname, c in cqm.constraints.items():
            if hasattr(c.lhs, "quadratic") and c.lhs.quadratic:
                raise ValueError(
                    f"HiGHSSampler does not support quadratic constraints "
                    f"(constraint {cname!r})"
                )

        variables = list(cqm.variables)
        n = len(variables)
        var_index = {v: i for i, v in enumerate(variables)}

        # ---- Build column vectors ----
        col_cost = np.zeros(n, dtype=np.float64)
        col_lower = np.zeros(n, dtype=np.float64)
        col_upper = np.zeros(n, dtype=np.float64)
        integrality = []

        for i, v in enumerate(variables):
            vt = cqm.vartype(v)
            if vt is dimod.BINARY:
                col_lower[i] = 0.0
                col_upper[i] = 1.0
                integrality.append(highspy.HighsVarType.kInteger)
            elif vt is dimod.INTEGER:
                col_lower[i] = float(cqm.lower_bound(v))
                col_upper[i] = float(cqm.upper_bound(v))
                integrality.append(highspy.HighsVarType.kInteger)
            elif vt is dimod.REAL:
                col_lower[i] = float(cqm.lower_bound(v))
                col_upper[i] = float(cqm.upper_bound(v))
                integrality.append(highspy.HighsVarType.kContinuous)
            else:  # SPIN was rejected above; any other vartype is unknown
                raise ValueError(f"unsupported vartype {vt!r} on variable {v!r}")

        for v, c in cqm.objective.linear.items():
            col_cost[var_index[v]] = float(c)

        # ---- Build rows (CSR via row-wise constraint translation) ----
        # HiGHS expects row_lower / row_upper bounds; equalities use
        # row_lower == row_upper.
        row_lower: list[float] = []
        row_upper: list[float] = []
        row_starts: list[int] = [0]
        row_indices: list[int] = []
        row_values: list[float] = []

        constraint_labels: list[Any] = []
        for cname, constraint in cqm.constraints.items():
            constraint_labels.append(cname)
            lhs = constraint.lhs
            sense = constraint.sense
            rhs = float(constraint.rhs)

            for v, c in lhs.linear.items():
                cv = float(c)
                if cv == 0.0:
                    continue
                row_indices.append(var_index[v])
                row_values.append(cv)
            row_starts.append(len(row_indices))

            if sense is dimod.constrained.Sense.Le:
                row_lower.append(-_HIGHS_INF)
                row_upper.append(rhs)
            elif sense is dimod.constrained.Sense.Ge:
                row_lower.append(rhs)
                row_upper.append(_HIGHS_INF)
            elif sense is dimod.constrained.Sense.Eq:
                row_lower.append(rhs)
                row_upper.append(rhs)
            else:  # pragma: no cover
                raise ValueError(f"unknown sense {sense!r} on constraint {cname!r}")

        num_rows = len(row_lower)

        # ---- Pass to HiGHS ----
        h = highspy.Highs()
        h.silent()

        lp = highspy.HighsLp()
        lp.num_col_ = n
        lp.num_row_ = num_rows
        lp.col_cost_ = col_cost.tolist()
        lp.col_lower_ = col_lower.tolist()
        lp.col_upper_ = col_upper.tolist()
        lp.row_lower_ = row_lower
        lp.row_upper_ = row_upper
        lp.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
        lp.a_matrix_.start_ = row_starts
        lp.a_matrix_.index_ = row_indices
        lp.a_matrix_.value_ = row_values
        lp.integrality_ = integrality
        # CQM objectives in this codebase are always minimize (the orchestrator
        # sign-flips maximize problems before solving).
        lp.sense_ = highspy.ObjSense.kMinimize
        # Objective offset (carried through to the returned energy).
        lp.offset_ = float(cqm.objective.offset)

        h.passModel(lp)
        h.setOptionValue("time_limit", float(time_limit))
        if seed is not None:
            h.setOptionValue("random_seed", int(seed))
        use_presolve = self._presolve if presolve is None else bool(presolve)
        h.setOptionValue("presolve", "on" if use_presolve else "off")

        started = time.perf_counter()
        h.run()
        elapsed = time.perf_counter() - started

        status = h.getModelStatus()
        # HiGHS returns kOptimal when MIP / LP solve to optimality; kSolveError
        # / kInfeasible mean no usable solution.
        ok_statuses = {
            highspy.HighsModelStatus.kOptimal,
            getattr(highspy.HighsModelStatus, "kSolutionLimit", None),
            getattr(highspy.HighsModelStatus, "kTimeLimit", None),
        }
        ok_statuses.discard(None)
        if status not in ok_statuses:
            return dimod.SampleSet.from_samples_cqm(
                [],
                cqm=cqm,
                aggregate_samples=False,
            )

        sol = h.getSolution()
        col_value = list(sol.col_value)
        # Round integers — HiGHS may return 0.9999999 for a 1.
        sample: dict[Any, int | float] = {}
        for i, v in enumerate(variables):
            val = col_value[i]
            if cqm.vartype(v) in (dimod.BINARY, dimod.INTEGER):
                sample[v] = int(round(val))
            else:
                sample[v] = float(val)

        sample_set = dimod.SampleSet.from_samples_cqm(
            sample,
            cqm=cqm,
            aggregate_samples=False,
        )
        sample_set.info.setdefault(
            "highs_status",
            str(status).rsplit(".", 1)[-1] if hasattr(status, "name") else str(status),
        )
        sample_set.info.setdefault("highs_wall_time_s", elapsed)
        sample_set.info.setdefault(
            "highs_objective", h.getInfo().objective_function_value
        )
        return sample_set
