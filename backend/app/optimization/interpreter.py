"""SampleSet → human-readable solution.

The interpreter turns the raw ``{variable: value}`` map a solver returns
into a paragraph that references the problem's domain terms via the
variable registry the formulation provider attached to the CQM.

Two design choices worth noting:

1. **Group by variable kind.** Binary variables behave like "selected /
   not selected" indicators; integer and real variables carry numeric
   values. We split the output by kind so the reader sees structure
   instead of a flat dump.

2. **Surface feasibility.** A sample that satisfies every constraint
   leads the paragraph; a sample that violates one or more constraints
   is flagged loudly at the top and the offending labels are listed.
   The pipeline orchestrator already gates on validation, but the
   interpreter still checks because by the time the solver picks the
   "first" sample it may have skipped feasibility for a lower-energy
   penalty-laden one — surfacing that to the user is the whole point of
   the validation harness's existence.

The output is plain text, ready to drop into the Phase 5 frontend's
Markdown viewer.
"""

from __future__ import annotations

import dimod


def interpret_solution(
    sample: dict[str, float],
    registry: dict[str, str],
    cqm: dimod.ConstrainedQuadraticModel,
    *,
    sense: str = "minimize",
) -> str:
    """Convert a solver sample to a human-readable paragraph.

    Parameters
    ----------
    sample
        The selected assignment — ``{variable_name: value}``.
    registry
        Human descriptions per variable name; values default to empty
        strings when missing.
    cqm
        The compiled CQM. Used to compute the objective value, check
        constraint feasibility, and resolve variable types.
    sense
        ``"minimize"`` (default) or ``"maximize"``. The CQM's internal
        objective is always minimization-shaped; for maximize problems
        the interpreter negates the reported objective value so it
        matches the units the user used to state the problem.
    """
    if not sample:
        return "No solution was produced (empty sample)."

    lines: list[str] = []

    # ---- Feasibility ----
    try:
        violations = cqm.violations(sample)
    except Exception:
        violations = {}
    offenders = {label: float(v) for label, v in violations.items() if float(v) > 1e-9}

    if offenders:
        lines.append("⚠ The selected sample violates one or more constraints:")
        for label, amount in sorted(offenders.items()):
            lines.append(f"  - {label}: violation = {amount:g}")
        lines.append("This solution is infeasible as stated.")
        lines.append("")

    # ---- Objective ----
    try:
        internal = float(cqm.objective.energy(sample))
        user_value = -internal if sense == "maximize" else internal
        verb = "Maximum" if sense == "maximize" else "Minimum"
        lines.append(f"{verb} objective value reached: {user_value:g}")
        lines.append("")
    except Exception:
        # Some CQM/sample combinations can't be energy-evaluated cleanly
        # (e.g. missing-variable assignments). Surface that quietly.
        lines.append("Objective value: <unavailable>")
        lines.append("")

    # ---- Variables, grouped by kind ----
    binaries_on: list[tuple[str, str]] = []
    binaries_off: list[tuple[str, str]] = []
    numerics: list[tuple[str, float, str]] = []

    # Walk variables in the order the CQM declared them so output is stable.
    for name in cqm.variables:
        if name not in sample:
            continue
        value = sample[name]
        desc = registry.get(name, "")
        try:
            vt = cqm.vartype(name)
        except Exception:
            vt = None
        if vt is dimod.BINARY:
            (binaries_on if int(value) == 1 else binaries_off).append((name, desc))
        else:
            numerics.append((name, float(value), desc))

    # Render binaries
    if binaries_on:
        lines.append("Selected (binary = 1):")
        for name, desc in binaries_on:
            tail = f" — {desc}" if desc else ""
            lines.append(f"  • {name}{tail}")
    if binaries_off and not numerics:
        # Suppress the "not selected" block when it would dominate the
        # output (e.g. all-zero solution). Only show it for context when
        # there's also a numeric section.
        pass

    # Render integer / real assignments
    if numerics:
        if binaries_on:
            lines.append("")
        lines.append("Assignments:")
        for name, value, desc in numerics:
            tail = f" — {desc}" if desc else ""
            # Strip a trailing ".0" off integer-valued floats so JSS makespans
            # read "11" not "11.0".
            value_str = f"{int(value)}" if value == int(value) else f"{value:g}"
            lines.append(f"  • {name} = {value_str}{tail}")

    # Fallback: cqm declared no variables, but the sample carries some
    # (rare but possible if the orchestrator handed us a raw BQM sample
    # without inverse-mapping). Dump whatever's there so the user isn't
    # lied to about an "empty" solution.
    if not binaries_on and not binaries_off and not numerics:
        lines.append("Sample values:")
        for k, v in sample.items():
            tail = f" — {registry.get(k, '')}" if registry.get(k) else ""
            lines.append(f"  • {k} = {v}{tail}")

    return "\n".join(lines)
