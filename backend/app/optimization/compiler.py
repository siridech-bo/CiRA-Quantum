"""CQM-JSON → ``dimod.ConstrainedQuadraticModel`` compiler.

The cqm_v1 JSON schema is the contract between the formulation provider
(an LLM, in production) and the solver pipeline. This module consumes
that JSON and returns a compiled CQM, a registry of human descriptions
for each variable, and the original objective sense — needed downstream
to convert between dimod's internal-minimization energies and the
user-facing values.

The schema (informally):

    {
      "version": "1",
      "variables": [
        {"name": "x", "type": "binary", "description": "..."},
        {"name": "y", "type": "integer",
         "lower_bound": 0, "upper_bound": 100, "description": "..."}
      ],
      "objective": {
        "sense": "minimize" | "maximize",
        "linear":    {"x": 1.0, "y": 0.5, ...},
        "quadratic": {"x*y": 0.5, ...}
      },
      "constraints": [
        {"label": "...", "type": "equality" | "inequality_le" | "inequality_ge",
         "linear": {...}, "quadratic": {...}, "rhs": <float>}
      ],
      "test_instance": {"description": "...", "expected_optimum": <float>}
    }

Quadratic terms are encoded as ``"u*v": coeff`` strings — a single ``*``
delimiter, no spaces. Splitting on ``"*"`` is the canonical decoder.

When ``objective.sense == "maximize"`` we negate the linear and quadratic
coefficients before pushing them into the CQM (which always minimizes).
The original sense is returned alongside the CQM so downstream code can
flip energies back into user-facing units.
"""

from __future__ import annotations

import dimod

_CONSTRAINT_TYPE_TO_SENSE: dict[str, str] = {
    "equality": "==",
    "inequality_le": "<=",
    "inequality_ge": ">=",
}


def _parse_quadratic_key(key: str) -> tuple[str, str]:
    """Decode ``"u*v"`` → ``("u", "v")``. Raises ``ValueError`` on a bad key."""
    parts = key.split("*")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"Malformed quadratic term key: {key!r} (expected 'u*v')")
    return parts[0], parts[1]


def compile_cqm_json(
    cqm_json: dict,
) -> tuple[dimod.ConstrainedQuadraticModel, dict[str, str], str]:
    """Compile a cqm_v1 JSON document into a ``dimod.ConstrainedQuadraticModel``.

    Returns
    -------
    cqm : dimod.ConstrainedQuadraticModel
        A model that always *minimizes* (dimod's only mode); maximize-sense
        objectives have their coefficients negated during compile.
    variable_registry : dict[str, str]
        Maps variable name → human description from the JSON.
    sense : str
        The original ``objective.sense`` value (``"minimize"`` or
        ``"maximize"``). Downstream code uses this to convert energies
        between minimization-internal and user-facing units.

    Raises
    ------
    ValueError
        On any schema violation: unsupported version, unknown variable or
        constraint type, undeclared variable references, malformed
        quadratic keys, etc.
    """
    if not isinstance(cqm_json, dict):
        raise ValueError(f"cqm_json must be a dict, got {type(cqm_json).__name__}")

    version = cqm_json.get("version")
    if version != "1":
        raise ValueError(
            f"Unsupported CQM JSON version: {version!r} (this build supports '1')"
        )

    cqm = dimod.ConstrainedQuadraticModel()
    registry: dict[str, str] = {}
    var_types: dict[str, str] = {}
    var_bounds: dict[str, tuple[float, float]] = {}

    # ---- Variables ----
    for v in cqm_json.get("variables", []):
        name = v["name"]
        vtype_str = v.get("type")
        if vtype_str == "binary":
            cqm.add_variable("BINARY", name)
            var_types[name] = "BINARY"
            var_bounds[name] = (0.0, 1.0)
        elif vtype_str == "integer":
            lb = float(v.get("lower_bound", 0))
            ub = float(v["upper_bound"])
            cqm.add_variable("INTEGER", name, lower_bound=lb, upper_bound=ub)
            var_types[name] = "INTEGER"
            var_bounds[name] = (lb, ub)
        elif vtype_str == "real":
            lb = float(v.get("lower_bound", 0.0))
            ub = float(v["upper_bound"])
            cqm.add_variable("REAL", name, lower_bound=lb, upper_bound=ub)
            var_types[name] = "REAL"
            var_bounds[name] = (lb, ub)
        else:
            raise ValueError(f"Unknown variable type: {vtype_str!r} for variable {name!r}")
        registry[name] = v.get("description", "")

    # ---- Objective ----
    obj = cqm_json.get("objective", {"sense": "minimize", "linear": {}, "quadratic": {}})
    sense = obj.get("sense", "minimize")
    if sense not in ("minimize", "maximize"):
        raise ValueError(f"Unknown objective sense: {sense!r}")
    sign = 1.0 if sense == "minimize" else -1.0

    for vname, coeff in obj.get("linear", {}).items():
        if vname not in registry:
            raise ValueError(
                f"Objective references undeclared variable: {vname!r}"
            )
        cqm.objective.add_linear(vname, sign * float(coeff))

    for key, coeff in obj.get("quadratic", {}).items():
        u, v = _parse_quadratic_key(key)
        if u not in registry or v not in registry:
            raise ValueError(
                f"Objective quadratic term references undeclared variable: "
                f"{u!r} or {v!r}"
            )
        cqm.objective.add_quadratic(u, v, sign * float(coeff))

    # ---- Constraints ----
    for cdata in cqm_json.get("constraints", []):
        label = cdata.get("label")
        ctype = cdata.get("type")
        if ctype not in _CONSTRAINT_TYPE_TO_SENSE:
            raise ValueError(
                f"Unknown constraint type: {ctype!r} for constraint {label!r}"
            )
        cqm_sense = _CONSTRAINT_TYPE_TO_SENSE[ctype]
        rhs = float(cdata.get("rhs", 0.0))
        linear = cdata.get("linear", {}) or {}
        quadratic = cdata.get("quadratic", {}) or {}

        # Build a per-constraint QuadraticModel containing only the variables
        # that appear in *this* constraint, with the same vartype/bounds as
        # in the parent CQM. Adding all CQM variables to every constraint QM
        # would multiply memory by len(constraints), which is unnecessary.
        seen: set[str] = set(linear.keys())
        for k in quadratic:
            u, v = _parse_quadratic_key(k)
            seen.add(u)
            seen.add(v)

        for vname in seen:
            if vname not in registry:
                raise ValueError(
                    f"Constraint {label!r} references undeclared variable: {vname!r}"
                )

        qm = dimod.QuadraticModel()
        for vname in seen:
            vtype = var_types[vname]
            if vtype == "BINARY":
                qm.add_variable("BINARY", vname)
            else:
                lb, ub = var_bounds[vname]
                qm.add_variable(vtype, vname, lower_bound=lb, upper_bound=ub)

        for vname, coeff in linear.items():
            qm.add_linear(vname, float(coeff))
        for key, coeff in quadratic.items():
            u, v = _parse_quadratic_key(key)
            qm.add_quadratic(u, v, float(coeff))

        cqm.add_constraint(qm, sense=cqm_sense, rhs=rhs, label=label)

    return cqm, registry, sense
