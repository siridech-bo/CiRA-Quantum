# CiRA Quantum — Optimization Module

> **Last major revision:** 2026-07-02 (classifier-routed hardcoded
> formulators, plain-English solutions, real-QPU pipeline).

This document is the elaborate reference for the **Optimization** app
inside CiRA Quantum — the plain-English → CQM → 10-solver comparison
pipeline that today ships to research customers on Origin Wukong-180
class hardware and QAOA-ready cloud/classical solvers.

The audience is a contributor or reviewer who has read
[`FEATURES.md`](FEATURES.md) at the top level and now wants the full
end-to-end architecture, the math behind each family, the routing
decisions, the failure modes, and the reasoning behind every non-obvious
choice.

---

## 1. Product goal

**Give a domain expert who cannot write QUBOs a way to submit a
combinatorial optimization problem in ordinary English, watch it be
formulated, validated, solved on ten solvers (classical → quantum-
inspired → simulator QAOA → real superconducting QPU), and receive an
answer phrased in the same vocabulary they used to ask the question.**

The user should never see:

- A LaTeX QUBO.
- A dimod SampleSet.
- A raw bitstring.
- Any solver's internal energy value without an explanation of what
  those units mean in their problem.

The user *may* see (and increasingly wants to see, once trust is
established):

- The exact structured translation of their prose.
- The CQM JSON that got compiled.
- Which QAOA tiers were skipped and why.
- The gate-level circuit that ran on the Origin QPU.
- The measurement histogram from real hardware.
- The classical top-K filter step that picked the winner.

Everything below is engineered so the beginner and expert views collapse
gracefully into the same UI.

---

## 2. Five-stage pipeline (high-level)

The pipeline is the orchestrator's `run()` method in
[`backend/app/pipeline/orchestrator.py`](../backend/app/pipeline/orchestrator.py).
Every request goes through the same five stages, each emitting a status
on the SSE event bus:

| # | Stage         | What happens                                                                 | Emitted status         |
|---|---------------|------------------------------------------------------------------------------|------------------------|
| 1 | Formulate     | Prose → structured CQM JSON. See §3 for the routing branch.                  | `formulating`          |
| 2 | Compile       | CQM JSON → `dimod.ConstrainedQuadraticModel` + variable registry + sense.    | `compiling`            |
| 3 | Validate      | Three-layer harness: brute-force oracle, cross-solver agreement, constraints.| `validating`           |
| ↳ | Approval gate | *(optional)* Pause with preflight summary; user reviews before solvers burn time. | `awaiting_approval` |
| 4 | Solve         | Fan out to N solvers, capture SampleSets + timing + QAOA telemetry.          | `solving`              |
| 5 | Interpret     | Deterministic assignment map + best-effort LLM plain-English rewrite.        | `complete`             |

Any stage failure lands on `status='error'` with the message attached.
The pipeline never re-raises — the orchestrator is the only code that
knows about the job row, so it owns the failure mode.

---

## 3. Stage 1 — Formulation

### 3.1 The routing decision (shipped 2026-07-02)

Stage 1 has **two ways** to produce a CQM:

- **Hardcoded route.** A registered deterministic formulator emits the
  CQM. Coefficients are exact by construction; no LLM in the loop past
  the classification step.
- **LLM route.** The formulation provider (Claude Sonnet 4.6 / GPT-5
  Mini) emits the CQM JSON directly from prose. Five-example few-shot
  prompt anchors the encoding style.

The routing branch decides which. It runs a small **classifier call**
first (same model as the formulation provider, but with a different
system prompt) that asks:

> *"Is this problem a genuine instance of one of these known families,
> and if so, what are the exact parameters?"*

The classifier returns
[`ClassificationResult`](../backend/app/formulation/base.py) with:

- `family` — one of `max_cut`, `number_partitioning`,
  `max_independent_set`, `portfolio_selection`, or `""` for no match.
- `parameters` — the extracted structured translation
  (`{"numbers": [4, 3, 2, 3, 1]}`).
- `confidence` — model self-report in `[0, 1]`.
- `reasoning` — one-sentence explanation, surfaced in the UI.

If `family` is non-empty and `confidence ≥ 0.85`
(`ClassificationResult.is_confident`), the orchestrator calls
[`hardcoded.formulate(family, parameters)`](../backend/app/formulation/hardcoded/registry.py)
and skips LLM CQM emission entirely. Otherwise it falls back to
`provider.formulate(problem_statement, api_key)` as before.

The audit trail is persisted on the job row as `formulation_route` JSON:

```json
{
  "route": "hardcoded",
  "family": "number_partitioning",
  "confidence": 0.95,
  "reasoning": "list of five positive numbers to partition into two groups",
  "classifier_tokens": 218,
  "parameters": { "numbers": [4, 3, 2, 3, 1] }
}
```

The Approval Panel renders this as a green *"Problem statement
(translated) — Number Partitioning"* card so the user can see exactly
what the pipeline is about to formulate before it commits.

### 3.2 Why hardcoded formulators exist

Before the routing layer landed, the Claude/GPT few-shot examples got
Number Partitioning **structurally correct** (5 vars, 0 constraints,
`objective.offset = S²` present) but **arithmetically wrong** — linear
coefficients drifted 17–32 % from the ideal `4·n_i·(n_i − S)` form. On
`[4, 3, 2, 3, 1]` the LLM's CQM produced an optimum of **5 or −39**
instead of the true **1**. Max-Cut had the same class of drift on
degree counts (2026-07-01 investigation).

Deterministic formulators fix this by construction — they compute
every coefficient in Python. The LLM is only trusted with the *soft*
task (parse prose → structured params), never the *arithmetic* task
(compute exact linear/quadratic coefficients).

### 3.3 The four hardcoded families

Each family lives in
[`backend/app/formulation/hardcoded/`](../backend/app/formulation/hardcoded/).
The registry auto-discovers them and exposes a parameter JSON-schema
that both the classifier and a defense-in-depth pre-call validator use.

#### 3.3.1 Number Partitioning

Given positive numbers `n_1..n_N` with sum `S`, split them into two
groups so the absolute difference of sums is minimized. Encoded as
*natural unconstrained binary*:

```
minimize (S − 2·Σ n_i · x_i)²           x_i ∈ {0, 1}
```

Expanded:

- `linear[x_i]  =  4·n_i·(n_i − S)`
- `quadratic[x_i · x_j]  =  8·n_i·n_j`  (i < j)
- `objective.offset  =  S²`

The offset lets the perfect-partition optimum evaluate to 0 (or the
smallest odd square for odd-`S` instances) rather than −S², so the
downstream validator's Layer A oracle can compare directly against a
brute-force ground truth. The `test_instance.expected_optimum` is
computed at formulator time via a 2^N brute force — fine up to ~20
numbers, comfortably above the qpu_ready-template range.

**Test case that broke the LLM (and now doesn't):** `[4, 3, 2, 3, 1]`,
`S = 13` (odd), optimum imbalance² = 1 with partition `{4, 3}` vs
`{2, 3, 1}`.

#### 3.3.2 Max-Cut

Split a graph's nodes into two groups to maximize edge-crossings:

```
maximize  Σ_{(i,j) ∈ E}  (x_i + x_j − 2·x_i·x_j)     x_i ∈ {0, 1}
```

Expanded:

- `linear[x_i]  =  deg(i)`  (weighted: Σ of incident weights)
- `quadratic[x_i · x_j]  =  −2 · w(i, j)`  per edge

Zero constraints — that's the point. Any indicator variables or
edge-slack terms would inflate the qubit count and get skipped by every
QAOA tier. The formulator is opinionated: multi-edges collapse (weights
sum), self-loops drop silently, node indices normalize to
`[0, node_count)`.

`expected_optimum` = brute-force max cut weight.

#### 3.3.3 Maximum Independent Set

Pick a maximum-cardinality set of nodes with no two selected nodes
sharing an edge:

```
maximize  Σ x_i  −  λ · Σ_{(i,j) ∈ E}  x_i · x_j     x_i ∈ {0, 1}
```

Expanded:

- `linear[x_i]  =  1`
- `quadratic[x_i · x_j]  =  −λ`  per edge

Penalty coefficient default: `λ = node_count + 1` — safe for any
graph density. The caller can lower it toward 2 for sparse graphs to
sharpen the optimization gap.

`expected_optimum` = size of the maximum independent set.

#### 3.3.4 Portfolio Selection (Markowitz shape + budget)

Pick assets to maximize expected return minus risk-scaled variance,
subject to a max-assets budget:

```
maximize  Σ μ_i · x_i  −  λ · Σ_{i,j}  Σ_{i,j} · x_i · x_j
subject to  Σ x_i  ≤  K
                                            x_i ∈ {0, 1}
```

Expanded (`x_i² = x_i` for binaries, so the variance diagonal folds
into linear):

- `linear[x_i]  =  μ_i  −  λ · Σ_{i,i}`
- `quadratic[x_i · x_j]  =  −2 · λ · Σ_{i,j}`  (i < j)
- Constraint `budget`: `Σ x_i ≤ K`

This family is the exception — it carries one real CQM constraint (the
budget), so the QAOA pipeline pays a slack-qubit cost when it lowers
the CQM into a BQM. That's expected: `K < N` portfolios are inherently
constrained and no simpler encoding avoids the slack qubits without
changing the problem.

`expected_optimum` = brute-force `μ·x − λ·xᵀΣx` over all budget-
respecting selections.

### 3.4 Feature flag and safety

Env var **`ENABLE_HARDCODED_ROUTING`** (default `1`) gates the whole
routing branch. Setting it to `0` restores the legacy LLM-emits-CQM
behavior for A/B comparison or for deployments that don't want the
extra classifier network call.

Failure isolation:

- Classifier call raising or returning malformed JSON → fall through to
  LLM path (`route: "llm_no_classifier"`).
- Classifier confident but the hardcoded formulator rejects the params
  (e.g. negative numbers for partition) → fall through to LLM path
  (`route: "llm"`).
- Confident classifier + happy formulator → hardcoded path
  (`route: "hardcoded"`).
- Classifier not confident (`confidence < 0.85`) → LLM path
  (`route: "llm"`).

The routing branch **never surfaces its own errors as user-facing job
failures.** The LLM path is always the safety net.

### 3.5 LLM formulation path (fallback)

When routing declines to use a hardcoded formulator, the request goes
to the provider's `formulate(problem_statement, api_key)`. Providers
implement the same async contract in
[`backend/app/formulation/base.py`](../backend/app/formulation/base.py):

- **Claude Sonnet 4.6** ([`claude.py`](../backend/app/formulation/claude.py))
  — `x-api-key` header, Anthropic Messages API.
- **GPT-5 Mini** ([`openai.py`](../backend/app/formulation/openai.py))
  — Bearer auth, Chat Completions API with `response_format=json_object`.
- **Local** ([`local.py`](../backend/app/formulation/local.py))
  — for offline dev; historically the least-reliable path, kept for
  demos without cloud credentials.

The system prompt + five-example few-shot lives in
[`prompts/`](../backend/app/formulation/prompts/) and includes the
Number Partitioning natural-form anchor that made structural (though
not always arithmetic) accuracy reliable.

---

## 4. Stage 2 — Compilation

[`app.optimization.compiler.compile_cqm_json`](../backend/app/optimization/compiler.py)
turns the validated CQM JSON into a triple:

```python
cqm: dimod.ConstrainedQuadraticModel
variable_registry: dict[str, str]           # name → human description
sense: Literal["minimize", "maximize"]
```

Key details worth calling out:

- Quadratic keys use `"u*v"` — a single `*` delimiter, no spaces. The
  Max-Cut example currently uses comma keys in `examples.json` (historical
  drift; low-priority cleanup).
- `objective.offset` is an optional constant term. Under `maximize` we
  negate it along with the linear/quadratic coefficients so the internal
  minimize form is a faithful negation of the user-facing maximize form.
- Constraint types map: `equality` → `==`, `inequality_le` → `<=`,
  `inequality_ge` → `>=`.
- Per-constraint `QuadraticModel`s only contain the variables that
  actually appear in that constraint — avoids O(len(constraints) · N)
  memory blowup.

Failures raise `ValueError` and stage 3 catches them into a
job-level error status.

---

## 5. Stage 3 — Validation harness

[`app.optimization.validation.validate_cqm`](../backend/app/optimization/validation.py)
runs three layers:

| Layer | Check                          | Skipped by default? |
|-------|--------------------------------|---------------------|
| **A** | Brute-force oracle vs `expected_optimum`. | Only when > ~20 vars. |
| **B** | Cross-solver agreement (3-solver majority). | Yes on live solves. Runs offline in the dashboard. |
| **C** | Constraint coverage & sanity checks. | Never — always runs. |

For live solves we deliberately skip Layer B (would add 30–60 s per
request); the Phase 5C dashboard runs it on archived RunRecords.

**Layer A is why the hardcoded formulator matters most.** On the
`[4, 3, 2, 3, 1]` case, the LLM's CQM triggered Layer A with
`oracle found −39, expected 1` — a red flag the classifier→hardcoded
route now avoids entirely.

Validation failures don't hard-abort at the approval gate; they render
as an advisory alert. Past the gate, a failed validation raises
`PipelineError` and the job marks `error`.

---

## 6. Approval gate

Toggled per-request via the `require_approval` param (defaults to
true from the UI). When set, the pipeline **pauses after stage 3** with
`status='awaiting_approval'` and writes a **preflight** payload:

```json
{
  "cqm_variables": 5,
  "cqm_constraints": 0,
  "lowered_qubits": 5,
  "tier_verdicts": {
    "qaoa_ibmq":    { "cap": 20, "fits": true },
    "qaoa_originqc":{ "cap": 12, "fits": true },
    "qaoa_sim":     { "cap": 12, "fits": true }
  },
  "qpu_footprint": null
}
```

- **Lowered qubits** — the CQM is lowered to a BQM once (with the same
  Lagrange penalty math the solvers use) so the count matches what the
  QAOA tiers will actually see. Slack qubits from inequality
  constraints are included.
- **Per-tier verdict** — QAOA tier caps are backend-aware
  (`full_amplitude=7`, real Wukong hardware = 12, IBM Brisbane = 20).
  A tier that "would skip" is not an error — it's a fit decision.
- **QPU footprint** (real hardware only, see §9) — estimated compute
  seconds, shot count, layer count, gate depth. Only compute time is
  billed by Origin; queue time is displayed separately.

The Approval Panel renders all of this, plus the routing card from §3.1
and the CQM JSON (expandable). Approve triggers
`POST /api/jobs/<id>/approve` which calls
`Orchestrator.resume_after_approval()` — same helper `run()` uses past
the gate, so semantics are identical.

The `solver_params_overrides` JSON (e.g. `{"qaoa_originqc": {"backend_name": "WK_C180_2"}}`)
is persisted on the paused row so the resume path picks up the same
choices the user made pre-approval.

---

## 7. Stage 4 — Solvers

The multi-solver fan-out lives in the same orchestrator. Each solver is
a dimod-compatible sampler with a per-solver wall-clock timeout
(`_SOLVER_WALLCLOCK_TIMEOUT_S`). The 10-solver lineup as of shipping:

| # | Solver             | Family              | Notes |
|---|--------------------|---------------------|-------|
| 1 | `gpu_sa`           | Classical           | GPU simulated annealing (JIT kernel). Default fallback. |
| 2 | `parallel_tempering` | Classical         | CPU replica-exchange. |
| 3 | `simulated_bifurcation` | Classical/inspired | Toshiba SBM-style adiabatic ODE. |
| 4 | `highs`            | Exact classical     | HiGHS MILP; only for small integer/binary problems. |
| 5 | `cpsat`            | Exact classical     | Google OR-Tools CP-SAT; often beats GPU SA on small dense instances. |
| 6 | `qaoa_sim`         | Quantum simulator   | Statevector QAOA; cap 12 qubits. |
| 7 | `qaoa_ibmq`        | Real QPU (cloud)    | IBM Quantum via `qiskit-ibm-runtime`; cap 20 qubits. Async submit + poll. |
| 8 | `qaoa_originqc`    | Real QPU (cloud)    | Origin Wukong-180 / 180-2 / HanYuan_01 via `pyqpanda3`. Cap 12 qubits on real hardware. |
| — | *(reserved)*       |                     | Two additional slots reserved for future annealer-family solvers. |

Each row in the results map carries: status, energy, feasibility,
elapsed ms, tier source, hardware name, and — for QAOA solvers — a
`qaoa_extras` block with trained (γ, β) angles, per-qubit linear
biases, pairwise couplings, top-K bitstrings, top-K probabilities, and
top-K energies. The Origin dashboard UI (`QaoaFullView.vue` and
supporting components) surfaces all of this.

**Cap enforcement.** Solvers whose lowered qubit count exceeds their
cap raise a specific `ValueError`; `_classify_solver_failure` maps
those to `status='skipped'` rather than red-chip errors — the sampler
correctly refused a workload it couldn't fit, not a bug.

**Real-QPU safety.** `ENABLE_ORIGIN_REAL_HARDWARE` (default off in dev)
gates any submission to Wukong-180 class backends. Without it, requests
for `WK_C180`, `WK_C180_2`, or `HanYuan_01` are rejected server-side.

---

## 8. Stage 5 — Interpretation

Two artifacts land on the row:

- **`interpreted_solution`** — a deterministic assignment map
  ([`interpret_solution`](../backend/app/optimization/interpreter.py))
  that walks the winning `SampleSet`, decodes variables using
  `variable_registry` descriptions, and prints
  `Number 0 (value 4): 0 = group A, 1 = group B` etc. This is the
  literal, trustworthy view.
- **`plain_english_solution`** — best-effort LLM rewrite via
  `provider.summarize_solution(problem, interpreted, key)`. System
  prompt enforces:
  1. Account for **every** variable/item/entity — never drop items to
     save space.
  2. Use the user's own vocabulary (nodes/items/groups/bins/etc.).
  3. Show concrete values.
  4. Ignore auxiliary/slack variables.
  5. Start with the answer directly, 2–4 short sentences.

The frontend renders the plain-English block above the technical view
when present, hides the section when absent. Failure is silent — the
solve still completes with the technical view.

---

## 9. QPU footprint estimator

Only rendered when the user selects a real Origin backend. The
[`_estimate_qpu_footprint`](../backend/app/pipeline/orchestrator.py)
helper computes compute-time seconds *and clearly notes that queue-
time is not billed*. The full metadata surfaced to the Approval Panel:

```
Estimated compute time  4–7 s
Shots                   1024
QAOA layers             1
Circuit depth (approx)  ~120 gates
Only the compute-time seconds are billed. Queue wait affects when
you see results, not how much they cost.
```

Origin doesn't expose a programmatic balance API, so the panel links
out to their web dashboard for quota — see the ApiKeyManager caution
callout about Origin's dual-queue quirk (Wukong-C180 and 180-2 have
independent queues; a healthy 180-2 doesn't imply 180 is available).

---

## 10. Template gallery

The Examples surface splits into two groups:

- **Real Quantum Friendly** — instances that lower to ≤ 12 qubits,
  fitting the Wukong-180 cap. `qpu_ready: true` in the template
  registry; a UI chip highlights them.
- **Classical & Larger Instances** — anything that needs > 12 qubits;
  runs on classical + `qaoa_sim` + `qaoa_ibmq` tiers only.

The
[`REQUIRED_TEMPLATE_IDS`](../backend/app/templates/registry.py) set
guarantees a curated baseline exists at startup and CI catches
regressions. Every template ships an `expected_optimum` so Layer A
validation has a ground truth even for hand-authored instances.

---

## 11. UI walkthrough (the happy path)

Landing → Solve tab:

1. User pastes: *"I have a set of 5 positive numbers: [4, 3, 2, 3, 1].
   Split them into two groups so the absolute difference is as small
   as possible."*
2. Selects **Claude (Sonnet 4.6)** as the formulation provider (BYOK
   stored key).
3. Solvers grid defaults to `RECOMMENDED` (7 of 10) — user can toggle
   any solver individually.
4. Origin backend selector defaults to `full_amplitude` simulator; a
   dropdown offers `WK_C180`, `WK_C180_2`, `HanYuan_01` when
   `ENABLE_ORIGIN_REAL_HARDWARE=1`.
5. Submit → Approval Panel opens with:
   - **Green routing card:** *"Problem statement (translated) — Number
     Partitioning"* showing `family: number_partitioning` and
     `numbers = [4, 3, 2, 3, 1]`.
   - Preflight: 5 vars, 0 constraints, 5 qubits, all QAOA tiers ✓.
   - Validation: passed (Layer A oracle agrees).
   - CQM JSON (expandable).
6. Approve → solvers run in parallel; SSE emits `solving` and
   incremental per-solver rows appear.
7. On the Origin row: a **"Full view"** button opens the Origin
   dashboard-style pop-up: submitted pyqpanda3 code, gate-level
   circuit diagram, measurement histogram top-29, top-K classical
   filter step.
8. Stage 5 completes → **Solution** card shows:
   - Plain English: *"Group A contains {2, 3, 1} (sum = 6) and Group B
     contains {4, 3} (sum = 7). The minimum absolute difference between
     the two groups is **1**."*
   - Below: the deterministic Solver Assignments block with per-variable
     detail.
   - Tabs: SOLUTION / CQM / VALIDATION / RAW LLM.

---

## 12. Data model (SQLite `jobs` table)

Columns relevant to the Optimization app, with their landing dates:

| Column                     | Type    | Since        | Purpose |
|----------------------------|---------|--------------|---------|
| `id`, `user_id`, `status`  | -       | Phase 4      | Base row identity. |
| `problem_statement`        | TEXT    | Phase 4      | Original prose. |
| `provider`                 | TEXT    | Phase 4      | `claude` / `openai` / `local`. |
| `cqm_json`                 | TEXT    | Phase 4      | Validated CQM (JSON string). |
| `variable_registry`        | TEXT    | Phase 4      | name → description map. |
| `validation_report`        | TEXT    | Phase 5B     | Full validation harness output. |
| `template_id`              | TEXT    | Phase 5B     | Gallery template id if selected. |
| `expected_optimum`         | REAL    | Phase 5B     | Ground truth for oracle. |
| `solver_results`           | TEXT    | Phase 5D     | Per-solver map with SolverResult rows. |
| `solvers_requested`        | TEXT    | Phase 5D     | List the user selected pre-submit. |
| `preflight`                | TEXT    | 2026-06-30   | Approval-gate payload (qubits, tiers, footprint). |
| `solver_params_overrides`  | TEXT    | 2026-07-01   | E.g. Origin backend selection. |
| `plain_english_solution`   | TEXT    | 2026-07-01   | LLM rewrite of interpreted solution. |
| `formulation_route`        | TEXT    | **2026-07-02** | Routing audit trail (family, confidence, parameters). |
| `solve_time_ms`            | INTEGER | Phase 4      | Winner elapsed. |
| `interpreted_solution`     | TEXT    | Phase 4      | Deterministic variable map. |

All ALTER TABLEs are idempotent and gated on
`PRAGMA table_info` — production DBs upgrade in-place.

---

## 13. Feature flags

| Env var                          | Default | Effect |
|----------------------------------|---------|--------|
| `ENABLE_HARDCODED_ROUTING`       | `1`     | Classifier→hardcoded stage 1 branch. `0` restores legacy LLM-only. |
| `ENABLE_ORIGIN_REAL_HARDWARE`    | (off)   | Allows submissions to Wukong-180 class backends. |
| `USE_REDIS_QUEUE`                | (off)   | Route solves through Redis-RQ instead of an in-process thread. |
| `REDIS_URL`                      | `redis://localhost:6379/0` | Queue backend URL. |
| `QUEUE_NAME`                     | `cira-quantum-solves`      | RQ queue name. |
| `QPANDA_API_KEY_FILE`            | (unset) | Bootstrap path for pyqpanda3 credentials. |
| `SESSION_COOKIE_SECURE`          | `false` | Enable HTTPS-only cookies in production. |
| `KEY_ENCRYPTION_SECRET`          | (env)   | Fernet key for BYOK encryption-at-rest. |

---

## 14. Testing strategy

The optimization module ships with layered tests:

| Test suite                             | Count | Scope |
|----------------------------------------|-------|-------|
| `test_hardcoded.py`                    | 26    | Round-trip: formulate → compile → energy at brute-force optimum matches `expected_optimum` for every family. |
| `test_classifier_prompt.py`            | 12    | Pure parser: builds prompt, parses response, clamps out-of-range confidence, downgrades unknown families. |
| `test_pipeline_routing.py`             | 7     | Orchestrator's `_formulate_with_routing`: hardcoded route, LLM fallback, missing classifier, flag toggle. |
| `test_compiler.py`                     | 7     | Schema violations, offset math, sense flipping. |
| `test_formulation_base.py`             | 9     | JSON extraction, schema validation, provider registry. |
| `test_formulation_claude.py` / `test_formulation_openai.py` | 9 + 8 | HTTP wiring with `pytest-httpx` mocks. |
| `test_pipeline.py`                     | 17    | End-to-end pipeline against SQLite fixture + stub provider + mock sampler. |
| `test_qaoa_*_sampler.py`               | ~30   | Sampler-level behavior for local QAOA, IBMQ, Origin (mocked). |
| `test_routes_solve.py` etc.            | 16    | Route-level auth, override validation, `_public_job` shape. |
| `test_validation.py`, `test_oracle_match.py`, `test_reproducibility.py` | -  | Validation harness + reproducibility contract. |

**Total contributing to optimization: 100 +** tests, all green on
2026-07-02.

Deterministic tests avoid the network via injected stubs; every LLM
call in production is best-effort with a safe fallback, so the LLM
being down never fails the pipeline.

---

## 15. What shipped this cycle (2026-07-01 → 07-02)

- **Real-QPU pipeline live on Wukong-180-2.** Confirmed successful
  end-to-end demo including QAOA angle training locally + one-shot
  submission to the real device + histogram interpretation.
- **Approval gate + preflight card.** Users see the qubit count and
  per-tier verdict before any solver burns time.
- **Server-side plain-English interpretation.** Replaced the failed
  WebGPU local LLM attempt with a Claude/OpenAI `summarize_solution`
  call.
- **Backend selector for Origin.** Full-amplitude simulator / Wukong-180
  / Wukong-180-2 / HanYuan_01 with per-backend cap awareness.
- **QPU footprint estimator.** Compute vs queue time distinction called
  out explicitly to prevent user confusion about billing.
- **Template gallery split.** "Real Quantum Friendly" vs
  "Classical & Larger Instances" sections; each template ships a
  `qpu_ready` flag.
- **Login moved to landing page** with dialog-based auth and a settings
  page containing API Keys + Change Password.
- **Number Partitioning few-shot anchor** in `examples.json` + optional
  `objective.offset` in `cqm_v1` schema.
- **The routing layer (this cycle's headline):** four hardcoded
  formulators (`max_cut`, `number_partitioning`, `max_independent_set`,
  `portfolio_selection`), a shared classifier prompt module, `classify_problem`
  method on `FormulationProvider`, orchestrator's `_formulate_with_routing`,
  `formulation_route` audit column, and the visible *"Problem statement
  (translated)"* card in the Approval Panel.

---

## 16. Roadmap notes

Deferred but tracked:

- Wire the routing chip / translation card into HistoryDetail (currently
  ApprovalPanel only).
- Extend the classifier prompt with Knapsack (natural form) and Bin
  Packing families once the anchoring math is settled.
- QAOA layer > 1 support in the local training loop (currently p=1).
- QUBO / QAOA cost cache for repeated identical CQMs.
- A "compare hardcoded vs LLM" diff view for family templates so
  contributors can see the exact coefficient delta at a glance.
- Histogram top-K display bug: some rows show `E = 1.000` when their
  decoded energy is higher — likely a top-K filter labeling artifact
  (2026-07-02, observed post-restart).

---

*Any coefficient here traces back to a Python file under
[`backend/app/formulation/hardcoded/`](../backend/app/formulation/hardcoded/).
Any UI element traces back to a Vue component under
[`frontend/src/components/`](../frontend/src/components/). Nothing in
this document is aspirational — every feature listed is shipped and
covered by tests as of the "Last major revision" date at the top.*
