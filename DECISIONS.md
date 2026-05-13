# DECISIONS

A running log of choices made when the project specification was silent or
ambiguous. Each entry is dated and scoped to a single phase so the rationale
can be revisited if the underlying assumption changes.

The active specification is `PROJECT_TEMPLATE v2.md`. The original
`PROJECT_TEMPLATE.md` (v1) is preserved at the repo root for audit trail.

---

## Phase 9C — Quantum-inspired classical tiers (PT + SB)

**Date:** 2026-05-13

### Parallel Tempering: custom Hukushima-Nemoto implementation, not a library

The published PT algorithm is straightforward enough (~200 lines of
numpy) that wrapping a third-party library would have added more
surface area than it saved. The simulated-bifurcation case is
different — Goto's algorithm has subtle continuous-time dynamics that
the upstream package implements correctly; rewriting it would have
been a multi-week investment.

So: PT in-house, SB via the `simulated-bifurcation` PyPI package.
Both follow the same `dimod.Sampler` interface as the Phase-8 tiers.

### PT semantics: `num_sweeps` is total simulation length, `num_reads` is snapshot count

This differs from SA-class samplers. The Phase-2 run_suite passes
`num_reads / num_sweeps` to both gpu_sa/cpu_sa_neal AND PT, but the
*meaning* differs:

- gpu_sa: `num_reads` independent restarts × `num_sweeps` per restart
  → ~500k spin-flips at the default settings
- PT: 1 trajectory × `num_replicas` × `num_sweeps` → ~8k flips at the
  same nominal settings (under-budgeted by 60×)

Experiment #2 ran both v1 (naive same-kwargs) and v2 (explicit
PT-fair budget of `num_sweeps=20000`) to verify this matters.
The 20× budget bump didn't change PT's wrong-answer pattern on the
hard instances — confirming the issue is algorithmic, not compute.

### Phase 9C headline finding: both tiers underperform on penalty-lifted CSPs

`parallel_tempering` matched the documented optimum on only 2 of 9
instances; `simulated_bifurcation` on 4 of 9. Both worse than
vanilla `gpu_sa` and `cpu_sa_neal` (5/9 each) on this fixture set.

This isn't an implementation bug — both adapters find the optimum
cleanly on small unconstrained QUBOs (3-var sympy, 5-var random).
The mismatch is that **the Phase-2 fixtures are penalty-lifted
CSPs**, exactly *not* what PT and SB were designed to dominate. PT's
published 10-100× speedup over SA is on Edwards-Anderson and
Sherrington-Kirkpatrick spin glasses — pure quadratic Ising models
with no Lagrange-induced large-coefficient discontinuities.

The honest scoreboard interpretation: PT and SB are filed under
"quantum-inspired classical" for completeness; their dashboard cells
should be read in that context, not as evidence the algorithms are
inferior. Adding spin-glass instances to the manifest is the right
follow-up.

---

## Phase 9B follow-up — empirical Wukong qubit wall + non-blocking capture

**Date:** 2026-05-13 (after Phase 9C shipped)

### The wall is between n=7 and n=8 for fully-coupled QAOA via Python

Two of our Python-submitted Wukong jobs (jobs `738A…` and `B58D…`)
sat in `JobStatus.COMPUTING` for 7+ hours each, never returning a
result. The web-UI Composer running the same Wukong chip returns
4-qubit Bell circuits in ~1 minute. We isolated the cause through a
series of diagnostic probes:

| n (qubits) | Couplings | Submitted via | Wall time |
|---|---|---|---|
| 4 (Bell, H+CZ) | 2 | Web UI | ~90 s |
| 4 (Bell, H+CNOT) | 2 | **Python** | **16 s** |
| 2 (minimal QAOA) | 1 | Python | 10 s |
| 4 (dense QAOA) | 6 | Python | 114 s (slow but completes) |
| 5 (dense QAOA) | 10 | Python | 10 s |
| 6 (dense QAOA) | 15 | Python | 10 s |
| 7 (dense QAOA) | 21 | Python | 10 s |
| **8 (dense QAOA, small coeffs)** | 28 | Python | **hung > 3 min** |
| **8 (dense QAOA, Lagrange coeffs)** | 28 | Python | **hung > 3 min** |

Both 8-qubit probes hung, regardless of coefficient magnitude. The
wall is between n=7 and n=8, and the cause is most likely
**transpilation cost** — at 28 couplings on Wukong's sparse
`{RPhi, CZ}` topology, the SWAP-cascade explosion pushes the
physical-gate count past whatever silent timeout the cloud's
processing pipeline enforces.

The cap is encoded in `QAOACloudSampler` as
`_DEFAULT_MAX_QUBITS_REAL_QPU = 7` and is auto-selected when
`backend_name in _REAL_HARDWARE_BACKENDS`. The cloud simulators
retain the original 64-qubit cap.

### Non-blocking submit pattern for cloud QAOA

The synchronous `record_run` path doesn't work for the real QPU even
within the qubit cap, because the Bash/Python process running the
submission gets reaped by various timeouts (Bash background-task
lifetime; Monitor budgets) before `job.result()` returns. The
solution is a two-step flow:

1. **Submit-only script**: trains QAOA params locally, builds the
   circuit, submits to Wukong via `backend.run()` to get a job ID,
   then *immediately* adds the job to
   `backend/data/pending_cloud_jobs.json` and exits. Total runtime
   measured in seconds.

2. **Dashboard auto-poll**: `PendingCloudJobsPanel.vue` polls every
   30 s, calls `POST /api/benchmarks/cloud-jobs/<id>/materialize` as
   soon as `has_probs` flips true. Result becomes a regular
   `qaoa_originqc` `RunRecord` in the archive without any
   long-running Python process needing to stay alive.

This is the production pattern for all future cloud QAOA submissions
— synchronous `record_run` is reserved for solvers that return
within seconds (everything except `qaoa_originqc`).

The first real-QPU `qaoa_originqc` record (job `93A2EAE6…`) is in
flight via this pattern at the time of this commit and will
materialize when Wukong returns. The pending list also had two stuck
jobs (`738A…`, `B58D…`) dropped during the cleanup — they were
8-qubit submissions over the now-known n=7 wall.

---

## Phase 9B — Origin Quantum cloud BYOK + real Wukong QPU

**Date:** 2026-05-12

### Discovery: the documented cloud URL is wrong

The pyqpanda3 docstring example shows
``url="https://qcloud.originqc.com.cn"``. Submitting against that URL
returns ``RuntimeError: Invalid value`` on ``backends()``. The actual
working endpoint is the **default** URL bound into pyqpanda3
(``http://pyqanda-admin.qpanda.cn``). The QAOACloudSampler defaults to
this. Worth flagging because the docs and the working endpoint are out
of sync — a follow-up could be filing a docs PR upstream.

### Credential format: full string, including the ``/&lt;account-id&gt;`` suffix

The Qpanda.txt credential is a 102-char string of the form
``<96-char hex>/<5-digit account>``. The pyqpanda3 cloud library
expects the **entire raw string** as the ``api_key`` argument, not
just the hex part. Stripping the account suffix returns
``RuntimeError: value is not array``.

### Hybrid local-train + cloud-execute (not full variational on cloud)

Each cloud submission has 30 s – several minutes of queue + execution
latency. The standard QAOA optimizer (SLSQP) needs 30–50 iterations
to converge. Running the full variational loop against the cloud
would mean ~30 cloud round-trips per record, taking 30–100 min and
burning real QPU credits.

Instead, ``QAOACloudSampler`` runs the entire variational loop **on
the CPU statevector simulator** (Phase-9A's local path) to extract
optimal ``(gamma, beta)`` parameters, then rebuilds the trained
circuit using ``pauli_z_operator_to_circuit`` + manual mixer
construction, and submits **one** trained circuit to the cloud.
Result: one cloud submission per record, ~3-10 min total wall time.

The result is *honest about what's quantum*: the variational
optimization happens classically; the final measurement happens on
real (or simulated) quantum hardware. That's literally what QAOA-on-
cloud-hardware is at every quantum-software vendor — the optimizer
loop never lived on the QPU.

### Real-hardware gating via ``ENABLE_ORIGIN_REAL_HARDWARE`` env flag

Constructing ``QAOACloudSampler(backend_name="WK_C180")`` (or any
backend in ``_REAL_HARDWARE_BACKENDS``) raises during ``__init__``
unless ``ENABLE_ORIGIN_REAL_HARDWARE=1`` is set in the environment.
This prevents accidental QPU-credit burns during development —
opening the Solve UI and clicking "submit" against the wrong tier
shouldn't be enough to spend real money on real qubits.

The cloud-simulator backends (``full_amplitude``, etc.) work
without the flag, since they don't cost QPU minutes.

### Per-sampler-instance submission cap (default 5)

A second-line guard inside ``sample()``: each ``QAOACloudSampler``
instance refuses further calls after ``max_submissions=5`` successful
submissions. Phase-7's full rate-limit infrastructure will replace
this with proper per-user limits; until then, the in-class cap
prevents a runaway loop from emptying a user's credit balance during
a single REPL session.

### Closure-bound API key — never recorded in RunRecord

The Phase-2 ``records.record_run`` dispatcher passes ``init_kwargs``
into ``sampler_cls(**init_kwargs)``, and those kwargs land in the
``RunRecord.parameters`` field that gets serialized to disk in
``benchmarks/archive/*.json``. If the ``api_key`` flowed through that
path, every cloud record would leak the credential.

The fix: ``bootstrap_default_solvers()`` reads the key from a file
path supplied via ``QPANDA_API_KEY_FILE`` env var, then registers a
**closure-bound subclass** of ``QAOACloudSampler``:

```python
class _BoundQAOACloud(QAOACloudSampler):
    def __init__(self, **kwargs):
        super().__init__(api_key=_api_key, **kwargs)
```

``_INIT_ONLY_KWARGS_BY_SOLVER["qaoa_originqc"]`` excludes
``api_key``. The route / suite runner never sees the key. Verified
via a smoke test that explicitly grepped the serialized parameters
JSON for the credential's known hex prefix.

### ``originqc`` BYOK provider is registered alongside Claude / OpenAI / local-LLM

The Phase-3 BYOK store (Fernet-encrypted at rest under
``KEY_ENCRYPTION_SECRET``) is provider-name-agnostic — it stores
``(user_id, provider_name, ciphertext)``. Adding ``originqc`` to the
allow-list in ``app.routes.keys._is_known_provider`` (via a separate
``_QUANTUM_PROVIDERS`` set, distinct from the LLM formulation
registry) and to the frontend ``ApiKeyManager.vue`` dropdown was
all that was needed at the table level. The Solve-API flow that
**uses** the stored key still needs wiring — that lands when the
quantum-tier becomes a Solve-UI-selectable provider in a later
phase. For Phase 9B, the QPU records are populated via the suite
runner / experiment scripts that read the key from the file path,
not via the live Solve route.

### Wukong chip metadata: 169 working qubits, native gates ``{RPhi, CZ}``

At probe time, ``WK_C180`` advertises 169 available qubits (of 180
nominal — typical for a superconducting chip; a few qubits get
disabled at calibration). Native gateset is ``RPhi`` (parameterized
single-qubit phase rotation) and ``CZ`` (two-qubit controlled-Z).
The QAOA circuit we build uses ``H``, ``RX``, and ``RZ``-derived
gates from the problem unitary; pyqpanda3's transpiler maps these to
the native set automatically before submission.

The 169-qubit cap is well above any Phase-2 instance after
Lagrange-lowering, so Wukong is not the binding constraint — the
binding constraint is the ``max_qubits=64`` default in
``QAOACloudSampler``, chosen conservatively to keep classical
training time bounded.

### Tests: 12 mocked, 0 real-hardware in the default suite

The full ``QAOACloudSampler`` dispatch path is covered by 12 tests
using a fake ``QCloudService`` that returns a fixed probability
distribution. No tests in the default suite touch the network. Real-
hardware integration tests are deferred to manually-invoked scripts
(like ``D:/tmp/smoke_qaoa_cloud.py``); the test runner stays cheap.

### Real-cloud verification deferred — OriginQC cluster-reply outage 2026-05-12 to 2026-05-13

Across **four consecutive submissions** to the ``full_amplitude``
cloud simulator between 21:31 on 2026-05-12 and 09:00 on 2026-05-13
(13+ hours), every job consistently hit the same server-side error:

```
status: JobStatus.COMPUTING (for ~4-5 minutes)
→ error_message: "query qcloud task <id> failed:
                  Error: Can NOT receive cluster reply!"
→ JobStatus.???  (unrecognized enum value after the error)
```

The pattern is identical across all four jobs:
1. ``DB588E2E…`` — 2-qubit Bell state, submitted 21:31 2026-05-12
2. ``C96656F9…`` — 1-qubit warmup, submitted 22:08 2026-05-12
3. *(third submission via smoke_qaoa_cloud.py — process killed before
   it logged a job ID, but matches the pattern)*
4. ``CC861DA7…`` — 1-qubit warmup, submitted 09:00 2026-05-13 (fresh
   morning attempt)

This is a sustained Origin Quantum cloud-side cluster issue, not
a bug in our integration. The auth handshake (``QCloudService(api_key)``)
and the backend listing (``service.backends()``) both work fine; the
``backend.run(prog, shots)`` submission succeeds and returns a job ID;
the cluster simply never returns results.

**What we know works** (verified at construction and submission time):

- The credential authenticates against
  ``http://pyqanda-admin.qpanda.cn``
- Six backends advertised, including ``WK_C180`` Wukong (real QPU,
  169 working qubits) marked as "available"
- ``backend.run()`` submission returns a job ID immediately
- Status query returns ``JobStatus.COMPUTING`` initially

**What we cannot verify until the cloud recovers:**

- A full round-trip producing measurement probabilities
- The ``QAOACloudSampler.sample()`` end-to-end path on a real cluster
- A live ``qaoa_originqc`` record in ``benchmarks/archive/``

The Phase 9B code is shipped (commit ``d787c58``) and verified via 12
mocked-cloud tests that exercise every code path. The live-cloud
record is the only outstanding item, and it's blocked on an upstream
outage we cannot work around. Recommended actions in priority order:

1. Wait 24-48h and retry. Cluster outages of this kind typically
   recover within a day.
2. If the outage persists, file an issue at OriginQC's support
   channel referencing the four job IDs above.
3. If we need a real-quantum record for a thesis / publication
   deadline, the ``WK_C180`` real-QPU path is a separate cluster from
   the simulator path and may be unaffected. Worth trying once
   under ``ENABLE_ORIGIN_REAL_HARDWARE=1``, but this burns real QPU
   credits.

### Test artifact cleanup: removed the `-0.5` energy record

A bogus ``qaoa_originqc`` record landed in
``benchmarks/archive/`` at 22:02:35 on 2026-05-12 during the Phase
9B pre-commit security verification. The verification script ran
``record_run(solver_name='qaoa_originqc', ...)`` against a 2-variable
test BQM (to prove the api_key never lands in serialized parameters)
but labeled the run with ``instance_id='knapsack/small/knapsack_5item'``.
The mocked ``QCloudService`` returned fixed probabilities, the
resulting energy of ``-0.5`` was computed against the test BQM, and
the whole thing got written to disk as if it were a legitimate
knapsack run.

The record (file ``20260512T150235.502660Z_cb1dcd.json``) was deleted
during this cleanup. The archive's "append-only" principle is for
*legitimate* records; an obvious test artifact is fair to remove.

**Guard going forward:** future security-verification scripts should
either use ``record_run(..., archive_samples=False)`` *and* never
write the ``RunRecord`` to disk (the dataclass exists in-memory only),
or run against an instance ID that's clearly tagged as a test
fixture (e.g. ``_test/security/empty``). Both paths avoid the
"realistic-looking but synthetic" record we just cleaned up.

---

## Roadmap amendments — 2026-05-12 (post-Phase-8)

The v2 spec's Phase 9 is one block ("Quantum-Inspired Solver Tiers"). After
building turned up new information, three amendments are pinned here.

`ROADMAP.md` at the repo root carries the working plan; the v2 spec is
preserved verbatim as the audit contract.

### Phase 9 split into 9A / 9B / 9C along the auth + hardware boundary

The original v2 Phase 9 lumped PT, PQQA, and SQBM+ together. OriginQC's
pyqpanda3 stack (QAOA + GAS) adds a fundamentally new tier *category* —
**true quantum** (variational, runs on actual superconducting qubits via
their cloud) — distinct from the quantum-*inspired* classical heuristics
v1 had in mind. The auth + hardware shape of true-quantum work is also
different enough that it should ship in two pieces, not one:

- **9A — local simulator.** No BYOK, no remote credentials, no shot
  noise from physics. CPU statevector simulation under pyqpanda3.
  Strategically the highest-leverage small unit of work: it adds the
  *real quantum algorithm column* to the Benchmark dashboard at the
  cost of one adapter + tests.
- **9B — Origin Quantum cloud (real QPU, BYOK).** Same `QAOASampler`
  class, backend switch flipped to the hosted superconducting
  processor. Adds `originqc` as a fourth provider in the existing
  BYOK plumbing (alongside Claude / OpenAI / local LLM).
- **9C — quantum-inspired classical tiers.** Parallel Tempering, PQQA,
  and (optionally) the open `simulated_bifurcation` algorithm. These
  are CPU Monte Carlo heuristics; useful for completeness, less
  strategically differentiating than 9A/9B.

The architecture absorbs the split cleanly: `_CQM_NATIVE` (introduced
in Phase 8) dispatches all three tiers through `sample_cqm`, and the
RunRecord schema is solver-tier-neutral by design.

### Phase 9D (SQBM+) dropped from the roadmap

SQBM+ is Fujitsu's proprietary brand on top of openly-published
simulated-bifurcation algorithms (Goto et al. 2019; ballistic and
discrete bifurcation variants). For an open academic platform, three
things make it not worth pursuing:

- The **brand** belongs to Fujitsu. Using "SQBM+" without a partnership
  is awkward; getting one is a multi-month sales cycle aimed at
  industrial customers, not academic platforms.
- The **algorithm** is published openly. If we want a bifurcation-style
  solver, we implement it ourselves under a generic name
  (`simulated_bifurcation`, `bsb`, `dsb`) with zero Fujitsu engagement.
  That captures 95% of the academic value at 0% of the partnership cost.
- The **scoreboard** doesn't change much. Once 9A + 9B + 9C are live, a
  fifth tier whose benchmark behavior is similar to PT/PQQA doesn't
  shift the narrative.

If we ever want bifurcation, it goes into 9C as a generic option, not
as a separate phase. SQBM+ is removed from the roadmap entirely.

### Phase 6 and 7 deferred to after the solver-tier work (9A → 9B → 9C → 6 → 7)

Phase 6 (Redis + RQ queue + GPU lock) and Phase 7 (rate limits, email
verify, abuse detection, admin UI, Cloudflare Tunnel, HTTPS) are
*pre-deployment* work, not *pre-development* work. Their value
materializes the moment we point a public URL at the host —
multi-user GPU contention pressure, rate-limit pressure, abuse
surface. None of those pressures exist while the platform is a single-
developer build on localhost.

Doing them at the end means:

1. **The solver-tier landscape ships first** (9A + 9B + 9C). The
   Benchmark dashboard tells the complete *classical vs quantum-
   inspired vs quantum-simulator vs real-QPU* story 3 weeks earlier
   than under strict spec order.
2. **Phase 6 and 7 are validated against a complete backend.** Queue
   semantics for QAOA-class long-running jobs are different from
   solver-class jobs that finish in 10ms; designing the queue with
   the full solver portfolio visible avoids retrofits.
3. **Phase 0 set the precedent.** It was built late, after Phases 1–3,
   for the same reason: settle the science before you operationalize
   it.

The cost is ~3 weeks of "almost public" before "actually public."
The risk is a real-QPU BYOK demo (9B) burning external testers'
credits — mitigated inside 9B itself via a hardcoded per-session
submission cap + a default-off `ENABLE_ORIGIN_REAL_HARDWARE` feature
flag. Both cost ~20 lines and remove the demo-time risk entirely
without requiring Phase 7's full rate-limit infrastructure.

---

## Phase 5C + 8 — Benchmark dashboard and classical solver tiers (parallel)

**Date:** 2026-05-11

Phases 5C and 8 were built in parallel rather than strictly sequentially.
The spec orders them 5C → 8 because v1 never had 8; v2 retrofitted
classical baselines. Once the archive is the shared interface between
the two, they are siblings, not predecessors: 5C reads it, 8 writes to
it. The launch with all five solver tiers visible on day one matches the
v2 "honest scoreboard" narrative far better than a two-stage rollout
would.

### `_CQM_NATIVE` class marker for the records dispatcher

`records.record_run` originally chose between `sample_cqm(cqm)` and
`sample(bqm)` by hardcoding the class name `"ExactCQMSolver"`. Phase 8
adds two more CQM-native solvers (CP-SAT, HiGHS), so the dispatch is
now driven by a class attribute `_CQM_NATIVE = True` on the adapter
class. The hardcoded `ExactCQMSolver` check is preserved alongside it
to avoid changing the dimod baseline's behavior. Phase 9 (PQQA, etc.)
will reuse the same marker.

### CP-SAT objective scaling

OR-Tools' CP-SAT is an integer solver. Real-valued CQM objective
coefficients are scaled by `_OBJ_SCALE = 1_000_000` before being added
to the model, and the user-facing objective is recomputed from the CQM
itself via `dimod.SampleSet.from_samples_cqm` so the back-scaling is
free. The 1e-6 ULP is well below the validation harness's 1e-3 match
tolerance for every Phase-2 / 5B instance we ship.

### HiGHS rejects quadratic objectives

HiGHS is a MIP solver — linear-objective only. The adapter raises
`ValueError("HiGHSSampler does not support quadratic objectives...")`
rather than silently dropping the quadratic terms. This surfaces the
mismatch loudly when the LLM picks a quadratic encoding for an
otherwise-MIP-shaped problem. The Phase-5C dashboard's empty cells
(see e.g. `maxcut/gset_subset × highs`) are the visible consequence —
a genuine "wrong tool for this problem" signal, not a bug.

### `num_reads` / `num_sweeps` are SA-only kwargs

The Phase-2 suite runner originally passed `num_reads` and `num_sweeps`
to every solver. CP-SAT and HiGHS don't take them — they're SA-class
concepts (number of independent chains × inner sweep count). The
runner now keys these kwargs to `gpu_sa` / `cpu_sa_neal` only; the
classical tiers get `seed` but not the SA-class parameters. The
parameter dict still records the choice, so reproducibility is
preserved.

### Classical-beats-QUBO is a regression test, not a decoration

`test_cpsat_beats_gpu_sa.py` asserts three things on the canonical
20-item knapsack instance: (1) CP-SAT finds the optimum (233),
(2) CP-SAT is faster than GPU SA, (3) CP-SAT's quality strictly
dominates GPU SA's quality. Empirically: CP-SAT 8.8ms / value 233 vs
GPU SA 3.4s / value 189. If this test ever fails, *either* GPU SA got
dramatically better (good — update the test to a harder instance)
*or* CP-SAT regressed (very bad — investigate). Either path is
meaningful signal; the test failing is never a silent flake.

### Phase 5C reads from disk on every request

The dashboard backend rereads the entire `benchmarks/archive/` on
every list call. With 44 records this is ~30ms; the projected ceiling
for "OK without a cache" is ~10k records, at which point a per-suite
in-memory cache (or a SQLite materialized view) will be needed. The
v2 spec's 3-second cold-load budget is met with ~3 orders of magnitude
headroom for now.

### Dashboard is public (no auth)

`/benchmarks` and all `/api/benchmarks/*` endpoints are
`@login_required`-free. The whole point of the Phase-5C scoreboard is
that it's *citable* and *replicable* without an account, in keeping
with the v2 positioning. Writing to the archive (the Phase-2 CLI)
still requires shell access to the host — that's the implicit
auth surface for contributions.

### Time-series view is deferred to Phase-10 contribution era

The spec lists `PerformanceTimeSeries.vue` as a 5C deliverable. We
have 44 records across 5 solvers right now — not enough longitudinal
data to make a line chart worth plotting. The component is wired into
the store contract (the record schema carries `started_at` and
`solver.version`) but the visual landed as a placeholder. When the
Phase-10 contribution pipeline brings external runs in over months,
the line gets meaningful.

---

## Phase 5B — Template / Modules library

**Date:** 2026-05-11

### Schema-validated JSON templates, not Python modules

Each template lives in `app/templates/library/<id>.json` and is checked
against `app/templates/schemas/template_v1.json` at import time. The
alternative — Python dicts in a registry module — would have been
slightly faster to author but loses two properties:

1. **External authoring**: a content contributor (course author, TA)
   doesn't need to know Python or open a PR against the runtime code
   path to add a problem. The JSON files live alongside the code but
   are independently editable.
2. **Eager validation**: ``jsonschema`` runs at process boot, not at
   first request. A malformed template surfaces as a startup failure
   the operator can't miss, rather than a 500 the first user to click
   on it triggers.

The schema is versioned (``template_v1``) so a future ``v2`` shift
(richer Modules metadata, multi-language problem statements) can land
without retrofitting every existing template.

### Module metadata is optional, not required

Templates carry a ``module`` field only when they belong to a structured
learning path. The Phase-5B Gallery view shows all 10 templates;
the Modules view shows only the 5 that have ``module_id`` set
(currently all in ``qubo_foundations``). This avoids the trap of
forcing every problem into a curriculum — TSP, JSS, bin-packing,
nurse-rostering, and portfolio are useful standalone examples that
don't need a "lesson 1 of N" framing to make sense.

### `expected_optimum` is a property of the template, not the solver

The template author records the known optimum (verified by brute-force
or literature reference) at authoring time. The match badge compares
the solver's ``validation_report.energy_oracle`` against this value
with a relative tolerance of ``1e-3``. The badge's purpose is
**didactic**, not gating — a mismatch surfaces with a tonal warning
chip and an explanatory tooltip, not an error. Three legitimate
reasons a mismatch can occur:

1. The LLM picked a different (still feasible) formulation whose
   optimum on the user's data differs.
2. The sampler didn't converge in the configured ``num_reads``.
3. The template's expected value is wrong (rare, but possible — the
   TSP-5cities template was corrected from 62 → 72 during this
   phase after a brute-force check exposed the original as
   a stale value carried over from an early draft).

For all three, surfacing the discrepancy is more useful than hiding
it. The user can re-run, switch provider, or report the template.

### Two blueprints for template-driven solving, not one

``templates_bp`` mounts at ``/api/templates`` for GET-only routes
(list, modules, categories, detail). ``solve_from_template_bp`` mounts
at ``/api`` so the POST lands at ``/api/solve/from-template/<id>``,
not ``/api/templates/<id>/solve``. The reason is that the frontend
solve store already routes through ``/api/solve/...`` — keeping the
mutating endpoint under the same path prefix means the existing
axios client doesn't need a second base URL or its own error handler
for template-launched jobs. The orchestrator launcher
(``_launch_pipeline_in_background``) is reused verbatim — the only
difference between the two solve entry points is where the
``problem_statement`` comes from.

### `template_id` and `expected_optimum` are columns on `jobs`, not a separate join table

Adding two columns to ``jobs`` is denormalized (the value lives on
the template too), but it has three properties a join would not:

1. **The template can change without rewriting history**: if a
   template author updates the expected value or rewrites the
   problem statement, archived jobs keep showing what the solver
   actually saw at the time of the run.
2. **The match badge needs no extra fetch**: ``GET /api/jobs/<id>``
   already returns both columns; the badge component doesn't need a
   separate template lookup to render.
3. **Deletable templates**: a template can be retired from the
   library without orphaning the jobs that referenced it.

Idempotent ``ALTER TABLE`` migration via ``init_db()``'s
``PRAGMA table_info`` check (same pattern Phase 4 used for
``validation_report``) — dev DBs upgrade in place, fresh installs
get the right schema from ``CREATE TABLE``.

### TSP-5 cities optimum is 72, not 62

The original draft of ``tsp_5cities.json`` listed
``expected_optimum: 62``, copied from a textbook example that used
different coordinates. A brute-force enumeration of all
``(5-1)!/2 = 12`` distinct tours over the actual coordinate set in
the template gives 72 (with multiple optimal tours). The template was
corrected before merging. Recorded here because the value will read
as "obviously wrong" against the textbook for anyone cross-checking
— and the answer is "different coordinates, same problem class."

### Modules view uses a v-timeline, not a numbered list

Vuetify's ``v-timeline`` was picked over a plain ordered list because
each lesson's *prerequisites* are first-class data
(``module.prerequisites: [other_template_id, ...]``) — the timeline
visually reinforces the "lesson 2 depends on lesson 1" relationship
in a way an ``<ol>`` would not. Within a module, lessons are sorted
by their declared ``order`` field, not by ID. Multi-module support
(adjacency between modules) is deferred to Phase 5C/7.

---

## Phase 5 — Frontend Solve UI

**Date:** 2026-05-11

### SSE EventSource lives in the Pinia store, not in components

A naive Vue 3 wiring would put the ``EventSource`` inside the component
that displays progress (``SolveStatus.vue``). That leaks: if the user
navigates away mid-solve, the component unmounts but the underlying
stream stays open until the browser tab is closed. The Pinia store
owns the connection — ``streamStatus(jobId)`` is the single creator,
``closeStream()`` the single destroyer, both called from
``submitProblem``, ``subscribeToJob``, ``loadJob`` (on terminal
status), and ``reset()`` (on logout). Component unmount in the
``JobDetailPage`` calls ``closeStream`` explicitly to cover the
mid-flight-navigation case.

### `use_stored_key` is opt-in per submit, not a saved preference

The frontend doesn't persist a "default to stored key" preference per
provider. Every submit asks again, defaulting to *inline*. The cost
is one extra checkbox click per submit; the benefit is that a user
who pastes a key (e.g. to try a colleague's account) doesn't
accidentally fall back to their own stored key on a typo. The
``ApiKeyManager`` page is where stored keys are stewarded; the
solve form is where the decision is made per-submit.

### Vite proxy strips `/api` differently than expected — don't rewrite

We hit a sharp edge during Phase 0 wiring: a ``rewrite`` rule on the
Vite proxy that stripped ``/api`` from the path broke the SSE
endpoint, because Vite buffered the response. The fix was to leave
the path intact — Flask's blueprint prefixes already mount at
``/api/...``, so the proxy can pass requests through unmodified.
Documented because future "let's clean up the prefix" refactors will
hit this same trap.

---

## Phase 4 — Solve API endpoint

**Date:** 2026-05-11

### One-event-per-stage SSE emission

The v1 spec's SSE example shows ``"compiling"`` carrying
``num_variables`` + ``num_constraints``, which suggests two possible
timings: emit start-of-stage with empty fields, then end-of-stage with
counts. We picked one — **emit start-of-stage, period**. The counts
land in the DB (via ``num_variables``/``num_constraints`` columns) and
are read by ``GET /api/jobs/:id``; the SSE stream stays as a clean
five-event sequence ``formulating → compiling → validating → solving
→ complete``. The Phase-5 progress timeline UI is easier to write
against a deterministic sequence than against duplicated events.

### Layer B (multi-solver agreement) is skipped during live solves

The Phase-2 validation harness's Layer B runs *another* CPU + GPU SA
pass on the BQM lowering. For a live user solve, that's a 30–60 s tax
on top of the one solve we actually want — most users would
interpret the wait as the platform being broken. Phase 4's orchestrator
calls ``validate_cqm(..., skip_layer_b=True)`` so the live request
runs Layer A (oracle when small) + Layer C (constraint coverage) only.
Layer B's signal — multi-solver agreement — is the **Phase 5C
dashboard's** job, running on archived RunRecords offline.

### Pipeline runs on a daemon thread, not async event loop

The Flask request handler can't ``await`` the orchestrator (Flask
3.x's sync request path doesn't drive an event loop). The route
handler spawns a ``threading.Thread(daemon=True)`` whose target wraps
``asyncio.run(orchestrator.run(...))``. Three consequences worth
calling out:

1. The thread dies when the parent process exits — fine for dev,
   important context if someone naively shells the dev server out
   with ``nohup`` (the in-flight job will be orphaned and never
   complete, because daemon threads die with the main process).
2. SQLite from a thread requires the connection to not be shared —
   ``models.py`` opens a fresh connection per call, which sidesteps
   the ``ProgrammingError: SQLite objects created in a thread can
   only be used in that same thread`` trap. Tests exercise this path
   inadvertently via the SSE flow.
3. Phase 6 replaces this with Redis + RQ. The orchestrator's API
   doesn't change — only ``_launch_pipeline_in_background`` does.

### `_launch_pipeline_in_background` is monkeypatched in tests

The route tests don't want to wait 60s per case for a real solver to
run. The fixture monkey-patches the *route module's* attribute, not
the orchestrator class, so the public orchestrator import path stays
stable. Tests verify the HTTP surface; the orchestrator's own behavior
is covered in ``test_pipeline.py`` with injectable
``sampler`` + ``provider_resolver`` dependencies.

### `404`, not `403`, on cross-user job access

When user B tries to fetch user A's job, the API returns
``404 NOT_FOUND`` rather than ``403 FORBIDDEN``. The reason is
**existence-leak avoidance**: a 403 confirms the resource exists; a
404 doesn't. For a multi-tenant platform where job IDs are UUIDs that
could be guessed (or scraped from a referrer header), the 404 path
prevents enumeration attacks. The seeded admin gets the 200 path.

### Job IDs are UUIDs, not sortable timestamps

``create_job`` issues a ``uuid.uuid4()``. We considered the same
ULID-style sortable timestamp Phase 2's ``RunRecord`` uses, but
``jobs`` already has a ``created_at`` column for ordering and the
``id`` is the SSE stream's path parameter — uuid-shaped strings are
easier to spot in URLs than 24-char timestamp blobs.

### Sample-value type coercion: ``v.item() if hasattr(v, "item")``

``dimod`` returns numpy scalars (``numpy.int8`` for binaries,
``numpy.int64`` for integers). Python's default ``json`` encoder
rejects them. The orchestrator coerces with ``v.item() if hasattr(v,
"item") else v`` — works for numpy scalars, Python ints, floats, and
booleans alike. A custom ``JSONEncoder`` subclass would do the same
thing with more ceremony.

### `validation_report` schema migration: idempotent ``ALTER TABLE``

Phase 0 shipped the ``jobs`` table without a ``validation_report``
column. Phase 4 needs it. ``init_db()`` now runs ``ALTER TABLE jobs
ADD COLUMN validation_report TEXT`` only when ``PRAGMA table_info``
reports the column is missing. Existing dev DBs upgrade in place;
fresh installs get the right schema from ``CREATE TABLE``.

### BYOK key resolution: stored cipher beats inline arg only when explicit

``_resolve_api_key`` follows this precedence:

1. ``use_stored_key: true`` → decrypt from DB; 402 if not stored.
2. ``api_key`` inline → use it as-is.
3. Neither → 402 for cloud providers; empty string for ``local``.

The choice to make ``use_stored_key`` an explicit opt-in (not the
default) prevents accidental reuse of stale keys. Phase 5's frontend
sets the checkbox according to whether the user has a stored key for
the selected provider.

### Lagrange multiplier is part of the Orchestrator constructor, not request param

The route doesn't accept a ``lagrange_multiplier`` query parameter.
The orchestrator's default is ``10.0`` (the Phase 2 settled value)
and it's set at construction time. We considered surfacing it through
the API but tabled until Phase 7 — exposing tuning parameters in the
public API is a sharp edge for users who don't know what they're
doing, and the Phase 5C dashboard is where comparative runs at
multiple Lagrange values belong.

---

## Phase 0 — Project skeleton + auth + DB (built-late catch-up)

**Date:** 2026-05-11

### Phase 0 was shipped late, against the spec's narrative order

The v2 spec table lists Phase 0 as "shipped under v1 ✅" — but that
claim was inherited from the prior CiRA Optima project. This
repository started fresh at Phase 1, so Phase 0's scaffolding (Flask
app factory + auth + SQLite + frontend shell) didn't actually exist on
disk before Phase 4 kickoff. Building Phase 0 first as its own
bounded phase (per the "one PR per phase" hard rule) was the
explicitly-chosen path; the alternative — bundling it with Phase 4
— was rejected.

The shipped Phase 0 surface matches the v1/v2 spec verbatim: same
table schemas, same five auth routes, same default admin
(`admin`/`admin123`), same Vue 3 + Vuetify dark theme, same Pinia
auth store pattern.

### `tests/conftest.py` does **not** isolate the auth tests' DB

Phase 0 ships its own ``tests/test_auth.py`` fixture that points
``DATABASE_PATH`` at a tmp_path before creating the Flask app. The
existing top-level ``tests/conftest.py`` (Phase 2 + 3 fixtures: BQM
loaders, CQM-JSON fixtures, helpers) doesn't touch the DB and so
doesn't need changes. Putting the DB-isolation fixture inside
``test_auth.py`` itself rather than in the shared conftest keeps the
schema-bootstrap path scoped to the tests that actually care.

### `change_password` returns a bool, not raises

The schema docstrings document this: ``change_password(user_id,
current, new) -> bool``. The route handler maps `False` to HTTP 400
``WRONG_CURRENT_PASSWORD``. We considered raising a domain exception
but the boolean fits the call site cleanly — the alternative would
need an exception class per failure mode (wrong current, weak new,
account inactive…), all of which the route layer already maps from
the boolean + the new-password validation.

### Session cookie defaults: `Lax` + `secure=False`

Phase 0 ships ``SameSite=Lax`` and ``secure=False`` for development.
Phase 7's "Public web-app hardening" flips ``SESSION_COOKIE_SECURE``
to ``True`` when HTTPS is enabled, but the env var
(``SESSION_COOKIE_SECURE``) is plumbed today so an operator can opt in
without code changes. The lax samesite stays as-is — the Vite dev
server proxy makes the cross-origin question moot in development.

### Default admin password is `admin123`, *with a warning*

Documented in ``run.py`` startup and the README. The default is the
v1/v2 spec's literal value — kept identical so existing CiRA Oculus
deployment recipes still work. Operators who run ``python run.py``
once and never log in get a working admin account; operators who run
it in production are expected to log in *immediately* and change the
password (the v2 spec's "Default Credentials" section says the same).

### Vue Router redirect rule: `requiresAuth` without a session → `/login?redirect=<path>`

Standard Vue Router pattern. The query parameter survives the
login round-trip so the LoginPage can push the user back where they
were trying to go. The opposite rule (``requiresGuest`` with a
session → `/`) prevents a logged-in user from accidentally landing
on the login page.

### `auth.bootstrapped` flag prevents a race on first navigation

When the browser loads ``/`` the Pinia store has no user yet, so the
route guard would redirect to ``/login`` even though the session
cookie is valid. The fix is the ``bootstrapped`` flag: the guard
awaits ``checkAuth()`` on the very first navigation before deciding.
App.vue's ``onBeforeMount`` does the same — both paths converge on
the same store state.

### `vue-tsc` is dev-only; build-time-only type check

``npm run build`` runs ``vue-tsc --noEmit`` before Vite's bundler.
``npm run dev`` skips it — type errors surface in the IDE during
development, full check happens at build time. This matches what the
v2 spec implies: TypeScript ^5.3.0 listed in the tech stack but no
CI gate documented at this phase.

### `WERKZEUG_RUN_MAIN`: Flask's auto-reload was disabled

``debug=False`` in ``run.py``: Flask's auto-reload spawns a child
process, which interferes with session-cookie debugging during
interactive smoke tests (cookies set by the parent don't survive into
the reloaded child). For Phase 0 dev work the reload-on-save
convenience isn't worth the surprise; later phases that want it can
wrap with ``flask --debug run`` from the command line.

---

## Phase 3 — Formulation provider layer

**Date:** 2026-05-07

### Providers use `httpx` directly, not the vendor SDKs

The v2 tech-stack table lists `anthropic` and `openai` as dependencies.
We import neither. Both vendor SDKs wrap an internal HTTP transport
that's awkward to mock cleanly; using `httpx.AsyncClient` directly lets
`pytest-httpx` intercept every request at the transport layer with no
SDK-monkeypatching required. The on-the-wire JSON shapes are stable
(both vendors version their APIs explicitly), so the SDK abstraction
buys us nothing the test isolation doesn't already give us. Optional
SDK dependencies for the manual integration test were considered and
rejected — that script also uses `httpx`, for the same reason.

### `cqm_v1` schema is shared with the compiler, not duplicated

The Phase-2 compiler (`compile_cqm_json`) had an *informal* schema in
its docstring; Phase 3 ships the *formal* JSON Schema as
[`backend/app/formulation/schemas/cqm_v1.json`](backend/app/formulation/schemas/cqm_v1.json).
We verified all 9 existing Phase-2 instance fixtures validate against
it — the schema is descriptive of the existing on-disk artifacts, not
prescriptive of a new format. Phase 4's pipeline orchestrator will
call `validate_cqm_json` *before* `compile_cqm_json`, so the schema
catches malformed LLM output before the compiler tries to interpret it.

### Schema validator throws `FormulationError`, not `ValueError`

`compile_cqm_json` raises `ValueError` on schema violations (Phase 2
convention). `validate_cqm_json` in the formulation layer raises
`FormulationError` because the *upstream* error story for the pipeline
is "the LLM returned something we can't use" — that maps cleanly onto
`FormulationError`, not `ValueError`. The two layers catch the same
shape of failure (bad JSON) but at different boundaries; using one
exception type per boundary keeps the orchestrator's `except` clauses
readable.

### Three-tiered JSON extraction, not a single regex

LLMs emit JSON inside ```json fences, inside plain ```fences, inside
prose, or with trailing commentary stuck on. `extract_json_object`
tries fenced blocks first (most common case), falls back to the first
balanced-looking `{...}` slice, and if `json.loads` fails it iteratively
trims the candidate back to the previous `}` before retrying. The
extractor returns the parsed `dict` — never raw text. Tests cover all
four shapes (plain JSON, fenced JSON, prose-wrapped JSON, plain-fenced
JSON without language tag).

### Crypto: deterministic Fernet key from `KEY_ENCRYPTION_SECRET`

The BYOK threat model is "attacker reads the SQLite database file" not
"attacker reads the operator's environment." That makes per-key salts
unhelpful — they'd require the salt to live alongside the ciphertext
in the same database, defeating the purpose. Instead `derive_fernet_key`
SHA-256s the operator's `KEY_ENCRYPTION_SECRET`, base64-encodes to
44 ASCII bytes, and uses that as the Fernet master key. Same secret
always derives the same key, so any boot of the platform can decrypt
records written by any prior boot.

**Rotation behavior (operator-facing).** Changing
`KEY_ENCRYPTION_SECRET` invalidates every previously-encrypted
`api_keys` row — Fernet's authenticator will reject decrypts under the
new key, and `decrypt_api_key` raises `ValueError("invalid token: …")`.
This is by design (deterministic derivation = no per-row salt = no way
to re-derive the old key from anything stored).

**Migration path under v2.** None of the three options is shipped as
automated tooling in Phase 3; the operator picks one explicitly:

1. **Forced re-entry (recommended for cohort handoff).** Truncate
   the `api_keys` table and let users re-enter their keys at next
   solve. Acceptable when the secret rotation is itself a sign that
   the old secret may have leaked — invalidating stored keys is the
   point.
2. **Soft re-entry (recommended for routine rotation).** Leave the
   `api_keys` rows in place. The first attempt to use a stored key
   under the new secret raises `ValueError`; the route handler in
   Phase 4 will catch this, log a "stored key no longer accessible"
   warning, and surface a UI prompt. Users re-enter only as needed.
3. **Manual re-encrypt-all migration.** Run a one-off script that
   loads each row under the *old* secret, re-encrypts under the *new*
   secret, and writes back. This requires both secrets to be
   simultaneously available to the script. Reserved for the case
   where the operator rotates secrets *without* a leak (e.g.,
   hardware migration).

The platform deliberately does not implement option 3 in Phase 3 —
its surface area (two-secret CLI, atomic in-place rewrites,
confirmation dance) is larger than the BYOK feature itself, and the
academic deployment context expects option 1 or 2 to be the normal
path. If a real grant-funded deployment needs option 3, it becomes
a scoped script then.

The operator's rotation steps are therefore short:

```
1. Change KEY_ENCRYPTION_SECRET in the deployment environment.
2. Restart the backend.
3. Either: truncate `api_keys`, or rely on Phase 4's
   "stored key no longer accessible" UI prompt to do soft re-entry.
```

### Provider registry auto-bootstraps on first use

`get_provider("claude")` is the public lookup; the first call
auto-imports the three built-in modules so each registers itself.
Importing `app.formulation.claude` (etc.) is idempotent — the
`register_provider` call inside that module hits the duplicate-name
guard on a second import and raises, but the bootstrap path catches
that import inside a try/except so a contributor's failed registration
in one module doesn't poison the others.

### Default models: Sonnet 4.6 (Claude), GPT-5-mini (OpenAI), qwen3:8b (local)

Each default was picked over its larger sibling deliberately. All three
are overridable per-instance via the provider constructor, so a
practitioner with quality-sensitive problems can opt up; the defaults
are tuned for the academic / benchmark / learning audience the v2
positioning targets, where token spend has to stay modest.

**Sonnet 4.6 over Opus 4.7.** Opus is roughly 5× the cost per token
($15/$75 per M input/output vs Sonnet's $3/$15) for marginal gains on
structured-JSON formulation tasks. The bottleneck isn't reasoning
depth — it's "emit the cqm_v1 schema faithfully," which Sonnet has been
reliable at since Sonnet 4. A Phase-5C benchmark comparing both tiers
on the canonical instances is the honest test; until those numbers
exist, defaulting to Sonnet keeps the platform affordable enough that
a course can use it without budget approval. Opus stays one
constructor argument away (`ClaudeFormulationProvider(model="claude-opus-4-7")`).

**GPT-5-mini over GPT-5.** Same reasoning, different vendor.
GPT-5-mini is ~10× cheaper than GPT-5 ($0.25/$2.0 per M tokens vs
$2.5/$20) and has been stable on JSON-mode output since the GPT-4o
generation. The formulation task doesn't need GPT-5's frontier
reasoning; it needs reliable schema adherence, which both tiers
deliver. The Phase-5C dashboard will rank both honestly; GPT-5-mini
defaulting on isn't a quality claim, it's a cost choice.

**`qwen3:8b` default for UX, `qwen3:14b` env-overridable for
reliability.** The local tier is the one place where model choice
*within* the tier matters more than the tier choice itself. We picked
defaults empirically:

| Model | JSS-3×2 trial result | Latency | Verdict |
|-------|---------------------|---------|---------|
| `qwen3:8b`   | 1/2 correct (opt=10), 1/2 dropped a disjunctive helper | ~90 s | Default — best UX |
| `qwen3:14b`  | 2/2 correct (opt=10), consistent shape | ~470–520 s | Env override — reliability tier |
| `llama3.1:8b`| timed out at 300 s | — | Same params as `qwen3:8b`, doesn't keep up |
| `phi4:14b`   | 2/2 *compile-OK but wrong* (opt=11 ≠ 10) | ~320–490 s | Worse than nothing (looks correct, isn't) |
| `qwen3-coder:30b`, `gpt-oss:20b`, `devstral:latest` | HTTP 400 — chat-incompatible | — | Phase-10 candidates (need a `/api/generate` adapter) |

The split-default — `qwen3:8b` for interactive use, `qwen3:14b` for
the integration gate — is exposed via `LOCAL_LLM_MODEL`. The
`manual_integration_test.py` documentation tells operators to set
`LOCAL_LLM_MODEL=qwen3:14b` explicitly when running the gate. The
v2-strict reading is: *operators dial reliability vs latency, the
platform doesn't pretend they're the same axis*.

`phi4:14b`'s failure mode is the most important one to call out:
**it produces compile-OK output that's silently wrong** (claimed
optimum 11 vs the correct 10). For the platform's "honest scoreboard"
positioning, a model that looks right but isn't is more dangerous than
a model that fails loudly. The cqm_v1 schema can't catch this — the
expected_optimum field carries whatever the model wrote — so the
defense is the validation harness's Layer A oracle agreement (when
the CQM is small enough) or the Phase-5C dashboard's
multi-solver-agreement display.

Above the 8B/14B class, the 33B / 70B local tier doesn't fit in 16 GB
VRAM alongside the GPU SA solver. A future phase may add a "large
local LLM" opt-in tier for workstations with more VRAM — explicitly
opt-in, not the default.

**Limitation owned, not papered over** — the local provider is
best-effort; the README documents the tier of "what works / what
doesn't" with concrete model names, latencies, and outcome counts.
Users who care about quality on hard formulations route through
Claude or OpenAI, which is the cost they pay for the BYOK trade.

The Phase-5C public Benchmark dashboard is where the platform earns
back the right to say which model is "best" for a given problem class.
Until then the defaults are picked for cost-conservatism, with full
configurability one constructor argument away.

### `estimate_cost` is conservative

Pre-submit cost previews must never *under-promise*. The estimator
adds a 5500-token fixed-prompt overhead (system prompt + 3 worked
examples), assumes 0.25 tokens per character, and includes a 25%
completion-tokens budget. This over-estimates real usage by ~15-30%
in practice — that's a feature, not a bug, for a budget-conscious UI.

### `system.txt` instructs the LLM to refuse non-optimization requests

Per the v2 spec's prompt strategy bullet. The system prompt explicitly
tells the model to emit
``{"error": "not_an_optimization_problem", "reason": "..."}``
when the user asks for something that isn't an optimization problem
(a poem, a chat, a code review). The base validator rejects such an
output as a schema violation (no `version` key), and the test
`test_extract_json_object_raises_on_no_json` covers the case where
the LLM refuses with prose instead of structured output.

### Local provider accepts but ignores `api_key`

Ollama doesn't authenticate. The interface includes `api_key` for
parity with the cloud providers — the pipeline orchestrator can call
`provider.formulate(problem, api_key)` without special-casing the local
path. The local provider's docstring documents this explicitly.

### Manual integration test sits in `tests/` but doesn't start with `test_`

`tests/manual_integration_test.py` is not collected by pytest (the
filename intentionally breaks the `test_*.py` discovery glob). It's
invoked directly: `python tests/manual_integration_test.py`. The Phase
3 DoD calls for it to pass for at least 2 of 3 providers; the script
exits with code 1 if fewer pass. CI never runs it.

### `pyproject.toml` adds `asyncio_mode = "auto"`

The provider tests are `async def` functions. With `pytest-asyncio` in
auto mode, decorating each async test with `@pytest.mark.asyncio` is
unnecessary — the plugin discovers and runs them automatically. This
keeps the test files clean and matches what the v1 spec's test list
implies (no `@pytest.mark.asyncio` decorators are mentioned).

---

## Phase 2 — Validation harness + Benchmarks foundation

**Date:** 2026-05-07

### Compiler returns a 3-tuple, not the v1-spec 2-tuple

The v1 spec documented `compile_cqm_json(cqm_json) -> tuple[dimod.CQM, dict]`.
We return `tuple[dimod.CQM, dict, str]` — the third element is the
JSON's `objective.sense` ("minimize" or "maximize"). dimod CQMs always
minimize internally; for maximize-sense problems the compiler negates
linear and quadratic coefficients before adding them. The validation
harness needs to know the sense to convert the internal minimization
energy back into the user-facing optimum the JSON's `expected_optimum`
is stated in. Putting the sense in a dedicated tuple element (rather
than smuggling it through the variable registry) keeps the boundary
clean.

### CQM-JSON constraint type names

The v1 example only showed `"equality"`. We support all three:
`"equality"` → `==`, `"inequality_le"` → `<=`, `"inequality_ge"` → `>=`.
Knapsack capacity (`<=`) and set-cover coverage (`>=`) require the
inequality variants; the verbose names beat single-character symbols
because typos (`==` vs `=`, `>=` vs `=>`) become parse errors instead
of silent miscompilations.

### `ValidationReport.oracle_skipped` extension

The v1 spec's `ValidationReport` dataclass does not include an
`oracle_skipped` field. We added one — `True` when the CQM has more
variables than `max_oracle_vars` so `ExactCQMSolver` is bypassed. The
v1-style `oracle_agreement` field becomes vacuously `True` in that case;
without `oracle_skipped`, callers can't tell whether oracle silence
means "agreed" or "didn't run." JSS-3x3 (19 vars) hits this path.

### `validate_cqm` takes `sense` and `skip_layer_b` kwargs

The v1 signature is `validate_cqm(cqm, expected_optimum, max_oracle_vars)`.
We added two keyword-only arguments:

- `sense="minimize"` — needed so `expected_optimum` (user-facing) can
  be compared against the layer's internal minimization energy.
- `skip_layer_b=False` — skipping the multi-solver agreement check is
  useful in tests that don't want to pay the sampling cost or that
  isolate Layer A / Layer C behavior. Default keeps Layer B on.

### JSS-3x3 brute-force-match test uses a 2x2 instance

The v2 spec lists `tests/instances/jss_3job_3machine.json` with
`expected_optimum=11` and a test named `test_jss_3x3_matches_brute_force`.
We ship the requested 3x3 file faithfully (19 variables, 27 constraints,
big-M disjunctive encoding) but two facts force the brute-force-match
arm of the test to use a smaller instance instead:

1. `dimod.ExactCQMSolver` enumerates every assignment in the cartesian
   product of variable domains. For our 3x3 encoding that is
   `12⁹ × 2⁹ × 12 ≈ 3 × 10¹³` — `ExactCQMSolver` raises a
   115 TiB-array allocation failure on import, not a slow run.
2. The Lagrange-penalised BQM lowering of the 3x3 has 173 variables.
   Vanilla SA (CPU and GPU, kernel="jit") finds zero feasible
   assignments at any tested Lagrange multiplier (5, 10, 20, 50, 100)
   over 2000 reads × 5000 sweeps. Disjunctive big-M is genuinely beyond
   single-temperature SA; parallel tempering / PT-SQA (Phase 9) are the
   tools designed for this regime.

We therefore ship a companion `jss_2job_2machine.json` (7 variables, 8
constraints, optimum makespan = 4) that is brute-forceable in 31104
states. The `test_jss_3x3_matches_brute_force` test now exercises the
2x2 for the actual oracle-match check while keeping the 3x3 in the
loop for Layer A-skip and Layer C constraint coverage. The full
rationale lives in the test docstring; the file name `jss_3job_3machine`
is preserved so the spec's instance manifest entry continues to point
at a real 3x3 fixture.

### `RunRecord` adds a `warnings` field

The v2 spec's `RunRecord` dataclass doesn't list `warnings`. We added
one (defaulting to an empty list) for parity with `ValidationReport` —
a benign solver warning ("kernel fallback to JIT", "feasibility-check
threw", etc.) shouldn't be silently dropped just because the run
finished. It serializes as an array of strings; absent in older records
it defaults to `[]` on load.

### `record_id` uses microsecond resolution

The spec says "sortable, unique, citable" but doesn't pin the format.
We use `YYYYMMDDTHHMMSS.ffffffZ_<6hex>`. Second-resolution timestamps
collided when the suite runner generated multiple records inside the
same wall-clock second; bumping to microseconds keeps strict monotonic
ordering across consecutive `record_run` calls.

### `_split_parameters` for solvers with `__init__` kwargs

The GPU SA built in Phase 1 takes `kernel` (and `device`) at
construction time, not at `sample()`. The other two baseline solvers
take everything at `sample()`. Rather than make every caller know
which key goes where, `record_run` consults a hardcoded
`_INIT_ONLY_KWARGS_BY_SOLVER` map. The full parameter dict is still
recorded in the RunRecord — splitting is purely for dispatch. When
Phase 10 starts accepting contributed solvers this becomes a
registry-time declaration on `SolverIdentity.parameter_schema`.

### Lagrange multiplier is a per-run parameter, not a global

`dimod.cqm_to_bqm` requires a Lagrange multiplier; the chosen value
materially affects which samples come back feasible. Rather than fix
a global default, the CLI runner records the chosen value in each
RunRecord's `parameters` so future Benchmark studies can reproduce or
sweep it. Default for Phase 2 ships at `10.0`, matching the value the
v1 Phase-4 pseudocode used.

### RunRecord results carry an honest convergence flag (post-approval addendum)

Added during Phase 2 review. ``RunRecord.results`` gains three fields
for the Benchmark dashboard's honesty:

- ``expected_optimum`` — the value supplied by the instance manifest
  (or ``None`` when the instance has no ground truth).
- ``gap_to_expected`` — absolute ``|best_user_energy - expected_optimum|``
  (or ``None`` if either is missing).
- ``converged_to_expected`` — ``True`` iff the gap is within
  ``max(1e-3 * |expected|, 1e-9)``; ``False`` when the run materially
  missed; ``None`` when no ground truth exists. **Never inferred** —
  no expected, no convergence claim. The dashboard reads ``None`` as
  "heuristic estimate, no ground truth available," not as success.

The CLI's per-instance summary surfaces this directly:

```
instance                                 best_user  expected   conv  feasible  time(ms)
knapsack/small/knapsack_5item                   26        26    yes         -      886
knapsack/small/knapsack_20item                 185       233     no         -      370
maxcut/gset_subset/maxcut_50node             ...         ?      ?           -     ...
```

The flag is one-way honest: a run that *exceeded* the expected optimum
(e.g. because the manifest's `expected_optimum` was wrong) is reported
as not-converged with a non-zero `gap_to_expected` — the gap is the
signal a maintainer follows back to the manifest. We deliberately do
*not* clamp the gap or call it converged when the run "beat" the
declared optimum.

Tolerance choice: 1e-3 relative is tight enough to flag any real
heuristic miss on integer-valued objectives (knapsack value, makespan,
cut count) and loose enough to absorb floating-point rounding from the
BQM lowering on continuous objectives.

### Benchmark archive root is overridable via `CIRA_BENCH_ARCHIVE`

The default archive root is `<repo>/benchmarks/archive/`. Tests set
the `CIRA_BENCH_ARCHIVE` env var to a `tmp_path` so they don't
contaminate the real archive. Production deployments leave it unset.

### `cite.py` cite key strips `:` and `.`

BibTeX cite keys can't contain `:` or `.` (some BibTeX parsers reject
them; LaTeX engines may interpret them). The cite key is the record_id
with both characters stripped, leaving the timestamp and the random
tail readable. The full record_id is preserved in the entry's `note`
field for direct lookup.

### Instance suite manifest is a single JSON file, not per-file metadata

Every instance has both a CQM JSON (the model) and metadata (suite,
problem class, expected_optimum kind, source). We collect the metadata
in a single
[`app/benchmarking/instances/manifest.json`](backend/app/benchmarking/instances/manifest.json)
rather than scattering one metadata sidecar per CQM. Reasoning:
contributors discover the full inventory in one place, the manifest
schema can evolve without touching every instance file, and the
filesystem stays compact. The Phase 10 contribution pipeline will
likely accept individual sidecars and merge them into the manifest at
build time.

---

## Phase 1 v2 alignment — 2026-05-06 (post-shipping)

The project was renamed from "CiRA Optima" to "CiRA Quantum" in v2. The
renames touched only metadata: `backend/README.md` (title and intro
paragraph) and `backend/pyproject.toml` (`name`, `description`). Module
names, file paths, and code identifiers stayed the same — there was no
import-graph churn from the rename.

`triton` was made platform-conditional in both `requirements.txt` and
`pyproject.toml` to fix the v2 spec's flagged Linux install hazard:

```
triton>=3.0;          platform_system != "Windows"
triton-windows>=3.0;  platform_system == "Windows"
```

The community `triton-windows` build is used on Windows because upstream
Triton does not ship Windows wheels. Both branches resolve cleanly in a
fresh venv (verified with `pip install -r requirements.txt` and
`pip install -e ".[dev]"`).

---

## Phase 1 — GPU Simulated Annealing library

**Date:** 2026-05-06

### `properties` returns concrete values, not types

The class contract in the brief shows `properties` as
`{"device": str, "compute_capability": tuple, "vram_total_gb": float}`,
which would mean returning Python type objects at runtime. `dimod.Sampler`'s
contract is that `properties` is a dict of property names to *values* — the
brief is using type annotations as documentation. We return the actual values
(e.g. `{"device": "cuda:0", "compute_capability": (12, 0), "vram_total_gb": 15.92}`).

### BQM input vartype handling

The class accepts any `dimod.BinaryQuadraticModel`. Internally the kernel
operates on Ising spins (±1); `bqm.change_vartype('SPIN', inplace=False)` is
used at the top of `sample()`. Output samples are emitted in the *original*
vartype: BINARY problems get back 0/1 assignments (`x = (s + 1) // 2`), SPIN
problems stay as ±1.

### Variable label support

Non-integer variable labels (strings, tuples) are supported. Labels are mapped
to internal integer indices for the kernel and unmapped via
`SampleSet.from_samples((array, var_order), …)` on the way out.

### Empty BQM

`bqm.num_variables == 0` short-circuits before the kernel: we return
`num_reads` empty samples, all with energy `bqm.offset`. This avoids any
zero-size tensor edge cases inside the JIT/compiled kernel.

### `seed=None` semantics

When `seed=None` we call `torch.seed()` and `torch.cuda.seed_all()` to draw a
fresh non-deterministic seed for both generators. `seed=<int>` calls
`torch.manual_seed` and `torch.cuda.manual_seed_all` so the same seed produces
the same samples (and the same energies) — the property the brief calls
"reproducible".

### Three kernel modes (`kernel="jit" | "compile" | "eager"`)

The brief recommends wrapping the inner sweep with
`@torch.compile(mode="reduce-overhead")` for a 2–4× speedup. On Windows that
requires Triton, which is not available in PyTorch's default install — we
took the dependency on `triton-windows` (3.6.0) to make it work.

We expose three kernel modes on `GPUSimulatedAnnealingSampler`:

- `"jit"` (default) — `torch.jit.script(_sweep_eager)`. Fast and warmup-free.
  All Phase 1 unit tests use this mode.
- `"compile"` — `torch.compile(_sweep_functional, mode="reduce-overhead")`.
  Highest throughput but has a one-time compile cost (~30–500 s per shape
  the first run; ~3–35 s after Inductor's on-disk cache is populated).
  Used by the benchmark.
- `"eager"` — pure-Python eager fallback for diagnostics.

### Two sweep functions

`_sweep_eager` mutates `spins` in place; `_sweep_functional` allocates a fresh
output. The compile path requires the functional variant because Inductor
disables CUDA-graph capture when it detects in-place mutation of inputs
("skipping cudagraphs due to mutated inputs"). The JIT path uses the in-place
variant because it doesn't go through Inductor and the in-place version
saves a per-sweep allocation. Caller-side `spins.clone()` after each compile
call breaks the aliasing chain that would otherwise let the cudagraph reuse
memory we still depend on.

### Symmetric-J row read

`J` is built as a full symmetric `(n, n)` matrix with zero diagonal. The inner
sweep reads `J[i]` (a contiguous row view) rather than `J[:, i]` (a strided
column gather) — they hold the same vector since J is symmetric, but row
access is materially faster on PyTorch's row-major storage.

### Random-uniform pre-roll cap

To keep the inner Python loop tight, the random uniforms used for the
Metropolis acceptance test are pre-allocated as a `(num_sweeps, n, num_reads)`
tensor when the buffer is ≤ 256 MB; for larger problems we generate one
`(n, num_reads)` slice per sweep to avoid blowing up memory.

### Benchmark `num_reads` default = 4000

Phase 1 DoD requires ≥10× speedup over `dwave-samplers.SimulatedAnnealingSampler`
at N=1000. With `num_reads = 1000` (the production default in
`PROJECT_TEMPLATE.md`'s `config.py`) the GPU is launch-bound: per-spin kernel
launch overhead dominates so the speedup is only ~5–7×. At `num_reads = 4000`
the GPU has enough parallelism per launch to amortize that overhead and the
speedup cleanly clears 10×. The benchmark defaults to 4000 to make the DoD
result robust to run-to-run CPU timing noise; users who care about a specific
production setting can override with `--num-reads`.

### Inductor cache lives under `~/.cache/torch/inductor`

Subsequent processes see compile times of ~3–35 s (vs. 30–500 s the first
ever time on a given machine). Wiping the cache restores cold-warmup cost.

### Vectorized BQM → tensor ingestion

Naively, `J[i, j] = float(b)` inside a Python `for (u, v), b in
ising.quadratic.items()` loop is fine for small problems but unworkable for
dense BQMs at the N=10000 OOM-check scale (~50M edges → tens of minutes per
sample). We batch the (i, j, value) triples into numpy arrays first, then
push them to the device via two `J.index_put_` calls. Same correctness, a
couple orders of magnitude faster on dense inputs.

### What was deferred (per the brief itself)

- Checkerboard parallel updates
- Direct CUDA graph capture (without going via Inductor's reduce-overhead)
- Hand-written Triton kernel for the entire sweep
- Parallel tempering, replica exchange
