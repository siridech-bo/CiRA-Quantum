# CiRA Quantum — Living Roadmap

> **Status as of 2026-05-12.** This document tracks the *current* phase plan
> and the next-step sequence. It diverges from `PROJECT_TEMPLATE v2.md`
> wherever the v2 spec needed amendment after building turned up new
> information — see `DECISIONS.md` for the rationale behind each divergence.
>
> `PROJECT_TEMPLATE v2.md` is the original contract. This file is the
> working plan. When the two conflict, this file wins for execution; v2
> wins for audit.

## Phase status

| Phase | Name | Status |
|---|---|---|
| 0 | Project skeleton + auth | ✅ shipped |
| 1 | GPU SA library | ✅ shipped |
| 2 | Validation harness + Benchmarks foundation | ✅ shipped |
| 3 | Formulation provider layer (Claude / OpenAI / local LLM + BYOK) | ✅ shipped |
| 4 | Solve API endpoint + orchestrator + SSE | ✅ shipped |
| 5 | Frontend Solve UI | ✅ shipped |
| 5B | Problem-template / Modules library | ✅ shipped |
| 5C | Public Benchmark dashboard | ✅ shipped |
| 8 | Classical solver tiers (OR-Tools CP-SAT + HiGHS) | ✅ shipped |
| 9A | Quantum tier — local simulator (pyqpanda QAOA) | ✅ shipped |
| 9B | Quantum tier — Origin Quantum cloud BYOK (full_amplitude + WK_C180 Wukong QPU) | ✅ shipped |
| 9C | Quantum-inspired classical tiers (Parallel Tempering + Simulated Bifurcation) | ✅ shipped — see [BENCHMARK_REPORT_002.md](./BENCHMARK_REPORT_002.md) |
| 9B+ | Non-blocking cloud-job capture flow + empirical Wukong qubit wall (n≤7) | ✅ shipped |
| **6** | **Async job queue + GPU contention** | ⏳ **next** |
| 9C | Quantum-inspired classical tiers (PT, PQQA, [bifurcation]) | ⏳ |
| 6 | Async job queue + GPU contention (Redis + RQ) | ⏳ deferred-until-9A-9B-9C |
| 7 | Public web app hardening | ⏳ deferred-until-9A-9B-9C |
| 10+ | Appendix items (cohort mgmt, contribution pipeline, federation) | 🔒 deferred-until-trigger |

**~~9D~~ SQBM+ dropped.** Fujitsu's proprietary brand on top of openly
published simulated-bifurcation algorithms; partnership cost
disproportionate to academic value. The underlying bifurcation math
remains available to us as a generic `simulated_bifurcation` solver
under Phase 9C if and when we want it.

## Sequence rationale

The v2 spec orders these phases sequentially. Building turned up two
amendments worth pinning in writing:

1. **Phase 9 is one phase in the v2 spec.** We split it into 9A
   (true quantum, local simulator — no BYOK), 9B (true quantum, real
   QPU — adds Origin Quantum as a BYOK provider), and 9C (quantum-
   inspired classical heuristics). The split is along the *auth and
   hardware* boundary, so 9A is offline / deterministic / lightweight
   while 9B brings in remote credentials and shot noise.

2. **Phase 6 and 7 are pre-deployment work, not pre-development.** They
   prevent multi-user contention and abuse; both pressures are absent
   on a localhost single-developer build. Doing them at the end means
   the solver tier landscape (9A + 9B + 9C) ships first, the dashboard
   tells the complete story, and 6+7 are then validated against a
   complete backend rather than a half-built one. Phase 0 followed
   the same "settle the science first, ship the shell when it
   matters" pattern.

The cost is delaying public exposure by ~3 weeks. The benefit is the
dashboard's headline view — *classical CP-SAT vs HiGHS vs quantum-
inspired vs quantum-simulator vs real Wukong QPU on the same instance*
— lands in roughly 2 weeks rather than 5.

---

# Phase 9A — Quantum tier (local simulator)

> **v2 status:** New sub-phase. Part of the v2 spec's monolithic Phase 9
> was carved out into this lighter, BYOK-free first step.

## Goal

Add a fourth solver-tier *category* to the Benchmark dashboard: **true
quantum optimization** (as opposed to classical or quantum-inspired
classical), running on local statevector simulation via OriginQC's
pyqpanda3 stack. The QAOA algorithm is the differentiated payoff: it
is the actual variational quantum algorithm that runs on real
superconducting qubits in Phase 9B; here we run the same code path
against a CPU statevector simulator so the dashboard can show the
algorithm's behavior on small instances without a cloud round-trip.

## Why this phase exists

Without 9A, the dashboard plots "QUBO solver vs QUBO solver vs
classical SOTA" — useful, but not what the v2 spec promises. The v2
positioning ("the honest scoreboard for quantum + classical
optimization") requires at least one actual-quantum algorithm in the
solver registry. 9A is the smallest unit of work that delivers that
positioning.

## Scope — In

- **`backend/app/optimization/qaoa_sampler.py`** — CQM-native
  `dimod.Sampler` adapter wrapping `pyqpanda_alg.QAOA.qaoa.QAOA`. Reads
  CQM directly (via the `_CQM_NATIVE = True` marker introduced in
  Phase 8), encodes the objective as a sympy expression or
  `pq.PauliOperator`, runs the variational loop, samples the resulting
  distribution into a `dimod.SampleSet`.
- **Backend selector** — `backend="cpu_sim"` (default). The same
  adapter class will get a `backend="originqc_cloud"` arm in Phase 9B.
- **Qubit budget enforcement** — adapter raises a clear `ValueError`
  when the CQM exceeds the simulator's capacity (default cap: 20
  variables). The dashboard surfaces this as an empty cell on the
  suite grid — same "honest gap" pattern HiGHS leaves on
  `maxcut/gset_subset`.
- **Conditional registration** — pyqpanda is an *optional* dependency
  (`pip install ".[quantum]"`). `bootstrap_default_solvers()` attempts
  the import and registers `qaoa_sim` only on success. Same conditional
  pattern as `gpu_sa`'s CUDA check.
- **Tests** — correctness on tiny knapsack (2-3 vars), agreement
  with `ExactCQMSolver` for instances small enough to enumerate,
  qubit-budget rejection test.
- **Archive population** — run QAOA across the Phase-2 small suites
  that fit under 20 variables (`knapsack_5item`, `setcover_4item`,
  `graphcoloring_4node`, `maxcut_6node`, `jss_2job_2machine`). Skip
  the rest.

## Scope — Out

- **Real quantum hardware** (deferred to Phase 9B)
- **Grover Adaptive Search** (`QUBO_GAS_origin`) — useful but
  rarely-best; defer to a follow-up rather than ship in 9A.
- **GPU statevector simulation** (`GPUQVM`) — would extend the qubit
  ceiling to ~28, but conflicts with our existing CUDA usage for
  `gpu_sa` and adds config surface. Add as a follow-up if the qubit
  ceiling becomes the bottleneck.
- **Custom mixer circuits** — default X-mixer only in 9A; constrained-
  problem mixers (XY, ring) deferred.
- **Dashboard UI changes** — the existing `SolverComparisonTable.vue`
  already renders empty cells for tier/instance combinations that
  don't apply. No frontend changes needed in 9A.

## Files to create / modify

```
backend/
├── app/optimization/qaoa_sampler.py          NEW
├── app/benchmarking/registry.py              MODIFIED (conditional register)
├── app/benchmarking/records.py               MODIFIED (param split for qaoa)
├── app/benchmarking/run_suite.py             MODIFIED (qaoa-aware kwarg routing)
├── requirements.txt                          MODIFIED (pyqpanda extras)
├── pyproject.toml                            MODIFIED (define quantum extras)
└── tests/
    └── test_qaoa_sampler.py                  NEW

DECISIONS.md                                  APPENDED
ROADMAP.md                                    UPDATED (move 9A to ✅)
README.md                                     UPDATED (phase table)
backend/README.md                             UPDATED (phase table + registered solvers)
```

## Definition of Done

- [ ] `pip install ".[quantum]"` installs pyqpanda cleanly on Windows.
- [ ] `register_solver` lists `qaoa_sim` after `bootstrap_default_solvers()`
      when pyqpanda is importable; gracefully *omits* it when not.
- [ ] `QAOASampler` passes a correctness test on tiny knapsack
      (3 items, capacity 5; optimum 7).
- [ ] Agreement check: QAOA's top sample on a 4-variable problem
      matches `ExactCQMSolver`'s optimal energy within 1e-3 relative
      tolerance (the same tolerance Phase 5B's match badge uses).
- [ ] Qubit-budget rejection: a 30-variable CQM raises `ValueError`
      with a message naming the qubit ceiling and pointing the reader
      at Phase 9B as the path forward.
- [ ] At least 3 QAOA records land in `benchmarks/archive/` and render
      on `/benchmarks/suites/knapsack/small` correctly.
- [ ] `pytest tests/ -q` passes (target 180 → ≥185 with new tests).
- [ ] `ruff check .` clean; `vite build` clean.
- [ ] `DECISIONS.md` carries one new entry covering: optional-dep
      strategy, qubit-budget design, QAOA non-determinism vs the
      Phase-2 replay model, why 9A ships before 6+7.

## Risks and mitigations

- **pyqpanda Windows install.** OriginQC's Windows wheel support has
  historically been the rougher edge of the stack. **Mitigation:** the
  Phase-9A first task is a clean pip install on a fresh-ish Python
  3.12 venv; if the wheel doesn't exist, fall back to building from
  source (~20 min on this hardware) or pin to an earlier compatible
  version. If neither works on Windows, document and defer 9A until
  upstream support catches up — we don't have to invent a workaround.
- **QAOA non-determinism vs replay model.** Phase 2's `replay_record`
  expects energy reproducibility within 1e-9 under a fixed seed. SPSA
  optimizer + shot-noise sampling won't meet that. **Mitigation:**
  record the optimizer trace and final loss alongside the energy;
  declare quantum records "reproducible-in-distribution, not
  reproducible-in-energy"; relax the replay tolerance for any solver
  carrying `_CQM_NATIVE` *and* a class-level `_STOCHASTIC = True`
  marker. The architecture absorbs this without changing the public
  schema.
- **Time per record.** QAOA at layer=3 on 10 qubits is ~10–60s. That's
  100–1000× slower than CP-SAT. **Mitigation:** archive on small
  instances only (≤ 10 vars by default for 9A's initial sweep). The
  Phase-5C dashboard's time-axis already handles a wide log range.

## Phase 9B preview (for reference)

The 9B sequel reuses the same `QAOASampler` class with
`backend="originqc_cloud"`, adds an `originqc` BYOK provider next to
`claude` / `openai` / `local`, and gates real-hardware submissions
behind a session-level submission cap + a `ENABLE_ORIGIN_REAL_HARDWARE`
feature flag (default off during dev). The simulator path remains
unchanged and BYOK-free.
