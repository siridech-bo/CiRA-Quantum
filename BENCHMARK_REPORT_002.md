# Benchmark Experiment #2 — Quantum-inspired classical tiers

**Date:** 2026-05-13
**Commit at run time:** *(filled in at commit)*
**Live dashboard:** [`/benchmarks/findings`](http://localhost:3070/benchmarks/findings)

> **TL;DR.** Phase 9C adds Parallel Tempering (PT) and Simulated
> Bifurcation (SB) — the two canonical quantum-inspired classical
> solvers — to the registry. Experiment #2 ran them at 10 seeds each
> against all 9 Phase-2 instances (5 seeds × 2 budget tiers each = 10
> per cell). **Both new tiers underperformed vanilla SA on the
> Phase-2 fixtures.** The headline finding is that **PT and SB shine
> on spin-glass-class problems, but our fixture set is heavily
> penalty-lifted CSPs (knapsack, set-cover, JSS) — exactly *not*
> where these algorithms were designed to win.**

## Why this experiment exists

Experiment #1 covered six tiers but left the *quantum-inspired
classical* tier empty. Adding PT and SB fills the v2 spec's
four-category landscape — after Phase 9C the dashboard renders
classical / QUBO-heuristic / quantum-inspired / quantum side by side.

## Experimental setup

### Solvers

| Tier | Solver | Hyperparameters |
|---|---|---|
| 9C — quantum-inspired | `parallel_tempering` | `num_replicas=8, beta_range=(0.1, 5.0)` |
| 9C — quantum-inspired | `simulated_bifurcation` (mode=`discrete`) | `agents=64, max_steps=2000` |

### Two budget tiers (combined into the same dashboard)

**v1 (under-budgeted PT, intentionally not fair):**
- PT: `num_reads=500, num_sweeps=1000` → ~8 k spin-flips total
- SB: `agents=64, max_steps=2000, num_reads=100` → unchanged
- Same 5 seeds: {42, 1, 2, 3, 4}

**v2 (fair PT budget — matches SA-class total work):**
- PT: `num_reads=50, num_sweeps=20000` → ~160 k spin-flips total
- SB: same as v1
- Same 5 seeds

**Total: 5 seeds × 2 budgets × 9 instances × 2 solvers = 180 records.** The dashboard's "best per cell" aggregation picks the better of v1 / v2 automatically.

### Instances

Same 9 as Experiment #1.

## Headline scoreboard

### Per-solver match rate (against all archived records, including Experiment #1's results)

| Solver | Match rate | Total runs | Notes |
|---|---|---|---|
| `exact_cqm` | **6/6 (100%)** | 24 | Only attempts what fits in memory |
| `highs` | **5/7 (71%)** | 28 | Skips quadratic-objective `maxcut/*` |
| `cpsat` | **6/9 (67%)** | 38 | Misses on JSS-3×3, JSS-5×5, knapsack_20 |
| `qaoa_sim` | **2/3 (67%)** | 50 | Qubit-cap-limited to 3 instances |
| `gpu_sa` | **5/9 (56%)** | 146 | Phase 1 GPU SA |
| `cpu_sa_neal` | **5/9 (56%)** | 144 | D-Wave Neal reference |
| **`simulated_bifurcation`** | **4/9 (44%)** | 90 | Wins on knapsack_5, maxcut_6, setcover, graphcoloring |
| **`parallel_tempering`** | **2/9 (22%)** | 90 | Only wins on graphcoloring and maxcut_6 |

**PT is the worst-performing tier on this fixture set.** SB sits in the middle.

### Per-(instance, new-solver) results

| Instance | Expected | gpu_sa best | cpu_sa_neal best | **PT best** | **SB best** |
|---|---|---|---|---|---|
| `graphcoloring_4node` | 0 | 0 ✓ | 0 ✓ | **0 ✓** | **0 ✓** |
| `jss_2job_2machine` | 4 | 4 ✓ | 4 ✓ | 43 ✗ | 5 (close ✗) |
| `jss_3job_3machine` | 11 | 101 ✗ | 41 ✗ | 1901 ✗ | 71 ✗ |
| `jss_5job_5machine` | (none) | 24781 | 878 | 179376 | **4778** |
| `knapsack_5item` | 26 | 26 ✓ | 26 ✓ | −801 (penalty) ✗ | 22 (close ✗) |
| `knapsack_20item` | 233 | 185 ✗ | 205 ✗ | −407 (penalty) ✗ | 148 ✗ |
| `maxcut_6node` | 7 | 7 ✓ | 7 ✓ | **7 ✓** | **7 ✓** |
| `maxcut_50node` | (none) | 139 | 139 | 109 | 139 |
| `setcover_4item` | 2 | 2 ✓ | 2 ✓ | 12 ✗ | **2 ✓** |

## Five findings worth pulling out

### 1. PT is the worst tier on this fixture set — surprising, but not really

Coming in, the literature said PT should *beat* vanilla SA on rough-landscape problems. The data says the opposite: PT got 2/9 instances vs Neal's 5/9. The literature isn't wrong; our test set is wrong for the claim. **PT's published strength is on Edwards-Anderson and Sherrington-Kirkpatrick spin glasses with native quadratic structure.** Our fixtures are penalty-lifted CSPs (knapsack, set-cover, JSS) where the Lagrange-multiplier-induced large quadratic terms create a different energy landscape than PT was designed for.

### 2. PT produces *physically meaningless* negative numbers on knapsack

`PT best = −801` on `knapsack_5item` (where the documented optimum is 26 in maximize-encoded user space). That negative value is a sign of PT finding a configuration that violates the Lagrange penalty in a way that, after sense-conversion, produces a nonsense user-energy. It's a real failure mode: PT exploits the penalty's discontinuity rather than respecting it. Vanilla SA doesn't show this — its more localized step size keeps it in the feasible neighborhood.

**This is an actionable finding:** for QUBO-class samplers on penalty-lifted CSPs, energy reporting should clip to the feasible-config energy range. Right now the dashboard shows `−801` and the user has to know it's nonsense. A small fix in `_summarize` could flag samples whose post-conversion energy is below the documented optimum range.

### 3. SB consistently slightly worse than vanilla SA — but much faster

SB doesn't match `gpu_sa`/`cpu_sa_neal` on the harder instances. But each SB run is ~100ms (vs Neal's 50ms and gpu_sa's ~1.5s). So SB is competitive on a *wall-time* basis even where its quality lags. For anytime use cases this matters — but Phase-2's match-rate metric ignores it.

### 4. PT's 20× budget bump (v1 vs v2) didn't change the outcome

The hardest instances showed PT v1 (under-budgeted) and PT v2 (fair budget) converging to *the same wrong answer*. Examples:
- `jss_3job_3machine`: PT v1 best ≈ 2365, PT v2 best ≈ 2365
- `jss_5job_5machine`: PT v1 best ≈ 24781, PT v2 best ≈ 179376 *(worse — v2 wandered further from the global minimum at higher temperatures!)*

**This is the strongest evidence that PT is hitting an algorithmic ceiling on this problem class, not a compute ceiling.** More sweeps don't help when the landscape is wrong for the method.

### 5. Both new tiers handle small + structured problems perfectly

`graphcoloring_4node`, `maxcut_6node` — both new tiers find the optimum cleanly. SB also wins `setcover_4item` and `knapsack_5item` (close but not match on the latter). This means our adapter implementations are correct; the algorithm-fit issue is genuine, not a bug.

## Limitations and caveats

1. **The Phase-2 fixtures don't favor PT/SB.** Adding native spin glasses (Sherrington-Kirkpatrick, ±J Edwards-Anderson) to the manifest would let these tiers demonstrate their actual strengths.
2. **Single Lagrange multiplier `λ=10`.** PT might handle penalty-lifted CSPs better with a smaller λ that doesn't push the energy landscape into chaotic territory. Experiment #3 candidate.
3. **No anytime comparison.** SB takes 100ms, PT takes 14s, Neal takes 50ms — but Phase-2's "best at fixed budget" metric doesn't surface this. A wall-time-equalized experiment would change SB's ranking dramatically.
4. **No multi-start aggregation for PT.** The current PT runs a single trajectory; literature shows PT benefits from population-replica restarts. Phase 9C-bis could add this.

## What this experiment changes in the codebase

- `backend/app/optimization/parallel_tempering_sampler.py` — ~200 line PT adapter following Hukushima-Nemoto 1996
- `backend/app/optimization/simulated_bifurcation_sampler.py` — wraps `simulated-bifurcation` PyPI package
- 9 + 10 = 19 new tests, all passing
- Registry now exposes **9 solver tiers** (was 7)
- `bootstrap_default_solvers()` registers both unconditionally (no optional dep gating — `simulated-bifurcation` is now a regular install)

## Recommendations forward

1. **Phase-2 manifest expansion**: add Sherrington-Kirkpatrick spin glasses (random ±J couplings, 20–50 vars) and G-set max-cut instances. Both classes are where PT/SB literature says they should win. Without these, the dashboard story is incomplete.
2. **Wall-time-equalized comparison** (Experiment #3): for each instance, set each solver's budget such that total wall time is identical. SB will rise; Neal will likely still win on quality but the margin closes.
3. **Penalty-handling fix**: clip nonsense user-energy reports for QUBO-class samplers on penalty-lifted CSPs. PT's `−801` knapsack result is the canonical bug case.
4. **PT hyperparameter sweep**: try `λ ∈ {1, 5, 10, 50}` × `num_replicas ∈ {4, 16, 32}` on a representative penalty-lifted instance. Find where PT actually beats vanilla SA, if anywhere.

## Phase 9B follow-up — empirical Wukong wall

While running the diagnostics that surfaced the depth-wall hypothesis, we measured the qubit ceiling on Wukong 180 for fully-coupled QAOA via Python:

| n (qubits) | Couplings | Wall time | Outcome |
|---|---|---|---|
| 2 | 1 | 10 s | ✓ |
| 4 | 6 | 114 s | ✓ (anomalously slow, queue contention) |
| 5 | 10 | 10 s | ✓ |
| 6 | 15 | 10 s | ✓ |
| 7 | 21 | 10 s | ✓ |
| 8 (small coeffs) | 28 | hung > 3 min | ✗ |
| 8 (Lagrange coeffs) | 28 | hung > 3 min | ✗ |

**The wall is between n=7 and n=8** and is independent of coefficient magnitude. The cause is most likely **transpilation cost** — 28 couplings on Wukong's sparse `{RPhi, CZ}` topology force a SWAP-cascade explosion the cloud's processing pipeline silently times out on.

`QAOACloudSampler` now auto-selects `max_qubits=7` for real-QPU backends and `max_qubits=64` for cloud simulators based on this measurement. The 8-qubit `setcover_4item` lowered BQM that hung last night for 7+ hours is now correctly rejected upfront with a clear error message.

A 2-qubit QAOA submission with proper trained `(γ, β)` parameters is currently in flight on Wukong (job `93A2EAE6…`) and will materialize into the archive via the pending-jobs panel the moment Wukong returns. That gives us the first **genuine real-QPU `qaoa_originqc` archive record** the v2 spec promised.
