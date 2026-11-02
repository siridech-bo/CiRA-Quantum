# De-risking plan + Gate 2 pre-registration: is the model-free policy robust to rich noise?

**STATUS: Gate 2 §§2.1–2.5 FROZEN (`4379613`) before any scoring. Regime a RESULT (§2.8): NOT GREEN
(drift fails; see the design-artefact disclosure); regime b pending the user's decision.** No threshold, prediction, configuration or metric definition changes after the
first scored run. The verdict (GREEN / AMBER / RED) is a decision output for the hardware
go/no-go, **not** a publication claim. Code: `backend/scripts/qrc_gate2_richness.py` (the
`analyze` subcommand implements §2.3–2.5 exactly and is committed with this freeze).

Companion to the Stage 1–3 pre-registrations (`QRC_AdaptiveEst.md`).

---

## The three gates (decision framework)

The sim→hardware step-up rests on one fact: on real hardware the noise matches **no** assumed
model, so "model-free robustness" becomes a genuine claim rather than a within-OU-family one, **but
only if the real noise departs from a simple OU/Lorentzian in a way that matters.** The gates test
that in the cheapest order.

| Gate | When | Cost | Question | Outcomes |
|---|---|---|---|---|
| **1. Kill criteria** | now | hours (analysis) | What must hardware show to be publishable, and what makes us walk away? | This document |
| **2. Rich-noise sim study** | **now** (A100) | one sim study | Does the model-free policy stay robust when the *true* noise is rich (multi-timescale, 1/f, drift, non-Gaussian)? | **GREEN**: hardware viability reduces to Gate 3. **AMBER**: works only if *trained* on richness. **RED**: fails even when trained for it, so no advantage paper |
| **3. Noise spectroscopy** | when the spectrometer arrives | hours of spectrometer time | Is the SpinQ's real noise rich or a clean single Lorentzian? (DD/CPMG filter-function S(ω); no RL) | **Rich**: the advantage paper is live. **Clean**: stop the advantage plan |

**Terminal outcomes:**
- **Advantage paper** (Gate 2 GREEN/AMBER **and** Gate 3 rich): model-free robustness to
  genuinely unknown hardware noise.
- **Demonstration paper** (Gate 3 clean, loop works): first NMR demonstration of learned adaptive
  noise estimation reaching the Cramér–Rao bound.
- **Sim-only methods paper** (Gate 2 RED or no hardware), which is the floor: the pre-registered
  benchmark plus the crossover quantification.

---

# Gate 2 — pre-registered rich-noise sim study

## 2.0 What it forecasts

Stage 3 showed the model-free policy beats a model-aware baseline under misspecified τc **within
the single-OU family**. This study replaces the *true* noise with richer families. The baseline
still assumes one OU at τ₀ = 1 ms, and the model-free policy is the **frozen Stage-3** one
(trained on single OU). It asks whether the policy degrades gracefully while the baseline breaks.

## 2.1 Frozen configuration

| Item | Value |
|---|---|
| Systems | S1, R3 (as Stages 1–3); **each criterion must hold on both** (Stage-3 convention) |
| Noise level | η = 10⁻¹ |
| Estimand | θ = log₁₀ σ_tot, σ_tot² = (1/2π)∫S(ω)dω = Σ_c σ_c². Drift: σ_tot² is the mean of σ_t² over the 49 shots, pinned to 10^(2θ) in every episode |
| Model-aware baseline | `bayes-misspec`: 1-D Bayesian posterior, greedy-Fisher design, single OU at **τ₀ = 1 ms** (ν₀ = −3.0, Stage 3) |
| Model-free policy | frozen Stage-3 checkpoints `qrc_adaptive_s3_{S1,R3}_eta0.1_{adaptive,fixed-cycle}_pilot20k.pt`, ν-blind, **unchanged** (primary: `nn-adaptive`; `nn-fixed` reported only) |
| Action space / budget | discrete 49-probe bank; shot N = 49 |
| Operating points | θ ∈ {1.627 (low-θ edge = Stage-3 θ₀), 1.977 (mid)} |
| Readout | z = y @ Q (Stage-3 line span), D = 2L |
| Episodes | **M = 2000** per configuration × θ × policy; shot noise `[900000, sys, k_θ, r]`, **shared by every configuration and policy at the same θ** (paired ρ, g, G) |
| Compute | regime a (primary): scoring only, cira3 A100, ≤ 2 CPU threads. Regime b only if regime a is not GREEN |

## 2.2 True-noise families (configurations fixed by seed 20261102 in `families()`)

All static families run on the exact SLE engine: the noise is a sum of components, each an
(N_c+1)-level collapsed-telegraph chain. A chain's autocorrelation is exactly σ_c² e^(−|t|/τ_c)
for any N_c, so its PSD is an exact Lorentzian; N_c sets only its Gaussianity.

1. **Multi-Lorentzian (Gaussian spectral-shape test).** C = 2 and C = 3, 4 draws each.
   - log τ_c is uniform in [0.2, 5] ms, and draws whose τ spread is under 4× are rejected.
   - The power fractions are drawn Dirichlet(1).
   - Levels: **N_c = 20 for τ_c ≥ 1 ms** (as Gaussian as the Stage-3 OU table) and 7 below. Up to K = 3528 states.
2. **1/f, discrete-fluctuator (McWhorter).** α ∈ {0.8, 1.0, 1.2}.
   - Five components, log-spaced over τ = 0.1–20 ms (ω band 50–10⁴ rad/s), with σ_c² ∝ τ_c^(α−1).
   - Levels (1, 2, 4, 8, 12) from short to long τ (K = 3510).
   - This is the physical microscopic model of 1/f noise (an ensemble of two-level fluctuators with a rate distribution). A Gaussian 1/f is not the target.
   - Its longest components lie outside the trained τ range (≤ 5 ms), by design.
3. **Non-stationary drift** (single OU at τ₀, per-shot signal from the Stage-3 table).
   - **σ-linear:** log σ ramps by log₁₀(1+δ) end to end.
   - **σ-random-walk:** Gaussian random walk in log σ, end-to-end std log₁₀(1+δ), one path per episode, rng `[910000, sys, k_θ, i_cfg]`.
   - **τc-linear:** log τc ramps by log₁₀(1+δ) around τ₀.
   - δ ∈ {5, 10, 20}% for each, giving 9 configurations.
4. **Telegraph (non-Gaussian distribution test).** One fluctuator (N_c = 1, levels ±σ_tot) at τ ∈ {0.5, 1, 2, 5} ms.
   - Each has a **same-τ Gaussian OU twin** (N = 20) as the factorial control: hold τ, flip only the distribution.
   - Twins are controls; they are excluded from the family medians.
5. **References** (the envelope): single OU (N = 20) at the 5 Stage-3 τc test values. REF-2 is τ₀ = 1 ms.

**Non-Gaussianity index d (reported per configuration).** For each component,
d_c = max_j ‖μ_j(N_c) − μ_j(N = 40)‖·√49 / σ_shot. This is the deviation from a near-Gaussian
chain, in shot-noise units, if all 49 shots hit probe j. The configuration's d is Σ_c d_c, a
first-order additive approximation. It is the *distribution* axis label; the families vary the
*spectral-shape* axis.

## 2.3 Metrics (MSE at shot 49; per configuration × θ × system)

- **ρ = MSE(baseline) / MSE(nn-adaptive).** Robustness ratio; the primary metric.
- **g = MSE(nn on config) / envelope_nn(θ).** envelope_nn(θ) is the **worst** nn MSE over the 5 single-OU references at the same θ, so g ≤ 2 means "no worse than twice what the policy already does on some plain OU". This isolates richness from an effective-τ shift.
- **G = MSE(baseline on config) / envelope_baseline(θ).** Same construction for the baseline.
- Reported, not decisive:
  - g_τ₀ and G_τ₀ (against REF-2 only);
  - nn-fixed MSE;
  - d;
  - for telegraph: dist_nn = MSE_nn(TEL τ)/MSE_nn(TWIN τ), dist_baseline (the same for the baseline), and ρ_twin.

## 2.4 Thresholds and advance predictions

**Aggregation: the median per family over its configurations × both θ, per system.**
Per-configuration rows are always reported, so bimodality stays visible.

| Family | Criterion | Role | Advance prediction |
|---|---|---|---|
| Multi-Lorentzian | median ρ ≥ 2 **and** median g ≤ 2 | required | **PASS**: generalised τc misspecification |
| Telegraph | median ρ ≥ 2 **and** median g ≤ 2 | required | **PASS**, but mostly through τ ≠ τ₀ at 0.5/2/5 ms. The distribution effect (dist_baseline > 1) is visible only at τ ≥ 2 ms, where d ≫ 1. **TEL-1ms ρ < 1** (there the baseline is the oracle up to distribution) |
| 1/f | median ρ ≥ 1 | desired | **UNCERTAIN**: no characteristic timescale, and components lie outside the trained range |
| Drift | median ρ ≥ 1 | desired | **LIKELY FAIL** at δ ≤ 10%: both methods assume stationarity, so ρ ≈ the Stage-3 matched-τ ratio, which is < 1 (Q3a) |

## 2.5 Decision rule (frozen)

- **GREEN:** every criterion in §2.4 holds on **both** systems in regime a. Proceed; hardware viability reduces to Gate 3.
- **AMBER:** regime a is not GREEN, but regime b is (the same configurations, criteria and seeds). The deployed hardware policy must be trained on a broad noise prior.
- **RED:** neither regime a nor regime b is GREEN. Drop the advantage plan and fall back to the demonstration or sim-only paper.

**Regime b** (only if regime a is not GREEN): train one fresh policy, at about the Stage-3 budget on cira3, on a *mixture* over the four families. The families are drawn from *training* seeds that are disjoint from `families()`. Its training configuration is written as a short addendum and committed **before training**; the thresholds and test configurations above do not change.

## 2.6 Order

1. ✅ Families implemented behind the engine/table interface. Generators validated (§2.7). Per-shot signal support added to both runners.
2. ✅ **§§2.1–2.5 frozen (this commit).**
3. Regime a: `gen` (engine signals + d, raw FIDs persisted to `artifacts/traces/qrc_gate2_{sys}.npz`), then `eval` (episodes persisted to `qrc_gate2_eval_{sys}_eta0.1.npz` before any metric), then `analyze`.
4. GREEN → done. Otherwise run regime b, re-score, and give the final verdict.
5. A short results section (§2.8) with the ρ/g/G/d table and the call.

## 2.7 Generator fidelity note (validation, before freeze; no policy run)

`validate` → `docs/QRC/data/qrc_gate2_validation.json`.
- **PSD.** The Monte-Carlo periodogram of the chain process matches the analytic Lorentzian sum over 30–2×10⁴ rad/s for ML3-0, 1f-a1 and TEL-1ms. The smoothed ratio is 1.00–1.03 after a factor-2 bookkeeping fix in the check itself. The 1/f realisation tracks A/ω^α to within ±0.08 decade over the inner band; the fitted slopes are −0.84, −1.00 and −1.16 for α = 0.8, 1.0 and 1.2 (the band edges flatten α ≠ 1 slightly).
- **Level count (Gaussianity), single component, d vs N = 40.**
  - **One fluctuator (N = 1)** is invisible at τ = 0.2 ms (d ≤ 0.08) and marginal at 1 ms (d = 0.1–4.7). It is glaring at 5 ms (d = 7.8–141).
  - **N = 7** is faithful at τ ≤ 1 ms (d ≤ 0.56) but not at 5 ms (d = 0.9–13).
  - **N = 3** is faithful only at τ ≤ 0.2 ms (d up to 1.5 at 1 ms and 36 at 5 ms).
  - Hence the level choices in §2.2. The residual d of every configuration is reported.
- **Disclosure for Stage 3.** The Stage-3 OU table (N = 20) differs from N = 40 by d ≤ 0.53 (S1) / 0.19 (R3) at θ₀, but up to **2.7 / 1.0 at τ = 5 ms, mid θ**. Stage 3 is internally consistent (the same N = 20 chain is the simulator and the model), but its "OU" is, strictly, a 21-level chain that is mildly non-Gaussian at the long-τ, high-σ corner.

## 2.8 Results (regime a; cira3 A100, ~1.5 h; prereg `4379613`)

Data: `docs/QRC/data/qrc_gate2_results.json` (every configuration × θ row). Traces:
`artifacts/traces/qrc_gate2_{S1,R3}.npz` (engine FIDs + d) and `qrc_gate2_eval_{S1,R3}_eta0.1.npz` (all episodes).
**Consistency check:** the 5 single-OU references reproduce the frozen Stage-3 MSEs at the same points within
noise (R3 τ₀: nn 1.10e-4 vs 1.13e-4; baseline 9.9e-6 vs 9.5e-6).

| Family (median over configs × θ) | S1 ρ | S1 g | R3 ρ | R3 g | Criterion | Result |
|---|---|---|---|---|---|---|
| Multi-Lorentzian | 13.8 | 0.86 | 29.8 | 0.41 | ρ ≥ 2, g ≤ 2 | **PASS** both |
| Telegraph | 685 | 0.034 | 286 | 0.11 | ρ ≥ 2, g ≤ 2 | **PASS** both |
| 1/f (McWhorter) | 36.2 | 3.08 | 33.9 | 4.74 | ρ ≥ 1 | **PASS** both |
| Drift | 0.017 | 0.042 | 0.033 | 0.15 | ρ ≥ 1 | **FAIL** both |

**Pre-registered verdict, regime a: NOT GREEN** (drift fails on both systems). Per §2.5, regime b decides AMBER / RED.

**Predictions:**
- Multi-Lorentzian PASS: ✅.
- Telegraph PASS mostly through τ ≠ τ₀: ✅. The same-τ twins give ρ_twin ≈ ρ at 0.5, 2 and 5 ms.
- TEL-1ms ρ < 1: ✅ (0.02–0.61).
- 1/f UNCERTAIN: it passes on ρ, but g is 3–5.
- Drift LIKELY FAIL: ✅, for the reason predicted.

**Reading (disclosures; none of this changes the verdict above):**
1. **The drift failure is a design artefact of this pre-registration.** All drift configurations sit at τ₀ = 1 ms, so apart from stationarity the baseline is the correctly specified oracle. ρ there is the Stage-3 matched-τ efficiency gap (0.01–0.1), not a drift effect.
   - Drift barely hurts either method: G_τ₀ is 0.95–1.6 for the baseline and g_τ₀ is 1–5 for nn. The exception is σ random walk at 20%, mid θ: G_τ₀ = 19 (S1) and 4.0 (R3).
   - Regime b cannot fix this. At matched τ, ρ is bounded by the baseline's near-optimal efficiency, so a mechanical regime-b run would almost surely end RED for a reason unrelated to robustness. This was not foreseen at the freeze; it was not one of the four decisions put to the user.
2. **A distribution effect exists, and it cuts both ways.**
   - At TEL-1ms, mid θ (τ correct, distribution wrong), the baseline is hurt 22× (S1) and 3.8× (R3) relative to its OU twin, while nn is unaffected (dist_nn ≈ 1).
   - At strong coupling (TEL-5ms, mid θ, d = 52–141) it is **nn** that breaks: dist_nn = 83 (S1) and 18 (R3). ρ stays 336–513 only because the baseline is also τ-misspecified.
3. **nn has real out-of-distribution failures that the medians hide.**
   - ML3-2 (84% of the power at 0.44 ms): ρ = 0.0004–0.002, g = 5–48. Here the baseline happens to fit (G_τ₀ = 1–60) while nn degrades 60–1300× from its τ₀ performance.
   - 1/f has median g of 3–5, and up to 46 at α = 0.8, mid θ.
   - These are what a broad-prior training (regime b) would target.
4. **The g envelope is lenient.** nn's worst single-OU MSE (τ = 0.28 ms) is about 40× its best (S1, R3 at θ₀), so g ≤ 2 against it admits large degradations.
