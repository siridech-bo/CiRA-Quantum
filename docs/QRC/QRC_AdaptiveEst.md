# ML-adaptive quantum estimation on the NMR engine

Build: an **estimation/metrology yardstick** (Fisher information and the Cramér–Rao
bound, CRB) and an **adaptive controller**, on top of the validated open-system engine.
The simulation shares its sequence definitions and gradient paths with the future
hardware loop, so it acts as a digital twin of that loop.
Simulation is preparation; the publishable result is the hardware demonstration.

Each stage is pre-registered in this file (systems, ranges, probe bank, readout + D,
thresholds, advance prediction) and committed **before** its data are generated.
Results are appended below each frozen pre-registration and never edit it.

**Status:**

- Module C done.
- Module A done; its unit-test gate passed.
- Stage 1 done (pre-registered `4d75d11`): **Gate Y PASS, Q1 NO, as predicted.** The M0 estimators are 3–4 orders of magnitude from the CRB; the readout keeps the information; the MLE attains the bound (§2.9).
- Module B arm (i) built: `app/qrc/adaptive.py`, tests pass.
- Stage 2 done (pre-registered `99f56de`): **Gate F, Q2a, Q2b PASS; Q2c FAIL (prediction miss: the adaptive gain over the best static probe is at the low-θ edge).** About 2.5× fewer shots than the fixed bank. A v1 posterior-grid bug was fixed and rerun as a disclosed deviation (§3.9).
- Batched GPU episode runner: `app/qrc/adaptive_batch.py`.

---

## 1. Modules

### 1.1 Module C — sequences as an engine primitive (`backend/app/qrc/sequences.py`)

- **What it is.** `Sequence` is an ordered tuple of `Pulse(axis, angle)`, `Delay(t)` and
  one final `Acquire(n, dwell)`.
- **Simulation.** Simulators execute it through a four-method backend:
  `init / pulse / free / acquire`.
- **Hardware.** A driver consumes the same object via `to_dict()`: an event list with
  absolute start times and RF phases. `from_dict` round-trips it.
- **Builders.** `cpmg(n_pi, T, n_acq, dwell)` is the M0 probe; `fid_delay` and
  `inversion_recovery` are the M1 kinds.
- **M0 now uses it.** `qrc_noisespec_m0.Batch.probe` runs `sequences.cpmg` through
  `sequences.run`.
- **Regression test** (`backend/tests/test_qrc_sequences.py`). The refactored probe equals
  a frozen copy of the pre-refactor code to ≤ 1e-12 relative, on S1 and R3, for
  N_π ∈ {0, 1, 4, 32}.
- **Not ported.** M1's `Propagator` (transverse regime, closed VOID) keeps its own
  step-count builder.

### 1.2 Module A — estimation yardstick (`backend/app/qrc/estimation.py`)

- **Measurement model.** A probe returns a deterministic ensemble signal μ(θ, probe),
  observed with additive Gaussian noise. Complex signals are real-ified as (Re, Im).
- **Fisher information.** FI = Jᵀ C⁻¹ J, with J = ∂μ/∂θ. J comes from autograd or from a
  4th-order central difference. FI adds over probes. CRB = FI⁻¹.
- **Functions.**
  - `classical_fisher(signal_fn, theta, probes, sigma | cov, method)`
  - `cramer_rao`, which raises on a singular FI
  - `linear_readout_fisher(J, M, sigma)`: the FI left after a linear readout F = M y,
    which bounds every estimator built on that readout
  - `report_vs_bound(estimates, θ, CRB)`: bias, variance, MSE, CRB/Var and CRB/MSE.
    A result below the bound is flagged; only a biased estimator can get there.
  - `m0_sigma(η, A) = ηA/√2`, the per-component noise std of the M0 convention
- **QFI.** Not implemented yet (the spec marks it as optional and later).

**Unit-test gate** (`backend/tests/test_qrc_estimation.py`, 8 tests): **PASSED.**

| Check | Required | Measured |
|---|---|---|
| Single-parameter analytic Gaussian CRB, complex decaying line, finite difference | < 1% | 3.7e-12 relative |
| Same, autograd | < 1% | exact (0.0) |
| Two-parameter (rate, amplitude) analytic FI matrix | < 1% elementwise | pass |
| Efficient linear-model MLE, Monte Carlo (M = 4·10⁵), Var vs 1/FI | within 1% | CRB/Var = 0.9994 |
| Two-parameter linear-model MLE covariance vs FI⁻¹ (M = 2·10⁵) | within 2% elementwise | pass |
| Nonlinear Gauss–Newton MLE at high SNR (M = 4000) | within 7% (≈ 3 s.e.) | pass |
| `linear_readout_fisher` | invertible map is lossless; compression never gains; equals the full-covariance form | pass |
| Singular FI | raises | pass |

### 1.3 Module B — adaptive controller

Arm (i) is built in `backend/app/qrc/adaptive.py`.

- **Posterior.** `GridPosterior` holds an adaptive 1-D grid that re-centres on the posterior. It is updated from the Gaussian sufficient statistics S_j = Σy and n_j, so it re-grids without replaying history.
- **Policies.** `FixedCycle` (the bank), `GreedyFisher` (posterior-expected one-shot FI, Bayesian D-optimal) and `GreedyEIG` (Monte Carlo expected information gain).
- **Episodes.** `run_episode`.
- **Tests** (`backend/tests/test_qrc_adaptive.py`, 3 tests): the posterior matches the closed-form linear-Gaussian posterior (std to 1e-3); the greedy policy picks the max-FI probe; posterior variances reach 1/(n FI_max) for greedy and 1/(n mean FI) for the cycle, to 1e-3.

Arm (ii), the RL / neural policy, is not built yet; it is Stage 3.

---

## 2. Stage 1 — yardstick: do the existing estimators reach the CRB? (PRE-REGISTRATION, frozen)

**Regime: commuting σz dephasing.** [L0, Lz] = 0, so the exact M0 fast path applies
(asserted at generation). No transverse noise and no class-mixing control.

### 2.1 Question

For **one** parameter, a known model and the fixed 49-probe M0 bank: does the variance (and
MSE) of the existing M0 estimators reach the CRB? The existing estimators are the ridge on
M0 features and the MLP on M0 features. The answer sets the gap that Stages 2–3
(adaptive and learned) must close.

### 2.2 Systems, parameter, noise

- **Systems.** S1 (C2 alone, on resonance) and R3 (crotonic C1–C3, protons decoupled).
  Both are exactly `qrc_noisespec_m0.make_systems()`.
- **Injected noise.** H = (ξ/2) Σσz, with ξ one OU component. It is an exact Lorentzian
  realised as a collapsed telegraph sum with N = 20 levels and solved exactly with the
  SLE, as in M0a.
- **Correlation time.** **τ_c = 1 ms, known.**
- **Parameter.** **θ = log10 σ** (σ in rad/s). Prior U[log10 30, log10 300], the M0a
  σ range.
- **Probe bank.** M0's 49 probes, N_π ∈ {0, 1, 2, 4, 8, 16, 32} × T ∈ {2, 5, 10, 20, 50,
  100, 200} ms, each built by `sequences.cpmg`. Acquisition is 512 points at 50 µs. One
  measurement record = one shot of all 49 probes.
- **Measurement noise.** M0 convention: complex white noise, std ηA/√2 per real
  component. A = noiseless |S(0)| after the π/2 pulse, per system: A_S1 = 0.05,
  A_R3 = 0.15.
  - η ∈ {1e-3, **1e-2 (primary)**, 3e-2, 1e-1}.
  - Noise comes from fixed seeds (`noise_seed` in the runner). Each test point uses
    M = 2000 realisations, drawn in chunks of 250.

### 2.3 Bound

- **Definition.** The CRB at θ0 is 1/FI(θ0), where FI is that of the **raw complex FID**
  of the full bank: 49 × 512 complex values, **D_raw = 50,176 real**.
- **Derivative.** J is a 4th-order central difference with step h = 1e-3 in θ, on
  complex128 noiseless FIDs at θ0 + h·{−2, −1, 1, 2}.
- **Method validation, done before this commit and disclosed.**
  - The FI is step-independent to < 1e-9 for h ∈ {5e-4, 1e-3, 2e-3}.
  - Bound values were computed at θ0 ∈ {1.627, 1.977, 2.327}, η = 1e-2: √CRB =
    1.6e-4, 6.4e-5, 4.3e-5 (S1) and 5.5e-4, 1.8e-4, 1.2e-4 (R3).
  - No estimator output on Stage 1 data has been seen.
- **Readout-limited bounds** (descriptive). The FI that survives the linear feature
  maps `lines-lin` and `time-lin` (`linear_readout_fisher`). This separates information
  lost by the readout from inefficiency of the estimator. The log families are
  nonlinear, so they get no exact readout bound.

### 2.4 Test points and estimators

- **Test points.** θ0 = 7 values evenly spaced in [log10 30 + 0.15, log10 300 − 0.15], so
  they stay in the interior and away from the prior edges, where prior-trained regressors
  are biased by truncation.
- **E0 — MLE, known model (reference).**
  - Gaussian likelihood on the raw FID.
  - Grid argmin over a model grid of 801 points spanning the prior (complex128), then
    Gauss–Newton on the 6-point Lagrange interpolant of the model, clipped to the prior.
  - It uses the full raw readout, **D = 50,176**.
  - Validated on an analytic model before this commit: CRB/Var = 1.006, bias −0.013 √CRB.
- **E1 — ridge (existing).** `qrc_noisespec_m0_analysis.ridge_method`.
  - Family ∈ {lines-lin, lines-log, time-lin, time-log} and α ∈ logspace(−4, 4, 17),
    chosen by 5-fold CV on train.
  - Train set: 1200 θ ~ prior (seed 501), each with its own measurement noise at the
    same η.
  - Selection and D are recorded per cell. D values:
    - S1: lines-lin 98, lines-log 49, time-lin 3136, time-log 1568
    - R3: lines-lin 1176, lines-log 588, time-lin 3136, time-log 1568
- **E2 — MLP (existing).** `mlp_method`: the same features and train set, and the M0 grid
  (hidden (128,128) or (256,256), L2 ∈ {1e-4, 1e-2}) with 3-fold CV.
- **Readout labelling.** These are the M0 noise-spectroscopy readouts, labelled with D
  above. They are **not** the forecasting SOP's 653-feature proton FID spectrum. That SOP
  readout belongs to the 9-spin forecasting reservoir and does not apply to these 1- and
  3-spin ¹³C sensing systems.

### 2.5 Metrics

At each (system, η, θ0) the metrics are: bias, Var, MSE, **e_mse = CRB/MSE** (primary)
and e_var = CRB/Var. The summary statistic is the median over the 7 test points.

### 2.6 Thresholds

**Gate Y (yardstick validated on the simulator).** For η ∈ {1e-3, 1e-2} and both
systems, at every θ0, all of the following must hold:

- E0 has e_var ∈ [0.9, 1.1]. This is ≈ 3 s.e. at M = 2000.
- E0 has |bias| ≤ 0.2 √CRB.
- The noiseless MLE offset, |MLE(μ(θ0)) − θ0| (the interpolation error), is ≤ 0.1 √CRB.

If Gate Y fails, Stage 1 is **VOID**: the CRB machinery is not validated on the
simulator, so no estimator is compared with it.

**Q1 (primary).** Per (system ∈ {S1, R3}) × (estimator ∈ {E1, E2}): the estimator
"reaches the CRB" iff median e_mse ≥ 0.8 at η = 1e-2.

Descriptive (no threshold):

- the other η levels;
- the readout-limited bounds;
- the R3 / S1 ratio of the CRB;
- the bias profile.

### 2.7 Advance prediction

- **Gate Y: PASS.**
- **Q1: NO for all four cells, and far from it.**
  - The M0a ridge reached R² ≈ 0.97, which is an RMSE of order 10⁻² in log10 σ.
  - The pre-commit √CRB is about 10⁻⁴.
  - Predicted: ridge median e_mse < 10⁻², MLP median e_mse < 10⁻¹.
  - The readout is not the main cause: the predicted lines-lin readout bound is within
    a factor of 10 of the raw CRB. The estimators are the bottleneck.
  - Consequence: the yardstick shows a gap of at least two orders of magnitude between
    the R²-style estimators and the bound. Stages 2–3 are then about closing it with
    fewer measurements or without knowing the model, not about beating a known-model MLE.

### 2.8 Persistence

`backend/artifacts/traces/qrc_adaptive_stage1_{S1,R3}.npz` holds the following, all
noiseless, all written **before** any metric:

- test-point FIDs and FD stencil FIDs (complex128);
- the MLE model grid (801, complex128);
- the train FIDs (1200, complex64);
- θ values, seeds, probes, A, dwell and `meta`.

Measurement noise is regenerable from fixed seeds. Results go to
`docs/QRC/data/qrc_adaptive_stage1_results.json`.

Runner: `backend/scripts/qrc_adaptive_est.py stage1-gen --system {S1,R3}`, then
`stage1-analyze --prereg <commit>`. It runs on CPU, about 10 min (S1) and about 20 min (R3)
for generation.

---

### 2.9 Stage 1 results (appended; pre-registration `4d75d11` unchanged)

Runs: generation on CPU, 283 s (S1) and 705 s (R3). Analysis on CPU, 1939 s.
Data: `docs/QRC/data/qrc_adaptive_stage1_results.json`.
Traces: `backend/artifacts/traces/qrc_adaptive_stage1_{S1,R3}.npz` (0.58 GB each, gitignored).

**Gate Y: PASS (all 4 cells).**

- E0 (known-model MLE) has CRB/Var between 0.93 and 1.07 at every test point and every η,
  including the descriptive η levels.
- Its bias is within 0.2 √CRB.
- The noiseless interpolation offset is ≤ 3·10⁻⁹ √CRB.
- So the yardstick is validated on the simulator: the bound is attainable, and the MLE
  attains it.

**Q1: NO for all four cells, as predicted.**

| η = 10⁻² | √CRB (mid θ0) | MLE CRB/MSE | ridge CRB/MSE | MLP CRB/MSE | ridge pick (D) | MLP pick (D) |
|---|---|---|---|---|---|---|
| S1 | 6.4e-5 | 1.00 | **1.5e-4** | **1.7e-3** | lines-lin (98) | lines-log (49) |
| R3 | 1.8e-4 | 1.01 | **5.6e-4** | **4.2e-4** | lines-lin (1176) | lines-lin (1176) |

Each CRB/MSE value is the median over the 7 test points.

At every η (descriptive):

- The ridge sits 10⁻⁶ to 2·10⁻² of the bound.
- The MLP sits 2·10⁻⁵ to 3·10⁻² of the bound.
- Both sit at an RMSE of order 10⁻² in log10 σ, almost flat in η.
- The bound and the MLE scale as η (Fig. AE1).

The gap therefore grows as the measurement improves.

**Where the information goes (descriptive, readout-limited bounds).**

- **`lines-lin` keeps all of it.** Its bound is 1.05× the raw CRB (S1) and 1.00× (R3).
  This is the readout both systems' ridges selected.
- **`time-lin` costs a factor of 16** in variance, which is exactly its 32/512
  subsampling.

So the estimators had the information and did not extract it. **The whole gap is the
estimator, not the readout.** This matches the prediction (the readout bound is within
10×).

**Bias.** It dominates the ridge error: bias² is 35–86% of the MSE. A linear map cannot
follow the nonlinear θ → signal relation. The MLP's error is mostly variance on R3 (bias²
1–6% of MSE) and mixed on S1 (17–96%).

**R3 vs S1.** Under the M0 noise convention (noise ∝ total signal A), the R3 √CRB is about
2.9× the S1 value. Three spins add no information per unit of noise about this
collective dephasing. This is consistent with the M0 symmetry argument.

**Verdict.**

- The M0 R²-style estimators are **3–4 orders of magnitude** from the CRB at the primary
  noise level.
- The readout keeps the information.
- A known-model MLE closes the gap entirely.
- Stage 2 therefore asks about *measurements*: whether adaptive probe choice reaches the
  bound with fewer shots.
- Stage 3 asks about *model knowledge*: whether a learned policy and estimator
  approach the bound without the known model, where the MLE cannot be built.

Fig. AE1: `docs/QRC/figures/figAE1_stage1_crb.png` (code
`docs/QRC/figures/code/qrc_adaptive_stage1_figure.py`). The `lines-lin` readout bound is
not drawn because it coincides with the raw CRB.


---

## 3. Stage 2 — adaptive, known model: is the CRB reached in fewer shots? (PRE-REGISTRATION, frozen)

**Regime:** commuting σz dephasing, as in Stage 1. The model is fully known (θ only
unknown, τ_c known).

### 3.1 Question

Does Bayesian-adaptive probe choice (Module B arm (i)) reach the CRB **in fewer shots**
than cycling the fixed 49-probe bank? Does it beat the best *static* single-probe design?

This is textbook Bayesian experimental design. It is the known-model baseline that Stage 3
must match without the model. It is not a novelty claim.

### 3.2 Setup

- **Systems, θ, prior, noise model, η convention:** as in Stage 1 (§2.2).
- **η.** **10⁻² (primary)**; 10⁻¹ descriptive.
- **Shot.** One execution of one probe (`sequences.cpmg`) gives one 512-point FID.
- **Readout per shot: z = y @ Q.** Q is an orthonormal basis (QR) of the line-phasor span
  `ma.phi(system)[+1]`, the same span as the Stage-1 `lines-lin` readout.
  - D = 2L real per shot: S1 L = 1 (D = 2), R3 L = 12 (D = 24).
  - Q is orthonormal, so the noise stays white with the same σ = ηA/√2. Simulating z
    directly is exactly equivalent to simulating y and projecting.
  - FI retained vs the raw FID: 93–98% for S1 and 99.1–99.7% for R3. This was computed
    before this commit and is disclosed.
  - All bounds below are for this readout.
- **Model.** The Stage-1 noiseless grid (801 points), projected, with the 6-point Lagrange
  interpolant (validated in Stage 1 Gate Y).
- **True signals.** The Stage-1 test-point FIDs, projected.
- **Posterior.** `GridPosterior`: uniform prior on [log10 30, log10 300], 201-point grid
  re-centred to mean ± 10 std. The estimate is the posterior mean.

### 3.3 Policies (shots chosen one at a time, n = 1…196 = 4 bank cycles)

- **fixed-cycle:** probes 0, 1, …, 48, 0, … (the Stage-1 bank, shot by shot).
- **greedy-fisher:** each shot uses argmax_j of the posterior-averaged one-shot FI.
- **static-best:** always the single probe with the largest prior-averaged one-shot FI.
  This is a good non-adaptive design.
- `GreedyEIG` is built but not run here (cost). It is out of scope for Stage 2.

### 3.4 Episodes, seeds, persistence

- **Episodes.** R = 200 per (system, η, θ0, policy), at the 7 Stage-1 test points θ0.
- **Seeds.** Shot noise comes from `default_rng([700000, sys, ie, j, r])`. It is
  **common across policies** (paired comparison).
- **Persistence.** Per-shot posterior mean, std and chosen probe go to
  `backend/artifacts/traces/qrc_adaptive_stage2_{S1,R3}_eta{η}.npz` **before** any metric.

### 3.5 Bounds and metrics

- **Per-probe FI.** FI_j(θ0) is the one-shot FI of probe j in the z readout. It comes from
  the Stage-1 FD stencil, divided by σ².
- **Bank bound.** B_bank(n) = 1/(n · mean_j FI_j): cycling the bank, n shots.
- **Locally optimal bound.** B_opt(n) = 1/(n · max_j FI_j). This is the best any policy can
  do with n shots, because the per-shot FI is at most max_j FI_j.
- **MSE.** MSE_p(n) = mean over the 200 episodes of (posterior mean − θ0)².
- **Efficiencies.** e_opt = B_opt(196)/MSE_p(196); e_bank = B_bank(196)/MSE_p(196).
- **Shots to target.** n*_p = smallest n with MSE_p(n') ≤ B_bank(49) for all n' ≥ n. The
  target B_bank(49) is the bound of one full bank record; n* is reported as "none" if it
  is not reached by 196.

### 3.6 Thresholds (η = 10⁻², per system)

The gate:

- **Gate F (sanity).** Median over θ0 of e_bank(fixed-cycle) ∈ [0.8, 1.25]. The posterior
  mean must be efficient for the bank. If it fails, Stage 2 is VOID.

The questions:

- **Q2a (greedy reaches the bound).** Median over θ0 of e_opt(greedy) ≥ 0.8.
- **Q2b (fewer measurements, primary).** Median over θ0 of n*_fixed / n*_greedy ≥ 1.5.
- **Q2c (adaptivity, not just a good probe).** At the edge test point θ0 = 2.327 (where the
  best probe differs most from the prior-optimal one), MSE_static(196) / MSE_greedy(196) ≥ 1.5.

### 3.7 Disclosed before this commit

These are bound values only; no Stage-2 episode data were generated:

- per-probe FI at the test points;
- the max/mean FI ratio: 2.0–3.1 (S1) and 2.0–3.0 (R3);
- the best probe moves from a 200 ms free delay at low σ to a 10 ms free delay at high σ;
- the FI-retention numbers above;
- two dev timing episodes at θ = 1.9, which is not a test point (seed 12345, not recorded).

### 3.8 Advance prediction

- **Gate F: PASS.**
- **Q2a: PASS** on both systems.
- **Q2b: PASS** on both. The ratio n*_fixed / n*_greedy should be about the FI max/mean ratio,
  predicted in [1.5, 3.2]. The greedy pays a few shots while the posterior is wide.
- **Q2c: PASS** on both systems.
- The adaptive gain on this bank is **modest (≈ 2–3× in shots)**. The bank already
  contains near-optimal probes for every θ, so the possible gain is capped by the FI
  max/mean ratio, and adaptivity's other contribution is only to track the θ-dependent
  best probe.
- Larger gains would need a richer action space, such as continuous T or N_π. That is
  noted for Stage 3, not tested here.

Runner: `backend/scripts/qrc_adaptive_est.py stage2-run --jobs 12`, then
`stage2-analyze --prereg <commit>`. CPU only, about 45 min.


### 3.9 Stage 2 results (appended; pre-registration `99f56de` unchanged)

**Verdict.** In both systems:

- **Gate F: PASS.**
- **Q2a (greedy reaches the bound): PASS.**
- **Q2b (fewer measurements): PASS.** About 2.5× fewer shots, inside the predicted [1.5, 3.2].
- **Q2c: FAIL.** This misses the prediction. The adaptive gain over the best static probe
  is at the *low*-θ edge, not at the pre-registered high edge θ0 = 2.327.

These are the v2 numbers. The frozen v1 run had a posterior implementation bug that
corrupted S1 at η = 10⁻²; it is reported below and was rerun as a disclosed deviation.

#### v1: the frozen protocol as run, and the bug

The frozen v1 scoring is in `docs/QRC/data/qrc_adaptive_stage2_results.json`.

- **R3: clean.** Gate F PASS (1.07), Q2a PASS (1.03), Q2b PASS (2.47×), Q2c FAIL (1.13).
- **S1 at η = 10⁻²: corrupted** at the four highest θ0, where one shot is very informative.
  - **What went wrong.** After a single shot, all posterior weight landed on one point of the
    201-point grid. The std was then 0, the re-centring step was skipped
    (`b − a < 1e-14`), and the grid never zoomed in again. The estimate froze on a grid point.
  - **Effect on the metrics.** Greedy and static MSE came out around 10⁻⁶, against 10⁻⁹ for
    fixed-cycle. At θ0 = 2.327, which happens to lie exactly on the prior grid, the MSE was
    exactly 0. So the S1 verdict was NaN for Q2b and an artefact (inf) for Q2c.
  - **Not affected:** S1 at η = 10⁻¹ and R3. Their posterior std never fell below the grid
    spacing.
- **Fix (v2).** `GridPosterior.update` now uses the half-width width·max(std, spacing). It
  zooms in until the std is at least 3 grid spacings, with at least 2 passes as before.
  - A regression test, `test_posterior_resolves_peak_narrower_than_prior_grid`, fails on v1
    and passes on v2.
  - Everything else is unchanged: seeds, episodes, policies, metrics, thresholds.
  - All four (system, η) cells were rerun with tag `_v2`.

#### Compute

- **v1:** numpy/joblib, on .167 (S1, R3 η = 10⁻²) and on cira3 (R3 η = 10⁻¹).
  - R3 η = 10⁻² was computed twice, on .167 and on cira3, with chunked tasks. The two copies
    agree: every probe choice is identical, and the means differ by at most 1.6·10⁻¹¹.
- **v2:**
  - S1 ran on cira3 with numpy.
  - R3 ran on the cira3 A100 with the batched torch runner `app/qrc/adaptive_batch.py`
    (17 s per 200-episode cell).
  - That runner uses the same seeds. Against the numpy v2 code, the means agree to within
    2·10⁻¹¹ and every probe choice is identical.
- **Traces:** `backend/artifacts/traces/qrc_adaptive_stage2_{S1,R3}_eta{0.01,0.1}{,_v2}.npz`.

#### v2 results

From `docs/QRC/data/qrc_adaptive_stage2_v2_results.json`. Medians are over the 7 θ0.

| system | η | e_bank(fixed) | e_opt(greedy) | n* fixed / greedy (median) | static / greedy MSE₁₉₆ at θ0 = 1.63 … 2.33 |
|---|---|---|---|---|---|
| S1 | 10⁻² | **0.97** | **0.98** | **2.56×** | 4.94, 3.85, 2.73, 1.73, 1.19, 1.00, **1.16** |
| R3 | 10⁻² | **1.07** | **1.03** | **2.47×** | 3.23, 2.78, 2.19, 1.81, 1.23, 1.00, **1.13** |
| S1 | 10⁻¹ | 1.04 | 1.05 | 2.21× | 5.01, 3.85, 2.73, 1.73, 1.19, 1.00, 1.16 |
| R3 | 10⁻¹ | 0.95 | 1.02 | 2.32× | 3.32, 2.82, 2.41, 1.87, 1.07, 1.00, 0.99 |

Shots to the target (MSE ≤ the one-bank-record bound) at η = 10⁻², for θ0 = 1.63 … 2.33:

| system | policy | n* |
|---|---|---|
| S1 | fixed | 54, 41, 60, 58, 50, 45, 51 |
| S1 | greedy | 19, 16, 19, 22, 22, 22, 23 |
| S1 | static | 73, 55, 53, 40, 24, 22, 26 |
| R3 | fixed | 54, 54, 42, 49, 40, 48, 52 |
| R3 | greedy | 19, 17, 17, 22, 21, 23, 18 |
| R3 | static | 62, 37, 46, 27, 25, 23, 29 |

![Fig. AE2](figures/figAE2_stage2_adaptive.png)

*Fig. AE2.* (a, b) MSE vs shots at θ0 = 1.63, η = 10⁻². The adaptive policy (green) follows
the best-probe-every-shot bound B_opt after about 3 shots. The fixed bank follows its own bound
B_bank, and the static design follows a bound for a probe that is wrong for this θ0. Markers
show n*. (c) n* vs θ0 for both systems.

#### Reading

1. **Q2a/Q2b (as predicted).** With the model known, greedy posterior-FI design is
   efficient: e_opt is 0.78–1.13 per point. It needs about 2.5× fewer shots than cycling the
   bank, ranging 1.9–3.2× per point.
   - This matches the bank's FI max/mean ratio (2.0–3.1) disclosed before the run.
   - The gain is capped by the action space, as predicted. The bank already contains
     near-optimal probes.
2. **Q2c (missed).** The pre-registration assumed the best probe differs most from the
   prior-optimal one at the high edge, θ0 = 2.327. That is wrong.
   - The prior-optimal static probe is the right probe at high θ (ratio 1.0–1.2).
   - It is badly wrong at low θ, where adaptivity is worth 3.2–5.0× in MSE.
   - The test point was mis-chosen; the effect itself is there. Per the protocol, Q2c is
     reported as FAIL.
3. **What Stage 2 establishes.** It is a known-model baseline only (textbook Bayesian design;
   no novelty claim). Stage 3 has to approach it **without** the model. The headroom beyond
   it lies in a richer action space (continuous T / N_π), not in better probe choice from this
   bank.
