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
- Stage 1 pre-registered (§2); its data are not yet generated.

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

Not built yet. It comes after Stage 1: `app/qrc/adaptive.py` and Stage 2+ in
`backend/scripts/qrc_adaptive_est.py`.

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
