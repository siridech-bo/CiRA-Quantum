# Milestone 0: can the simulated NMR reservoir recover an injected noise spectrum?

Feasibility test for a hybrid QRC noise-spectroscopy direction (NMR reservoir plus
on-hardware parameter-shift training). Simulation only. A null result is a valid outcome.

**Status:** complete. **M0a PASS, M0b FAIL (as predicted): no multi-spin advantage.**
The pre-registration (§1–3, commit `09e3bfd`) was committed before any data and is unchanged;
the results (§4) and the verdict (§5) are appended.

---

## 1. Question

Does a coupled multi-spin NMR reservoir recover a known injected noise spectrum S(ω)
from its FIDs **better, or with fewer measurements**, than:

- standard single-spin filter-function (FF) inversion on the same noise, and
- a classical neural network on single-spin data (deep-learning noise spectroscopy,
  cf. arXiv:2301.05079)?

If it does not, there is no contribution in this direction, and we say so.

## 2. Prediction, stated before any data

In this engine the free Hamiltonian is H0 = Σ π ν_i σz_i + Σ (π/2) J_ij σz_i σz_j
(weak coupling, `app/qrc/system.py`). The dissipators (T1 amplitude damping, T2
dephasing) are phase-covariant. The injected noise is H_noise = (ξ(t)/2) Σ_i σz_i.
So [H0, Σσz] = 0, and the noise superoperator commutes with the whole free Liouvillian.
The CPMG π pulses act on all spins and only flip the sign of Σσz.

Consequence: during any probe sequence, a coherence of total order p picks up the phase
p·φ_f with φ_f = ∫ f(t) ξ(t) dt, where f = ±1 is the sequence's toggling function. The
FID detects only single-quantum coherences (p = ±1). The π/2 pulse on a near-thermal
state, J evolution and all-spin π pulses never take them anywhere else (multi-spin
ε² terms aside). So **every detected line, on every spin, is multiplied by the same
noise factor ⟨e^{iφ_f}⟩**: the noiseless multi-spin FID times one common envelope. That
envelope is the very quantity a single spin measures.

**Predicted outcome:**

- **M0a** (recover σ, τ_c): PASS, or failing only at the edges of the range where the
  noise is unidentifiable.
- **M0b** (multi-spin beats single-spin on binned S(ω)): FAIL. The multi-spin FIDs
  carry no information about S(ω) that the single-spin FIDs lack, at equal
  signal-to-noise ratio.

Routes that could break the prediction (NOT tested here, listed for the verdict):

- spin-dependent coupling to the noise: heteronuclear γ, or spatially varying fields
  (cross-correlation spectroscopy);
- multiple-quantum probe sequences (phase p·φ, sensitivity ∝ p²);
- non-Gaussian noise;
- strong (non-secular) coupling, where [H0, Σσz] ≠ 0.

## 3. Pre-registration (frozen)

### 3.1 Systems

- **R3 (reservoir):** ¹³C crotonic acid C1, C2, C3. Taken from `crotonic_acid_paper4()`
  indices 0–2; protons decoupled, as in the dccb7 setup.
  - Shifts are re-referenced to a carrier at their mean: −7876.5, 5303.3, 2573.1 Hz.
  - J couplings: C1–C2 40.8, C2–C3 69.5, C1–C3 1.6 Hz.
  - T1 5.9 / 4.9 / 5.6 s; T2* 212 / 231 / 208 ms.
  - Weak coupling, |J| ≪ |Δν|, so this is a real molecule, not an abstract Ising chain.
- **S1 (single spin):** C2 alone, on resonance (ν = 0), T1 4.9 s, T2* 231 ms.
- Initial state: the engine's thermal product state.
- Scale-up (4–5 spins, a crotonic subset) only if M0b passes.

### 3.2 Injected noise model

**Option used:** a time-dependent stochastic field (the preferred option).
H_noise = (ξ(t)/2) Σ_i σz_i, with ξ in rad/s, so each spin's precession frequency is
shifted by ξ.

The ensemble average is computed **exactly** with the stochastic Liouville equation
(Kubo–Anderson):

- ξ is the sum of N independent random telegraph processes, collapsed to N+1 levels:
  ξ_k = σ(2k−N)/√N, switching rate 1/(2τ_c) per unit.
- Its autocorrelation is exactly σ² e^{−|t|/τ_c}, i.e. a Lorentzian PSD,
  S(ω) = 2σ²τ_c / (1 + ω²τ_c²) (two-sided). The amplitude distribution is Gaussian to
  O(1/N).
- This is the infinite-ensemble limit of the Monte-Carlo average: no Monte-Carlo
  sampling noise.
- The generator is G = I⊗L0 + diag(ξ)⊗L_z + W⊗I, with L0 the engine Liouvillian
  (`_ensure_liouvillian`). The code checks numerically that [L0, L_z] = 0 and then
  propagates exactly in factorized form. A brute-force `expm_multiply` on the full G
  and a Monte-Carlo OU-trajectory average are both checked against it (§4.1).

Noise ranges:

- **M0a:** one OU component, N = 20 levels. σ log-uniform on [30, 300] rad/s;
  τ_c log-uniform on [0.1, 30] ms. Targets: log10 σ, log10 τ_c.
- **M0b:** the sum of two independent OU components, N = 10 levels each. σ_j
  log-uniform on [20, 200] rad/s; τ_j log-uniform on [0.05, 50] ms. Target: log10 of the
  bin-averaged two-sided S(ω) in **6 log-spaced bins over [10, 3·10⁴] rad/s**.

### 3.3 Probe bank (identical for R3 and S1)

- Each probe: π/2_x on all spins, then a CPMG block of N_π ideal π_y pulses on all
  spins, with spacing τ = T/N_π (τ/2 – π – τ – … – π – τ/2) over a total time T.
  N_π = 0 is free induction.
- **N_π ∈ {0, 1, 2, 4, 8, 16, 32} × T ∈ {2, 5, 10, 20, 50, 100, 200} ms = 49 probes.**
- After the block, acquire **512 points at a 50 µs dwell (25.6 ms)**. The noise keeps
  acting during acquisition.
- **One acquisition per probe = 49 measurements per noise sample.**

### 3.4 Measurement noise

- Complex Gaussian noise on every sampled point, with per-point std = η·A. A is the
  system's own noiseless |S(0)| (no injected noise, t = 0), so R3 and S1 are compared at
  **equal signal-to-noise ratio**.
- **η ∈ {0, 10⁻³, 3·10⁻³, 10⁻², 3·10⁻², 10⁻¹}. Realistic (primary) level: η = 10⁻².**
- One fixed-seed noise draw per sample and level.
- A descriptive equal-per-spin-noise variant (R3 noise std η·A_S1, so R3 gets its
  3-spin signal advantage) is reported and labelled, but is not part of any threshold.

### 3.5 Readouts and methods

All feature sets are labelled with their D in the results.

Feature families, each computed for both systems (`A`-normalised):

- **lines-lin:** per probe, complex line amplitudes at the analytic J-multiplet lines,
  read by direct-DFT projection (`app/qrc/spectral_lines.py`). Re and Im parts.
  D = 49·D_eff·2.
- **lines-log:** log(|line amplitude| + 10⁻⁶). D = 49·D_eff.
- **time-lin:** the FID decimated to 32 points per probe (every 16th sample). Re and Im
  parts. D = 49·32·2.
- **time-log:** log(|FID| + 10⁻⁶) at the same points. D = 49·32.

Methods:

| | Method | Data | Readout |
|---|---|---|---|
| **M1** | Reservoir (primary) | R3 | Closed-form ridge (regime A; `app/qrc/training._fit_ridge`). Feature family and α ∈ logspace(−4, 4, 17) chosen by 5-fold CV on the dev split. |
| **M2** | Single-spin ridge control | S1 | M1's pipeline exactly. Isolates the multi-spin effect from the readout effect. |
| **M3** | Single-spin FF inversion | S1 echo/acquisition magnitudes, at offsets {0, 2.5, 5, 10, 20} ms into each acquisition (245 points) | χ = −ln(\|S\|/A) − t/T2, with A and T2 known (calibrated). \|S\| is floored at η·A; weighted least squares with weights (\|S\|/ηA)². Filter functions are exact for each sequence: χ = (1/2π)∫₀^∞ S(ω)\|F(ω)\|² dω. **M0a:** nonlinear least squares in (log σ, log τ_c) from a 15×15 grid start. **M0b:** non-negative least squares over the 6 bins plus 2 nuisance edge bins ([0.1, 10] and [3·10⁴, 10⁶] rad/s), with ridge λ and log floor chosen on dev. |
| **M4** | Classical net (DL noise spectroscopy) | S1 | `sklearn` MLPRegressor, standardised in and out. Feature family, hidden sizes {(128,128), (256,256)} and L2 {10⁻⁴, 10⁻²} chosen by 3-fold CV on dev; early stopping. |
| M5 | Descriptive only | R3 | M4's net. |

All selections are made per η level, on dev only.

### 3.6 Splits and statistics

- **dev:** 1200 noise samples, seed 101 (M0a) / 201 (M0b).
- **confirm:** 1200 noise samples, seed 102 (M0a) / 202 (M0b). Used once: fit on all of
  dev, score on confirm.
- **Metric:** R² of each log target on confirm. M0a is reported per target; M0b uses the
  mean over the 6 bins.
- **CIs:** paired bootstrap over confirmation samples (iid), 2000 resamples, 95%.

### 3.7 Success thresholds (at η = 10⁻², confirmation split)

- **M0a PASS** iff M1 has R² ≥ 0.90 on **both** log10 σ and log10 τ_c.
- **M0b PASS** iff the mean-bin R² of M1 exceeds each of M2, M3 and M4 by ≥ 0.05, and
  the paired-bootstrap 95% CI of each difference has its lower bound above 0. Beating M2
  is required: it is the multi-spin-vs-single-spin comparison with the readout held
  fixed.
- **"Fewer measurements"** (descriptive, no threshold): R² vs probe budget, using nested
  random probe subsets of 7, 14, 28 and 49 (seed 7), for M1–M4 at η = 10⁻².
- **Regime B** (PSR-trained probes) runs only if M0b passes.

### 3.8 Persistence

Raw noiseless ensemble FIDs (complex64, samples × 49 × 512) are saved for R3 and S1,
per task and split, to `backend/artifacts/traces/qrc_noisespec_m0{a,b}_{dev,confirm}.npz`
**before any metric is computed**. Each file holds `fids_R3`, `fids_S1`, `fid_dwell`,
`split`, `seed`, `noise_params`, `targets`, `probes` and `meta`. Measurement noise is
added at analysis time from fixed seeds.

---

## 4. Results

Confirmation run: commit `1c55562` (analysis), on traces generated after the
pre-registration `09e3bfd`. A dev-only dry run came first (the dev split cut in two), then
one confirmation pass. Nothing in the pre-registration was changed.

### 4.1 Simulator validation

- [L0, L_z] = 0 exactly in the engine, for R3 and S1 (`data/qrc_noisespec_m0_check.json`).
- The factorized exact propagation matches a brute-force `expm_multiply` on the full SLE
  generator to 2e-14 (relative), for FID, CPMG-2 and CPMG-8 probes.
- SLE with N = 20 levels vs a Monte-Carlo average over 20,000 OU trajectories (single spin,
  σ = 150 rad/s, τ_c = 1 ms): coherence 0.8165 vs 0.8179 (FID, 10 ms) and
  0.7736 vs 0.7777 (CPMG-4, 20 ms). The MC sampling error is ±0.007.
- The standard filter-function forward model reproduces the SLE decay of S1 to a median
  relative error of 0.17% (M0a) and 0.26% (M0b), on χ > 0.05. The residual is
  the non-Gaussianity of the finite telegraph sum.
- Initial FID amplitudes: A(R3) = 0.150, A(S1) = 0.050 (3 spins vs 1, as expected).
  R3 has D_eff = 12 resolvable J-multiplet lines.

### 4.2 M0a: recover (σ, τ_c). Confirmation R², mean of log10 σ and log10 τ_c

| η | M1 R3 ridge | M2 S1 ridge | M3 S1 FF inversion | M4 S1 MLP | M5 R3 MLP | M1eq R3 ridge, equal per-spin noise |
|---|---|---|---|---|---|---|
| 0 | 0.991 | 0.990 | 1.000 | 0.995 | 0.995 | 0.991 |
| 0.001 | 0.981 | 0.982 | 0.998 | 0.993 | 0.995 | 0.984 |
| 0.003 | 0.977 | 0.980 | 0.985 | 0.994 | 0.994 | 0.980 |
| 0.01 | 0.970 | 0.979 | 0.925 | 0.993 | 0.990 | 0.976 |
| 0.03 | 0.960 | 0.976 | 0.766 | 0.989 | 0.985 | 0.970 |
| 0.1 | 0.938 | 0.965 | 0.201 | 0.987 | 0.970 | 0.958 |

Selected at η = 10⁻²: M1 lines-lin (D = 1176); M2 lines-lin (D = 98); M4 lines-log (D = 49); M5 lines-lin (D = 1176); M3 NLLS (log sigma, log tau), grid start, prior-range clip (245 echo/acquisition points).

M1 at η = 10⁻²: R²(log σ) = 0.967, R²(log τ_c) = 0.973.

Paired-bootstrap differences in mean R² at η = 10⁻²:

- M1 − M2: -0.0091 [-0.0108, -0.0076]
- M1 − M3: +0.0451 [+0.0334, +0.0573]
- M1 − M4: -0.0233 [-0.0262, -0.0208]

### 4.3 M0b: binned S(ω), 6 log bins over [10, 3·10⁴] rad/s. Confirmation R², mean over bins

| η | M1 R3 ridge | M2 S1 ridge | M3 S1 FF inversion | M4 S1 MLP | M5 R3 MLP | M1eq R3 ridge, equal per-spin noise |
|---|---|---|---|---|---|---|
| 0 | 0.964 | 0.954 | 0.824 | 0.987 | 0.977 | 0.964 |
| 0.001 | 0.942 | 0.929 | 0.787 | 0.969 | 0.947 | 0.946 |
| 0.003 | 0.933 | 0.927 | 0.563 | 0.953 | 0.941 | 0.941 |
| 0.01 | 0.916 | 0.922 | 0.154 | 0.948 | 0.940 | 0.932 |
| 0.03 | 0.894 | 0.912 | -0.283 | 0.947 | 0.911 | 0.917 |
| 0.1 | 0.848 | 0.892 | -0.760 | 0.940 | 0.860 | 0.887 |

Selected at η = 10⁻²: M1 lines-lin (D = 1176); M2 lines-lin (D = 98); M4 lines-log (D = 49); M5 lines-lin (D = 1176); M3 weighted NNLS over 6 bins + 2 edges (245 echo/acquisition points).

Paired-bootstrap differences in mean-bin R² at η = 10⁻²:

- M1 − M2: -0.0063 [-0.0118, -0.0008]
- M1 − M3: +0.7612 [+0.6883, +0.8412]
- M1 − M4: -0.0325 [-0.0384, -0.0270]

The highest-frequency bins are the hard ones for every method. At η = 10⁻² the per-bin R²
(low to high ω) is 0.99, 0.99, 0.99, 0.96, 0.84, 0.71 for M1 and 0.99, 0.99, 0.99, 0.99, 0.92, 0.80 for M4.

### 4.4 Probe budget at η = 10⁻² (descriptive). Confirmation R², nested random probe subsets

M0a:

| probes | M1 | M2 | M3 | M4 |
|---|---|---|---|---|
| 7 | 0.918 | 0.911 | 0.880 | 0.994 |
| 14 | 0.951 | 0.959 | 0.879 | 0.991 |
| 28 | 0.969 | 0.977 | 0.908 | 0.993 |
| 49 | 0.970 | 0.979 | 0.925 | 0.993 |

M0b:

| probes | M1 | M2 | M3 | M4 |
|---|---|---|---|---|
| 7 | 0.848 | 0.856 | -0.960 | 0.930 |
| 14 | 0.864 | 0.877 | -1.024 | 0.962 |
| 28 | 0.914 | 0.917 | -0.484 | 0.962 |
| 49 | 0.916 | 0.922 | 0.154 | 0.948 |

M4 reuses its full-budget configuration here; M1 and M2 re-select their family and α by
dev CV at each budget.

### 4.5 Figure

`figures/figNS0_noisespec_m0.png` (`figures/code/qrc_noisespec_m0_figure.py`):

- (a–c) true vs recovered binned spectrum for three confirmation samples;
- (d, e) recovery vs measurement noise, M0a and M0b;
- (f) multi-spin minus single-spin with the readout held fixed (M1 − M2), with 95% CIs.

### 4.6 Raw traces

- `backend/artifacts/traces/qrc_noisespec_m0{a,b}_{dev,confirm}.npz`: 4 × 0.48 GB,
  complex64, saved before any metric. A copy is also kept on cira2. To be listed in the
  trace manifest for the Zenodo archive.
- Confirmation predictions at η = 10⁻²: `data/qrc_noisespec_m0{a,b}_preds.npz`.

## 5. Verdict against the pre-registered thresholds

**M0a: PASS.** The multi-spin reservoir recovers both Lorentzian parameters with
R² ≥ 0.90 at η = 10⁻² (0.967, 0.973). So the reservoir can do noise spectroscopy.

**M0b: FAIL, as predicted.** At η = 10⁻² the reservoir is *worse* than the single spin
read out the same way (M1 − M2 = -0.0063 [-0.0118, -0.0008]) and worse than the single-spin classical net
(M1 − M4 = -0.0325 [-0.0384, -0.0270]). The same holds for M0a (M1 − M2 = -0.0091 [-0.0108, -0.0076]). It beats only the
filter-function inversion, and that baseline is weak in our implementation (see caveats).
With fewer probes it does not do better either: at 7 probes the single-spin net is ahead
by 0.08 (M0b) and 0.08 (M0a).

**There is no contribution in this direction as tested.** The coupled multi-spin reservoir
does not recover S(ω) better, or with fewer measurements, than a single spin.

### Why: the predicted mechanism, confirmed

[H0, Σσz] = 0 in this engine, so every detected line carries the same noise factor
⟨e^{iφ_f}⟩. The R3 FIDs contain no information about S(ω) that the S1 FIDs lack.

- **Low-noise wrinkle (M0b):** at η ≤ 3·10⁻³ the R3 ridge is slightly *ahead* of the S1
  ridge (+0.006 to +0.013, CI above 0). We read this as a linear-readout effect, not extra
  information. R3 offers 12 lines, each carrying the same envelope, and that changes what
  a linear ridge can fit. The single-spin net, given the S1 data only, is better than both
  at every η, so the information is already in the single-spin signal. The edge reverses
  by η = 10⁻² because at equal SNR R3's signal is spread over 12 lines.
- **Equal per-spin noise (M1eq):** here R3 keeps its 3× signal advantage. M1eq stays below
  the single-spin net at every η, by 0.004–0.029 (M0a) and 0.012–0.054 (M0b). Against the
  S1 ridge it is mixed:
  - M0a: within +0.003 up to η = 3·10⁻³, then below;
  - M0b: +0.005 to +0.017 up to η = 3·10⁻², below at 10⁻¹.

  A molecule with more spins simply gives more signal; that is not a reservoir effect, and
  it never beats the single-spin net.

### Caveats (honest)

- **M3 is a weak baseline in M0b.** Our binned NNLS filter-function inversion is poorly
  conditioned (λ = 0 and log floor −1 selected on dev; R² 0.15 at η = 10⁻²). A
  better-regularized inversion (smoothness priors, Bayesian spectral estimation) would do
  much better. This does not affect the verdict: M0b fails on M2 and M4 regardless.
- **M3 in M0a:** the parametric inversion is the best method at low noise (R² 1.000 at
  η = 0) but degrades with noise. Our floor and weighting of weak echoes is a simple choice,
  not an optimized one.
- **Simulation only:** 3 spins, one molecule subset, ideal instantaneous pulses, Gaussian
  measurement noise.
- **Readout:** lines and time-domain FID features, labelled D per method. This is not the
  653-bin standard of the forecasting work, which does not apply to this task.
- **Not tested:** the routes in §2 that could break the null.
  1. Spin-dependent noise coupling (heteronuclear γ, spatially varying fields).
  2. Multiple-quantum probe sequences (p·φ phase, p² sensitivity).
  3. Non-Gaussian noise.
  4. Strong coupling.

  Any follow-up should start from one of these, because the commuting-noise case is closed
  by a symmetry argument, not just by this simulation.

**Per the pre-registration:** no scale-up to 4–5 spins and no regime B (PSR-trained probes),
because M0b failed.
