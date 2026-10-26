# Milestone 0: can the simulated NMR reservoir recover an injected noise spectrum?

Feasibility test for a hybrid QRC noise-spectroscopy direction (NMR reservoir plus
on-hardware parameter-shift training). Simulation only. A null result is a valid outcome.

**Status:** pre-registered (this section is committed before any data is generated).
Results and verdict are appended below the pre-registration and do not edit it.

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

(appended after the confirmation run; the pre-registration above is not edited)
