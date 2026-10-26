# Milestone 1: does transverse noise let the multi-spin reservoir earn its place?

Follow-on to the M0 negative (`QRC_NoiseSpec_Milestone0.md`). Simulation only. A null
result is a valid outcome.

**Status:** **VOID.** The pre-registered mechanism gate (G2) failed, so no M1a/M1b data was generated
or scored. The pre-registration (§1–4, commit `62005a8`) is unchanged; the gate result, a post-hoc
diagnosis and options are appended in §5.

---

## 1. Question and win axis

**What M0 settled.** Under longitudinal noise ξ(t)Σσz/2, [H0, Σσz] = 0. Every detected
line then carries the same noise factor, and the 3-spin reservoir gave no advantage
over a single spin.

**Hypothesis.** With transverse noise ξ(t)Σσx/2, relaxation becomes frequency-selective:

- A line at rotating-frame offset ω relaxes at a rate set by S(|ω|) (Redfield).
- So a multi-spin system whose lines span many offsets is a *comb*. It samples S(ω) at
  several frequencies in one acquisition.
- A single spin gets the same information sequentially, by changing its offset.

**Win axis: measurement efficiency, not information.**

- The measurement budget **B** is the number of runs. A run is one probe sequence plus
  one 25.6 ms acquisition.
- The reservoir reads all its lines in every run.
- A single spin must spend its runs across offsets.
- **Primary question:** at a fixed small budget B, does the reservoir recover the binned
  S(ω) better than a single spin given the same B?
- Raw accuracy at an unbounded budget is kept as a secondary comparison, for continuity
  with M0.

## 2. Advance prediction (stated before any data)

**Mechanism.**

- In secular Redfield theory, a single spin at offset Ω under A = σx/2 noise has an excess
  transverse decay rate of S(Ω)/4 (verified in simulation for the check case: 21.4/s
  simulated vs 21.5/s from S/4).
- R3's lines sit in 5 resolvable clusters at about 3 distinct frequencies (§3.1).
- So one R3 run samples S(ω) at ~3 frequencies, and one single-spin run samples 1.
- Resolved partner lines are coupled: the H2 lines split by J(C3, H2) = 163 Hz exchange when
  C3 flips, at a rate set by S(Ω_C3)/2. So the cluster rates are linear *mixtures* of S at
  the different frequencies. This is still invertible, but it is not a clean
  one-line-one-frequency comb.

**Predicted outcome.**

- **Mechanism gate:** PASS. The comb exists, and the SLE cluster rates match Redfield.
- **M1a** (σ, τ_c at full budget): PASS.
- **M1b at B = 1** (secondary): PASS. The comb sees ~3 frequencies in one run; one
  single-spin run sees one.
- **M1b at B = 3** (primary): FAIL against the single-spin net, and probably against the
  single-spin ridge too. Reasons:
  - One acquisition already measures the decay rate at its frequency, so a single spin
    with 3 runs covers 3 frequencies, about the comb's count.
  - The single spin can place its offsets freely (bin centres, chosen greedily on dev).
    The comb's 3 clusters are fixed, and two of the 6 bins (bin 3, 9.3·10³ rad/s; bin 5,
    3.2·10⁴ rad/s) have no comb line.
  - The comb shares the signal over 5 clusters at equal SNR, and its rates are mixed.
- **Large B:** the single spin wins. Its sweep covers all 6 bins; the comb stays at its
  3 frequencies.
- **Relaxation dispersion (classical sweep):** weak at B < 3, where it has too few
  frequencies to interpolate 6 bins. Competitive at B ≥ 6.

Your prior was that the comb beats the single spin at small B. **We confirm it only at
B = 1, and predict a fail at the pre-registered primary B = 3.**

## 3. Pre-registration (frozen)

### 3.1 Systems and line frequencies

**R3 (reservoir):** crotonic acid C1, C3, H2 (`crotonic_acid_paper4()` indices 0, 2, 5).
Heteronuclear, chosen for frequency span.

- Offsets: C1 −7749.7 Hz, C3 2699.9 Hz, H2 520.3 Hz. These are the paper offsets, each
  relative to its own carrier (doubly rotating frame, as in the engine).
- J couplings: C1–C3 1.6, C1–H2 6.6, C3–H2 162.9 Hz.
- T1: 5.9, 5.6, 3.4 s. T2*: 212, 208, 332 ms.
- 12 analytic single-quantum lines, merged within the acquisition resolution
  1/25.6 ms = 39 Hz into **5 clusters**: 4.36·10², 6.02·10², 2.62·10³, 2.78·10³ and
  −7.75·10³ Hz.
  - |ω| ≈ 2.76·10³, 3.78·10³, 1.65·10⁴, 1.75·10⁴, 4.87·10⁴ rad/s.
  - About 3 distinct frequencies: H2 ~3·10³, C3 ~1.7·10⁴, C1 ~4.9·10⁴ rad/s.

**S1 (single-spin control):** one ¹³C with C3's T1/T2 (5.6 s, 208 ms).

- Its carrier is set per run, so its offset is one of the **6 bin centres**: 2720, 5030,
  9302, 17201, 31811 and 58828 rad/s (433, 801, 1480, 2738, 5063 and 9363 Hz).
- This is the sweep a single spin needs to cover the band.

**Noise coupling.** Equal per-spin coupling, ξ(t)·Σσx/2, in the (doubly) rotating frame.

- This is an abstract model: in practice, RF-amplitude-like noise on each channel's
  carrier, the same ξ on all spins.
- The single spin's offset sweep is the classical off-resonance relaxation-dispersion
  analogue.

### 3.2 Transverse noise model and ranges

- H_noise = (ξ(t)/2) Σ_i σx_i, with ξ in rad/s. ξ is a sum of independent OU components.
  Each is an (N+1)-level collapsed telegraph sum with exactly Lorentzian
  S(ω) = 2σ²τ_c/(1+ω²τ_c²) (two-sided).
- The ensemble average is exact, by the stochastic Liouville equation (SLE). Because
  [L0, Lx] ≠ 0 (relative 0.35), it is propagated in full: dense
  exp(G·25 µs) on the GPU, binary powers, state-vector products.
- **M1a:** one component, N = 12. σ log-uniform on [100, 1000] rad/s; τ_c log-uniform on
  [10 µs, 1 ms]. Targets: log10 σ, log10 τ_c.
- **M1b:** two components, N = 6 each. Same ranges for each component. Target: log10 of
  the bin-averaged two-sided S(ω) in **6 log bins over [2·10³, 8·10⁴] rad/s** (same bin
  count as M0, range matched to the comb).
- **Noise bandwidth vs line spread (pre-registered check, computed from the prior, no
  simulation):** the fraction of prior draws in which S(ω) varies by ≥ 2× between the
  lowest (2π·440 Hz) and highest (2π·7750 Hz) comb frequency.
  - M1a: **84.4%**. M1b: **94.4%**. Median ratio 24 and 21.
  - Required: ≥ 50%. Met, so the comb has structure to resolve.

### 3.3 Probe bank (14 sequences, same for R3 and every S1 offset)

All pulses are ideal and act on all spins. Every timing is a multiple of 25 µs.

- **FID-delay** (5): π/2_x, wait T, acquire. T ∈ {0, 5, 20, 50, 100} ms.
- **Inversion recovery** (5): π_x, wait T, π/2_x, acquire. T ∈ {5, 20, 50, 100, 200} ms.
  Reads each spin's z relaxation, rate ∝ S(Ω)/2.
- **CPMG** (4): π/2_x, then N π_y over T, then acquire. (N, T) ∈ {(4, 20), (4, 100),
  (16, 20), (16, 100)} ms.
- **Acquisition:** 512 points at a 50 µs dwell (25.6 ms) after every sequence, with the
  noise acting throughout.

Run pools: **R3 has 14 candidate runs; S1 has 84** (14 probes × 6 offsets).

### 3.4 Readouts and D

Per run, A-normalised. Each label below is reported with its D.

- **lines-lin:** complex amplitudes at the analytic cluster lines (direct-DFT projection),
  Re and Im parts. R3: 5 clusters, D = 10 per run. S1: 1 line at its offset, D = 2 per run.
  The R3 cluster amplitudes carry the J-multiplet pattern, which is the two-spin zz order
  (multi-spin correlation) reachable through an FID. Correlations are not read
  separately.
- **lines-log:** log(|amplitude| + 10⁻⁶). R3: D = 5 per run. S1: D = 1 per run.
- **time-log:** log(|FID| + 10⁻⁶) at 32 decimated points. D = 32 per run.

Total D = (D per run) × B.

### 3.5 Measurement noise and budget

- Complex Gaussian noise, per-point std η·A, where A is the system's own noiseless
  amplitude right after π/2. This puts R3 and S1 at equal SNR per acquisition. There is
  an independent draw per (sample, run).
- **η ∈ {10⁻³, 10⁻², 3·10⁻²}. Primary η = 10⁻².**
- **Budgets B ∈ {1, 2, 3, 6, 12}**, plus "all": R3 14 runs, S1 84 runs.
- **Primary B = 3.** This is the smallest budget at which a sweeping single spin can
  cover as many frequencies as the comb has (~3). **Secondary B = 1.**

### 3.6 Methods

| | Method | How the B runs are chosen | Readout |
|---|---|---|---|
| **M1** | Reservoir, R3 | Greedily on dev: add the run that most increases the 5-fold-CV R² of ridge, for each feature family | Per B: the best by dev CV of {ridge with the best family and α ∈ logspace(−4, 4, 17); MLP} |
| **M2** | S1, same ridge | Same greedy procedure on the S1 pool of 84 (offset × probe) | Ridge as in M1 |
| **M3** | Relaxation dispersion (RD), S1, classical | Fixed order: FID-delay T = 0 at the offsets ordered bin 1, 6, 3, 4, 2, 5; then FID-delay T = 20 ms in the same order (B up to 12) | No training. Each run: decay rate R from a weighted log-linear fit of \|S(t)\| over the acquisition (points with \|S\| > 3ηA), so S(Ω) = 4·max(R − 1/T2, 10⁻³). Rates at the same offset are averaged. **M1b:** log S interpolated linearly in log ω to the 6 bin centres (flat extrapolation). **M1a:** Lorentzian least-squares fit in (log σ, log τ_c), grid start, prior-box bounds. |
| **M4** | S1 classical net | M2's greedy order | MLP (hidden (128,128) or (256,256), L2 10⁻⁴ or 10⁻², early stopping), family and size chosen by 3-fold CV on dev |
| M5 | Descriptive | M1's greedy order | R3 MLP alone |

The MLP that M1 can select is the same architecture search as M4.

### 3.7 Splits and statistics

- **dev:** 1200 samples, seed 301 (M1a) / 401 (M1b). Used for every choice: greedy
  orders, families, α, nets.
- **confirm:** 1200 samples, seed 302 / 402. Used once.
- **mechanism gate:** 40 draws from the M1a prior, seed 399.
- **CIs:** paired bootstrap over confirmation samples, 2000 resamples, 95%.

### 3.8 Thresholds

**Mechanism gate (must pass, or the result is VOID).** Computed on the gate draws with
στ_c ≤ 0.3 (Redfield regime), using per-cluster *excess* decay rates. The excess is the
rate with noise minus the rate without, fitted from the FID-delay series.

- **G1** (Redfield theory): median |SLE − Redfield| / SLE excess ≤ 0.20, over clusters
  with an excess > 2/s.
- **G2** (the comb exists, i.e. no M0-style degeneracy): the median over draws of
  max/min cluster excess is ≥ 1.5, **and** the median per-draw correlation between SLE
  and Redfield cluster excesses is ≥ 0.9.

**M1a PASS** iff M1 at the full budget (14 runs), η = 10⁻², has R² ≥ 0.90 on both log σ
and log τ_c.

**M1b PRIMARY PASS (B = 3, η = 10⁻²)** iff the mean-bin R² of M1 exceeds **each** of M2,
M3 and M4 by ≥ 0.05, with each paired-bootstrap 95% CI lower bound > 0.

**M1b SECONDARY (B = 1, η = 10⁻²):** the same rule, reported separately.

**Descriptive:** R² vs B for all methods; large-B (all runs) comparison; η ∈ {10⁻³,
3·10⁻²} at B = 3; and the same budget table for M1a.

**Scope:** 3-spin and single spin only. No scale-up and no PSR-trained probes unless the
M1b primary passes.

### 3.9 Persistence

- Raw noiseless ensemble FIDs, complex64, are saved **before any metric** to
  `backend/artifacts/traces/qrc_noisespec_m1{a,b}_{dev,confirm}.npz`.
  - R3: samples × 14 × 512. S1: samples × 6 offsets × 14 × 512.
  - Each file also holds `fid_dwell`, `split`, `seed`, `noise_params`, `targets`,
    `probes`, `s1_offsets_hz`, the amplitudes and `meta`.
- Mechanism-gate outputs: `docs/QRC/data/qrc_noisespec_m1_mech.json`.

## 4. Validation done before pre-registration (method checks, no task data)

`docs/QRC/data/qrc_noisespec_m1_check.json` (`qrc_noisespec_m1.py check`):

- [L0, Lx] ≠ 0 (relative 0.35), so the general propagator is required.
- Squaring propagator vs brute-force `expm_multiply` on the full sparse SLE generator: ≤ 2.3·10⁻¹²
  relative (FID, inversion recovery, CPMG-16).
- Single spin at 600 Hz, σ = 700 rad/s, τ_c = 100 µs, intrinsic relaxation off. SLE
  (N = 12) vs Monte-Carlo with 10,000 trajectories at t = 2, 10 and 20 ms: |S| = 0.0479,
  0.0405, 0.0326 vs 0.0479, 0.0404, 0.0326 for the same telegraph process, and
  0.0479, 0.0405, 0.0326 for Gaussian OU (MC error ±0.0005).
- The implied excess rate is 21.4/s, against Redfield S(Ω)/4 = 21.5/s.

---

## 5. Results

### 5.1 Mechanism gate (pre-registered): FAILED, so M1 is VOID

Output: `data/qrc_noisespec_m1_mech.json` (40 draws, seed 399; 38 in the Redfield regime,
στ_c ≤ 0.3).

| Gate | Statistic | Threshold | Value | Result |
|---|---|---|---|---|
| G1: SLE vs Redfield | median rel. error of cluster excess rate | ≤ 0.20 | 1.3e-04 | PASS |
| G2: comb exists | median max/min cluster excess (clusters > 2/s) | ≥ 1.5 | 1.38 | **FAIL** |
| G2: comb exists | median corr(SLE, Redfield) per draw | ≥ 0.9 | 1.00000 | pass |

**Per §3.8 the M1 result is VOID.**

- Trace generation was stopped as soon as the gate failed.
- **No dev or confirmation data was generated; no M1a/M1b R² exists.**
- No R² is reported, as the task specified.

### 5.2 Diagnosis (post hoc; labelled, not part of the pre-registration)

Figure: `figures/figNS1_gate.png` (`figures/code/qrc_noisespec_m1_gate_figure.py`).

The gate failed for two reasons. Neither is "there is no frequency-selective relaxation".

1. **With the real J couplings, cluster amplitudes are not exponential in the delay T.**
   - The C1 multiplet (J = 6.6 and 1.6 Hz) and the H2 sub-structure are unresolved in a
     25.6 ms window. In-phase and antiphase parts then interfere as cos(πJT), even with
     no noise (Fig. NS1a, solid).
   - Example: the C1 cluster amplitude goes 1 → 0.21 (50 ms) → 0.37 (100 ms).
   - The pre-registered single-exponential rate fit is therefore wrong for those clusters.
   - The C1 and C3 clusters also pick up decay their own S(ω) does not explain. Their
     fitted excess rates are far above S(|ω|)/4 (Fig. NS1b), while the H2 lines track it.
   - The exact simulation and Redfield theory agree on this (G1 rel. error 1e-4). So it is
     real physics of the model, not a solver error. The likely cause is spin flips of the
     noisy, low-frequency H2 (S large) reaching C1 and C3 through the J couplings. The
     J = 0 diagnostic below does not separate this from the fit artefact.
2. **The G2 statistic was badly chosen.**
   - The "> 2/s" mask usually removes the C1 and C3 clusters, whose own rates are small.
   - That leaves the two H2 clusters, which sit at almost the same frequency, so max/min
     ≈ 1.
   - Even the uncoupled diagnostic below fails G2 (max/min 1.17), although its
     comb is perfect.

**Post-hoc J = 0 diagnostic** (same 40 draws, J switched off; `data/qrc_noisespec_m1_mech_posthoc_j0.json`):

- Every cluster's excess decay rate equals its own Redfield rate S(|ω|)/4 across four
  decades (Fig. NS1c; SLE vs Redfield median rel. error 1.2e-04).
- So the frequency-selective comb **does** exist for uncoupled spins.

### 5.3 What this means for the hypothesis

- **The comb works only when the spins do not talk to each other.** Uncoupled spins at
  different offsets are a clean comb: several single spins read in parallel.
- **The coupling that makes it a "reservoir" is what blurs the comb.** In the coupled
  molecule, the C1 and C3 clusters decay mostly through something other than their own
  S(ω), most likely the partner-flip and J mechanism above, and the decays stop being
  exponential.
- **A parallel array of uncoupled spins is not a reservoir advantage.** It is
  multiplexed single-spin relaxometry: one sample with several non-interacting probe spins
  (or several samples).
- This reverses the premise of M1. A multi-spin comb can be efficient. But it is
  efficient *because* of parallel uncoupled sensors, and in spite of the coupling.
- The pre-registered efficiency comparison was not run, so this is a reasoned conclusion,
  not a measured one.

### 5.4 Options (not run)

1. **Close M1 as VOID** (recommended reading of the pre-registration), with the diagnosis
   above.
2. **Amend and re-register** (a disclosed protocol deviation: the 40 gate draws were seen,
   but no task data exists). Fix the extraction and the statistic:
   - per-spin rates from J-free quantities (z relaxation via inversion recovery, or full
     time-trace fits with a J-modulated model);
   - selectivity as the correlation between excess rate and own S(|ω|), with no mask.

   Optionally, add a heteronuclear-decoupled variant. Then run M1a/M1b. Expect the
   coupled comb to be weaker than the J = 0 comb.
3. **Test the uncoupled comb (J = 0) on the efficiency axis** as a separate, labelled
   question: "does a multiplexed spin array beat a sweeping single spin?" It probably does
   at small B, since that is parallelism. But it is not a quantum-reservoir result.

**Decision (2026-10-26): closed as VOID (option 1).** No amendment and no further runs. The
noise-spectroscopy direction stops here unless a new mechanism is proposed.
