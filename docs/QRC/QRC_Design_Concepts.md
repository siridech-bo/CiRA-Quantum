# QRC design concepts — closing the gaps found in the real-world benchmark

Status: concept / pre-registration draft (2026-10-23). Nothing here is tested yet.
Basis: `QRC_RealWorld_Benchmark.md` (9 datasets, QRC vs tuned ESN / LSTM / linear AR) and
`QRC_RealWorld_Factors.md` (why the QRC wins, is comparable, or loses).

## 0. The gaps to close

| Gap | Evidence | Where it costs |
|---|---|---|
| **G1 Memory** | QRC−ESN falls with horizon (Spearman −0.47, 101 targets); memory function ERA5 Delhi: QRC R² 0.92 / 0.77 / 0.53 at lag 10 / 30 / 90 days vs ESN 0.97 / 0.94 / 0.88 | long horizons, seasonal tasks (Mumbai −0.17 long-horizon temperature) |
| **G2 Robustness** | ETTh1: unbounded FFT features drift to 16–29 σ under regime shift → R² −0.2 … −8.6; ESN states stay in range | any deployment with drift |
| **G3 Expressiveness** | Santa Fe (strongly nonlinear): gain over linear AR +0.20 (QRC) vs +0.24 (ESN) / +0.25 (LSTM) | chaotic / strongly nonlinear signals |
| **G4 Data efficiency of the readout** | adding features (dual channel, multi-τ) *hurt* on 600-day weather; QRC beats LSTM only when data are scarce or noisy | small-data regimes — a strength to protect |

## 1. Physics of the QRC's memory (what the concepts act on)

Each step: encode (R_x pulses, angle θ per spin) → free evolution τ = 30 ms (chemical shifts,
J-coupling, T₁/T₂ relaxation) → read the proton FID on a copy of the state.

| Memory carrier | Lifetime (crotonic-9) | in steps (τ = 30 ms) | Read by the current FID? |
|---|---|---|---|
| Transverse coherences | T₂* 0.20–0.33 s | **7–11** | yes (directly) |
| Populations (z-magnetisation) | T₁ 2.2–27.5 s (C4: 27.5 s) | **73–917** | protons: yes (the π/2 readout pulse converts z → transverse); carbons: indirectly, via J-multiplet amplitudes |

The measured memory (R² 0.77 at 30 days) outlasts T₂ alone, so population memory already reaches the
readout. The likely limiter is **input overwrite**: every step rotates every spin by θ = arcsin(√s), up
to π/2, which mixes the stored populations into coherences that decay in ~10 steps. In ESN language,
the QRC runs with a large, fixed **input scale** and no leak control. (Hypothesis H1 below tests this.)

## 2. Concepts

### C1 — Encoding gain: a tunable "input scale" (targets G1; cheapest)
θ = g · arcsin(√s), g ∈ (0, 1]. Smaller g rotates less per step → less overwrite → longer population
memory, at the cost of weaker input injection — the same trade-off as the ESN input scale, which the
tuned ESNs set to 0.1–0.3. With the learned encoding, g can also be learned (it already can, implicitly;
C1 makes the operating point explicit and searchable).
*Test:* memory function and pooled validation R² vs g ∈ {0.1, 0.2, 0.35, 0.5, 0.75, 1}; arcsin
readout only, no training → minutes per dataset with the block propagator.

### C2 — Memory spins: separate "write" and "store" roles (G1)
Encode the inputs into the fast, readable spins and leave long-T₁ spins (crotonic C4, T₁ = 27.5 s,
≈ 900 steps) **unpulsed**, so they integrate history through J-coupling and are read through the
multiplet amplitudes. Physical analogue of a reservoir with a slow sub-population (multi-timescale ESN).
*Prior evidence to respect:* Phase 2.3 found protons-only encoding hurt short-term capacity (carbons feed
nonlinearity). C2 therefore tests *partial* role separation (subsets), on long-horizon targets, not just
short-memory metrics.

### C3 — Population readout channel (G1)
Add a direct longitudinal readout: the ¹³C channel's z-magnetisation (π/2 on carbons) and/or a
small-flip-angle proton readout that weights populations over coherences. Earlier dual-channel features
hurt because they *doubled* the feature count on 600 days of data, so C3 is paired with C7 (compact
readout): replace, not append.

### C4 — Echo-protected coherence memory (G1; needs a model extension)
The crotonic T₂ values are **T₂\*** (they include static field inhomogeneity, which is reversible).
Inserting refocusing π pulses (spin echo / CPMG) inside each free-evolution window removes the
reversible part, extending coherence memory toward the true T₂ (typically several × T₂\* in liquids).
*Requires:* modelling inhomogeneous broadening explicitly (ensemble average over static offsets) plus a
true-T₂ value for the molecule from the literature; the block propagator still applies (H stays diagonal
during free evolution; π pulses are instantaneous rotations like the encoding).

### C5 — Step time τ as the physical leak rate (G1, G3)
τ sets both coupling cycles per step (J·τ, expressiveness — dccb7 τ = 100 ms beat 30 ms by 3–30×) and
decay per step (T₂/τ, memory). Choose τ per task on validation; for long-memory tasks a *longer* τ with
a smaller C1 gain may give both more coupling and less overwrite per unit time. Molecule choice follows:
long-T₂ systems (dccb7, T₂ up to 1.9 s ≈ 64 steps) sit in a better memory regime than crotonic-9.

### C6 — Bounded, drift-robust readout features (targets G2)
Map the FFT magnitudes through a bounded monotone transform before the ridge: log(1 + |X|), clipped
z-score (±3 σ of training statistics), tanh(z), or a quantile/rank transform fitted on training data.
Plus **shift-aware regularisation**: choose ridge α on a validation window that is *later* in time and
longer (rolling-origin validation), so a too-light α is not picked on in-distribution data.
*Fairness:* the same options are offered to the ESN/LSTM/linear baselines.
*Test:* ETTh1 is the known failure — so the transform family is fixed in advance and selected per
dataset on validation only; ETTh1's test set is not used to choose it.

### C7 — Compact, physics-selected readout (G4, enables C3/C8)
The data limit means the readout must stay small. Use the physics-informed line basis (D_eff lines,
SOP §3.2) or a validation-selected subset / low-rank projection of the FFT features, and spend the freed
budget on the most informative extra channel (C3) rather than appending everything.

### C8 — More expressive reservoirs (targets G3)
(a) **More spins:** 10–12-spin real molecules (e.g. ¹³C-labelled histidine, 12 qubits; dccb with its
5 protons coupled), now tractable with the block propagator (12 spins ≈ 15 h per learned condition).
(b) **Stronger coupling per step:** molecules with large J (¹⁹F-rich systems) or longer τ (C5).
(c) **Richer observables:** two-body correlations / multiple-quantum coherences, kept compact via C7.

### C9 — Learned reservoir control, beyond the input map (G1–G3)
The simulator is differentiable end-to-end, so beyond the encoding we can learn a **fixed control
sequence** applied every step (e.g. learned refocusing/decoupling pulses that choose which couplings act
and how much is overwritten). This turns memory and nonlinearity into trainable design parameters while
the reservoir stays physical and the readout stays linear. Fairness: compare against an ESN whose
recurrent matrix is likewise partially trained.

## 3. Hypotheses and pre-registered tests

| # | Hypothesis | Measurement | Success criterion |
|---|---|---|---|
| H1 | Input overwrite limits memory | memory function vs encoding gain g (C1) | memory at lag 30 rises monotonically as g falls, until injection fails |
| H2 | Longer memory closes the long-horizon gap | long-horizon (h ≥ 60 % max) QRC−ESN on validation | gap halves (−0.041 → ≥ −0.02) without losing short horizons |
| H3 | Bounded features fix drift failures | C6 on all datasets | no dataset with negative R²; no loss > 0.005 elsewhere |
| H4 | Expressiveness closes the Santa Fe gap | C8 / C5 on Santa Fe | gain over linear AR ≥ ESN's (+0.24) |
| H5 | The QRC's small-data advantage survives the changes | Kaggle Delhi + Bangkok | still comparable to ESN and better than LSTM |

**Protocol (to avoid tuning on test sets we have already seen).**
1. Develop C1–C9 on *validation* periods only.
2. Freeze the design (one configuration rule per concept, chosen on validation across datasets).
3. Confirm on **untouched data**: new ERA5 cities never opened (e.g. Chennai, Cairo, Sydney,
   São Paulo, Moscow) plus a held-out later window, with the full baseline set (tuned ESN, tuned LSTM,
   tuned linear AR) and block-bootstrap CIs. Report all concepts tried, including failures.

## 4. Order and cost

| Step | Concepts | Compute | Why first |
|---|---|---|---|
| 1 | C1 (gain sweep) + memory function | CPU/GPU minutes per dataset | tests H1 cheaply; may close most of G1 alone |
| 2 | C6 (bounded features) | CPU minutes | cheap, directly targets the only catastrophic failure |
| 3 | C5 + C7 (τ choice, compact readout) | GPU minutes–hours | refines the operating point |
| 4 | C2, C3 (memory spins, population channel) | GPU hours | structural memory changes |
| 5 | C4, C8, C9 (echo model, bigger molecules, learned control) | model work + GPU days | largest potential, most effort |
| 6 | Frozen-design confirmation on new cities | ~1 GPU-day | the publishable test |

## 5. Realism note
The simulation reads the FID from a copy of the state each step. On real NMR hardware the readout
pulse disturbs the ensemble, so experiments typically re-run the input history per readout point or use
weak/partial readout. Concepts that add readout channels (C3) or pulses (C4, C9) should be costed in
experiment time, not only simulation time.

## 6. Confirmation 1 — frozen encoding-gain rule on untouched cities (PRE-REGISTERED 2026-10-23)

Written and committed **before** the confirmation data were downloaded or opened.

**Frozen rule (from the 8 development datasets, validation only, `QRC_Design_Evidence.md` step 1).**
- **Primary:** fixed encoding gain **g = 0.35** (θ = 0.35·arcsin√s). Chosen because it has the best mean
  rank over the 8 development datasets (2.12 of 6 gains), the best mean long-horizon validation R² (0.618),
  ties the best mean pooled validation R² (0.700), and beats g = 1 on all 8.
- **Secondary:** per-dataset gain selected on that dataset's own validation period from
  {0.2, 0.35, 0.5, 0.75, 1} (0.1 excluded: unstable, never best on pooled R² in development).

**Untouched data.** Open-Meteo ERA5 daily temperature + relative humidity, 1970–2025, for three cities
never downloaded or examined in this project, chosen for climatic diversity: **Chennai** (13.08 N,
80.27 E; tropical coastal), **Cairo** (30.04 N, 31.24 E; hot desert), **Moscow** (55.76 N, 37.62 E;
humid continental).

**Protocol** (identical to the real-world benchmark, `qrc_realworld_bench.py`). Train 1971–1999,
validation 2000–2004, test 2005-01-01 to end−45 days; normalisation from pre-test data; targets
temperature and humidity at h = 1, 5, 10, 15, 20, 30, 45 days; readout (standardisation, ridge α) chosen
on validation, refit on train+validation, test read once. Models: QRC g = 1 (arcsin), QRC g = 0.35,
QRC per-city-selected g, tuned ESN (+ size-matched), tuned LSTM (3 inits each), tuned linear AR.
Significance: 95% moving-block bootstrap CI (30 days, 2000 resamples).

**Success criteria.**
- **S1 (primary):** pooled test R²(QRC g = 0.35) − R²(QRC g = 1) > 0 with the 95% CI excluding 0, in at
  least 2 of the 3 cities.
- **S2:** mean over cities of the long-horizon (h ≥ 30 days) test R² improvement ≥ +0.0205; and
  short-horizon (h ≤ 10 days) test R² not worse by more than 0.005 on average.
- **S3 (descriptive, no success claim):** QRC g = 0.35 vs tuned ESN, tuned LSTM and tuned linear AR,
  with CIs, reported whatever the outcome.
All results are reported, including a failure of S1/S2.

## 7. Step 2 — C6 bounded readout features (PRE-REGISTERED 2026-10-23, before ETTh2 was downloaded)

**Problem.** ETTh1's failure appears only in its test period (its validation months resemble training),
so validation data cannot show whether a robustness fix works, and re-scoring ETTh1's test set after
designing a fix for it would be test-set tuning.

**Transform family (fixed now).** Applied to the QRC readout features (653 FFT magnitudes) before the
ridge, all fitted on the training block only:
- `clipz`: z = (X − μ_train)/σ_train, clipped to [−3, 3]
- `tanhz`: tanh(z / 3)
- `rank`: per-feature empirical CDF of the training block (quantile transform); values beyond the
  training range map to 0 or 1
- `raw`: identity (reference; the current design)
The readout stays the benchmark protocol (standardisation flag and ridge α chosen on validation, refit
train+validation). The tuned ESN is offered the same transform choice (selected on validation).

**Freeze rule (validation only, 8 development datasets, QRC at the frozen gain g = 0.35).** Adopt the
bounded transform (clipz / tanhz / rank) with the highest mean pooled validation R², provided its mean is
not below raw by more than 0.005; otherwise keep raw and report C6 as not adopted.

**Confirmation on untouched data with real distribution shift: ETTh2** (ETDataset, Zhou et al. 2021;
sister series to ETTh1 from a different transformer; never downloaded or examined in this project).
Same task and split as ETTh1 (inputs OT + HUFL, target OT at h = 1, 6, 12, 24, 48 h; standard
12/4/4-month split). Models: QRC g = 1 raw, QRC g = 0.35 raw, QRC g = 0.35 + frozen transform, tuned
ESN (± transform), tuned LSTM, tuned linear AR. 95% moving-block bootstrap CIs.

**Success criteria.**
- **R1 (primary, ETTh2):** QRC g = 0.35 + frozen transform has pooled test R² ≥ 0, and is not worse
  than QRC g = 0.35 raw (CI lower bound of the difference > −0.005). If raw fails (R² < 0), the transform
  must beat it with the CI excluding 0.
- **R2 (post-hoc, labelled; the 9 development datasets' test sets, already seen):** with the frozen
  transform, no dataset has negative pooled test R², and no dataset loses more than 0.005 against raw.
- **R3 (descriptive):** ETTh2 comparison against the tuned ESN / LSTM / linear AR.
All outcomes are reported, including failures.

### Step 2 outcome (2026-10-23) and next hypothesis
C6 **not supported** (R1 FAIL on untouched ETTh2, R2 FAIL post-hoc): bounded transforms do not fix the
drift failure, which replicates out of sample. Diagnosis: extrapolation, not feature size. Candidate
next concept (not yet pre-registered or run): **C6b linear skip connection** — let the readout see the
raw inputs (and a few lags) alongside the QRC features, giving it a linear extrapolation path, as is
standard in reservoir computing. Fairness: the ESN gets the same skip connection.
