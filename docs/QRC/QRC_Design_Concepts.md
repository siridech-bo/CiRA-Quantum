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

## 8. Step 3 — C5 step time τ + C7 compact readout (PRE-REGISTERED 2026-10-23, before confirmation data)

**Current frozen design:** crotonic-9, encoding gain g = 0.35, τ = 30 ms, readout = 653 top FFT-magnitude
bins of the proton FID (benchmark readout protocol).

**Search space (fixed now), all at g = 0.35:**
- **C5 τ ∈ {15, 30, 60, 100} ms.**
- **C7 readout ∈** `bins653` (current) | `bins200`, `bins100` (top-K of the same ranking) | `pca128`,
  `pca64` (PCA fitted on the training block) | `lines` (physics-informed line projection of the FID onto
  the D_eff analytic proton lines, real + imaginary parts; SOP §3.2).
FIDs for every τ are saved (persist rule); readout variants are computed from the saved FIDs.

**Freeze rule (validation only, 8 development datasets; Kaggle Delhi reported separately as the
small-data check).** Score every (τ, readout) by mean pooled validation R² (benchmark protocol). Adopt the
best combination **only if** it beats the current design (τ = 30 ms, bins653) by ≥ 0.005 in mean; otherwise
keep the current design and report C5/C7 as not adopted (then no confirmation run).

**Confirmation on untouched data** (only if a change is adopted): Open-Meteo ERA5 daily T + RH,
1970–2025, for three cities never downloaded in this project: **Sydney** (−33.87, 151.21), **São Paulo**
(−23.55, −46.63), **Toronto** (43.65, −79.38). Benchmark protocol (train 1971–1999, validation
2000–2004, test 2005–2025 read once). Models: QRC current design, QRC adopted design, tuned ESN, tuned
LSTM, tuned linear AR; 95% moving-block bootstrap CIs.

**Success criteria.**
- **T1 (primary):** pooled test R²(adopted) − R²(current) > 0 with the 95% CI excluding 0 in ≥ 2 of 3 cities.
- **T2:** no city loses more than 0.005 (point estimate).
- **T3 (descriptive):** both QRC designs vs tuned ESN / LSTM / linear AR.
- **T4 (small-data check, descriptive):** adopted vs current on Kaggle Delhi validation (H5).
All outcomes are reported.

### Step 3 outcome (2026-10-23)
C5/C7 **not adopted**: the current design (τ = 30 ms, 653 bins) is the best of 24 (τ, readout) combinations
on development validation (mean 0.7059; next best 0.7028), below the +0.005 adoption threshold. Per the
pre-registration no confirmation was run, so Sydney / São Paulo / Toronto remain untouched for a later
confirmation. Findings: τ is a flat optimum near 30 ms once the gain is set (15 / 60 / 100 ms: 0.701 /
0.703 / 0.698); compact readouts cost a little on large datasets (bins100 0.697, pca64 0.695) but the
physics line projection is near-equal with ~3× fewer features (0.702); on the small Kaggle set compact
readouts help slightly (0.590 vs 0.582), consistent with the data-limit factor.

## 9. Step 4 — learned encoding on top of the frozen gain (PRE-REGISTERED 2026-10-23, before confirmation data)

Written and committed **before** any learned-g = 0.35 run and before the confirmation cities were
downloaded. Question: does the published learned encoding (C9, input map) add to the frozen gain rule,
i.e. are the two improvements complementary?

**Arm.** θ = 0.35·arcsin√s + 0.5·tanh(MLP(s)), per-spin MLP (16 tanh units), last layer zero-initialised
so training starts exactly at the current design. Training identical to the published learned encoding
(`qrc_realworld_bench.py learn --gain 0.35`): train-period slice only (last 3000 training steps; fit on
the first 2000, inner validation on the last 1000, 500-step washout), Adam lr 0.02 with cosine decay,
30 steps, keep the best inner-validation checkpoint; seeds 0, 1, 2. Readout features: the 653 bins of the
g = 0.35 pass. Raw FID trace, θ, and encoder weights saved for every seed.

**Development (validation only)** on the 8 development datasets, the same score as Step 3 (pooled
validation R², ridge α and standardisation chosen on validation). Reference: g = 0.35 fixed (current
design, mean 0.7059). Descriptive: the existing learned g = 1 encoders (3 seeds).

**Freeze rule.** Adopt the learned g = 0.35 encoding if its seed-mean validation R², averaged over the 8
datasets, is ≥ g = 0.35 fixed + 0.005. Otherwise keep the current design, report "not adopted", and run no
confirmation.

**Confirmation (only if adopted).** Untouched Open-Meteo ERA5 daily T + RH, 1970–2025: **Sydney**
(33.87 S, 151.21 E), **São Paulo** (23.55 S, 46.63 W), **Toronto** (43.65 N, 79.38 W). The protocol is
identical to Confirmation 1 (splits, normalisation, targets, readout selection, test read once), with the
tuned ESN, tuned LSTM and tuned linear AR run as in the benchmark.
- **L1 (primary):** pooled test R²(learned g = 0.35, seed-mean) − R²(g = 0.35 fixed) > 0 with the 95%
  moving-block bootstrap CI (30 days, 2000 resamples) excluding 0, in at least 2 of 3 cities.
- **L2 (descriptive):** learned g = 0.35 vs g = 1, tuned ESN, tuned LSTM and linear AR, with CIs.
All results are reported, including failures.

### Step 4 outcome (2026-10-24)
**Not adopted.** Learning the encoding on top of g = 0.35 gives mean development-validation R² 0.7038 vs
0.7059 for the fixed gain (Δ −0.002, threshold +0.005; better on 4/8, all gains ≤ +0.004, Bangkok −0.017,
Santa Fe −0.005 driven by one seed). Per the pre-registration no confirmation was run; Sydney / São Paulo /
Toronto remain untouched. Interpretation: the gain rule and the learned input map are **not complementary** —
once input overwrite is reduced by g, the learned correction mostly fits the training slice (inner NMSE
falls 4–35 %) without transferring to the validation period. The fixed g = 0.35 also beats the published
learned encoding at g = 1 on 6/8 datasets (mean 0.7059 vs 0.6927; Beijing and Berlin are the exceptions), so the earlier learned-encoding gain is
largely explained by reduced input overwrite. Current design unchanged: g = 0.35, τ = 30 ms, 653 bins.

## 10. Confirmation 2 — the final design on untouched cities (PRE-REGISTERED 2026-10-24, before the data were downloaded)

Written and committed **before** the Sydney / São Paulo / Toronto data were downloaded or opened.

**Frozen final design** (Steps 1–4; nothing is selected on these cities): crotonic-9 reservoir, fixed
encoding gain **g = 0.35** (θ = 0.35·arcsin√s, no learned correction), τ = 30 ms, proton FID, |FFT| at
**653 bins** (SOP standard readout, D = 653), ridge readout (α and standardisation chosen on validation,
refit on train+validation, test read once).

**Untouched data.** Open-Meteo ERA5 daily T + RH, 1970–2025: **Sydney** (33.87 S, 151.21 E; humid
subtropical coastal, southern hemisphere), **São Paulo** (23.55 S, 46.63 W; subtropical highland),
**Toronto** (43.65 N, 79.38 W; humid continental, lake-moderated). Protocol identical to Confirmation 1
(§6): train 1971–1999, validation 2000–2004, test 2005-01-01 to end−45 days; T and RH at h = 1, 5, 10, 15,
20, 30, 45 days. Models: QRC final design, QRC g = 1 (published design), tuned ESN (+ size-matched ≤ 653),
tuned LSTM (3 inits), tuned linear AR, all tuned on validation exactly as in the benchmark. Significance:
95% moving-block bootstrap CI (30 days, 2000 resamples), paired on the same test days.

**Pre-registered criteria** (Δ = pooled test R² difference, mean over the 14 targets).
- **F1 (replication of the gain effect):** Δ(final − g = 1) > 0 with the CI excluding 0 in ≥ 2 of 3 cities.
- **F2 (competitiveness; non-inferiority, margin δ = 0.02 R²):** separately for each baseline
  b ∈ {tuned ESN, tuned LSTM, tuned linear AR}: the CI lower bound of Δ(final − b) > −0.02 in ≥ 2 of 3
  cities. F2 is reported per baseline (F2-ESN, F2-LSTM, F2-AR); no superiority over the ESN is claimed
  unless a CI lies wholly above 0, and that would be reported as descriptive. *Disclosure:* δ = 0.02 was set
  with knowledge of the Confirmation 1 results (§6, where the final design's CI lower bounds vs the tuned
  ESN were −0.022, −0.005, −0.016); the Confirmation 2 data were untouched when it was set.
- **F3 (descriptive):** short (h ≤ 10) and long (h ≥ 30) horizon breakdown, size-matched ESN, per-target
  R², and the QRC − baseline differences pooled over all six confirmation cities (Confirmation 1 + 2).
All results are reported, including failures of F1 or F2.

### Confirmation 2 outcome (2026-10-24)
**F1 PASS (3/3), F2 PASS for every baseline (ESN 3/3, LSTM 3/3, linear AR 3/3).** Pooled test R², final
design vs g = 1: Sydney 0.448 vs 0.424, São Paulo 0.282 vs 0.231, Toronto 0.480 vs 0.454 (all CIs > 0;
long-horizon gains +0.047 to +0.104). Against the tuned baselines the final design is within 0.009 R² of the
best model in every city: the tuned ESN remains significantly better in Sydney (−0.009) and Toronto
(−0.005) and is tied in São Paulo; the tuned LSTM is significantly better only in Sydney (−0.005); the QRC
beats linear AR in Sydney and São Paulo. Over all six confirmation cities (descriptive F3): mean Δ vs tuned
ESN −0.006 (ESN significantly better in 4/6, QRC in 0/6), vs tuned LSTM +0.001 (2/6 vs 1/6), vs linear AR
+0.015 (QRC better in 4/6). Claim supported: **the final physical QRC design is non-inferior to tuned
classical reservoirs and LSTMs on untouched real-world forecasting, but not superior to the tuned ESN.**
Details: `QRC_Design_Evidence.md` (Confirmation 2).

## 11. Step 5 — readout-side concepts R1–R5 (PRE-REGISTERED 2026-10-25, before any R-result was computed)

Basis: literature review `reports/QRC improvements for forecasting.md` (2026-10-25): the drift failure
(ETTh1/ETTh2) looks like a *readout* limitation — the 653 |FFT| features are non-negative, bounded and
carry no linear copy of the input, so the linear readout cannot follow a shifted level — and the
long-horizon gap may be partly closable by readout-side memory. All variants are **offline** on the saved
frozen-design features (`design_c1/g0.35/X.npz`; g = 0.35, τ = 30 ms, 653 bins, D = 653); no new
reservoir evolution. **Fairness:** every variant is applied identically to the tuned ESN (its
validation-selected configuration, states recomputed, 3 inits); the linear AR baseline is unchanged
(it already is the lag model).

Variants (F = the model's base features; hyper-parameters chosen on VALIDATION only, refit TRAIN+VAL):
- **base** — current readout (ridge on F; α and standardisation on VAL, per target).
- **R1 delta** — last-value re-centring (NLinear-style, Zeng et al. 2023): target y(t+h) − w_v(t),
  prediction adds w_v(t) back; R² on the original scale.
- **R2 lags** — [F, raw input lags w(t..t−L+1)], L ∈ {1, 7, 30}, lag block scaled by s ∈ {1, 3, 10}
  after standardisation (smaller effective penalty), (L, s) on VAL.
- **R3 ar_residual** — two-stage: the tuned linear AR (lags/α on VAL) first, ridge on F fitted to the AR
  residual; prediction = AR + residual model.
- **R4 rff** — kernel readout: random Fourier features (RBF, 2000 features, fixed seed 0) of the
  standardised F, bandwidth γ ∈ {0.3, 1, 3} × 1/D, then ridge.
- **R5 delay** — [F, PCA32(F)(t−7), PCA32(F)(t−30), PCA32(F)(t−90)] (PCA fitted on TRAIN). *Adds classical
  memory (SOP §6) — reported with that label.*
- **R1+R2** — delta target with the lag-augmented features.

**Freeze rule (P1):** for the QRC, the variant with the highest mean pooled VALIDATION R² over the 8
development datasets (era5 Delhi/Mumbai/Bangkok/Beijing/Berlin, Beijing PM2.5, ETTh1, Santa Fe) is
adopted only if it beats base by ≥ 0.005 in the mean **and** is not worse than base by > 0.005 on more than
2 of the 8 datasets (so ETTh1 alone cannot decide). The ESN gets the same variant (descriptive at dev).
**P2 (descriptive, dev validation):** QRC_v − ESN_v vs QRC_base − ESN_base, per dataset.
ETTh2 (seen in Step 2) is reported post-hoc only, labelled.

**Confirmation 3 (pre-registered here; run only if P1 adopts a variant):** three new untouched ERA5 cities,
fixed now, data not yet downloaded — **Lagos** (6.52 N, 3.38 E), **Mexico City** (19.43 N, 99.13 W),
**Stockholm** (59.33 N, 18.07 E); protocol identical to Confirmations 1–2 (train 1971–1999, val 2000–2004,
test 2005–2025 read once; 30-day moving-block bootstrap, 2000 resamples). Needs one reservoir pass per city
(GPU; asked for separately).
- **G1:** QRC_v − QRC_base CI > 0 in ≥ 2/3 cities.
- **G2:** non-inferiority QRC_v − ESN_v, CI lower bound > −0.02 in ≥ 2/3 cities (same margin as §10).
- **G3 (descriptive):** QRC_v − ESN_v, − LSTM (base), − AR_v; counts of significant wins either way.
All outcomes are reported, including a P1 that adopts nothing.

### Step 5 outcome (2026-10-25)
**P1 adopts `rff` (kernel readout) — by the letter of the rule, but driven by one dataset.** Mean development
VALIDATION R² (QRC): base 0.7061, rff **0.7115** (+0.0054 ≥ 0.005; worse than base by > 0.005 on 2/8 —
Bangkok −0.012, ETTh1 −0.011 — within the ≤ 2 allowance). The gain is Santa Fe (+0.059); without Santa Fe the
mean change is **−0.002** (ERA5 cities −0.012 … +0.004). Other variants: lags 0.7080, delta_lags 0.7080, delay
0.7104 (classical memory), ar_residual 0.7056, delta 0.7027. P2 (descriptive): QRC − ESN, same variant, mean
−0.018 (base) → −0.001 (rff), but mostly because rff *hurts* the ESN (0.7243 → 0.7125); against the ESN's
base readout the QRC-rff gap is −0.013.
**POST-HOC drift check (ETTh1/ETTh2 test sets already seen; labelled, not confirmatory):** validation cannot
see drift, and the adopted rff **fails catastrophically under shift** (test R² ETTh1 −1.87, ETTh2 −17.0).
The level-handling variant does what the literature predicts: **delta_lags** raises the QRC from −0.21 → **+0.23**
(ETTh1) and +0.36 → **+0.74** (ETTh2; lags alone 0.72), against tuned ESN 0.66 / 0.79 — most of the ETTh2 gap
closes, ETTh1 remains far behind. The ESN gains little from the same variants (it already extrapolates).
Base scores reproduce Step 2 exactly (−0.2120 / +0.3632). Data `qrc_design_r_{dev_*,frozen,posthoc_*}.json`.
Note: the Santa Fe base here (0.9076) differs from the remote Step 3 run (0.9060) by numerical/host effects;
all variants were scored in one place.

## 12. Drift test — level handling on untouched shifted series (PRE-REGISTERED 2026-10-25, before the data were downloaded)

Basis: Step 5 post-hoc (§11 outcome): `delta_lags` (last-value re-centring + raw input lags appended to the
653 features) lifted the QRC from +0.36 → +0.74 test R² on ETTh2 and −0.21 → +0.23 on ETTh1 — but those test
sets had been seen. This step tests it on series never touched by any model here.

**Datasets (fixed now; standard LTSF sources, laiguokun/multivariate-time-series-data):**
- **exchange** — daily exchange rates 1990–2016 (8 currencies). Target = column 0, second input = column 1.
  Horizons 1, 5, 10, 20 days.
- **electricity** — hourly consumption 2012–2014 (321 clients). Target = last client (the LTSF "OT"),
  second input = mean over all clients. Horizons 1, 6, 12, 24, 48 h.
Split: chronological 70 / 10 / 20 % (LTSF standard), washout = first 240 steps of train; min-max
normalisation from pre-test data, clipped to [0, 1] (same protocol as every dataset here). QRC: frozen
design (g = 0.35, τ = 30 ms, 653 bins, D = 653), var0 → protons, var1 → carbons; one reservoir pass per
dataset, FID trace saved. Variants and their validation-only selection exactly as §11.
Bootstrap: moving-block, 2000 resamples, block 30 (exchange, days) / 168 (electricity, hours).

- **D1 (primary):** QRC delta_lags − QRC base, pooled test R², 95 % CI > 0 on **both** datasets.
- **D2 (descriptive):** QRC delta_lags vs tuned ESN (base and delta_lags readout), tuned LSTM, tuned linear
  AR, persistence (ŷ(t+h) = y(t)); test-input shift reported (mean test input in training s.d.).
All outcomes are reported.
