# Why the QRC wins, is comparable, or loses — factor analysis of the real-world benchmark

Data: `docs/QRC/data/qrc_realworld_factors.json` (from `backend/scripts/qrc_realworld_factors.py`) and
`docs/QRC/data/qrc_realworld_memory_function.json`; benchmark results in `QRC_RealWorld_Benchmark.md`.
QRC = simulated crotonic-9 NMR reservoir, τ = 30 ms, proton-FID FFT readout, learned encoding (3 seeds).
"Comparable" = 95% CI of the R² difference contains 0. Analysis is post-hoc and correlational over
9 datasets / 101 forecast targets; treat it as mechanism-guided evidence, not proof.

## 1. Outcomes and dataset characteristics

| Dataset | vs tuned ESN | vs tuned LSTM | best R² | linear AR R² | nonlinear gain | train size | train→test shift (sd) | input memory (steps) |
|---|---|---|---|---|---|---|---|---|
| Delhi, Kaggle station | comparable (QRC highest) | **QRC better** | 0.731 | 0.689 | +0.042 | 300 | 0.16 | 35 |
| Delhi, ERA5 | ESN better | LSTM better | 0.769 | 0.737 | +0.032 | 10,592 | 0.33 | 49 |
| Mumbai, ERA5 | ESN better | LSTM better | 0.706 | 0.666 | +0.040 | 10,592 | 0.31 | 48 |
| Bangkok, ERA5 | comparable | **QRC better** | 0.333 | 0.299 | +0.034 | 10,592 | 0.31 | 31 |
| Beijing, ERA5 | comparable | comparable | 0.651 | 0.635 | +0.016 | 10,592 | 0.06 | 47 |
| Berlin, ERA5 | comparable | LSTM better | 0.616 | 0.605 | +0.011 | 10,592 | 0.17 | 43 |
| Beijing station, hourly | ESN better | LSTM better | 0.931 | 0.930 | +0.001 | 25,804 | 0.11 | 1,379 |
| ETTh1, hourly | ESN better | LSTM better | 0.732 (linear AR) | 0.732 | −0.070 | 8,400 | **1.31** | 571 |
| Santa Fe laser | ESN better | LSTM better | 0.971 | 0.717 | **+0.254** | 6,000 | 0.00 | 2 |

*Nonlinear gain* = best nonlinear model − tuned linear autoregressive model (ridge on lagged inputs,
lags and regularisation chosen on validation; added for this analysis).

## 2. The factors

### F1 — Forecast horizon / memory length (the main factor against the ESN)
Across 101 targets (ETTh1 excluded), the QRC−ESN difference falls with horizon (Spearman −0.47):

| horizon (fraction of max) | targets | QRC − ESN (mean) | QRC ahead on |
|---|---|---|---|
| short (≤ 25 %) | 44 | −0.0001 | 43 % — **comparable** |
| medium | 30 | −0.022 | 30 % |
| long (≥ 60 %) | 27 | −0.041 | 7 % — **ESN better** |

Mechanism, measured directly (memory function = test R² of reconstructing the input L steps back from
the current features, ERA5 Delhi):

| lag (days) | 1 | 10 | 30 | 60 | 90 |
|---|---|---|---|---|---|
| QRC arcsin | 1.00 | 0.92 | 0.77 | 0.61 | 0.53 |
| QRC learned | 0.99 | 0.91 | 0.74 | 0.58 | 0.55 |
| tuned ESN (leak 0.3) | 1.00 | 0.97 | 0.94 | 0.91 | 0.88 |

The QRC is a **short-memory reservoir** relative to a tuned ESN, and its memory is not tunable like an
ESN's leak rate. Correction (2026-10-23): it is not simply T₂-limited — coherences last only ~7–11 steps
(T₂* 0.20–0.33 s, τ = 30 ms) yet the measured memory (R² 0.77 at 30 days) outlasts that, because proton
populations (T₁ ≈ 73–917 steps) reach the readout through the π/2 readout pulse and carbon populations
through the J-multiplet amplitudes. The likely limiter is that every input pulse (up to π/2 per spin)
partly overwrites the stored state — to be tested (see `QRC_Design_Concepts.md`, H1). Long horizons need weeks of context, so
the QRC falls behind exactly there; the worst case (Mumbai, monsoon-driven, long-horizon temperature
−0.17) is the most seasonal, long-memory task.
The flip side (Kaggle Delhi): the ESN selected on that small dataset used a very slow leak (0.05), whose
memory is long but **blurred** (only 0.72 R² at lag 1); the QRC's sharp short-term memory is why it has
the highest score there.

### F2 — Data size and noise (the main factor against the LSTM)
The QRC beats the tuned LSTM only where the LSTM is data- or signal-starved: **Kaggle Delhi** (300
training days) and **Bangkok** (lowest predictability, best R² 0.33 — noise-dominated). A fixed
reservoir with a closed-form ridge readout cannot overfit the way a trained recurrent network can.
With ≥ 6,000 training steps and a structured signal (ERA5 Delhi/Mumbai/Berlin, hourly data, Santa Fe)
the LSTM wins.

### F3 — Nonlinearity of the task
Most weather targets are nearly linear: a tuned linear AR model is within 0.01–0.04 R² of the best
nonlinear model (and is the best model on ETTh1). Where the task is strongly nonlinear (Santa Fe chaotic
laser, +0.25 over linear), the QRC captures most of it (+0.20) but less than the ESN (+0.24) or LSTM
(+0.25): at 9 spins its nonlinear feature map is less expressive than a 650–1000-node ESN.

### F4 — Distribution shift and feature boundedness (failure mode)
On ETTh1 the test period shifts regime (inputs 1.31 sd from the training mean). The QRC's FFT-magnitude
features are unbounded and drift far outside their training range (6 % beyond 4 sd, up to 16–29 sd),
and the lightly regularised readout chosen on in-distribution validation data amplifies the drift
(R² −0.2 to −8.6). The ESN's tanh-bounded states stay in range (0 % beyond 4 sd) and degrade gracefully.

### F5 — Strong simple baselines
A tuned linear AR model is competitive on every weather/hourly dataset here. Any reservoir claim should
include it; without it, small nonlinear gains can be mistaken for reservoir quality.

## 3. Practical guidance (from this evidence)

| Situation | Prefer | Why |
|---|---|---|
| Little training data, noisy signal, short horizons | QRC ≈ ESN (both fixed-reservoir + linear readout) | no overfitting; QRC's sharp short memory helps; LSTM is data-starved |
| Long horizons / seasonal context | tuned ESN (tune the leak) or LSTM | needs long, tunable memory; QRC memory is physics-limited |
| Large data, strongly structured or nonlinear signal | LSTM (or a large ESN) | trainable dynamics win with enough data |
| Possible regime shift between train and deployment | ESN or linear AR, or a QRC with bounded features | unbounded QRC features extrapolate badly |
| Any case | always include a tuned linear AR | often within 0.01–0.04 R² of the best model |

## 4. Implications for QRC design (testable next steps)
1. **Extend memory**: tune the encoding strength (input overwrite), protect long-T₁ spins, add a compact
   population readout, or choose τ; the ESN result says tunable memory is what wins long horizons.
   Full concepts and pre-registered tests: `QRC_Design_Concepts.md`.
2. **Bound the features**: rank, log or clipped-z transforms of the FFT readout for robustness to
   regime shift (test as a pre-registered change, not tuned on these test sets).
3. **Expressiveness**: larger or more strongly coupled reservoirs for strongly nonlinear tasks
   (Santa Fe), now feasible to 10–12 spins with the block propagator.
4. **Selection hygiene**: validation periods must look like deployment; a single short validation slice
   misled ESN selection earlier (slow-leak failures) and can mislead QRC readout regularisation (ETTh1).

## Caveats
Nine datasets and 101 targets from one QRC system; correlational, post-hoc. ERA5 is reanalysis (smoother
than station data). The memory function on smooth series includes the signal's own autocorrelation, but
the comparison between models on the same data is fair.
