# QRC vs fair classical baselines — combined comparison (2026-10-22)

Test NMSE (lower is better), mean ± s.d. over data seeds 1-3; same tasks, T=1500,
washout 30, leakage-free 50/25/25 split for every model. QRC rows are regime B
(learned per-spin encoding + ridge readout) on the physics-informed reduced-FID
readout; classical baselines are tuned on validation (SOP §6.5).

| System (learned per-spin encoding) | NARMA-2 | NARMA-10 | Mackey-Glass h=10 |
|---|---|---|---|
| 6-spin ×2 (abstract Ising), τ=30 ms | 0.0052 | 0.12 | 0.00155 |
| crotonic-9 (real), τ=30 ms | 0.007919 ± 0.0035 | 0.2766 ± 0.069 | 0.01024 ± 0.0031 |
| **dccb7 (real), τ=100 ms** | **0.002756 ± 0.00046** | — | **0.002515 ± 0.00057** |
| dccb7, τ=30 ms | 0.0861 ± 0.042 | — | 0.008613 ± 0.0055 |
| dccb7, τ=100 ms, **J=0 ablation** | 0.3465 ± 0.029 | — | 0.09506 ± 0.011 |
| *arcsin encoding (dccb7 τ=100 / crotonic)* | *0.0991 / 0.317* | *— / 0.506* | *0.0193 / 0.0894* |
| *τ→0 ablation, learned (dccb7 τ=100 / τ=30 / crotonic)* | *1.00 / 1.00 / 1.00* | *— / — / 1.00* | *0.88 / 0.88 / 0.88* |
| Tuned LSTM (seeds 1-3) | 0.004351 ± 0.0027 | 0.2453 ± 0.022 | 0.002738 ± 0.00059 |
| Size-matched tuned ESN (≤350 nodes) | 2.5e-05 | 0.02 | 1.1e-05 |
| **Fully tuned ESN** (≤1000 nodes) | **1.1e-05** | **0.011** | **2.8e-06** |

Sources: `qrc_crotonic9_campaign.json`, `qrc_dccb7_tau100.json`, `qrc_dccb7_tau30.json`,
`qrc_dccb7_tau100_j0.json`, `qrc_classical_fair_esn.json`, `qrc_classical_fair_lstm.json`
(6-spin values from the confirmed 6-spin campaigns). dccb7 was run on NARMA-2 and
Mackey-Glass only.

## Verdict
1. **Learned encoding vs arcsin:** wins on every seed of every system and task; every
   τ→0 ablation collapses to the mean predictor → the gain is reservoir-carried.
2. **Coupling-mediated:** with J=0 (same readout basis) dccb7 is ~38× (MG) and ~120×
   (NARMA-2) worse → the spin-spin coupling carries the computation.
3. **Physical design matters:** dccb7 at τ=100 ms (J·τ≈6) beats τ=30 ms (J·τ≈1.7) by
   ~3× (MG) and ~30× (NARMA-2), and is the best real molecule.
4. **vs classical:** the *best* QRC configuration per task beats a fairly tuned LSTM
   (NARMA-2: dccb7 0.0028 vs 0.0044; NARMA-10: 6-spin 0.120 vs 0.245; Mackey-Glass:
   6-spin 0.00155 and dccb7 0.0025 vs 0.0027) — but not every system does (crotonic-9
   loses to it on all three; the 6-spin on NARMA-2). A fairly tuned ESN, even a
   size-matched one, beats every QRC by 1-3 orders of magnitude on these synthetic
   benchmarks. On real-world weather (separate study) the
   learned QRC is ahead of a tuned ESN on 3/3 seeds, not yet significant.
