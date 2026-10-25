# Fix the readout before redesigning the spins

The fastest and best-supported improvements to our 9-spin crotonic-acid reservoir are on the classical readout side, and they can be tested now on the saved FID traces with no GPU time. The ETTh1/ETTh2 failure looks like a failure to extrapolate the level of a shifted series, not a shortage of memory or nonlinearity. The 653 |FFT| features are non-negative, bounded and discard phase, and they carry no linear copy of the input. The forecasting literature fixes exactly this with last-value re-centring (NLinear) and with linear-plus-nonlinear feature concatenation (hybrid RC-NGRC) ([Zeng et al. 2023](https://ar5iv.labs.arxiv.org/html/2205.13504); [Chepuri et al. 2024](https://arxiv.org/html/2403.18953v2)). Delay-embedding the saved features will measure how much of the lag-90 memory gap (R² 0.53 vs ESN 0.88) can be closed classically. A kernel readout is the one lever that Hou et al. showed lifts an NMR reservoir from about ESN(1000) to above ESN(10000) ([arXiv:2508.12383](https://arxiv.org/html/2508.12383v2)). The physically new ideas need fresh reservoir evolution. The strongest are population (z) readout and write/store separation using the long-T1 degrees of freedom, because T1, not T2, carries memory in this system. Spin echo, more spins and learned pulses rank low: our Markovian simulation cannot show an echo gain honestly, 12 spins cost about 64x as much, and the closest prior attempt (learned per-spin encoding) gave nothing. Every readout-side gain must also be given to the ESN, LSTM and AR baselines. Otherwise it repeats the positivity bias the benchmarking literature warns about. The realistic aim is "closes the 0.006 R² gap under matched controls", not "quantum advantage".

## We are a tuned-baseline stress test of Hou et al. 2026, with one encoding caveat

The closest prior work is the experimental reservoir of **Hou, Hua, … Peng, Du, Phys. Rev. Lett. 136, 120602 (2026), arXiv:2508.12383**. Its setup is ([arXiv HTML](https://arxiv.org/html/2508.12383v2); [PRL](https://journals.aps.org/prl/abstract/10.1103/r8ww-qw7j)):

- **System:** 13C-labelled crotonic acid, 9 spins (4 13C + 5 1H), on a 400 MHz Bruker spectrometer.
- **Encoding:** an Rx pulse-area encoding on the protons.
- **Step time:** τ = 0.03 s for weather.
- **Readout:** a proton FID, Fourier-transformed, reduced to **653 spectral-peak features**, z-scored, with a ridge readout under 10-fold CV.
- **Washout:** a "rewinding" protocol that replays inputs k−n_wo..k, with n_wo = 100 to 1000.
- **Data:** a single real-data task, Delhi daily climate 2013–2017 with 600 training and 600 test steps, and one readout per horizon.
- **Result:** QRC roughly matches ESN(1000) on temperature and beats the ESN average at longer horizons. It beats ESN(10000) only with RBF-SVR post-processing, and roughly matches ESN(10000) on humidity.
- **Baselines:** ESNs of 500 to 10000 nodes **averaged over 100 random realisations**. There is **no linear/AR baseline and no treatment of data drift** ([arXiv:2508.12383](https://arxiv.org/html/2508.12383v2)).

The research notes disagree on how far the ESN was tuned. One note records a grid search over connectivity and spectral radius (ρ = 0.99). Another found no tuning described in the main text, because the appendix was not read. Check the appendix before characterising their baseline in print.

Against that, our study adds pre-registration, six untouched cities, tuned ESN/LSTM/linear-AR baselines, paired significance tests, and two stress tests the PRL never ran: a drift-heavy dataset (ETT) and a chaotic lab signal (Santa Fe). Our result is non-inferior, with a tuned ESN ahead by 0.006 R² and significantly so in 4 of 6 cities. That is the same pattern as the most careful recent comparisons:

- **Pandey (2026).** Matched size and tuning effort leave "a well-tuned classical network" slightly ahead, even after feedback rescues the quantum reservoir ([arXiv:2607.09905](https://arxiv.org/abs/2607.09905)).
- **IBM Heron multivariate QRC.** It loses to NVAR and a clustered ESN on Lorenz and on a (synthetic, ODE-generated) "ENSO" set ([arXiv:2510.13634](https://arxiv.org/html/2510.13634v1)).
- **Real hydrology data.** QRC reached an NSE of 0.69–0.74, and the ESN had slightly lower errors throughout ([Springer](https://link.springer.com/article/10.1007/s00477-026-03383-z)).
- **Survey view.** Tariq et al.'s 2026 survey states that current results "do not establish a broad quantum advantage over well matched classical reservoirs" ([arXiv:2607.18552](https://arxiv.org/abs/2607.18552)).

The honest framing is that under a tuned-baseline protocol, the PRL-style NMR reservoir is non-inferior but not superior. That gap is a publishable contribution in its own right, and the notes found no study that does it.

**The encoding discrepancy must be resolved before we claim replication.** The arXiv HTML of Hou et al. writes the encoding as U = ∏ R_x[arcsin(s_k)] on protons 5–9, with s in [0,1] ([arXiv HTML v2](https://arxiv.org/html/2508.12383v2)). We use θ = g·arcsin(√s) with g = 0.35. These differ in two ways that matter:

- **Different nonlinearity.** arcsin(s) makes sin θ, the transverse component, linear in s. arcsin(√s) makes sin²θ, the population transfer, linear in s.
- **Much smaller angles.** With g = 0.35, our maximum flip angle is about 0.35·π/2 ≈ 0.55 rad (≈31°), while Hou's reaches π/2. Our reservoir is therefore driven in a much smaller-angle regime.

That cuts both ways. Small angles should reduce overwrite, yet we still diagnose overwrite. They also reduce the encoding-generated nonlinearity, which is the quantity Schütte et al. show bounds the number of independent nonlinear functions a QRC can produce, regardless of Hilbert-space size ([arXiv:2501.15528](https://arxiv.org/html/2501.15528v2)). That is a plausible contributor to our weaker Santa Fe performance. Hou also encodes humidity on the **carbons** (a second channel, φ) while temperature goes on the protons, and caps input dimension at two ([arXiv HTML v2](https://arxiv.org/html/2508.12383v2)). If our multivariate T+RH run pulses only protons, that is a second protocol difference. Two further things are unverified: whether our relaxation parameters (T2* 0.20–0.33 s, T1 2.2–27.5 s) come from Hou's Table 2, which is an image the notes could not extract, and whether their quoted T2 values are really T2*, as they state. In the manuscript, describe our system as "a PRL-style reservoir with a modified encoding", unless an arcsin(s), g = 1 control run reproduces their regime.

## Readout-side fixes can test every gap without new reservoir runs

### The ETT failure is a level-extrapolation problem the readout can fix

On ETTh1/ETTh2 the test period is shifted by about 1.3 s.d. and QRC scores −0.15 to 0.36 R², against 0.75–0.81 for ESN/LSTM/AR. The best-documented cause is the one Zeng et al. identified on exactly these datasets. Inputs normalised with training statistics push predictions outside the test distribution. **NLinear's last-value subtraction "greatly improves" on plain Linear on ETTh1, ETTh2 and ILI**, while on unshifted data the two tie ([Zeng et al., AAAI 2023](https://ar5iv.labs.arxiv.org/html/2205.13504)).

Our bounded-feature attempts (tanh, clip, rank) bounded the *features* but left the *target level* untouched, which explains why they did nothing. Three readout-side transfers follow, all on saved traces:

- **Level-free target.** Predict y(t+h) − y(t) and add the level back.
- **Linear pass-through.** Append raw input lags and a bias to the 653 FID features.
- **Residual stacking.** Fit the tuned AR first, then fit ridge on the FID features to the AR residual.

The concatenation idea has strong support, though only on synthetic data. Hybrid RC-NGRC, with reservoir state ⊕ [linear delayed inputs ⊕ monomials] in one ridge fit, roughly doubled Lorenz valid prediction time: **4.13 Lyapunov times vs 2.06 for NG-RC and 0.98 for RC**. A 100-node hybrid matched a 500-node pure RC ([Chepuri et al., Chaos 2024](https://arxiv.org/html/2403.18953v2)).

The notes found no paper applying RevIN, NLinear-style re-centring or differencing to any physical or quantum reservoir. A clean result here would be new. RevIN itself is a weaker bet than last-value re-centring. A 2026 re-examination finds its affine layer unhelpful and its benefit inconsistent, and on Traffic it *increased* distribution distance ([Berthelier et al.](https://arxiv.org/html/2603.11869v1)). Forgetting-factor RLS readouts are the established adaptation tool for physical reservoirs, for example photonic ones ([arXiv:2012.10613](https://arxiv.org/pdf/2012.10613)). But they cannot fix a level shift that the features cannot represent, so they come second. Input-side differencing or windowed normalisation before the arcsin encoding would keep test angles inside the training range. It changes the drive, however, so it needs new evolution.

### Delay embedding will show how much of the lag-90 memory gap is classical

The lag-90 deficit is consistent with expected physics. The FID reads transverse coherences, and with T2* ≈ 0.2–0.33 s at τ = 30 ms those last only about 7–11 steps. Markovian Lindblad reservoirs provably lose past information exponentially ([Sannia et al., npj QI 2026, arXiv:2505.02491](https://arxiv.org/abs/2505.02491)). In erase-and-write protocols, the information loss comes entirely from the injection map ([Keenan & Zambrini, arXiv:2608.07677](https://arxiv.org/html/2608.07677)), which is our "input overwrite".

The cheapest experiment is to PCA-compress the saved features and concatenate [X_t, X_{t−7}, X_{t−15}, X_{t−30}, X_{t−60}, X_{t−90}]. Three cautions apply:

- **The baselines need the same delays.** A 2026 theorem shows that reservoirs whose measured features evolve linearly cannot create fixed-delay capacity beyond what the preprocessed input already holds. Summed capacity can also hide per-delay limits ([Nokkala et al., arXiv:2605.29071](https://arxiv.org/abs/2605.29071)). Our dynamics are linear in ρ, with the input pulse as the only nonlinearity. So the per-lag memory function we already use is the right diagnostic, and the ESN must receive the same delayed states.
- **The result must be labelled.** This is "quantum + classical memory".
- **The result is itself a finding.** It separates quantum-dynamical memory from readout-side memory on real forecasting data. The notes found no study that has done this.

### A kernel readout is the most direct published lever for the chaotic and weather gaps

Hou et al.'s RBF-SVR stage moved temperature from about ESN(1000) to above ESN(10000) ([arXiv:2508.12383](https://arxiv.org/html/2508.12383v2)). That is the one real-data intervention with a reported effect on this exact platform. For Santa Fe, hybrid concatenation with NG-RC monomials is the second offline option. The NG-RC "catch-22" warns that polynomial libraries fail badly when the true nonlinearity is not in the library ([Zhang & Cornelius, arXiv:2210.10211](https://ar5iv.labs.arxiv.org/html/2210.10211)), so the reservoir block should stay in the fit. Fairness is non-negotiable here. Bowles et al. found that results swing from random to near-perfect across hyperparameters, and that out-of-the-box classical models ranked first once everyone got the same CV grid ([arXiv:2403.07059](https://arxiv.org/html/2403.07059v1)). An SVR on QRC features must be compared against an SVR on ESN states.

## Long memory lives in populations (T1), so the physical redesigns target z-order

### Population readout and write/store separation lead the physical changes

The physical lever with the best support is the gap between lifetimes. At τ = 30 ms, populations with T1 = 2.2–27.5 s survive about 70–900 steps, while transverse coherence survives about 7–11. Hou's appendix finds that **unitary dynamics give zero STM, T2 dephasing alone gives none, and T1 amplitude damping is what produces fading memory**. STM peaks at an intermediate τ ([arXiv:2508.12383 appendix](https://arxiv.org/html/2508.12383v2)). Our FID-only readout therefore looks at the short-lived part of the state.

**Population (z) readout** fixes this. Apply a π/2 read pulse to a *copy* of the state, which is free in simulation because we have no back-action, and append the z-spectrum or 13C-channel features. The physical precedent is Negoro et al.'s solid-state NMR reservoir, which read the **13C Z component** ([arXiv:1806.10910](https://ar5iv.labs.arxiv.org/html/1806.10910)). On a real spectrometer this costs a separate scan or rewinding, and 13C has lower SNR, which is why Hou read only protons.

**Write/store separation** attacks overwrite at its source. The template is Monomi et al.'s NMR-oriented proposal, with 2 input qubits and 4 memory qubits plus feedback rotations on memory. There, weak measurement plus feedback beat projective feedback on STM and NARMA5–20 ([arXiv:2503.17939](https://arxiv.org/html/2503.17939)). In crotonic acid the carbons are *already* an unpulsed, unread store in Hou's single-input design. Pushing further means:

- **Selective writes.** Write with a selective soft pulse on a proton subset, such as the methyl group, and leave the other spins as store.
- **Carbon store with readout.** Keep the carbons as a long-T1 store and read their populations.

This is also the practical form of Sannia's non-Markovian embedding: an unread spin "environment" feeding information back to the read spins ([arXiv:2505.02491](https://arxiv.org/abs/2505.02491)). Sasaki et al. caution that strong system-environment backflow slows STM decay but can break the echo-state property, which makes linear readouts unstable ([arXiv:2505.14450](https://arxiv.org/abs/2505.14450)). The washout length must therefore be re-validated.

The memory-nonlinearity trade-off literature points the same way. Linear long-delay memory favours slow loss and weak scrambling, while high-degree capacity needs the opposite ([Keenan & Zambrini](https://arxiv.org/html/2608.07677); [Cindrak et al., arXiv:2603.21371](https://arxiv.org/abs/2603.21371)). A low-angle store channel combined with the existing write channel is therefore the trade-off-aware design.

### Spatial multiplexing and output feedback are the next tier

**Spatial multiplexing** means running 2–4 reservoir copies on the same input, at different g, different τ (30 and 300 ms) or different pulsed nucleus, then concatenating the spectra into one ridge. It improved memory capacity and NARMA monotonically in the original ensemble-NMR proposal, although only on synthetic tasks against no classical baseline ([Nakajima et al., arXiv:1803.04574](https://arxiv.org/html/1803.04574v1)). This differs from the τ variants we already tried, which *swapped* τ rather than concatenating. Multiplexing more time points of the same timescale saturates through collinearity ([arXiv:2510.00171](https://arxiv.org/html/2510.00171v1); [arXiv:2609.28441](https://arxiv.org/pdf/2609.28441)), and that explains why our compact readouts did nothing. What we need is diversity of timescale.

**Output feedback** is the second option. Set θ_t = g·arcsin(√s_t) + a_fb·f(ŷ_{t−1}), or drive a 13C pulse from the previous readout. In Kobayashi et al., memory peaks at the edge of chaos as a function of feedback strength ([PRX Quantum 5, 040325, arXiv:2406.15783](https://arxiv.org/abs/2406.15783)). In Pandey's matched comparison, feedback made a useless reservoir work but did not beat a tuned classical network ([arXiv:2607.09905](https://arxiv.org/abs/2607.09905)). Feedback is sequential, so it needs new evolution, and an ESN with output feedback is the fair control.

### Spin echo, more spins and learned pulses rank low

These are deprioritised for specific reasons, not dismissed:

- **Spin echo / CPMG.** On hardware, refocusing would extend transverse lifetime from T2* (0.2–0.33 s) to T2 (crotonic acid ≈0.7–1.1 s), about 3x or roughly 23–37 steps ([arXiv:1210.4760](https://arxiv.org/pdf/1210.4760); [IMSERC](https://imserc.northwestern.edu/guide/eNMR/eNMR1D/cpmg.html)). But if our Lindblad model folds inhomogeneity into Markovian dephasing, a simulated echo recovers nothing, and editing the rate would be a modelling choice rather than a technique. An honest test needs an explicit field-offset ensemble. Even then, Hou's finding that dephasing contributes no STM caps the benefit, and π pulses also refocus the chemical-shift and heteronuclear-J dynamics that generate features.
- **More spins and stronger couplings.** The neutral-atom study argues that physical size beats multiplexing ([arXiv:2609.28441](https://arxiv.org/pdf/2609.28441)). But a 12-spin exact Lindblad simulation is about 64x the 9-spin cost. Memory is T1-limited rather than size-limited. 19F molecules with roughly 2x stronger J ([quant-ph/9905041](https://arxiv.org/pdf/quant-ph/9905041)) add mixing, not memory. On the critical side, the Schütte bound means more Hilbert space does not raise the nonlinear-function count without richer encoding ([arXiv:2501.15528](https://arxiv.org/html/2501.15528v2)).
- **GRAPE / learned control pulses.** No QRC precedent exists ([arXiv:2412.05856](https://arxiv.org/abs/2412.05856v3) is the closest NMR control work). Our learned per-spin encoding was null, and learned pulses erode the "natural dynamics" story. The notes also found no QRC use of singlet states, which last 10–37x T1 ([Levitt, Annu. Rev. Phys. Chem. 2012](https://eprints.soton.ac.uk/347311)). Singlets would require spin-locking the strongly inequivalent crotonic carbons, which conflicts with free evolution and with fading memory.

### Technique catalogue

| Technique | Evidence quality | NMR feasibility | Memory (lag-90) | Drift (ETT) | Chaos (Santa Fe) / weather gap |
|---|---|---|---|---|---|
| Last-value re-centring / Δ-target | REAL (ETTh1/2), linear & deep models, tuned; never on a reservoir | Readout only | none | **high** | low |
| Linear lags ⊕ FID (hybrid RC-NGRC) | SYN chaotic, tuned ridge; ESN only | Readout only | medium | **high** | medium |
| AR-residual stacking | REAL, low-quality venues; no reservoir | Readout only | low | high | low–medium (weather) |
| Delay-embedded FID PCs | Theory (Nokkala) + our metric | Readout only | **high** (classical memory) | low | low |
| RBF-SVR / kernel readout | REAL (Hou, single city), untuned ESN | Readout only | low | low | **medium–high** |
| Population (z) / 13C readout | SYN (Hou appendix STM); Negoro EXP non-temporal | Extra scans / low 13C SNR on hardware; free in sim | **high** | low | medium |
| Write/store separation | SYN, CB− (Monomi, Sannia, Sasaki) | Selective pulses standard | **high** | low | medium |
| Spatial multiplex (multi-g/τ/nucleus) | SYN, CB− (Nakajima 2019) | k× scans or molecules | medium–high | low | medium |
| Output feedback | SYN, CB− (Kobayashi); matched sim (Pandey: still behind) | Classical feedback between scans | medium | low–medium | medium |
| Input-side differencing / windowed norm | REAL indirectly (NLinear logic) | Changes drive only | none | medium–high | low |
| Spin echo / CPMG | No QRC evidence; lifetime ratios only | Trivial on hardware; not meaningful in Markovian sim | low | none | low (may reduce) |
| More spins / 19F | SYN (neutral atoms); no NMR QRC | Molecule synthesis hard; 64× sim cost for 12 | low | none | medium |
| Learned control pulses | None | GRAPE mature | unknown | none | unknown |

## Ranked next steps, with offline/GPU status and the matching baseline control

Ranking is by expected effect on our stated gaps per unit cost and by evidence quality. Items marked **GPU** need new reservoir evolution. Under the project rules, each one requires an explicit go-ahead with a wall-clock estimate, a launch via the launcher API, and the raw FID waveform persisted to `artifacts/traces/*.npz` before any metric is computed. Items marked **offline** run on existing traces. Results on any readout other than the standard 653-feature FID spectrum must state that readout and its feature count D.

| Rank | Step | Compute | Gap targeted | Fairness control the baselines must also receive |
|---|---|---|---|---|
| 1 | Level handling on ETT: Δ-target / last-value re-centring, plus linear-lag pass-through ⊕ 653 FID, with per-block ridge penalties | **Offline** | Drift | The same re-centring and lag block for the ESN, LSTM and AR. Add NLinear/DLinear and persistence baselines. Report MSE on the standard LTSF split as well as R² |
| 2 | AR-residual stacking: ŷ = AR + ridge(FID → AR residual); report ΔR² over AR per city | **Offline** | Weather gap, drift | Stack the ESN on the same AR residual, with the same CV budget |
| 3 | Delay-embedded FID PCs (lags 7–90); per-lag memory function with and without | **Offline** | Lag-90 memory | The ESN gets identical delayed states. Label the result "quantum + classical memory" |
| 4 | Kernel readout (RBF-SVR / kernel ridge) on 653 features; Santa Fe and weather | **Offline** | Chaos, weather gap | The same kernel readout on ESN states, with a matched hyperparameter grid |
| 5 | Encoding-replication control: arcsin(s), g = 1, humidity on 13C, as in Hou | **GPU** (short; quote measured time) | Positioning, expressiveness | None needed (quantum-side control); report it beside the g = 0.35 result |
| 6 | Population (z) and 13C-channel readout appended to the FID spectrum | **GPU** (compute ⟨Iz⟩, ⟨Sz⟩ and 2-spin zz alongside the FID in the same pass) | Lag-90 memory | A capacity-matched ESN (readout D equal to the new total D). Label the new D |
| 7 | Write/store separation: selective write on a proton subset or low-angle store channel; re-validate washout and echo-state property | **GPU** | Memory (overwrite) | A matched-D ESN and a separable / classical-spin reservoir control |
| 8 | Spatial multiplex: concatenate a slow copy (τ = 300 ms or low g) with the standard copy. Feedback sweep of a_fb as a follow-on | **Offline** if same-input traces at other g/τ already exist; otherwise **GPU** | Memory, chaos | An ESN ensemble or ESN at matched total D. For feedback, an ESN with output feedback |

Three controls apply across all eight steps:

- **Same tuning budget.** Give the baselines the same hyperparameter-search budget and the same seeds ([Pandey 2026](https://arxiv.org/abs/2607.09905); [Bowles et al.](https://arxiv.org/html/2403.07059v1)).
- **Matched readout dimension.** The ESN readout dimension should be near D = 653, or equal to the new D wherever we add features. Advantage claims elsewhere rest on 50-node ESNs ([arXiv:2505.13933](https://arxiv.org/html/2505.13933v1)).
- **A de-quantised control.** Run a separable or non-interacting spin reservoir before attributing any gain to correlations. In the 12-model QML benchmark, removing entanglement gave similar or better performance for most models ([arXiv:2403.07059](https://arxiv.org/html/2403.07059v1)).

## Conclusion

The three weaknesses have different causes, and only one of them is fundamentally quantum. Drift is a readout-design omission: no linear path from the input level to the output. Long-horizon memory is a mismatch between what we read (T2*-limited coherences) and where the memory lives (T1-limited populations). The chaotic-signal deficit traces back mainly to the encoding, including our small-angle arcsin(√s) choice, rather than to spin count. So the cheapest experiments are also the most informative. If steps 1–4 close the 0.006 R² gap only when the baselines are denied the same treatment, that is a negative result about quantum value and should be reported as one. If z-readout or write/store separation then raises lag-90 memory with the ESN given a matched readout, we would have the first real-data evidence separating quantum-dynamical memory from classical readout memory.

Strategically, the defensible paper is not "NMR QRC beats ESN". It is "a PRL-style NMR reservoir, re-encoded and stress-tested under pre-registered, tuned, multi-city protocols, is non-inferior on stationary weather. It fails under drift only for a readout reason we can name and fix. Its memory ceiling sits where the physics (T1 vs T2*) says it should." That framing turns our gaps into mechanism, and it turns the encoding discrepancy from a liability into a controlled variable.
