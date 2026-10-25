# Honest benchmarks: QRC / QML for time series vs tuned classical baselines (2019-2026, emphasis 2023-2026)

Scope note: ~17 tool calls; most items verified from arXiv abstracts or HTML full text (partial reads). Where only an abstract was read, this is stated. "Tuned?" = whether the classical baseline got a documented hyperparameter search comparable to the quantum model's.

**Most important finding for CiRA:** the closest published analogue to our setup is an *experimental* NMR QRC: Hou, Hua, ..., Xinhua Peng, Jiangfeng Du, "High-Accuracy Temporal Prediction via Experimental Quantum Reservoir Computing in Correlated Spins", arXiv:2508.12383, **Phys. Rev. Lett. 136, 120602 (2026)**. It uses 9 spins in 13C-labelled crotonic acid on a 400 MHz Bruker spectrometer, an FID -> Fourier transform -> spectral-peak readout with **653 features** (the same D as our SOP), and **Delhi daily climate data (2013-2017)** for temperature and humidity forecasting. Our project reproduces and extends that paper's protocol. Its ESN comparison is much weaker than ours (details below), so our pre-registered, multi-city, tuned-baseline result is a real methodological step beyond it.

## Q1. Which papers claim a QRC advantage on real-world forecasting, and does it survive a tuned ESN?

### Takeaway
No paper I found shows a QRC advantage on real-world forecasting that holds up against a tuned ESN of matched capacity, checked with statistics across several datasets. Every "advantage" or "comparable" claim on real data rests on one of these: a single dataset, ESNs averaged over random realisations instead of tuned, baselines copied from another paper, or a simulated benchmark presented as a real one. When the classical baseline is tuned as carefully as the quantum model, the reported gap disappears or reverses. That matches our result (ESN ahead by about 0.006 R^2).

### Cited Findings
**NMR, correlated spins (PRL 2026), closest analogue**
- Platform and features: 9-spin NMR reservoir (4 13C + 5 1H in crotonic acid). Readout is protons only; carbons act as an inaccessible bath. The time-multiplexed readout Fourier-transforms the FID and selects 653 spectral-peak features. The single-time readout uses 9 Pauli expectation values — [arXiv:2508.12383 HTML](https://arxiv.org/html/2508.12383)
- Data and protocol: daily climate data for Delhi, 2013-2017, predicting temperature and humidity. 374 washout, 600 train and 600 test steps. Input interval tau = 0.03 s. Multi-step forecasts use the direct strategy, with one readout per horizon ("multitasking") — [arXiv:2508.12383 HTML](https://arxiv.org/html/2508.12383)
- Baselines: ESNs with 500, 1000, 5000 and 10000 nodes, each **averaged over 100 random realisations**. The main text does not describe any hyperparameter tuning (the ESN appendix was not in the part I read) — [arXiv:2508.12383 HTML](https://arxiv.org/html/2508.12383)
- Reported results, qualitative only (no table of weather errors in the main text): temperature QRC ~ ESN(1000), and better than the *average* ESN(1000) at longer horizons. With SVR post-processing, QRC beats ESN(10000). Humidity QRC ~ ESN(10000). The abstract claims errors 1-2 orders of magnitude lower than previous QRC *experiments*, a comparison with other quantum experiments, not with classical models — [arXiv:2508.12383](https://arxiv.org/abs/2508.12383)
- Caveat: one city, one split, and a baseline averaged over random draws rather than tuned. This is a "comparable to an untuned ESN" result, not evidence of advantage.

**Finance: S&P 500 realized volatility (Li, Mukhopadhyay, Bayat, Habibnia; Phys. Rev. Research 2025)**
- Data: monthly S&P 500 realized volatility, Feb 1950 to Dec 2017, 815 observations. 245 rolling one-step out-of-sample forecasts (1997-2017) — [arXiv:2505.13933 HTML](https://arxiv.org/html/2505.13933v1)
- Model: 10-qubit transverse-field Ising reservoir (7 input + 3 memory qubits, chosen by forward selection), 3 lags, ridge readout. It appears to be simulated, not run on hardware — [arXiv:2505.13933 HTML](https://arxiv.org/html/2505.13933v1)
- Baselines: AR1, AR3, ARMAX, HAR, HARX, LSTM (2x60), LSTMX, and classical RC/ESN with only **50 neurons** (RCX 20), leak 0.6, spectral radius 0.9. Hyperparameters chosen by "time-series-aware cross-validation". Evaluation uses MCS (alpha = 0.05) and Diebold-Mariano tests — [arXiv:2505.13933 HTML](https://arxiv.org/html/2505.13933v1)
- Claim: the quantum reservoir "consistently outperforms benchmark models" — [Phys. Rev. Research abstract](https://journals.aps.org/prresearch/abstract/10.1103/rbj7-4wnq). Caveat: the ESN is very small (50 nodes) and the quantum side got feature and architecture selection. Its rigour (MCS/DM on 245 forecasts) is otherwise good. I did not read the numeric results section.

**Finance: QLSTM vs QRC (2026 preprint)**
- On real financial series, quantum models roughly match classical LSTM/RC baselines on univariate inputs and do modestly better on multivariate, correlated inputs. Only the abstract or summary was seen; tuning is unknown — [arXiv:2605.02656](https://arxiv.org/pdf/2605.02656)

**IBM hardware multivariate QRC (Hamhoum, Cherkaoui et al., 2025)**
- Hardware: IBM Heron R2 (156-qubit chip), 12-qubit configuration, about 1,858 gates, 4,096 shots, no error mitigation — [arXiv:2510.13634](https://arxiv.org/html/2510.13634v1)
- Data: Lorenz-63 and an "ENSO" set. The ENSO set is generated from a **three-variable conceptual ENSO ODE model**, so it is not observational climate data, although it is often described as climate data — [arXiv:2510.13634](https://arxiv.org/html/2510.13634v1)
- Baselines were **not retrained**. RMSEs were digitised from plots in Shahi et al. 2022. RMSE results: Lorenz NVAR 0.042 < MTS-QRC 0.093 < ESN 0.168. ENSO clustered ESN 0.007 < NVAR 0.014 < ESN 0.018 < MTS-QRC 0.060. So **tuned classical models (NVAR, clustered ESN) win**, and QRC beats only LSTM/GRU — [arXiv:2510.13634](https://arxiv.org/html/2510.13634v1)
- On ENSO, hardware MSE (0.0030) was lower than the noiseless simulator (0.0036). The authors suggest, without confirming it, that noise acts as an implicit regulariser (effective rank 11 -> 10) — [arXiv:2510.13634](https://arxiv.org/html/2510.13634v1)

**Retail demand (Walmart M5), variational quantum reservoir (Nonlinear Dynamics 2026)**
- Compares ESNs with hybrid quantum-classical reservoirs under a *shared* hyperparameter protocol across smooth, erratic, intermittent and lumpy demand regimes. I did not see the headline numbers — [Springer](https://link.springer.com/article/10.1007/s11071-026-12707-1)

**Neutral-atom hardware (QuEra/Harvard, Kornjača et al. 2024)**
- The only time-series task is **Santa Fe laser one-step prediction**: a real chaotic lab signal, comparing encodings only. NMSE values are given only in figures. There is no claim of beating a classical model on time series — [arXiv:2407.02553](https://arxiv.org/html/2407.02553v1)

**Gap-closing study (Pandey, July 2026)**
- With matched size and matched tuning effort, using exact simulation up to 11 qubits: (i) "extra quantum measurements add nothing beyond a classical formula of the same size"; (ii) a feedback loop turns a useless quantum reservoir into a working predictor, but "a well-tuned classical network still predicts slightly more accurately". The gap is reported as statistically reliable. It offers a reusable matched-capacity/matched-tuning checklist. Single-author preprint, tasks not named in the abstract — [arXiv:2607.09905](https://arxiv.org/abs/2607.09905)

### Inferences
- Our finding (QRC non-inferior; tuned ESN slightly ahead at -0.006 R^2, significant in 4/6 cities) is the same *pattern* as Pandey 2026 and the IBM ENSO paper. Once baselines are tuned, QRC lands close to but just behind the best classical model. That is the honest current state of the field, not a failure of our system.
- The PRL 2026 NMR paper is the natural benchmark to cite and to "stress-test". Our study adds pre-registration, 6 untouched cities, tuned ESN/LSTM/AR baselines and paired significance tests, none of which that paper reports. Framing: "under a tuned-baseline protocol, the PRL-style NMR reservoir is non-inferior but not superior".
- Volatility-forecasting claims of advantage use 50-node ESNs. A fair replication would need ESNs at matched feature count (our D = 653, so roughly 650-node ESNs or matched readout dimension).

### Gaps
- No QRC papers found on real energy-load, solar, traffic or ECG forecasting with tuned baselines. One search turned up none. The closest real-world items are finance, Delhi weather and the Santa Fe laser.
- Exact numeric weather errors and the ESN tuning protocol of the PRL 2026 paper (appendix not read).
- Results of the M5 demand paper and the neurodynamical forecasting preprint ([arXiv:2608.00139](https://arxiv.org/html/2608.00139v1), whose abstract says the ESN was tuned separately with matched readouts) were not read.
- Digital QRC for ATM cash time series ([arXiv:2606.04686](https://arxiv.org/html/2606.04686)) was not read.

## Q2. What do critical and benchmark studies recommend (tuning, size matching, statistics)?

### Takeaway
The critical literature agrees on a few points. Give classical baselines the same hyperparameter-search budget as the quantum model. Match capacity, meaning model size and readout dimension. Include "de-quantised" controls, such as non-entangling, separable or classical-spin versions. Report rankings across many datasets and seeds rather than the best case. Account for the full cost of shots and readout. Under these rules, quantum models usually lose or tie.

### Cited Findings
- Bowles, Ahmed, Schuld (Xanadu, 2024): 12 QML models on 6 binary classification tasks gave 160 datasets. The same five-fold CV grid search was used for every model, quantum and classical. Out-of-the-box classical models (MLP, Gaussian SVC, CNN) **ranked first in all three model families** — [arXiv:2403.07059](https://arxiv.org/html/2403.07059v1); [PennyLane blog](https://pennylane.ai/blog/2024/03/benchmarking-near-term-quantum-algorithms-is-an-art/)
- In the same study, **removing entanglement** gave similar or better performance for most models; the separable kernel ranked second among kernels. Performance ranged from random to near-perfect across hyperparameters, so poor tuning alone can flip conclusions — [arXiv:2403.07059](https://arxiv.org/html/2403.07059v1)
- Same study, recommendations: ask narrow questions rather than "does it beat classical?"; use controllable synthetic data; look for patterns across settings; tune classical baselines broadly; guard against **positivity bias**, i.e. reporting only the best candidate model. Time series and reservoirs were not covered — [arXiv:2403.07059](https://arxiv.org/html/2403.07059v1)
- Fellner, Kreplin, Tovey, Holm, *Mach. Learn.: Sci. Technol.* 7 (2026) 010501: 27 time-series tasks from 3 chaotic systems, with extensive hyperparameter optimisation for both sides. "Quantum models struggle to match the accuracy of simple classical counterparts" of comparable complexity. This covers variational QML, not reservoirs — [arXiv:2504.12416](https://arxiv.org/abs/2504.12416)
- Tariq et al. survey (July 2026): "Current results do not establish a broad quantum advantage over well matched classical reservoirs"; "a large state space alone does not guarantee useful computation". It calls for resource accounting and benchmark standards, and the search snippet says baselines should be ESNs or structured state-space models, tuned under a comparable search budget, with the full quantum execution and readout cost counted — [arXiv:2607.18552](https://arxiv.org/abs/2607.18552)
- Wringe, Trefzer, Stepney, reservoir-computing benchmark tutorial and critique (2024): benchmark choice in RC is often justified by citation rather than reasoning. They propose a task taxonomy and best practices — [arXiv:2405.06561](https://arxiv.org/html/2405.06561v2)
- Pandey 2026 checklist: matched size, matched tuning effort, fixed seeds, statistically tested gaps — [arXiv:2607.09905](https://arxiv.org/abs/2607.09905)
- Kornjača et al. 2024 report their "comparative quantum kernel advantage" (7.1 +/- 3.1% test accuracy over a classical spin reservoir) **only on synthetic labels engineered from kernel geometry**. They state it is dataset-specific and comparative, not absolute, and that a classical-favouring dataset could be built just as easily — [arXiv:2407.02553](https://arxiv.org/html/2407.02553v1)
- Kornjača et al. baselines were reasonably fair: linear SVM/SVR with C and epsilon grid-searched on an 80/20 split, a 2x100 ReLU MLP tuned the same way, and a classical-spin reservoir control. The QRC matched the nonlinear classical methods on 10-class MNIST rather than beating them — [arXiv:2407.02553](https://arxiv.org/html/2407.02553v1)

**Theory: classical simulability, expressivity and universality**
- Schütte et al., "Expressivity Limits of QRC" (2025): the number of orthogonal nonlinear functions a QRC can generate from classical data is **bounded linearly by the number of input-encoding gates, independent of Hilbert-space size**, for typical single-qubit rotation encodings. Any advantage would need multi-qubit input schemes or quantum-native data — [arXiv:2501.15528](https://arxiv.org/html/2501.15528v2)
- Sannia, Giorgi, Zambrini (2025-26): **exponential concentration**. Finite-shot noise means exponentially many samples are needed to extract outputs from scrambling reservoirs. Hamiltonian symmetries suppress this — [arXiv:2505.10062](https://arxiv.org/html/2505.10062v2)
- Gonon & Jacquier: universal approximation and explicit error bounds for QNNs and *random* quantum circuits as reservoirs (IEEE TNNLS 2025) — [arXiv:2307.12904](https://arxiv.org/abs/2307.12904v2). Gonon, Martínez-Peña, Ortega (ICLR 2026): feedback-driven recurrent QNNs approximate regular state-space systems **without the curse of dimensionality**, with qubits growing logarithmically in 1/accuracy — [arXiv:2506.16332](https://arxiv.org/html/2506.16332v2). Universality shows QRC *can* represent the target class. It does not show QRC is better than ESNs, which are also universal (cf. unified universality conditions, [arXiv:2401.15067](https://arxiv.org/html/2401.15067v3)).
- Simulability admissions: the 2025 Gaussian boson sampler QRC experiment says its lossy system is likely efficiently classically simulable — [arXiv:2505.13695](https://arxiv.org/pdf/2505.13695)

### Inferences
- A 9-spin NMR reservoir, like ours, is classically simulable by construction, since we simulate it. Any "advantage" claim must therefore be about physical or energy implementation, not computational intractability. Reviewers will know this.
- The Schütte bound applies directly to us. Our 653 FID features do not mean 653 independent nonlinear functions of the input. The effective nonlinear-feature count is limited by the encoding. Improving the encoding (multi-spin or richer input injection) is the theoretically motivated lever, more than adding readout features.
- Our study already meets most of the recommendations: pre-registration, untouched test cities, tuned baselines, paired tests. Missing controls worth adding: a capacity-matched ESN (same readout D = 653), a "classical spin" or non-interacting (separable) reservoir control, and reporting the shot and measurement budget.

### Gaps
- I found no paper that formally "dequantizes" QRC for time series (proving a classical algorithm reproduces it efficiently). The critique works through simulability arguments, expressivity bounds and empirical fairness instead.
- The full text of Pandey 2026 (tasks, tuning budget, test used) was not read.

## Q3. What ingredients were credited for gains, and what helps QRC in practice?

### Takeaway
Ingredients credited with real, repeatable effects are: time-multiplexed or rich readout, matching dissipation and relaxation to the task's memory, feedback or recurrence, and the measurement protocol (weak vs repeated measurement). Moderate hardware noise sometimes acts as a regulariser. Claims about entanglement or "quantumness" are contested: separable controls often match the entangled model. Evidence for sample efficiency or small-data advantage is thin, task-specific and mostly on synthetic tasks.

### Cited Findings
- **Time multiplexing:** in the NMR PRL, time-multiplexed FID readout "markedly improves" NARMA performance, especially at short input intervals (tau = 0.01 s). At tau = 0.3 s the benefit shrinks because relaxation washes out high-order correlations — [arXiv:2508.12383 HTML](https://arxiv.org/html/2508.12383)
- **Relaxation as memory:** the same paper says T1 relaxation plays a "pivotal role" and provides fading memory without qubit resets — [arXiv:2508.12383 HTML](https://arxiv.org/html/2508.12383)
- **Quantum correlations:** the same paper reports that a classical-spin counterpart averaged R^2 < 0.75 on NARMA2-20 and credits the gap to quantum correlations. This is a synthetic task, compared against a classical-spin model rather than a tuned ESN — [arXiv:2508.12383 HTML](https://arxiv.org/html/2508.12383)
- **Feedback/recurrence:** in Pandey 2026, feedback turns a useless reservoir into a working predictor, but a tuned classical network still wins slightly — [arXiv:2607.09905](https://arxiv.org/abs/2607.09905). Gonon et al. give theory for feedback-driven universality — [arXiv:2506.16332](https://arxiv.org/html/2506.16332v2). Kornjača et al. list recurrence as future work — [arXiv:2407.02553](https://arxiv.org/html/2407.02553v1)
- **Measurement protocol:** Yasuda et al. (2023) use repeated mid-circuit measurements on an IBM superconducting device. They report better accuracy *and* shorter run time than conventional QRC, pick the measurement strength by temporal information-processing capacity, run a soft-robot task for over 1,000 timesteps, and give a preliminary 120-qubit result — [arXiv:2310.06706](https://arxiv.org/abs/2310.06706v1). Mujal et al. (npj QI 2023) give theory for weak vs projective measurement in time-series QRC — [arXiv:2205.06809](https://inspirehep.net/arxiv/2205.06809)
- **Noise as a resource:** non-unital (amplitude-damping) noise on superconducting hardware counteracts information loss from repeated measurement — [arXiv:2409.07886](https://arxiv.org/html/2409.07886v3). Hardware noise acting as an implicit regulariser on ENSO is suggested but unconfirmed — [arXiv:2510.13634](https://arxiv.org/html/2510.13634v1)
- **Shot-noise resilience:** reservoirs with finite entanglement and coherence degrade less under statistical (shot) noise than incoherent ones, and the size of this benefit depends on the measurement budget (Kora & Simon 2025/26) — [arXiv:2504.17837](https://arxiv.org/abs/2504.17837). Against this, exponential concentration says scrambling reservoirs need exponentially many shots unless symmetries are used — [arXiv:2505.10062](https://arxiv.org/html/2505.10062v2)
- **Entanglement and memory:** high entanglement is a prerequisite for the complex dynamics that unlock the large phase space and higher short-term memory; dephasing hurts (Götting, Lohof, Gies, PRA) — [arXiv:2302.03595](https://www.arxiv.org/abs/2302.03595). In the general QML benchmarks, by contrast, removing entanglement often did not hurt — [arXiv:2403.07059](https://arxiv.org/html/2403.07059v1)
- **Encoding is the bottleneck:** see Schütte et al. — [arXiv:2501.15528](https://arxiv.org/html/2501.15528v2). In Kornjača et al., encoding choice (global pulse vs position vs local pulse) changed Santa Fe performance and global-pulse encoding was worst — [arXiv:2407.02553](https://arxiv.org/html/2407.02553v1)
- **Readout post-processing:** in the NMR PRL, adding SVR post-processing on top of the QRC features moved it from about ESN(1000) to above ESN(10000) on temperature — [arXiv:2508.12383 HTML](https://arxiv.org/html/2508.12383)
- **Sample efficiency:** a single-node finitely-sampled quantum reservoir shows a "modest but statistically significant" sample-efficiency advantage for one quantum-state-learning task, while other tasks favour classical reservoirs — [Phys. Rev. Research](https://journals.aps.org/prresearch/pdf/10.1103/wsyq-jyxd). The NARMA-10 "sustainability" study claims competitive NRMSE vs ESN/LSTM with benefits in low-data, compute-constrained regimes, but it is synthetic and small — [arXiv:2510.25183](https://arxiv.org/html/2510.25183v1)
- **Scale:** in the neutral-atom experiment, accuracy rose with system size from 9 to 108 qubits on a tomato-disease image task (498 images). Performance plateaus near 1,000 shots, and about 60% of shots in the largest arrays had sorting defects — [arXiv:2407.02553](https://arxiv.org/html/2407.02553v1)

### Inferences
- For our weather gap (-0.006 R^2 vs ESN), the levers best supported by the literature are: (1) a nonlinear or kernel readout on the 653 features (the PRL's SVR result), compared against an ESN given the same readout; (2) tuning tau and dwell to the weather's memory scale (the relaxation and memory trade-off); (3) richer or multi-spin input encoding; (4) feedback of past outputs. Adding qubits or features alone is not supported by the theory.
- Applying the nonlinear readout fairly means giving the ESN the same readout, or the comparison falls back into the positivity-bias trap.
- "Noise helps" findings are hardware-specific and unconfirmed. For a simulated NMR reservoir, the honest equivalent is a test of robustness to added shot noise or decoherence, which the Kora & Simon result suggests could favour coherent reservoirs.

### Gaps
- No controlled ablation found that isolates time multiplexing vs relaxation vs encoding on *real-world* data, as opposed to NARMA.
- No head-to-head study of small-data learning curves for QRC vs tuned ESN on real forecasting data. This is an open niche.
- Suzuki et al. ("natural" noise-driven QRC on IBM devices) and Fujii & Nakajima were not checked in this pass.

## Q4. Hardware QRC demonstrations 2023-2026, and which niches are most credible for positioning

### Takeaway
Hardware QRC now spans NMR (9 spins, PRL 2026), superconducting (IBM, repeated measurement and Heron R2 multivariate runs; transmon-cavity analog microwave processing), neutral atoms (up to 108 qubits) and photonics or Gaussian boson sampling. None claims a computational advantage over tuned classical models on real forecasting. The most credible positioning is honest benchmarking, non-inferiority, physical or in-sensor processing of signals that are already quantum or analog, and resource or energy accounting. "Beats classical on weather" is not credible.

### Cited Findings
- **NMR:** 9-spin crotonic acid, Delhi weather, ESN-comparable — [PRL 136, 120602 (2026) / arXiv:2508.12383](https://arxiv.org/abs/2508.12383)
- **Neutral atoms (QuEra/Harvard):** up to 108 qubits, gradient-free reservoir learning, billed as the largest QML experiment. Tasks were MNIST, tomato leaf disease and Santa Fe laser. Matches tuned nonlinear classical models on MNIST-10; the "advantage" exists only on engineered synthetic labels — [arXiv:2407.02553](https://arxiv.org/html/2407.02553v1); [QuEra blog](https://www.quera.com/blog-posts/large-scale-quantum-reservoir-learning-with-an-analog-quantum-computer)
- **Superconducting, IBM:** repeated-measurement QRC with a soft-robot time series and 120-qubit preliminary runs (Yasuda et al. 2023) — [arXiv:2310.06706](https://arxiv.org/abs/2310.06706v1). Keio University and Mitsubishi Chemical are extending this toward utility-scale QRC — [Quantum Computing Report](https://quantumcomputingreport.com/?p=33890). Heron R2 multivariate runs on Lorenz/ENSO lose to NVAR and clustered ESN — [arXiv:2510.13634](https://arxiv.org/html/2510.13634v1)
- **Superconducting analog, in-sensor:** Senanian et al., Nature Communications 2024. A transmon coupled to a long-lived cavity processes ~6 GHz microwave signals directly, without discretisation, and classifies ultra-low-power signals with high accuracy. They explicitly "do not attempt to address" computational advantage on pre-recorded classical signals and frame this as a step toward a *quantum sensing-computational* advantage — [arXiv:2312.16166](https://arxiv.org/pdf/2312.16166); [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11364634)
- **Photonic / GBS:** large-scale GBS QRC (2025) admits its lossy system is likely classically simulable — [arXiv:2505.13695](https://arxiv.org/pdf/2505.13695). A scalable photonic platform for real-time QRC — [arXiv:2207.14031](https://arxiv.org/pdf/2207.14031)
- **Energy and thermodynamics angle:** "Thermodynamics of QRC" (2026) — [arXiv:2607.02157](https://arxiv.org/pdf/2607.02157) (not read). The NARMA-10 sustainability benchmark claims energy benefits — [arXiv:2510.25183](https://arxiv.org/html/2510.25183v1)

### Inferences
- **Most credible niches for a CiRA paper:**
  1. **Rigorous non-inferiority benchmark** of a PRL-style NMR QRC on real multi-city weather, with pre-registration, tuned baselines and paired tests. This fills the exact gap that Bowles et al., Fellner et al. and the 2026 survey describe, and it stress-tests a PRL claim. A result of "non-inferior, ESN slightly ahead" is publishable and in line with the field's direction.
  2. **Physical or in-sensor reservoir:** an NMR spectrometer already produces FIDs, so "computing in the sensor" (Senanian-style framing) is plausible for magnetic-resonance or spin-sensor data, not for weather tables.
  3. **Resource accounting:** time per sample, shots, energy. Only credible with real numbers; simulated GPU evolution is not an energy argument.
  4. **Small data and shot-noise robustness:** under-tested, so learning-curve and noise-robustness experiments against a tuned ESN would be novel. The evidence so far is weak, so claim it only if our data shows it.
- **Avoid:** claiming computational advantage (the system is simulable), entanglement as the cause without a separable control, and comparisons against untuned or undersized ESNs.

### Gaps
- No trapped-ion QRC demonstration on real-world tasks found in this pass.
- No hardware QRC on real energy, traffic or ECG forecasting found.
- Rigetti-specific QRC demonstrations were not found.
