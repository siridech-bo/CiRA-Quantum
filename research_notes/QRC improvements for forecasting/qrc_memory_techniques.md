# QRC techniques for fading memory, long-horizon forecasting and nonlinear expressiveness (2019-2026, emphasis 2023-2026)

Scope note: about 23 search/fetch calls. Most 2024-2026 items were checked against arXiv abstract or HTML pages. Items marked **[unverified, from prior knowledge]** were not re-fetched in this session, so the report writer should treat their bibliographic details as needing a check. Evidence tags used below:
- **SYN**: synthetic tasks only (STM/memory capacity, NARMA, IPC, Mackey-Glass, chaotic maps).
- **REAL**: real-world data.
- **CB+**: compared against a classical baseline that was tuned, at least by grid search.
- **CB-**: no classical baseline, or an untuned one.

**Critical prior art for our system (read first).** Hou, Hua, Wu, Xia, Chen, Li, Li, Peng, Du, "High-Accuracy Temporal Prediction via Experimental Quantum Reservoir Computing in Correlated Spins", arXiv:2508.12383, Phys. Rev. Lett. 136, 120602 (2026). It is essentially our configuration run *experimentally*:
- 13C-labelled crotonic acid, 9 spins (4 C, 5 H), 400 MHz Bruker.
- Input as an x-rotation RF pulse on the protons, tau = 0.03 s.
- Readout = the FFT of the FID with **653 spectral peaks** ("time-multiplexed readout").
- Fading memory comes from T1, and a rewinding/washout protocol is used.
- Delhi daily climate (Kaggle 2013-2017) temperature and humidity, horizons up to 45 days, with a separate readout per horizon.
- NARMA2-20 NMSE about 1e-5. Baselines are ESN(500/1000/5000/10000) via reservoirpy, grid-searched on connectivity and spectral radius (rho = 0.99), averaged over 100 realisations.
- Result: QRC is comparable to ESN(1000) on temperature and exceeds it on average at longer horizons. With RBF-SVR post-processing it beats ESN(10000), and it is comparable to ESN(10000) on humidity.
- Its appendix says unitary dynamics give zero STM, T2 dephasing alone gives no useful fading memory, and T1 amplitude damping is the key ingredient. STM peaks at an intermediate tau.
- Sources: [arXiv abs](https://arxiv.org/abs/2508.12383); [HTML](https://arxiv.org/html/2508.12383).
- Our project's baseline therefore *is* this PRL's protocol. Any improvement technique below should be framed as "beyond Hou et al. 2026".

---

## 1. Time multiplexing (virtual nodes) and spatial multiplexing (ensembles of reservoirs)

### Takeaway
Time multiplexing gives large gains over single-time readout but saturates quickly from collinearity. Our 653-bin FID spectrum already is an extreme form of it. Spatial multiplexing (several independent reservoirs on the same input) improved memory capacity and NARMA monotonically in the original work. That evidence is SYN only, with no tuned classical baseline.

### Cited Findings
- **Spatial multiplexing.** Nakajima, Fujii, Negoro, Mitarai, Kitagawa, "Boosting computational power through spatial multiplexing in quantum reservoir computing", arXiv:1803.04574 (Phys. Rev. Applied 11, 034021, 2019 [journal ref unverified]).
  - C different quantum reservoirs, each with its own qubit count and input interval, are driven by a common input. Each also uses temporal multiplexing with V_c virtual nodes.
  - "In all cases, the increase of the order of spatial multiplexing induces the improvement of performance" on memory capacity and NARMA.
  - The paper explicitly targets ensemble (NMR) quantum systems.
  - Evidence: SYN, CB-.
  - [arXiv HTML](https://arxiv.org/html/1803.04574v1)
- **Time multiplexing saturates.** In a Jaynes-Cummings QRC, V = 5 virtual nodes give a "drastic improvement" over V = 1, but further increases in V do not significantly improve capacity. Evidence: SYN. [arXiv:2510.00171](https://arxiv.org/html/2510.00171v1)
- **Physical scaling beats multiplexing.** A neutral-atom QRC study finds that time multiplexing saturates quickly because of multicollinearity among the extracted observables. It concludes that larger physical system size is the more robust route to expressivity. Evidence: SYN. [arXiv:2609.28441](https://arxiv.org/pdf/2609.28441)
- **Monitored systems.** A general theory of monitored QRC states that time multiplexing amplifies the information-disturbance trade-off when the readout disturbs the state. [arXiv:2607.14311](https://arxiv.org/pdf/2607.14311)
- **Experimental NMR.** Hou et al. 2026: the 653-peak FID readout clearly beats single-time Pauli readout (9 values), especially at tau = 0.01 s. At tau = 0.3 s, stronger relaxation diminishes high-order correlations and reduces the benefit of time multiplexing. Evidence: SYN + REAL, CB+ (grid-searched ESN). [arXiv HTML](https://arxiv.org/html/2508.12383)
- **Original framework [unverified, from prior knowledge].** Fujii & Nakajima, "Harnessing disordered-ensemble quantum dynamics for machine learning", Phys. Rev. Applied 8, 024030 (2017), arXiv:1602.08159. It introduced QRC with virtual-node time multiplexing on ensemble spin systems.
- **Review [unverified, from prior knowledge].** Fujii & Nakajima, "Quantum reservoir computing: a reservoir approach toward quantum machine learning on near-term quantum devices", arXiv:2011.04890 (2021 book chapter/review).
- **Information processing capacity.** IPC of spin-based QRC, with a parameter study of multiplexing and dynamics. [arXiv:2010.06369](https://arxiv.org/pdf/2010.06369)

### Inferences
- Adding more FFT bins is unlikely to fix our lag-90 deficit. The FID readout is already a dense time multiplex, and compact readouts didn't help us. What we need is feature *diversity in timescale*, not more features of the same timescale.
- **Spatial multiplexing maps cheaply onto our simulator.** Run 2-4 copies of the crotonic reservoir in parallel on the same input but with different settings:
  - (a) different input gain g;
  - (b) different tau, e.g. 30 ms and 300 ms. Hou et al. show long tau gives a different memory/correlation profile;
  - (c) different pulsed channel, 1H vs 13C. In 13C the populations relax slowly.
  - Concatenate the spectra and use one ridge readout. Cost is k times the GPU, but the copies are trivially parallel.
  - This is the closest "ensemble of reservoirs" analogue. In particular, a long-tau or weak-gain copy acts as a slow-memory reservoir, and the standard copy supplies nonlinearity.
- Our earlier result that "tau variants gave no gain" was presumably about a *single* reservoir with different tau. A multi-tau *concatenation* (spatial multiplex) is a different experiment and is untested.

### Gaps
- I found no paper giving tuned-ESN-vs-spatially-multiplexed-QRC numbers on real data.
- I did not verify the journal reference for 1803.04574, or exact effect sizes (MC values) from it.

---

## 2. Weak / partial / continuous measurement, back-action, feedback, repeated or reset protocols

### Takeaway
This is the most active 2023-2026 line. Weak measurement plus feedback of past outputs improves STM and NARMA over projective/reset protocols. One 2025 proposal is explicitly designed for NMR ensembles. Nearly all of this evidence is SYN-only with no tuned classical baseline. For our simulator, "feedback of the previous readout into the next input pulse" is the most directly transferable idea, because our readout has no back-action at all.

### Cited Findings
- **Weak and projective measurement protocols.** Mujal, Martinez-Pena, Giorgi, Soriano, Zambrini, "Time-series quantum reservoir computing with weak and projective measurements", npj Quantum Inf. 9, 16 (2023), arXiv:2205.06809, DOI 10.1038/s41534-023-00682-z.
  - Proposes and compares measurement protocols (restart, rewinding, online weak measurement) on resource efficiency.
  - Weak online measurement can keep memory while extracting information.
  - Evidence: SYN, CB-.
  - [npj](https://www.nature.com/articles/s41534-023-00682-z); [arXiv](https://arxiv.org/pdf/2205.06809)
- **Feedback-driven QRC.** Kobayashi, Fujii, Yamamoto, "Feedback-driven quantum reservoir computing for time-series analysis", PRX Quantum 5, 040325 (2024), arXiv:2406.15783.
  - Uses projective measurement on all qubits, and the outcomes are fed back into the reservoir to restore memory of prior inputs. This avoids the time cost of repeated re-initialisation.
  - There are three phases versus feedback strength, and **memory performance is maximised at the edge of chaos**.
  - Evidence: SYN, including signals from quantum spin systems. No ESN comparison was seen in the abstract.
  - [arXiv](https://arxiv.org/abs/2406.15783); [PRX Q](https://link.aps.org/doi/10.1103/PRXQuantum.5.040325)
- **Weak measurement plus feedback, designed for NMR.** Monomi, Setoyama, Hasegawa, "Feedback-enhanced quantum reservoir computing with weak measurements", arXiv:2503.17939 (2025, v1 only, no journal found). [abs](https://arxiv.org/abs/2503.17939); [HTML](https://arxiv.org/html/2503.17939)
  - Setup: 6 qubits (2 input, 4 memory) with a transverse-field Ising Hamiltonian, h = 5 J_s, dt = 10/J_s.
  - Weak measurement of X and Z of strength g (coherences damped by e^{-g^2/2}), with g = 0.3 by default.
  - The previous step's Z expectations are fed back as rotation angles a_fb * z on the memory qubits. a_fb of about 0.1-0.3 is best for STM.
  - Weak+feedback beats projective-feedback on STM (sum over tau = 0..20) and on NARMA5/10/15/20, with a smaller gain on NARMA20. The projective model's memory drops sharply after tau of about 4, while the weak one decays gradually.
  - It is fragile to depolarising noise.
  - The **framework assumes an ensemble system such as NMR**: averaging over about 1e18-1e20 molecules gives small statistical error, and back-action can be made nearly negligible.
  - Evidence: SYN, CB-. No exact numbers are in the text.
- **Indirect measurement and back-action.** Franceschetto, Plodzien, Lewenstein, Acin, Mujal, "Harnessing quantum back-action for time-series processing", arXiv:2411.03979, Phys. Rev. X 16, 021002 (2026).
  - Indirect measurements trade information extraction against disturbance, and the measurement strength is tunable.
  - Claims improved memory over "state-of-the-art classical feedback protocols" and better execution-time scaling.
  - Evidence: SYN (two benchmarks). [arXiv](https://arxiv.org/abs/2411.03979)
- **Moderate measurement as a resource.** Keenan & Zambrini, "Storage, Scrambling, and Loss of Information in QRC", arXiv:2608.07677 (v3 Oct 2026). [HTML](https://arxiv.org/html/2608.07677)
  - Moderate measurement strength (g of about 0.3) *raises* total IPC above g = 0. Median loss rate lambda goes from 0.146 to 0.335 while scrambling gamma stays near 1.
  - At g of about 1, IPC returns to the no-measurement level.
  - Evidence: SYN (IPC), 6 qubits.
- **Reset-based protocol on hardware.** Hu, Khan, Bronn, Angelatos, Rowlands, Ribeill, Tureci, "Overcoming the coherence time barrier in QML on temporal data" (NISQRC), Nature Communications 2024, arXiv:2312.16165.
  - Mid-circuit measurement plus deterministic reset keeps a usable memory, so arbitrarily long signals can be recovered beyond T1/T2. Demonstrated on 7 qubits.
  - Task: channel equalisation. Evidence: SYN/engineering task.
  - [arXiv](https://arxiv.org/html/2312.16165v3); [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11364873/)
- **Repeated measurement on hardware.** Yasuda et al., "Quantum reservoir computing with repeated measurements on superconducting devices", arXiv:2310.06706 (2023). I did not retrieve its contents. [cited in Hu et al.](https://arxiv.org/html/2312.16165v3)
- **Artificial memory restriction.** Cindrak, Donvil, Ludge, Jaurigue, "Enhancing the performance of QRC and solving the time-complexity problem by artificial memory restriction", Phys. Rev. Research 6, 013051 (2024), arXiv:2306.12876.
  - Only the last few inputs re-initialise the reservoir after each measurement. This gives linear time complexity, and the window length tunes nonlinearity.
  - Evidence: SYN.
  - [arXiv](https://arxiv.org/abs/2306.12876)
- **Measurement-enabled processing.** Related work on measurement-enabled online processing with amplitude encoding, arXiv:2606.18991. [link](https://arxiv.org/pdf/2606.18991) Not read in detail.

### Inferences
- Our simulator reads a *copy* of the state, so we already have zero back-action, which is the ideal that weak-measurement papers try to approach. The weak-measurement memory gain therefore does not transfer directly; we already have it.
- What does transfer is **output feedback**: feed the previous step's readout (for example a few principal components of the FID spectrum, or the total proton z magnetisation) back into the next pulse angle or onto a second channel. Examples:
  - theta_t = g * arcsin(sqrt(s_t)) + a_fb * f(y_{t-1});
  - a separate 13C pulse driven by the previous readout.
- In NMR this is physically realistic as "classical feedback" between scans. It amounts to a closed-loop reservoir. Kobayashi et al. find memory peaks at the edge of chaos versus a_fb, so sweep a_fb.
- Cindrak-style windowed restarts would *reduce* long memory by design. They are useful only for nonlinearity tuning, not for our lag-90 weakness.

### Gaps
- Exact STM or NARMA numbers for Monomi et al. and Kobayashi et al. weren't extracted, since they appear only in figures.
- None of these papers compares to a tuned ESN on real-world data.

---

## 3. Input re-uploading, multiple injections, delay embedding / concatenating past readouts, multi-tau readouts

### Takeaway
The strongest evidence that long memory can come "for free" is classical: concatenating past reservoir readouts, or past inputs as in NG-RC/NVAR-style delay embedding, adds memory at the readout. A 2026 theorem shows linear reservoirs cannot create fixed-delay capacity beyond what the preprocessed input already contains, so honest comparisons must include the same delay features in the classical baselines. Explicit QRC re-uploading studies for temporal memory are sparse.

### Cited Findings
- **Linear reservoirs cannot add capacity.** Nokkala, Centrone, Arzani, "A hidden bottleneck in classical and quantum linear reservoir computing", arXiv:2605.29071 (May 2026).
  - For reservoirs whose measured features evolve linearly, with a linear readout, capacity at any fixed delay is capped by what the preprocessed input already contains. Linear dynamics "cannot create new fixed-delay expressive power on their own".
  - Global summed capacity can hide strong per-delay limits.
  - Evidence: theory + numerics (CV systems).
  - [arXiv](https://arxiv.org/abs/2605.29071)
- **Our dynamics are linear in the density matrix.** A Lindblad map is linear in rho, and our input pulse is the only nonlinearity, via theta(s). So a per-delay memory analysis (the memory function at lag k, which we already use) is the right diagnostic, not total MC. This inference follows from the result above.
- **Concatenating QELM features with an ESN.** Tested on Lorenz-63 and ENSO. [arXiv:2510.13634](https://arxiv.org/abs/2510.13634)
  - Gate-based QRC on NISQ hardware gave MSE 0.0087 on Lorenz and 0.0036 on ENSO.
  - It was on par with classical RC on Lorenz and beat learned RNNs, but NVAR and a clustered ESN remain stronger on some settings.
  - Evidence: REAL (ENSO), CB+.
- **Time-delayed QELM.** A 2026 time-delayed QELM (TD-QELM) encodes several past inputs at once, so circuit depth is independent of sequence length. It reportedly beats conventional QRC on NARMA in simulation and on an IBM 127-qubit device.
  - Evidence: SYN. I could not confirm its arXiv ID; the search snippet linked 2503.05535 or related entries, which is ambiguous.
  - [search context](https://arxiv.org/html/2503.05535v3)
- **Classical reference [unverified, from prior knowledge].** Gauthier, Bollt, Griffith, Barbosa, "Next generation reservoir computing", Nat. Commun. 12, 5452 (2021). It uses delay embedding plus polynomial features without any reservoir, and is a strong baseline for Lorenz-type and short-memory tasks.

### Inferences
- **Delay embedding of readouts (cheap, offline, no GPU).**
  - Build features [X_t, X_{t-d1}, X_{t-d2}, ...] from the saved FID traces, PCA-compressed first to avoid a 653 x k blow-up, with d in {7, 15, 30, 60, 90}.
  - This directly attacks the lag-90 memory deficit, and can be computed entirely from existing `artifacts/traces/*.npz` per the evolve-once rule.
  - **But** it must be labelled as "quantum + classical memory". The ESN baseline must get the same delay features, otherwise the comparison is unfair (Nokkala et al.).
- **Multiple injections per step.**
  - Option: pulse the same input on 1H and then on 13C, or split theta into two half-pulses separated by a delay.
  - This injects into the slow-relaxing 13C populations, giving a write channel decoupled from the proton readout. It is a cheap route toward "memory spins" (write/store separation).
- **Multi-tau readouts.**
  - Take FIDs at two evolution times within a step, or run two copies with different tau (spatial multiplex, §1).
  - This differs from the "tau variants" already tried, because the features are concatenated rather than swapped.

### Gaps
- I found no QRC paper that quantifies memory-function R^2 at long lags (>50 steps) with readout delay embedding versus without, on real data.
- I found no dedicated 2023-2026 study of data re-uploading for *temporal* QRC memory. Most re-uploading literature is on static QML classifiers, outside scope.

---

## 4. Dissipation engineering, phase transitions / MBL, disorder, memory-nonlinearity trade-off, IPC

### Takeaway
The consensus framework (2021-2026) is a tunable memory-nonlinearity trade-off, measured by IPC.
- Linear and long memory wants slow information loss and weak scrambling; nonlinearity wants strong scrambling and faster loss.
- Dissipation (amplitude damping, T1) is necessary for fading memory, and can be engineered as a resource.
- Dynamical or dissipative phase transitions mark the optimal regimes.
- Evidence is almost entirely SYN.

### Cited Findings
- **Phase transitions [unverified details].** Martinez-Pena, Giorgi, Nokkala, Soriano, Zambrini, "Dynamical phase transitions in quantum reservoir computing", Phys. Rev. Lett. 127, 100502 (2021). Bibliographic reference confirmed via search; content from prior knowledge.
  - In a disordered transverse-field Ising reservoir, the ergodic (thermal) side near the MBL transition gives the best performance.
  - Deep MBL kills fading memory and echo state property.
  - [citation seen](https://arxiv.org/html/2501.15528)
- **Dissipation as a resource.** Sannia, Martinez-Pena, Soriano, Giorgi, Zambrini, "Dissipation as a resource for Quantum Reservoir Computing", Quantum 8, 1291 (2024), arXiv:2212.12078.
  - Engineered local dissipation (amplitude damping) gives fading memory and nonlinearity without measurement/reset protocols.
  - Evidence: SYN.
  - [Quantum](https://quantum-journal.org/papers/q-2024-03-20-1291/); [arXiv](https://arxiv.org/abs/2212.12078)
- **Unified IPC view.** Cindrak, Giebeler, Gotting, Gies, Ludge, "Memory-Nonlinearity Trade-off across QRC Frameworks", arXiv:2603.21371 (Mar 2026).
  - Memory restriction, weak measurements, operation near the edge of quantum chaos, and dissipative dynamics all share "a tunable balance between memory retention and nonlinear response".
  - IPC identifies regimes that beat the standard protocol.
  - Evidence: SYN, no numbers in abstract.
  - [arXiv](https://arxiv.org/abs/2603.21371)
- **Storage, scrambling and loss.** Keenan & Zambrini 2026, arXiv:2608.07677. [HTML](https://arxiv.org/html/2608.07677)
  - In an erase-and-write protocol, unitary dynamics cannot reduce memory; loss comes entirely from the dissipative injection map.
  - IPC_1 (linear memory) favours weak scrambling and slow decay, especially at long delays. Higher-degree IPC needs strong scrambling and fast decay.
  - Stored information saturates by about 50 steps at about an order of magnitude below the theoretical maximum.
  - Evidence: SYN, 6 qubits.
- **Entanglement and memory.** Gotting, Lohof, Gies, "Exploring quantumness in quantum reservoir computing", Phys. Rev. A 108, 052427 (2023), arXiv:2302.03595.
  - High entanglement is a prerequisite for complex dynamics that unlock the exponential phase space and higher short-term memory capacity. Dephasing removes this.
  - Evidence: SYN.
  - [ADS](https://ui.adsabs.harvard.edu/abs/2023PhRvA.108e2427G/abstract)
- **Noise as a resource.** Kubota, Suzuki, Kobayashi, Tran, Yamamoto, Nakajima, "Temporal information processing induced by quantum noise", Phys. Rev. Research 5, 023057 (2023), arXiv:2207.07924.
  - Abstract noise channels (amplitude damping etc.) provide temporal processing.
  - IPC *rose* with device noise and error rates on IBM hardware.
  - Evidence: SYN.
  - [DOAJ](https://doaj.org/article/ac7cc7576d8c407184f682dff6f87f3f); [arXiv](https://arxiv.org/pdf/2207.07924)
- **Natural QRC.** [arXiv:2107.05808](https://arxiv.org/pdf/2107.05808)
- **Optimised noise-induced QRC.** Fry et al., Sci. Rep. 2023, on chaotic time series. [Nature](https://www.nature.com/articles/s41598-023-45015-4)
- **Hou et al. 2026 appendix.** T1 amplitude damping is key; dephasing alone drives toward the maximally mixed state with no useful memory. STM vanishes for tau >> T1 and in the near-unitary limit, and peaks in between. [HTML](https://arxiv.org/html/2508.12383)
- **Dissipation and encoding near criticality.** In two coupled Kerr oscillators near a bifurcation, encoding shifts from redundant to synergistic. Strong dissipation favours redundant encoding that supports long-term memory. Evidence: SYN. Entropy 27, 88 (2025). [DOI](https://doi.org/10.3390/e27010088)
- **Dissipative phase transitions.** Near the transition, the enhanced memory is carried mainly by finite-rate Liouvillian sectors rather than by the asymptotically slow gap mode. Evidence: SYN, preprint. [arXiv:2609.20345](https://arxiv.org/pdf/2609.20345)
- **Role of scrambling and noise.** Further temporal-processing analysis. [arXiv:2505.10080](https://arxiv.org/pdf/2505.10080)
- **Krylov complexity and observability.** A design tool for estimating effective phase-space dimension. [arXiv:2502.12157](https://arxiv.org/pdf/2502.12157)

### Inferences
- Our crotonic reservoir has fixed Hamiltonian parameters. The "phase transition" knob we control is the *effective* coupling-to-shift ratio, via tau and via pulse decoupling/recoupling sequences. Disorder or Stark-field tuning is not natural in liquid-state NMR, so I'd rank this low for us.
- The trade-off literature predicts our g = 0.35 choice sits on the "nonlinear, fast-overwrite" side. A small-angle secondary channel (very low g) dedicated to memory, combined with the existing channel, is the trade-off-aware design. This matches §1 (spatial multiplex) and §6 (memory spins).
- Because the Keenan-Zambrini result says loss comes from the injection map, the overwrite we diagnosed is expected physics. The fix is to make injection *not* act on the storage degrees of freedom.

### Gaps
- I couldn't fetch full results for Martinez-Pena 2021 in this session.
- I found no paper using Stark or disorder fields in NMR QRC.
- No 2024-2026 IPC study compares against a tuned ESN on real data.

---

## 5. Quantum extreme learning machines (QELM), kernel views, memory only from the classical readout

### Takeaway
A QELM (no recurrence) has no intrinsic memory, so any temporal memory comes from classical delay inputs or readout. Fourier-series analyses show the encoding fixes the accessible frequencies, while the reservoir and measurement only set coefficients. Concentration of observables limits scaling. On real data, QELM/QRC are usually competitive with, but rarely clearly better than, tuned classical RC.

### Cited Findings
- **Fourier view of QELM.** Xiong et al., "On fundamental aspects of quantum extreme learning machines", Quantum Machine Intelligence (2025), arXiv:2312.15124.
  - QELM predictions are a Fourier series. The encoding sets frequencies, and the reservoir and measurement set coefficients.
  - The limit is the number of frequencies and observables.
  - Exponential concentration of observables arises from randomness, noise, entanglement and global measurement.
  - [arXiv](https://arxiv.org/abs/2312.15124v2)
- **QELM at scale.** Dao, Yilmaz, ..., Woerner, Tacchino, "Breaking concentration barriers for quantum extreme learning on digital quantum processors", arXiv:2603.13005 (Mar 2026).
  - Up to 124 IBM qubits and over 5,000 two-qubit gates, with multi-objective hyperparameter tuning (observable variability, capacity, performance) and eigentask feature selection.
  - Performance is "competitive with leading classical baselines" on time-series forecasting.
  - Evidence: REAL+SYN, CB+, no numbers in abstract.
  - [arXiv](https://arxiv.org/abs/2603.13005)
- **Pauli-transfer-matrix view.** Theory and interpretability of QELMs. [arXiv:2602.18377](https://arxiv.org/pdf/2602.18377)
- **Expressivity limits.** Expressivity of quantum reservoir computers, arXiv:2501.15528. [link](https://arxiv.org/html/2501.15528)
- **Quantum reservoir kernel.** QuaRK, a quantum reservoir kernel for time series. [ResearchGate](https://www.researchgate.net/publication/400855526_QuaRK_A_Quantum_Reservoir_Kernel_for_Time_Series_Learning). Not read.
- **Real-world QRC typically ties ESN.**
  - Hydrology: QRC test NSE 0.693-0.738, while the ESN achieved slightly lower errors throughout. [Springer](https://link.springer.com/article/10.1007/s00477-026-03383-z)
  - Lorenz/ENSO: QRC is on par with classical RC, but NVAR and clustered ESN are stronger on some settings. [arXiv:2510.13634](https://arxiv.org/abs/2510.13634)
- **Benchmarking across quantum models.** [arXiv:2412.13878](https://arxiv.org/pdf/2412.13878)

### Inferences
- In our pipeline, the input encoding arcsin(sqrt(s)) with fixed g sets the "frequency" content. The learned per-spin encoding gave no gain, which is consistent with the Fourier view: the encoding can't create memory, only nonlinearity.
- If we add readout delay embedding (§3), our system partially becomes "QELM + classical memory". The paper must say so explicitly, and must give the ESN/NVAR baselines the same delays.

### Gaps
- I found no study that cleanly separates "quantum-dynamical memory" from "readout-side memory" on real forecasting data. This would be a novel contribution for us.

---

## 6. Long-term memory enhancement mechanisms (DFS, conserved quantities, slow modes, non-Markovianity, memory qubits)

### Takeaway
The firmest 2025-2026 result is negative: Markovian QRC loses past information exponentially (Sannia et al. 2026, npj QI). Proposed remedies are:
- non-Markovian embedding, which revives memory at specific delays;
- separating write and store subsystems (input qubits versus memory qubits, as in Monomi et al.);
- partial-SWAP memory networks;
- slow decay channels.

I found no peer-reviewed QRC paper that uses decoherence-free subspaces or conserved quantities explicitly as long-memory stores. A 2026 preprint even finds that the slowest Liouvillian gap mode is *not* the main memory carrier.

### Cited Findings
- **Markovian memory decays exponentially.** Sannia, Ravell Rodriguez, Giorgi, Zambrini, "Non-Markovianity and memory enhancement in quantum reservoir computing", arXiv:2505.02491, npj Quantum Information 12, 111 (2026).
  - "All architectures proposed so far rely on Markovian dynamics, which ... inherently lead to an exponential decay of past information."
  - An embedding scheme makes the evolution non-Markovian, and such reservoirs outperform Markovian ones on tasks mixing short- and long-term correlations.
  - Evidence: SYN, CB-.
  - [arXiv](https://arxiv.org/abs/2505.02491); [npj](https://www.nature.com/articles/s41534-026-01257-4)
- **Write/store separation.** Monomi et al. 2025: 2 input qubits plus 4 qubits storing past inputs, with feedback rotations on the storage qubits. [HTML](https://arxiv.org/html/2503.17939)
- **Partial-SWAP memory networks.** "Controllable Quantum Memory Capacity in Quantum Reservoir Networks with Tunable partial-SWAPs", arXiv:2605.12713 (2026). It controls memory capacity through tunable partial-SWAP links between reservoir modules, an explicit write/store mechanism. Not read in detail. [arXiv](https://arxiv.org/pdf/2605.12713)
- **Fisher-orthogonal memory.** "Fisher-Orthogonal Memory in QRC", arXiv:2607.29219 (2026). Not read in detail. [arXiv](https://arxiv.org/pdf/2607.29219)
- **Slow gap mode is not the main carrier.** Dissipative-phase-transition preprint: enhanced memory lives in finite-rate sectors rather than the slow gap mode. [arXiv:2609.20345](https://arxiv.org/pdf/2609.20345)
- **Slow loss favours long-delay memory.** Keenan & Zambrini 2026: linear memory at long delays favours slow information loss and weak scrambling. [HTML](https://arxiv.org/html/2608.07677)
- **Relaxation sets the memory window.** Hou et al. 2026: T1 sets the fading-memory window, and STM peaks at intermediate tau/T1. [HTML](https://arxiv.org/html/2508.12383)
- **Conserved quantities: weak support only.** The only source suggesting conserved quantities or slow scrambling as a memory design rule was a non-peer-reviewed GitHub issue. I am excluding it as a citation and flagging it as low quality.
- **Other memory-related links.**
  - Connection between memory performance and optical absorption in QRC, PRL (2025/26). [APS](https://journals.aps.org/prl/abstract/10.1103/vp79-8t1l)
  - Energetic cost of temporal information processing. [arXiv:2608.08279](https://arxiv.org/pdf/2608.08279)

### Inferences (mapping onto our NMR simulator; ranked by expected value per GPU-hour)
1. **Memory spins / write-store separation (highest priority).**
   - Pulse only the protons, and let J-coupling transfer information into 13C *populations*. The carbons have long T1, and we already measured populations living 70-900 steps.
   - Then read 13C populations through a protected readout (next item).
   - Variant: a selective soft pulse on a subset of protons (e.g. the methyl group as a "write" register) leaves the others as "store".
   - This is the NMR analogue of Monomi's input/memory qubit split. It directly targets the overwrite cause.
2. **Population (z) readout.**
   - The FID spectrum reads transverse coherences, which live only 7-11 steps. Long memory sits in populations, which the FID does not see without a read pulse.
   - Add features from a pi/2 read pulse on the *copy*, which our no-back-action simulator allows, giving a spectrum of z magnetisations or a population-transfer readout.
   - This is the cheapest test, because it may be computable from stored density matrices if they're saved. Otherwise it needs a re-run.
   - It is consistent with Hou et al.'s finding that T1, not T2, carries memory.
3. **Output feedback (Kobayashi; Monomi).** Feed y_{t-1} (or a few FID PCs) into theta_t or into a 13C pulse, and sweep a_fb for the edge of chaos.
4. **Spatial multiplex of a slow copy** (long tau or low g) plus the standard copy (§1).
5. **Spin echo / refocusing.**
   - pi pulses mid-step refocus chemical-shift dephasing, extending coherence lifetime from T2* toward T2. In an exact Lindblad simulation with no inhomogeneity, the gain is only whatever the model's dephasing is refocusable.
   - It would also suppress shift-driven mixing, which may reduce nonlinearity. Expected effect is modest for memory and uncertain overall.
   - I found no QRC paper that tests spin echo.
6. **Non-Markovian embedding (Sannia 2026).** In our system this would mean treating some spins as an environment that we do not read but that feeds back into the readout spins. It is effectively the same as item 1.
7. **More spins.**
   - The neutral-atom study suggests physical size beats multiplexing.
   - It is expensive in an exact Lindblad simulation: 9 to 10+ spins means a 4x larger Liouvillian per spin.
8. **Learned control pulses.** I found no QRC temporal-memory evidence for this in 2023-2026. The adjacent "learned per-spin encoding" we tried gave nothing. Lowest priority.

- **Readout delay embedding (§3)** is the zero-GPU first step. It should run offline on existing traces before any new GPU experiment, to bound how much of the lag-90 gap is closable purely classically.

### Gaps
- I found no QRC paper with long-lag (>50) memory curves on real data with a tuned ESN.
- I found no QRC use of decoherence-free subspaces, singlet/long-lived states, or explicit conserved quantities as memory. NMR long-lived singlet states are a natural candidate, but that is NMR-specific and outside my scope.
- I did not obtain full numbers for arXiv:2605.12713, 2607.29219 or 2609.20345.
- I did not retrieve the Mujal et al. 2021 Adv. Quantum Technol. review, the Fujii & Nakajima 2021 review, or the 2026 review by Tariq et al. (arXiv:2607.18552). The last one's PDF was not text-extractable via fetch: [arXiv:2607.18552](https://arxiv.org/pdf/2607.18552).
