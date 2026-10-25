# Drift / nonstationarity fixes, NG-RC and quantum NG-RC, hybrid quantum-classical reservoirs, and real-world RC benchmarking practice

Scope: concrete fixes that could transfer to a fixed physical (NMR) reservoir with 653 FFT-magnitude FID features and a linear ridge readout, which fails on ETTh1/ETTh2 (test period shifted about 1.3 s.d.) but is level with ESN/LSTM on weather. Each item notes whether the evidence comes from **real nonstationary data** and whether it was tested on a **physical/quantum reservoir**.

Context finding up front: the platform this project simulates appears to be the one in Hou et al., Phys. Rev. Lett. 136, 120602 (2026), arXiv 2508.12383. That paper uses a 9-spin 13C-labelled crotonic acid NMR reservoir (Bruker 400 MHz), FFT of the FID, 653 spectral-peak features, z-scored, ridge readout with 10-fold CV. It reports QRC about level with ESN(1000) on Delhi daily climate temperature (better at longer horizons), above ESN(10000) only with an RBF-SVR post-processor, and comparable to ESN(10000) on humidity. Inputs are normalized to [0,1] and encoded as arcsin-angle RF rotations. It reports **no linear/AR baseline and no handling of nonstationarity** (washout/rewinding address initial-state dependence, not data drift) — [Hou et al. 2026, arXiv 2508.12383v2](https://arxiv.org/html/2508.12383v2). This matters: the published weather result was never tested against a drift-heavy dataset or a linear baseline, and the project's ETT results are the first stress test of that kind.

## Q1. Drift / nonstationarity handling in RC (differencing, RevIN, online/RLS readouts, rolling refits, FORCE, online normalisation, residual learning)

### Takeaway
The best-supported fix for a train/test level shift is a *forecasting-side* change, not a reservoir change: re-centre the target and inputs on the last observed value (NLinear-style), or use instance normalisation (RevIN), so that the readout predicts *deviations* rather than absolute levels. Evidence for this is strong on real data including ETTh1/ETTh2, but only for linear and deep models; none of it was tested on reservoirs. Adaptive readouts (RLS with forgetting) are well established for physical reservoirs, but they are motivated by hardware drift, and benchmarks on real nonstationary data are thin.

### Cited Findings
**Last-value re-centring / instance normalisation (strongest real-data evidence; ETT-specific)**
- NLinear subtracts the last value of the input window, applies one linear layer, and adds the value back. It is motivated explicitly by train/test distribution shift: inputs normalised with training statistics push predictions outside the test distribution. The paper reports clear train-test shift on **ETTh1, ETTh2 and ILI**, where NLinear greatly improves on plain Linear. On datasets without obvious shift (Electricity), Linear is about as good — [Zeng et al., AAAI 2023, arXiv 2205.13504](https://ar5iv.labs.arxiv.org/html/2205.13504).
- Univariate horizon-96 MSE (Table 9): ETTh1 NLinear 0.053, DLinear 0.056, against best Transformer 0.071 (Autoformer). The reported Linear 0.189 may be misread because the table alignment is ambiguous. ETTh2: Linear 0.133, NLinear 0.129, DLinear 0.131, FEDformer 0.128 — [Zeng et al. 2023](https://ar5iv.labs.arxiv.org/html/2205.13504). Evidence: real nonstationary data, yes; reservoir, no.
- LTSF-Linear (one-layer linear models) beat all the sophisticated Transformer LTSF models tested, often by a large margin, across nine real datasets. The authors attribute the Transformers' earlier gains mainly to direct multi-step (DMS) forecasting rather than to the attention architecture — [Zeng et al. 2023](https://doi.org/10.1609/aaai.v37i9.26317).
- RevIN (Kim et al., ICLR 2022) normalises each input instance with its own mean and std (plus a learnable affine), then denormalises the output with the same statistics. It is model-agnostic and reported significant gains across forecasting models — [Kim et al. 2022, ICLR](https://mlanthology.org/iclr/2022/kim2022iclr-reversible); [code](https://github.com/ts-kim/RevIN).
- A critical re-examination (Berthelier et al., EDF R&D/Inria, arXiv 2603.11869, Mar 2026) finds:
  - The **affine layer is not beneficial** in practice.
  - Training in normalised space is what matters.
  - Instance normalisation reduces mean/variance heterogeneity "neither completely nor consistently". On Traffic it *increased* distribution distance, and it does not address conditional shift.
  - On Electricity 168-24, test MSE fell from 78.28 (standard normalisation) to 22.04 (RevIN with normalised-space loss), and DLinear showed the same pattern.
  - ETT was not tested — [Berthelier et al. 2026](https://arxiv.org/html/2603.11869v1).
- I found no paper applying RevIN to ESNs or physical/quantum reservoirs — (search returned nothing; see Gaps).

**Online / adaptive readouts**
- Jaeger (NeurIPS 2002) introduced online RLS adaptation of the ESN readout. Convergence rate and misadjustment can be set at design time — [Jaeger 2002](https://papers.neurips.cc/paper_files/paper/2002/file/426f990b332ef8193a61cc90516c1245-Paper.pdf).
- An exponentially weighted (forgetting-factor λ) RLS readout for ESNs: a smaller λ forgets faster and adapts to sudden changes. This was tested in RL control settings, not forecasting benchmarks — [arXiv 2602.06326, 2026](https://arxiv.org/pdf/2602.06326).
- RLS costs O(N^2) per step and is "notorious for numerical stability issues". Instability of recursive ESN identification grows with the norm of the output weights — [Saadat et al.](https://www.aut.upt.ro/~rprecup/IJAI_42.pdf); search summary of a [UFC 2019 thesis](https://repositorio.ufc.br/bitstream/riufc/70718/1/2019_eve_gabarreto.pdf) (secondary — not verified in full text).
- **Physical reservoir evidence:** online training of analogue readouts in photonic reservoir computers keeps the model updated as the input varies and also compensates for gradual hardware drift — [arXiv 2012.10613](https://arxiv.org/pdf/2012.10613). Tested on a physical reservoir: yes (photonic). Real nonstationary forecasting data: not established from the abstract.
- RC embeddings can detect concept drift without supervision in dynamical systems, beating nonlinear-time-series-analysis baselines (Thorne, Corrêa, Zaitouny, Small, Jüngling, *Chaos* 2025) — [UWA repository](https://research-repository.uwa.edu.au/en/publications/reservoir-computing-approaches-to-unsupervised-concept-drift-dete/). This is detection, not correction. Data were mainly synthetic dynamical systems.

**Residual learning / linear + reservoir decomposition**
- Several ARIMA+ESN hybrids exist on real data:
  - Seasonal ARIMA handles the seasonal part and an ESN the remainder, for mobile traffic (Neural Comput. Appl. 2014) — [HIT](https://scholar.hit.edu.cn/en/publications/a-novel-hybridization-of-echo-state-networks-and-multiplicative-s/).
  - HAR+ESN beat ARIMA, HAR, MLP and a plain ESN on Nasdaq realised volatility on most cases — (search-result summary; primary not fetched).
  - AR-order-selected inputs to a deep ESN for wind speed — (Iraqi journal, low evidence quality).
  - None of these are physical reservoirs. Evidence quality is low to moderate (small venues).
- I found no paper that specifically trains a reservoir on the *residual of a tuned AR model* with a 2019-2026 benchmark. The classical template is the Zhang 2003 / Khashei-Bijari ARIMA-ANN residual hybrids — (Gaps).

### Inferences
- ETTh1/ETTh2 are the canonical "shifted" datasets in the LTSF literature (Zeng et al. say so explicitly). A model that predicts absolute levels from bounded/saturating features will fail there. The project's tanh/clip/rank transforms bound the *features* but leave the *target level* unshifted, which is consistent with their not helping. The NLinear lesson is to move the level into the readout via the last value.
- Concrete transfers for a non-trainable reservoir with a ridge readout, roughly in order of expected payoff and cost (all offline on saved traces, except where noted):
  1. **Level-free target.** Predict y(t+h) − y(t) (or y(t+h) − mean of the last window) and add the level back. This needs no new GPU evolution and can be applied to existing traces.
  2. **Skip/linear terms in the readout.** Append the raw last k inputs (and a bias) to the 653 FID features, giving a single ridge fit over [FID ⊕ lags]. The linear part can then extrapolate the level while the reservoir supplies nonlinear corrections. See Q2 for evidence. This is also offline.
  3. **Residual stacking.** Fit the tuned AR first, then fit ridge on the FID features to the AR residual, with a separate ridge penalty for that stage. Offline.
  4. **Input-side instance normalisation / differencing** before encoding into RF rotation angles: feed Δu or a window-normalised u into the reservoir, so that the [0,1]→arcsin encoding stays inside its trained range. This changes the reservoir drive, so it needs **new GPU evolution** and fresh traces (see CLAUDE.md GPU/trace rules).
  5. **Rolling-origin or forgetting-factor RLS readout refits** on the 653-D features (cheap at D=653). RLS is the established tool for physical reservoirs (photonic), but on real drift it may only partly recover; it does not fix a level shift the features cannot represent.
- Under drift, the gap between QRC (−0.15 to 0.36) and ESN/LSTM/AR (0.75-0.81) is most likely an *extrapolation-of-level* failure of a feature map with no linear pass-through, not lack of memory or nonlinearity. Fix 2 or 1 should therefore be tried before any reservoir re-design.

### Gaps
- No source found applying RevIN, NLinear-style re-centring, or differencing to ESN/physical/quantum reservoir inputs. The idea is unvalidated in the RC literature.
- No systematic benchmark found of forgetting-factor RLS ESN readouts on real nonstationary forecasting data (ETT, weather), and no FORCE-learning evidence on real drift data. FORCE was not searched in depth.
- No 2019-2026 reservoir-on-AR-residual benchmark found on standard datasets.

## Q2. Next-generation RC (NG-RC), quantum NG-RC, and linear + nonlinear feature concatenation (skip connections)

### Takeaway
NG-RC's feature vector is explicitly [constant ⊕ linear delayed inputs ⊕ polynomial monomials], so it always contains a linear pass-through. Concatenating reservoir states with NG-RC features into one ridge readout (hybrid RC-NGRC, Chepuri et al. 2024) beats either component alone on chaotic benchmarks, about doubling valid prediction time. However, all of this evidence is synthetic and stationary. NG-RC is also fragile when the true nonlinearity is not in the feature library. Quantum NG-RC variants exist (theory plus one quantum-optics experiment) but are not tested on real drifting data.

### Cited Findings
- NG-RC (Gauthier et al., Nat. Commun. 2021) is nonlinear vector autoregression. Its features are the current input plus k−1 delayed inputs spaced by s, plus nonlinear (polynomial) monomials of those terms. It needs no random matrices, has fewer hyperparameters, and is interpretable — [Gauthier et al. 2021, arXiv 2106.07688](https://arxiv.org/pdf/2106.07688).
- **Hybrid RC-NGRC** (Chepuri, Amzalag, Antonsen, Girvan; *Chaos* 34(6), 2024):
  - Method: the reservoir state is concatenated with all NG-RC feature terms, H(t) = r(t) ⊕ O(t), and fit with a single ridge readout.
  - Lorenz median valid prediction time: hybrid 4.13 Lyapunov times, NGRC 2.06, RC 0.98.
  - A hybrid with N=100 reservoir nodes roughly matches a pure RC with N=500.
  - Climate (long-term statistics) error on Lorenz: hybrid 6.0e-3, NGRC 3.9e-2, RC 8.8e-1.
  - The hybrid is robust over a wide range of input scaling.
  - Gains are small on Mackey-Glass. When a well-specified NGRC exists, NGRC alone is more data-efficient.
  - Data: synthetic chaotic systems only. Reservoir: classical ESN — [Chepuri et al. 2024, arXiv 2403.18953](https://arxiv.org/html/2403.18953v2).
- **NG-RC catch-22** (Zhang & Cornelius, Phys. Rev. Research 2023): NG-RC reconstructs basins of attraction with very little warm-up *if* the exact nonlinearity is in the feature library. A small uncertainty in the nonlinearity can make its accuracy "no better than chance". Standard ESNs need warm-up covering almost the whole transient. Data: synthetic multistable systems — [arXiv 2210.10211](https://ar5iv.labs.arxiv.org/html/2210.10211); [summary](https://comdig.cssociety.org/2023/10/04/catch-22s-of-reservoir-computing/).
- NG-RC extrapolation: Köglmayr & Räth (2023 preprint) used NG-RC trained on stationary data to extrapolate bifurcation/tipping behaviour. Data: synthetic dynamical systems — (reported in search summary of NG-RC literature; primary not fetched; see [arXiv 2106.07688 lineage](https://arxiv.org/pdf/2106.07688)). Treat as unverified.
- **Physical NG-RC:**
  - Optical NG-RC via light scattering in disordered media, driven directly with time-delayed inputs, forecasts Lorenz63 and Kuramoto-Sivashinsky (arXiv 2404.07857, 2024) — [arXiv 2404.07857](https://www.arxiv.org/pdf/2404.07857).
  - Memristor NG-RC on Mackey-Glass/Duffing — [TU Dresden](https://fis.tu-dresden.de/portal/en/publications/next-generation-memristor-reservoir-computing(b67ab15a-e0d2-4a98-9ea7-5c9985894797).html).
  - Physical reservoir: yes. Real nonstationary data: no.
- **Quantum NG-RC:**
  - Sornsaeng, Dangniam, Chotibut (arXiv 2308.14239, 2023): a quantum algorithm for NG-RC that needs only quantum time-series data, and forecasts integrable and chaotic many-body quantum dynamics. Data: simulated quantum dynamics — [arXiv 2308.14239](https://arxiv.org/pdf/2308.14239).
  - Wang, Sun, Kong, Sun, Zhang (Phys. Rev. A 111, 022609, 2025): a QRC scheme implementing quantum nonlinear vector autoregression. It extracts linear and nonlinear features through measurements instead of long quantum evolution, needs less training data than other QRC schemes, and is experimentally verified in quantum optics. The datasets are not named in the abstract — [arXiv 2502.16938](https://arxiv.org/pdf/2502.16938).
  - Attribution check: the search did not find a quantum NG-RC paper by Čindrak. Čindrak et al. work on QRC artificial memory restriction (2023) and on the "memory-nonlinearity trade-off across QRC frameworks" (2026, with Lüdge). The latter argues that memory restriction, weak measurement, edge-of-chaos operation and dissipation are all governed by one memory-vs-nonlinearity balance — [arXiv 2306.12876](https://arxiv.org/abs/2306.12876); [arXiv 2603.21371](https://arxiv.org/abs/2603.21371v1). The brief's "Cindrak et al. 2024 quantum NG-RC" attribution is likely a misattribution.

### Inferences
- The project's readout has **no linear pass-through of the input**. The 653 FFT-magnitude features are nonnegative, bounded, and phase-discarding nonlinear functions of the drive history. NG-RC and hybrid RC-NGRC always keep the linear delayed-input block. Under a level shift, only that block can extrapolate linearly. Appending raw lags to the 653 features (the "hybrid RC-NGRC" recipe with the NMR reservoir in the RC slot) is the most direct transfer. It is also cheap: offline, on existing traces, with a single ridge fit or block-wise penalties.
- Chepuri et al. show the concatenation is never worse than its best component on their benchmarks, given CV-tuned ridge. On ETT this predicts QRC+lags ≥ linear AR (0.75-0.81) if regularisation is tuned per block. This is unconfirmed on real data.
- The catch-22 result warns against expecting polynomial NG-RC monomials to fix ETT. The dominant failure is level, not nonlinearity.
- FFT *magnitude* discards sign and phase. A level shift in input changes RF rotation angles, which may move features non-monotonically, so bounding features cannot restore a linear relation to the target. This is speculative and could be checked offline by regressing features on input level.

### Gaps
- No NG-RC or hybrid RC-NGRC evaluation on real nonstationary data (ETT, weather) found.
- No quantum NG-RC result on real data found. The datasets of Wang et al. 2025 were not confirmed.
- No direct test found of "skip connection fixes extrapolation under covariate shift" for any reservoir. The evidence is indirect (NLinear/DLinear plus hybrid RC-NGRC).

## Q3. Hybrid quantum-classical reservoirs (QRC + ESN/linear ensembles, stacking, physics-informed hybrid RC, quantum features appended to classical models)

### Takeaway
Hybridisation via concatenated features or a knowledge-based model plus RC is well supported classically (Pathak 2018; Chepuri 2024). Quantum-classical hybrids exist (ensembles of small quantum reservoirs with classical feedback, deep hybrid QRC, multi-dimensional hybrid QRC) but are tested almost entirely on synthetic chaotic systems. Independent benchmarks on real data mostly find classical models matching or beating quantum ones. No study found stacks QRC features onto a strong linear/AR model on real drifting data.

### Cited Findings
- **Pathak et al. 2018** (*Chaos* 28, 041101): an RC combined with an imperfect knowledge-based model predicts much longer than either alone. Tests: a low-dimensional chaotic system and a spatiotemporal one, both synthetic — [arXiv 1803.04779](https://arxiv.org/pdf/1803.04779). Extended to partial, noisy observations with ETKF data assimilation — [arXiv 2102.07819](https://arxiv.org/abs/2102.07819v1).
  - Transfer: the "knowledge-based model" can be the tuned AR. Its prediction can be fed both as an input feature to the readout and as the base for the output.
- **Hybrid QRC ensemble** (IIT Delhi / Penn State, arXiv 2407.21602, 2024): an ensemble of small quantum systems linked by classical feedback. Benchmarked on NOAA sea-surface temperature (real data) against other ML methods, claiming reduced training time and memory — [arXiv 2407.21602](https://arxiv.org/pdf/2407.21602). Abstract-level only; drift handling not reported.
- Deep hybrid classical-quantum RC (Nokkala et al., MLST 2024) retrieves past quantum features. Only the output layer is optimised — (search summary; [utupub](https://www.utupub.fi/bitstreams/fec60a9c-47f0-461a-8b74-5722238cc363/download)).
- Multi-dimensional hybrid QRC (5 qubits, input-modulated, temporal multiplexing) was tested on Lorenz-63 and reduced Navier-Stokes. Data: synthetic — [arXiv 2509.04006](https://arxiv.org/html/2509.04006v1).
- Distributed QRC claims MAE/RMSE reductions of up to 78.8%/72.3% from quantum-enhanced configurations — [arXiv 2605.04991](https://arxiv.org/pdf/2605.04991). This is a preprint with an abstract-level claim; baselines were not checked.
- **Real-data benchmarks** of quantum models (mostly variational, some QRC):
  - BASF/Aqarios/SAP univariate benchmark: classical models generally beat current quantum models even in noiseless simulation, with occasional wins over ARIMA — [arXiv 2412.13878](https://alphaxiv.org/overview/2412.13878v1).
  - A 27-task chaotic benchmark: quantum models struggle to match simple classical models of comparable complexity — arXiv 2504.12416 (search summary).
  - Financial QRC/QML (2026 preprint): quantum-enhanced models match classical models in univariate settings and do modestly better in correlated multivariate settings — [emergentmind summary of 2605.02656](https://www.emergentmind.com/papers/2605.02656). Secondary source.
- The NMR QRC paper (Hou et al. PRL 2026) pairs QRC with an RBF-SVR post-processor to exceed ESN(10000) on temperature. This is effectively a classical nonlinear stage stacked on quantum features — [arXiv 2508.12383v2](https://arxiv.org/html/2508.12383v2).

### Inferences
- The highest-value "hybrid" for the project is classical-first stacking: ŷ = AR(t) + ridge(FID features ⊕ lags → AR residual). The AR carries the level/trend that drifts, and the quantum reservoir is judged only on whether it adds signal beyond the AR. This also gives a clean, publishable incremental-value test (ΔR² over AR) on weather, where the AR is already within 0.01-0.04 of the best model.
- An ensemble/average of QRC and ESN can only help where their errors are uncorrelated. Under drift, QRC errors are dominated by a level bias, so stacking with a learned weight (not equal averaging) or residual stacking is preferable.

### Gaps
- No real-data study found of QRC features appended to a classical AR/ESN readout under distribution shift.
- Quantitative baselines of arXiv 2407.21602, 2605.04991 and 2504.12416 were not verified in full text.

## Q4. Real-world benchmarking practice for RC on ETT/weather, and what simple baselines teach about drift

### Takeaway
In the LTSF literature, ETTh1/ETTh2 are known shifted datasets. Trivially normalised linear models (NLinear/DLinear) beat Transformers there, and per-instance normalisation (RevIN) is now standard. RC papers on ETT are rare. The one found (ParalESN, ICML 2026) uses train-split normalisation plus clipping, with results not verified here. Any QRC claim on ETT should report Linear/NLinear/DLinear, a tuned AR, a persistence/Repeat baseline, and an ESN under the same normalisation protocol.

### Cited Findings
- Zeng et al. 2023 report explicit train/test shift on ETTh1, ETTh2 and ILI, and NLinear's last-value subtraction is the remedy. LTSF-Linear beats all the Transformer LTSF models tested on nine real datasets. Multivariate Repeat (persistence) MSE at h=96: ETTh1 1.295, ETTh2 0.432 — [Zeng et al. 2023](https://ar5iv.labs.arxiv.org/html/2205.13504).
- ParalESN (Pinna, Lagomarsini, Ceni, Gallicchio; ICML 2026, arXiv 2601.22296) evaluates on ETTh1/ETTh2/ETTm1/ETTm2, all 7 variables, horizon 192:
  - Normalisation is fitted on the training split, and normalised data is clipped to (−10, 10).
  - ETTh1 split: 8640 train / 2880 val / 2880 test.
  - It claims accuracy competitive with traditional RC and fully trainable sequence models.
  - Numbers were not retrieved — [arXiv 2601.22296](https://arxiv.org/abs/2601.22296).
- RevIN-style normalisation is the de facto drift remedy in deep LTSF, but its benefit is inconsistent across datasets and does not cover conditional shift — [Kim et al. 2022](https://mlanthology.org/iclr/2022/kim2022iclr-reversible); [Berthelier et al. 2026](https://arxiv.org/html/2603.11869v1).
- In the NMR QRC weather paper, the classical baselines are only ESNs (500-10000 nodes). There is no linear/AR baseline — [Hou et al. 2026](https://arxiv.org/html/2508.12383v2).

### Inferences
- The project's observation that a tuned linear AR is within 0.01-0.04 R² of the best nonlinear model on weather, and at 0.75-0.81 on ETT, matches the LTSF lesson: on these real series most predictable variance is linear and level-like. Reservoir nonlinearity can add at most a small margin, and only if the level is handled linearly.
- For ETT, report R² *and* MSE on the standard LTSF split, and include NLinear as a baseline. If the QRC+lags or AR-residual-stack does not beat NLinear/AR, the honest claim is "no added value under drift".
- Because the project's ETT test set is shifted about 1.3 s.d., train-split z-scoring followed by the [0,1]→arcsin angle encoding likely drives test inputs outside the range seen in training, or saturates them. A RevIN/NLinear-style input transform keeps the encoding in-distribution. This is an inference; it needs new reservoir evolution to test.

### Gaps
- No published RC/QRC results with numbers on ETTh1/ETTh2 were retrieved (ParalESN numbers not fetched).
- No RC paper found that reports a linear/NLinear baseline alongside the reservoir on ETT.
