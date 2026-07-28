# QRC Next-Stage Experiments: Systematic Comparison Plan

**Purpose.** Actionable specification for the coder. Each experiment is self-contained: what to
run, what to measure, what to compare against, and what the result means. Ordered by
cost (cheapest first). Every experiment reuses the existing reservoir trace where possible
(re-extract features, don't re-evolve).

**Baseline to beat.** Weather temperature R² from §4.3 v2 (FID-653 magnitude, ridge):

| horizon | QRC baseline | QRC+RBF | best ESN |
|---------|-------------|---------|----------|
| 1       | 0.950       | 0.953   | 0.956    |
| 10      | 0.888       | 0.866   | 0.849    |
| 20      | 0.860       | 0.815   | 0.820    |
| 30      | 0.812       | 0.834   | 0.752    |
| 45      | 0.786       | 0.778   | 0.697    |

NARMA-10 NMSE baseline: 2.46×10⁻⁵ (ridge).

**Evaluation protocol.** Every experiment reports: (a) weather temperature R² at h=1,10,20,30,45;
(b) NARMA-10 NMSE; (c) feature count; (d) wall-clock time. Same train/val/test split, same seed.
Tabulate deltas vs baseline.

---

## Phase 0 — Instrumentation: Trace Cache + FID/Progress UI (do FIRST, parallel with Phase 1)

**Rationale.** Two things gate everything else and are therefore built up front: (a) a **raw-FID
trace cache** — currently only the 653-feature matrix `X` is saved, not the per-step complex FID,
so the "re-extract without re-evolving" premise of Phases 1/3/4 does not yet hold; and (b) a **UI**
on quantum.cira-core.com to display the FID (time-domain + frequency content) and trace all run
progress live. Both consume the same artifacts, so they ship together.

### 0.A Raw-FID trace cache (foundation for all feature experiments)
- Add a `save_traces` option to the reservoir/runner that dumps, during a pass, an `.npz`:
  `fids: complex64 [n_steps, fid_points]`, `targets`, and config/split metadata.
- One-time generation run (weather ≈14.6 h + NARMA ≈4 h at fid_points=2048) produces the cached
  traces; thereafter Phase 1/3/4 feature experiments re-extract offline in minutes.

### 0.B Offline evaluation harness
- `evaluate_features(X, task, split, seed) -> dict` implementing the Evaluation Protocol exactly
  (weather R² @ h=1,10,20,30,45; NARMA-10 NMSE; feature count; wall-clock; delta-vs-baseline);
  standardized JSON + auto summary table.

### 0.C Backend API + run control (Flask blueprint `/api/qrc`, mirrors `app/routes/qml.py`)

**Deployment note — GPU locality.** QRC runs require the local GPU (RTX 5070 Ti); the Cloudflare
production container has no GPU. Therefore the `qrc_bp` **launcher + execution run on the dev box**,
exposed to the SPA via a **Cloudflare Tunnel** (auth-gated), with the dev box acting as the worker.
The SPA (on Cloudflare/R2) calls this dev-box API for QRC control, progress, FID and results.

Register `qrc_bp` at `/api/qrc` in `app/__init__.py`. Endpoints:

*Monitor (read-only):*
- `GET /api/qrc/runs` — list runs (id, task, config, status) from the run registry.
- `GET /api/qrc/runs/<id>/progress` — progress events (poll `events.jsonl`; SSE optional, matching
  the existing solve-stream pattern): phase, step/total, ETA, event log, per-order/horizon results.
- `GET /api/qrc/runs/<id>/fid?step=k` — time-domain FID (t, real, imag, |.|) **and** frequency
  content (freq_Hz, magnitude, selected 653 peak positions), read from the trace `.npz`.
- `GET /api/qrc/runs/<id>/results` — results JSON (NARMA/weather tables, ESN comparison).

*Control (auth-gated — reuse `auth_bp`/admin; these launch GPU jobs):*
- `POST /api/qrc/runs` — launch a run from a JSON config (`task` ∈ trace-gen / narma / weather /
  phase1; fid_points, splits, horizons, feature method, …). Spawns the existing runner as a managed
  subprocess on the dev box (mirror the QML/solve launcher: run registry + PID + `events.jsonl`),
  returns a run id. Enforce a **single-GPU-job lock** (one heavy run at a time) + a config allow-list.
- `POST /api/qrc/runs/<id>/stop` — terminate a running job.
- `GET /api/qrc/runs/<id>/status` — running / done / failed + exit info.

This makes the UI a **remote control plane**: submit, monitor, and stop GPU runs entirely from the
browser while away from the dev box. Operational requirement: the dev box stays powered and the
tunnel stays up (it is the worker).

### 0.D Frontend UI (Vue 3 + Vuetify + Pinia, mirrors the QML pages)
- `stores/qrc.ts` (axios), router entries `/qrc` and `/qrc/runs/:id`.
- `QrcDashboardPage.vue` — list of runs with live status (task, progress %, ETA), **plus a
  "New run" control** (choose task + config → `POST /api/qrc/runs`) and per-run **Stop** buttons
  (auth-gated). This is the remote command surface.
- `QrcRunDetailPage.vue` — the core view:
  - **FID time-domain plot** (real + imaginary vs time) with a **step slider** to scrub input steps.
  - **Frequency-domain plot** (magnitude spectrum vs Hz) with the 653 selected peaks highlighted.
  - **Live progress trace** — progress bar + ETA + scrolling event log (polls `.../progress`).
  - **Results panel** — the NARMA/weather-vs-ESN tables and comparison figures.
- Charting: add a performance chart lib (uPlot recommended for 2048-point FID/spectrum; chart.js
  acceptable) — currently the frontend has no chart dependency.
- Deploy: build (`npm run build`) served by the existing site; no new infra (see `deploy/`).

**Deliverable of Phase 0:** a working `/qrc` section on quantum.cira-core.com showing the FID, its
spectrum, and live progress for any run — plus the trace cache + harness that Phase 1 needs.

---

## Phase 1 — Flag-Level Experiments (reuse cached FID traces from Phase 0)

**Cost: minutes to hours each. No new reservoir evolution needed — re-extract features from the
Phase-0 cached FID traces.**

### Experiment 1.1: Add FID Phase Information

**What.** Turn on `FeatureConfig.fid_complex = True`. This adds real and imaginary FFT
coefficients at the same 653 frequency bins, giving ~1959 features (653 magnitude + 653 real +
653 imaginary).

**Why.** Phase carries information about coherence order and spin correlations that magnitude
discards. Das et al. 2026 (Phys. Rev. Research) showed higher-order observables improve
nonlinear memory — phase is a higher-order observable.

**Config change.**
```python
feature_cfg = FeatureConfig(
    mode='fid',
    fid_complex=True,   # was False
    # everything else unchanged
)
```

**Measure.** Weather R² and NARMA NMSE with the expanded feature set. Compare vs 653-magnitude
baseline. If phase helps → it is nearly free information gain.

**Expected outcome.** Modest improvement (5–15% error reduction) on weather, especially at
long horizons where phase-encoded correlations carry memory information.

---

### Experiment 1.2: Enable Multimodal FID Features

**What.** Turn on the existing multimodal toggles:
```python
feature_cfg = FeatureConfig(
    mode='fid',
    fid_complex=True,
    time_domain=True,      # was False — adds mean, var, skew, kurtosis, envelope decay
    wavelet=True,           # was False — adds CWT coefficients
    nonlinear=True,         # was False — adds sample entropy, permutation entropy
)
```

**Why.** The project's own finding #1: "readout is the primary determinant." More independent
features = more capacity. The multimodal features capture aspects of the FID that pure FFT
bins miss (decay dynamics, complexity, multi-scale structure).

**Measure.** Same protocol. Record total feature count. Compare vs Exp 1.1 and baseline.

**Expected outcome.** Further improvement, especially if the new feature types are genuinely
independent of FFT magnitude. If feature count exceeds ~2000 with only 600 training samples,
watch for overfitting (monitor train vs test gap).

---

### Experiment 1.3: Feature Selection After Expansion

**What.** Take the full expanded feature set from Exp 1.2 and apply selection/reduction. Run
each of these independently on the SAME expanded features:

**Method A — LASSO (embedded selection):**
```python
from sklearn.linear_model import LassoCV
lasso = LassoCV(cv=10, max_iter=10000).fit(X_train, y_train)
selected = np.where(np.abs(lasso.coef_) > 0)[0]
# retrain Ridge on selected features only
```

**Method B — PCA (linear projection):**
```python
from sklearn.decomposition import PCA
for n in [50, 100, 200, 400, 653]:
    pca = PCA(n_components=n).fit(X_train)
    X_train_pca = pca.transform(X_train)
    # train Ridge on X_train_pca
```

**Method C — UMAP (nonlinear projection):**
```python
import umap
for n in [50, 100, 200]:
    reducer = umap.UMAP(n_components=n, random_state=42)
    X_train_umap = reducer.fit_transform(X_train)
    # train Ridge on X_train_umap
```

**Method D — Mutual information filter:**
```python
from sklearn.feature_selection import mutual_info_regression
mi = mutual_info_regression(X_train, y_train, random_state=42)
for k in [50, 100, 200, 400]:
    top_k = np.argsort(mi)[-k:]
    # train Ridge on X_train[:, top_k]
```

**Why.** With 1500–2000+ features and ~600 training samples, dimensionality reduction is
essential. The question is whether simple PCA suffices or nonlinear methods (UMAP) or
task-aware selection (LASSO, MI) add value.

**Measure.** For each method and each n_components/k: weather R² and NARMA NMSE. Plot
performance vs feature count for all methods on one chart.

**Expected outcome.** LASSO or MI selection likely outperforms PCA at the same feature count
(task-aware > task-blind). UMAP may help if the feature manifold is nonlinear. The optimal
feature count is likely 100–400 (sweet spot between information and overfitting).

**Key deliverable.** A single plot: x-axis = number of features, y-axis = weather R²@h=30,
one curve per method. This plot is a publication figure.

---

## Phase 2 — Encoding Sweep (requires new reservoir evolution)

**Cost: one full reservoir run per configuration (~15h each for weather at N=9).**

### Experiment 2.1: Encoding Function Sweep

**What.** Run the weather benchmark with each of the 7 implemented encoding functions,
everything else fixed at baseline settings (FID-653 magnitude, ridge):

```python
for func in ['arcsin_sqrt', 'arccos', 'linear', 'sinusoidal',
             'logarithmic', 'polynomial', 'exponential']:
    cfg = QRCConfig(
        encoding=EncodingConfig(function=func),
        # ... everything else = baseline
    )
    run_weather_benchmark(cfg)
```

**Why.** Different encoding curvatures map different input ranges to different quantum
rotations. Some may be better for weather data than the default arcsin(√s). This is a
straightforward systematic study that nobody has published for NMR QRC.

**Measure.** Weather R² at all horizons for each function. Rank functions. Also run
NARMA-10 for each to see if ranking is task-dependent.

**Expected outcome.** 2–3 functions will cluster near the top; very different functions
(e.g., logarithmic vs exponential) may suit different tasks. Task-dependent ranking is
itself a finding.

---

### Experiment 2.2: Phase-Amplitude Encoding

**What.** Enable the coded-but-unused phase-amplitude mode:
```python
encoding_cfg = EncodingConfig(
    function='arcsin_sqrt',
    phase_amplitude=True,  # was False — adds R_z(2πs) before R_x(θ)
)
```

**Why.** Doubles the encoding DOF per input step (angle + phase). May produce richer
reservoir dynamics. Paper 4 doesn't use this.

**Measure.** Same protocol. Compare vs amplitude-only baseline.

---

### Experiment 2.3: Protons-Only vs All-Spins Encoding

**What.** Compare encoding targets:
```python
# Option A: protons only (Paper 4's actual choice for NARMA)
encoding_cfg = EncodingConfig(target_qubits='protons')

# Option B: all spins (current default)
encoding_cfg = EncodingConfig(target_qubits='all')
```

**Why.** The state document notes this deviation from Paper 4 is "unexamined." It may
matter because encoding on bath (carbon) spins changes their contribution to the dynamics
even though they are not directly read out.

**Measure.** NARMA NMSE for both. Weather R² for both (weather already uses frequency-selective
encoding, so this mainly affects NARMA).

---

## Phase 3 — External Feature Extraction Libraries (new code required)

**Cost: days to implement. Operates on saved FID traces (no re-evolution).**

### Experiment 3.1: TSFRESH Automatic Feature Extraction

**What.** Apply TSFRESH to the raw FID time-domain signal (real and imaginary parts
separately) at each time step.

**Implementation outline:**
```python
import tsfresh
from tsfresh.feature_extraction import (
    MinimalFCParameters,
    EfficientFCParameters,
    ComprehensiveFCParameters,
)

def extract_tsfresh_features(fid_signal_complex, step_id):
    """Extract TSFRESH features from one FID signal."""
    real_part = np.real(fid_signal_complex)
    imag_part = np.imag(fid_signal_complex)
    mag_part  = np.abs(fid_signal_complex)

    # Format for TSFRESH: DataFrame with id, time, value columns
    df = pd.DataFrame({
        'id': step_id,
        'time': np.arange(len(real_part)),
        'real': real_part,
        'imag': imag_part,
        'mag':  mag_part,
    })

    # Three levels of extraction
    for level_name, params in [
        ('minimal',       MinimalFCParameters()),
        ('efficient',     EfficientFCParameters()),
        ('comprehensive', ComprehensiveFCParameters()),
    ]:
        features = tsfresh.extract_features(
            df, column_id='id', column_sort='time',
            default_fc_parameters=params,
            n_jobs=0,  # single-threaded for reproducibility
        )
        # save features for this level
```

**Also apply TSFRESH to the FFT magnitude spectrum** (treat the spectrum as a 1D signal and
extract features from it).

**Three extraction levels to compare:**
- `MinimalFCParameters` (~10 features per channel → ~30 total)
- `EfficientFCParameters` (~200 features per channel → ~600 total)
- `ComprehensiveFCParameters` (~800 features per channel → ~2400 total)

**Feature sets to evaluate:**
- TSFRESH-only (minimal / efficient / comprehensive)
- Baseline FFT-653 only
- FFT-653 + TSFRESH-efficient (combined)
- FFT-653 + TSFRESH-comprehensive (combined)

**Measure.** Weather R² and NARMA NMSE for each feature set. Use LASSO/PCA from Exp 1.3
on combined sets.

**Why.** Tests whether data-driven features add value over Paper 4's simple FFT peak
selection. The combined set tests complementarity.

**Expected outcome.** TSFRESH-comprehensive alone likely underperforms FFT-653 (it doesn't
know about NMR physics). Combined set may outperform both by adding genuinely independent
features. TSFRESH's `fft_coefficient` features will partially overlap with FFT-653.

**Key deliverable.** Table comparing feature sets. If combined > either alone → paper figure.

---

### Experiment 3.2: nmrglue Physics-Informed Processing

**What.** Process the FID through a proper NMR pipeline before feature extraction:

```python
import nmrglue as ng

def nmrglue_features(fid_complex, dwell_time, num_points):
    """Physics-informed feature extraction from FID."""
    # 1. Apodization (exponential line broadening)
    fid_apod = ng.proc_base.em(fid_complex, lb=5.0)  # 5 Hz broadening

    # 2. Zero-fill to next power of 2
    fid_zf = ng.proc_base.zf_size(fid_apod, num_points * 2)

    # 3. FFT
    spectrum = ng.proc_base.fft(fid_zf)

    # 4. Auto phase correction
    spectrum = ng.proc_autophase.autops(spectrum, 'acme')

    # 5. Baseline correction
    spectrum = ng.proc_bl.baseline_corrector(spectrum, wd=20)

    # 6. Peak picking
    peaks = ng.peakpick.pick(spectrum, pthres=0.1, algorithm='downward')

    # 7. Extract physics features per peak:
    features = []
    for peak in peaks:
        features.extend([
            peak['position'],    # chemical shift (Hz)
            peak['amplitude'],   # peak height
            peak['linewidth'],   # FWHM → T2* information
            peak['integral'],    # peak area → population
        ])

    # 8. Also extract: spectral moments, total integral, peak count
    features.extend([
        len(peaks),                    # number of peaks
        np.sum(np.abs(spectrum)),      # total spectral power
        spectral_centroid(spectrum),   # center of mass
        spectral_spread(spectrum),     # spectral width
    ])

    return np.array(features)
```

**Feature types extracted:**
- Per-peak: position, amplitude, linewidth (FWHM), integral (4 per peak)
- Global: peak count, total power, spectral centroid, spectral spread
- Phase-corrected spectrum samples (analogous to FFT-653 but phase-corrected)

**Feature sets to evaluate:**
- nmrglue physics features only
- FFT-653 + nmrglue physics features (combined)
- TSFRESH + nmrglue (combined)
- All three combined (FFT-653 + TSFRESH + nmrglue)

**Why.** Tests whether proper NMR processing (apodization, phase correction, baseline
correction) improves over the raw FFT approach. Paper 4 does NOT do these processing
steps (just raw FFT → magnitude peaks), so any improvement is a novel contribution.

**Caveat.** In simulation, the FID is noise-free and phase is known analytically.
Phase/baseline correction may have minimal effect in simulation but would matter on
real hardware. Worth doing for methodology even if the sim improvement is small.

---

## Phase 4 — Dimensionality Reduction Comparison (on best feature set from Phase 3)

**Cost: hours. Operates on pre-extracted features.**

### Experiment 4.1: Systematic Reduction Benchmark

**What.** Take the best combined feature set from Phase 3 (likely 1500–3000 features).
Apply each reduction method. Evaluate downstream prediction with both Ridge and SVR-RBF.

**Methods to compare:**

| ID | Method | Type | Params to sweep |
|----|--------|------|-----------------|
| R0 | Raw (no reduction) | — | — |
| R1 | PCA | Linear projection | n = 50, 100, 200, 400, 653 |
| R2 | Kernel PCA (RBF) | Nonlinear projection | n = 50, 100, 200 |
| R3 | UMAP | Nonlinear projection | n = 50, 100, 200; n_neighbors = 15, 30 |
| R4 | LASSO selection | Sparse selection | alpha via CV |
| R5 | MI top-k | Filter selection | k = 50, 100, 200, 400 |
| R6 | Random projection | Baseline | n = 50, 100, 200, 400 |

**For each (method, params, downstream_model) triplet, record:**
- Weather R² at h = 1, 10, 20, 30, 45
- NARMA-10 NMSE
- Number of output features
- Fit time + predict time (wall-clock)

**Key deliverable.** A heatmap or table: rows = reduction methods, columns = horizons,
cells = R². Plus the performance-vs-feature-count plot from Exp 1.3 extended to all methods.

**Why.** Answers the question: "Is simple PCA enough, or do we need nonlinear/task-aware
methods?" This is a publishable comparison that nobody has done for QRC features.

---

## Phase 5 — Self-Supervised Representation Learning (only if Phase 4 shows headroom)

**Cost: days to weeks. Requires PyTorch training loops.**

**Gate condition.** Only proceed to Phase 5 if Phase 4 shows that:
(a) nonlinear methods (UMAP/Kernel PCA) outperform PCA by > 3% R², OR
(b) the best reduced feature set still leaves > 5% gap to a theoretical ceiling.

If PCA at n=200 already matches the best achievable R², SSL will not help — stop here.

### Experiment 5.1: Denoising Autoencoder

**What.** Train a denoising autoencoder on the full feature set (unlabeled — uses ALL
available FID feature vectors, not just labeled training samples).

```python
class DenoisingAE(nn.Module):
    def __init__(self, input_dim, bottleneck_dim=100):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, bottleneck_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, input_dim),
        )

    def forward(self, x, noise_std=0.1):
        x_noisy = x + noise_std * torch.randn_like(x)
        z = self.encoder(x_noisy)
        x_recon = self.decoder(z)
        return x_recon, z

# Train on ALL feature vectors (unlabeled)
# Loss = MSE(x_recon, x_clean)
# After training: z = encoder(features) gives reduced representation
```

**Sweep:** bottleneck_dim = [50, 100, 200], noise_std = [0.05, 0.1, 0.2]

**Compare with:** PCA at same n_components (from Phase 4).

**Why.** Tests whether learned nonlinear compression outperforms PCA. The denoising
objective specifically teaches robustness to noise, which is relevant for eventual
sim-to-real transfer.

---

### Experiment 5.2: Contrastive Learning (if 5.1 shows promise)

**What.** Apply TS2Vec or a simple temporal contrastive objective to learn representations
that capture temporal structure in the feature sequences.

**Positive pairs:** Feature vectors at time t and t+1 (temporal neighbors)
**Negative pairs:** Feature vectors at time t and t+k for large k

**Compare with:** Denoising AE from 5.1 and PCA from Phase 4.

**Why.** Tests whether temporal contrastive learning discovers structure that purely
spatial methods (PCA, AE) miss.

---

## Phase 6 — Encoding × Feature Interaction (if time permits)

**Cost: multiple full runs. The expensive but highest-novelty experiments.**

### Experiment 6.1: Best Encoding × Best Features Cross-Study

**What.** Take the top-2 encoding functions from Exp 2.1 and the best feature set from
Phase 3. Run the full weather benchmark with each combination:

```
encodings:  [best_1, best_2, arcsin_sqrt (baseline)]
features:   [FFT-653, best_combined, best_reduced]
models:     [Ridge, SVR-RBF]

→ 3 × 3 × 2 = 18 runs
```

**Why.** Tests whether encoding and feature improvements are additive (independent) or
interact (some encodings work better with certain feature types).

**Expected outcome.** If improvements are additive → total gain = encoding_gain +
feature_gain. If they interact → some combinations may be synergistic. Either result
is a finding.

---

### Experiment 6.2: τ Sweep with Best Configuration

**What.** Take the best encoding × feature × model from Exp 6.1. Sweep τ:
```python
for tau in [0.005, 0.01, 0.02, 0.03, 0.05, 0.1]:
    # run weather benchmark
```

**Why.** τ controls the balance between coherent evolution and dissipation. Different τ
values access different memory regimes. This has never been studied systematically.
T1 and T2 are fixed physical constants — τ is the only free time parameter.

**Measure.** Weather R² vs τ for each horizon. Memory capacity vs τ. Plot the memory
capacity curve — this is a physical characterization of the reservoir.

---

## Summary: Experiment Priority and Cost

| Phase | Exp | Description | New evolution? | Est. time | Priority |
|-------|-----|-------------|:-:|-----------|:--------:|
| 0 | 0.A/B | Trace cache + eval harness | trace-gen ×1 (~19 h once) | 1–2 days code | **P0** |
| 0 | 0.C/D | FID/progress UI on quantum.cira-core.com | No | 3–5 days | **P0** |
| 1 | 1.1 | Add FID phase | No | 30 min | **P0** |
| 1 | 1.2 | Enable multimodal | No | 1 hr | **P0** |
| 1 | 1.3 | Feature selection comparison | No | 2 hr | **P0** |
| 2 | 2.1 | Encoding function sweep (×7) | Yes (×7) | 4–5 days | **P1** |
| 2 | 2.2 | Phase-amplitude encoding | Yes (×1) | 15 hr | **P1** |
| 2 | 2.3 | Protons-only vs all-spins | Yes (×1) | 15 hr | **P1** |
| 3 | 3.1 | TSFRESH features | No | 1 day | **P1** |
| 3 | 3.2 | nmrglue processing | No | 1–2 days | **P1** |
| 4 | 4.1 | Reduction benchmark | No | 4 hr | **P2** |
| 5 | 5.1 | Denoising autoencoder | No | 1 day | **P3** |
| 5 | 5.2 | Contrastive learning | No | 2 days | **P3** |
| 6 | 6.1 | Encoding × Features cross | Yes (×18) | 2 weeks | **P3** |
| 6 | 6.2 | τ sweep | Yes (×6) | 4 days | **P2** |

**P0** = do immediately (flag changes, hours).
**P1** = do next (days, either new evolution or new extraction code).
**P2** = do after P1 results are in.
**P3** = do only if earlier phases show headroom / justify complexity.

---

## Decision Criteria

### After Phase 1: decide readout vs encoding priority

- If Phase 1 improves weather R²@h=45 by >5% (0.786 → >0.825): readout enrichment is the
  primary lever → prioritize Phase 3 (TSFRESH/nmrglue) over Phase 2 (encoding sweep).
- If Phase 1 improvement is <2%: current readout may be near saturation → encoding sweep
  (Phase 2) becomes more valuable.

### After Phase 3: decide reduction strategy

- If combined features >2000 and LASSO selects <300 with no loss: LASSO is sufficient,
  skip Phase 5 (SSL).
- If PCA at n=200 matches LASSO: linear structure dominates, skip UMAP and SSL.
- If UMAP significantly outperforms PCA: nonlinear manifold exists, Phase 5 worth trying.

### After Phase 4: decide SSL investment

- If best reduction already matches or exceeds Paper 4 experimental R²: publish without SSL.
- If gap remains and unlabeled data is plentiful: Phase 5 may close it.

---

## Required Dependencies (for the coder)

```
# Phase 1-2: already available
# nothing new needed

# Phase 3:
pip install tsfresh
pip install nmrglue

# Phase 4:
pip install umap-learn

# Phase 5:
# PyTorch already available (used by training.py)
# TS2Vec: pip install ts2vec  OR  clone from github.com/yuezhihan/ts2vec
```

---

## Output Format

Each experiment should produce a JSON results file:
```json
{
    "experiment": "1.1_fid_phase",
    "timestamp": "...",
    "config": { ... },
    "feature_count": 1959,
    "weather_r2": {
        "h1": 0.955, "h10": 0.901, "h20": 0.872,
        "h30": 0.831, "h45": 0.810
    },
    "narma10_nmse": 1.8e-5,
    "wall_clock_s": 3600,
    "delta_vs_baseline": {
        "h1": "+0.005", "h10": "+0.013", ...
    }
}
```

Plus a summary table auto-generated across all experiments for comparison.

---

## Publication Framing

The systematic comparison itself is the novel contribution. Regardless of which method
wins, the paper answers:

1. **Does FID phase information improve QRC?** (Exp 1.1)
2. **Do multimodal features exceed spectral-only?** (Exp 1.2)
3. **Which feature selection method is best for QRC?** (Exp 1.3 + 4.1)
4. **Is encoding function choice task-dependent?** (Exp 2.1)
5. **Does physics-informed NMR processing help?** (Exp 3.2)
6. **Do data-driven features complement physics features?** (Exp 3.1)
7. **Do encoding and readout improvements interact?** (Exp 6.1)
8. **Is τ optimization a viable lever?** (Exp 6.2)

Each question is answerable regardless of whether the answer is positive or negative.
Negative results (e.g., "PCA is sufficient, SSL doesn't help") are equally publishable
as methods guidance for the QRC community.

---

**Document version:** 1.0
**Based on:** QRC_Development_State.md (July 2026)
**Prepared for:** AI coder implementation
