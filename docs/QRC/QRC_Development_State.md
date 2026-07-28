# QRC Development — State of Play

**Purpose of this document.** A factual snapshot of the Quantum Reservoir Computing (QRC)
sub-project so another agent can recommend the next stage. It records what is built, what has
been run (with numbers), what is coded-but-unused, what is genuinely open, and the concrete
design space for the two levers currently under discussion (input encoding and FID feature
extraction). No decisions are pre-made here.

Branch: `qrc-simulation` (all work pushed). Package: `backend/app/qrc/`. Docs: `docs/QRC/`.

---

## 1. What this project is

A GPU-accelerated simulator for QRC on nuclear-magnetic-resonance (NMR) spin networks, built to
(a) prototype experiments for the SPINQ Gemini Lab and (b) reproduce/extend
**Hou et al. 2026, PRL 136, 120602** ("High-Accuracy Temporal Prediction via Experimental QRC in
Correlated Spins"). The reproduction of that paper is complete; the project is now deciding what
novel extension to pursue.

---

## 2. Architecture and capabilities (all implemented and tested)

Package `backend/app/qrc/` (mirrors the existing `app/qml`, `app/qldpc` module style):

| Module | Role |
|---|---|
| `config.py` | Dataclasses (System/Sim/Encoding/Training/QRCConfig); presets: `spinq3`, `generic_nqubit(N)`, `crotonic9`, **`crotonic9_paper4`** (exact 9-spin ¹³C crotonic acid, SM Table II) |
| `system.py` | N-qubit rotating-frame Hamiltonian `H=Σπν σz + Σ(π/2)J σzσz`; Lindblad collapse ops (T1 amplitude damping + T2 dephasing); **four evolution backends**; **`fid_signal`** (FID readout) |
| `encoding.py` | Input → RF pulse; 7 encoding functions; multi-nucleus parallel; phase-amplitude (coded) |
| `evolution.py` | Reservoir loop; observable-multiplex readout **and** FID readout modes; GPU fast paths |
| `features.py` | `FeatureConfig`; observable readout; **`from_fid`** (FFT→peak features); multimodal add-ons (coded) |
| `training.py` | Ridge regression on GPU (torch), k-fold CV |
| `tasks.py` | Memory-capacity, NARMA (uniform + Paper-4 sine), weather loader, `nmse_paper` |
| `benchmarks.py` | Orchestration: memory-capacity gate, NARMA (+multitask), ESN sweep, RBF-SVR, weather, scaling study |
| `validate_sleepy.py` | Independent cross-check vs the SLEEPY NMR engine |
| `main.py` | CLI |

**Physics core.** Open-system Lindblad dynamics (dissipation = fading memory). Fixed exact
molecule parameters (`crotonic9_paper4`): chemical shifts, T1, T2*, full 21-entry J table, all
from the paper's SM. Protons are the readout; carbons are an inaccessible bath.

**Evolution backends** (cross-validated to ≤10⁻³):
- `propagator` — dense Liouvillian superoperator (exact, small N; 4ᴺ memory wall ≈550 GB at N=9).
- `action` — exact sparse Krylov `exp(t·L)` (scipy `expm_multiply`); removes the wall; CPU.
- `gpu` — Taylor + sub-stepping `exp(t·L)` with sparse CUDA matvecs; **≈28× faster than CPU at N=9**.
- `mesolve` — QuTiP adaptive ODE (reference; stiff past ~7 qubits).
An exact reduced-Liouvillian trick (drop bath chemical-shift terms, which commute with the proton
FID) cuts the FID cost ~6–8× (verified to ~1e-14).

**Reproducibility infra.** Durable JSONL event logs + self-contained auto-refreshing
`progress.html`; per-order/per-run checkpointing; fixed seeds. Delhi climate dataset committed.

---

## 3. Current default configuration (the "as-run" settings)

### Input encoding (as used)
- **Function:** θ = **arcsin(√s)** (`arcsin_sqrt`), s∈[0,1] → θ∈[0,π/2].
- **Pulse:** single **global R_x(θ)** (x-axis), once per input step.
- **Target spins:** NARMA → **all 9 spins**; weather → temperature on the 5 protons, humidity on
  the 4 carbons (frequency-selective, multivariate).
- **Phase:** off (amplitude/angle only).
- *Note / deviation:* Paper 4 encodes NARMA on **protons only**; our default encodes on all 9. It
  still reproduced their NARMA numbers, but the choice is unexamined.

### FID feature extraction (as used)
- **Signal:** complex FID `S(t)=⟨Σ_protons(σy+iσx)⟩(t)`, `fid_points` samples at `fid_dwell`=0.3 ms.
- **Transform:** full complex FFT → magnitude spectrum.
- **Features:** the **top-653 magnitude bins**, selected once and **frozen** across steps.
- **Phase discarded** by default (magnitude only; `fid_complex` flag to add real+imag is OFF).
- **Multimodal** (time-domain / wavelet / entropy / peak-width) toggles exist but are **OFF**.

### Other knobs
- τ (evolution/step): NARMA 0.01 s, weather 0.03 s (paper values).
- `fid_points`: reference runs use 2048; a 1024 run exists for the readout study.
- Readout training: ridge + 10-fold CV; QRC+RBF uses CV-tuned RBF-SVR.

---

## 4. Experiments run and results

### 4.1 N-qubit scaling study — observables-only readout (superseded framing)
Swept N=3–9 with a **coarse observable readout** (⟨σx,y,z⟩ × N × V). Findings: NARMA accuracy
improves ~5× with N; **memory capacity peaks at N=4 and saturates**; a size-matched ESN wins on
NARMA at every N. **Conclusion later shown to be a readout artifact** (see 4.2–4.3). The scaling
data remain valid as an ablation *of the observable-only readout*. Report:
`QRC_Simulation_Report.md/.html` (carries a superseding-update banner).

### 4.2 NARMA reproduction — FID-653 readout
`crotonic9_paper4`, τ=0.01 s, sine input, FID-653 (fid_points=2048), 100/400/100 split, GPU,
multitask (one reservoir pass, one readout per order). NMSE:

| order | this work | Hou et al. (expt) |
|---|---|---|
| 2 | 5.19×10⁻⁶ | 1.74×10⁻⁷ |
| 5 | 5.21×10⁻⁵ | 4.44×10⁻⁵ |
| 10 | 2.46×10⁻⁵ | 5.84×10⁻⁵ |
| 15 | 1.82×10⁻⁵ | 6.37×10⁻⁵ |
| 20 | 3.24×10⁻⁶ | 4.34×10⁻⁵ |

Reproduces the paper's 10⁻⁵–10⁻⁶ regime (R²≈0.999). Observable-only readout on the same system
gives NMSE ≈0.25 — the FID readout is decisive. **On NARMA a classical ESN still wins** (2×10⁻⁷ →
9×10⁻⁹ for 500→10000 nodes); this is consistent with the paper (NARMA is an ESN's strong task; the
paper's advantage claim is on weather, not NARMA).

### 4.3 Weather forecasting — the quantum-advantage task
Delhi daily climate; temp→protons, humidity→carbons; single reservoir pass + per-horizon readouts
(1–45 days); ESN(500–10000) + CV-tuned RBF-SVR baselines. **Reference run (v2):** fid_points=2048,
374/600/500 split.

Temperature R² (v2):

| horizon | QRC | QRC+RBF | best ESN | v1 (fid=1024) QRC | Paper expt (digitized) |
|---|---|---|---|---|---|
| 1 | 0.950 | 0.953 | 0.956 | 0.936 | ~0.92 |
| 10 | 0.888 | 0.866 | 0.849 | 0.773 | ~0.85 |
| 20 | 0.860 | 0.815 | 0.820 | 0.741 | ~0.84 |
| 30 | 0.812 | 0.834 | 0.752 | 0.745 | ~0.83 |
| 45 | 0.786 | 0.778 | 0.697 | 0.572 | ~0.82 |

Humidity: QRC beats ESN at long range too (h=45: 0.374 vs 0.233). **Findings:**
- From h≥10 the quantum reservoir beats every ESN size; the **ESN saturates** (500≈10000). This is
  the paper's quantum-advantage signature, **reproduced and quantitatively matched to the experiment**.
- **Readout richness is the lever:** doubling `fid_points` 1024→2048 lifted 45-day temperature QRC
  0.572→0.786 (onto the experiment). This is the paper's own thesis ("capacity limited by the number
  of independent readout functions") made quantitative.
- With the full 653-feature readout, **linear QRC ≈ QRC+RBF** — the readout richness, not the RBF
  post-processing, carries the advantage.

### 4.4 Independent physics validation (SLEEPY)
Two-proton subsystem built identically in our QuTiP engine and SLEEPY (independent Liouville-space
NMR library). FID peak positions agree to **0.27–0.49 Hz**; normalized FID RMS ≈**0.3%**. Confirms
the Lindblad implementation reproduces genuine NMR dynamics.

---

## 5. Key scientific findings (for the recommender)

1. **The readout is the primary determinant** of QRC performance here — not qubit count. FID-653 vs
   observable-only changes NARMA error by ~1000×.
2. **Task-dependent, honest advantage:** none on NARMA (ESN wins; not claimed by the paper), present
   and reproduced on chaotic long-horizon weather.
3. **Readout richness quantified:** performance scales with the number of independent FID readout
   functions (`fid_points`); the sim matches the experiment once resolution matches.
4. **Physics validated three ways:** internal cross-backend, SLEEPY, and the paper's numbers.

---

## 6. Coded but NOT yet exercised (low-effort experiments available)

These exist in code and can be turned on via flags/config — each is roughly one GPU run (the
reservoir features can often be re-extracted rather than re-evolved):

- **FID phase** (`FeatureConfig.fid_complex=True`) — adds real+imag at the 653 bins (currently phase
  is discarded). ~2–3× features, nearly free.
- **Multimodal FID features** (`time_domain/wavelet/nonlinear=True`) — the plan's "beyond-653"
  novel contribution (target 1000–2000+ features). Never evaluated.
- **6 alternative encoding functions** (`arccos, linear, sinusoidal, logarithmic, polynomial,
  exponential`) — never swept.
- **Phase-amplitude encoding** (`EncodingConfig.phase_amplitude=True`) — R_z(2πs)·R_x(θ), a second
  DOF per input. Never evaluated.
- **Encoding target set** — protons-only vs all-spins is a config change (`target_qubits`), never
  compared.
- **Encoding axis** (x vs y).

---

## 7. Genuinely open / not yet coded (higher-effort directions)

From the original plans (`QRC_Simulation_Plan.md` §8–§9, `QRC_Reproduction_Plan.md` §3.4):

- **Composite / multi-pulse encoding** — sequences of pulses per input (more expressive input map).
- **ML-optimized encoding** — learn θ(s) end-to-end through the differentiable simulator (plan §8.6).
- **Learned FID features** — autoencoder/CNN on the FID (plan §9.5).
- **Apodization / windowing** of the FID before FFT (standard NMR SNR/line-shape trick).
- **Physics-informed features** — peak widths (T2*), integrals (populations), cross-peaks,
  J-coupling extraction.
- **τ / memory sweep** — τ is fixed at the paper's values; its effect on long-horizon memory is
  unstudied (T1/T2 are fixed physical constants and must not change).
- **Weather multi-step strategy variants**, larger test splits, ESN hyperparameter parity with the
  paper's grid search.

---

## 8. The two levers under active discussion (design space)

### 8.1 Input encoding
Current: single global R_x, θ=arcsin(√s), amplitude-only. Options (roughly increasing effort):
sweep the 7 functions ▸ phase-amplitude (2nd DOF) ▸ protons-only vs all-spins ▸ composite/multi-pulse
▸ ML-optimized. Motivations: encoding "curvature" suits different tasks; richer encoding raises state
expressivity; the plan frames "encoding as an optimization problem" as a distinct contribution.

### 8.2 FID feature extraction
Current: FFT magnitude, top-653 fixed bins, phase discarded, no multimodal. Options: add phase (cheap)
▸ multimodal wavelet/entropy/time-domain (the 1000–2000-feature extension) ▸ apodization ▸ peak
width/integral/cross-peak features ▸ learned features. Motivation: the number and independence of
readout functions is the demonstrated performance lever, so enriching the readout is the most direct
route to *exceeding* the paper.

**Two framing questions for the next stage:**
- Goal = **beat Paper 4** (accuracy above their 653-feature result) or **study encoding/readout as an
  optimization problem** (methods contribution)?
- Start with the **flag-level experiments** (phase, multimodal, phase-amplitude — days) before the
  **from-scratch** directions (ML-optimized encoding, learned features — weeks)?

---

## 9. Caveats and limitations

- Noise-free simulation (no cross-correlated relaxation / RF inhomogeneity / drift). The sim can
  therefore undercut or (with a rich readout) match the experiment, but does not model device error.
- Weather test split is 500 (not the paper's 600) — the Delhi series (1576 days) cannot fit
  374+600+600+45 for horizon-45 targets.
- GPU backend uses complex64 (validated to ≤2×10⁻³ vs exact CPU).
- NARMA encoded on all 9 spins vs the paper's protons-only (unexamined).
- Paper-4 weather comparison values are **digitized from their Fig 4b** (approximate ±0.03); NARMA
  comparison uses their exact Table I.
- N=9 FID readout is ~24 s/step at fid_points=2048 → weather reference run ≈14.6 h on one GPU.

---

## 10. Reproducibility

- **Install:** `cd backend && pip install -e ".[qrc]"` (+ `.[validation]` for SLEEPY).
- **Hardware/software:** Python 3.12; QuTiP 5.3; PyTorch 2.10 (CUDA 12.8) on an NVIDIA RTX 5070 Ti
  (17 GB); SciPy 1.17; scikit-learn; sleepy-nmr 1.1.2. Seed 42.
- **Runners:** `scripts/qrc_reproduce_paper4.py --task {narma,weather}` (durable logs + progress.html
  + checkpoints); `scripts/qrc_scaling_run.py`; figure scripts `scripts/qrc_make_*.py`.
- **Tests:** `pytest tests/test_qrc.py tests/test_qrc_physics.py tests/test_qrc_sleepy.py` (16 pass;
  physics/SLEEPY auto-skip without deps).
- **Raw data:** `docs/QRC/data/qrc_paper4_narma.json`, `qrc_paper4_weather.json` (v1, fid=1024),
  `qrc_paper4_weather_v2.json` (v2, fid=2048).

---

## 11. Deliverables (current)

- **Journal manuscript:** `docs/QRC/QRC_Manuscript.docx` (+ `QRC_Manuscript_SI.docx`) — abstract,
  methods, results (fading memory, NARMA, weather + advantage, fid-points study, sim-vs-experiment),
  discussion, references; 4 figures + tables.
- **Reproduction write-up:** `docs/QRC/QRC_Reproduction_Results.md`.
- **Scaling report:** `docs/QRC/QRC_Simulation_Report.md/.html` (superseded framing, banner added).
- **Plans:** `docs/QRC/QRC_Simulation_Plan.md` (original, broad), `docs/QRC/QRC_Reproduction_Plan.md`
  (Tier-A spec, complete).
- **Figures:** `docs/QRC/figures/fig1..fig7` (scaling, fading memory, timings, weather QRC-vs-ESN,
  NARMA sim-vs-expt, weather sim-vs-expt, fid-points study).

---

## 12. One-line status

**Paper-4 reproduction is complete and validated (NARMA regime + weather quantum advantage,
quantitatively matched to experiment, SLEEPY-verified). The open question is which extension to
pursue next — richer readout (phase/multimodal/learned) and/or richer encoding
(functions/phase-amplitude/composite/ML-optimized) — to move from *reproduction* to a *novel
contribution*.**
