# QRC Tier-A Reproduction + SLEEPY Validation — Implementation Spec

**Purpose.** Reproduce Hou et al. 2026 (PRL 136, 120602) faithfully in our simulator,
then triangulate correctness three ways: **our QuTiP/GPU sim ↔ SLEEPY (independent
NMR engine) ↔ the paper's published numbers.** This document is the shared spec for the
coder agents; each owns disjoint files and codes to the interface contracts below.

---

## 0. Interface contracts (all coders code to these, not to each other's live code)

- **FID array:** `np.ndarray`, dtype `complex128`, shape `(sim.fid_points,)`. Produced by
  `QRCSystem.fid_signal(rho) -> np.ndarray`. Index 0 = t=0.
- **FeatureConfig** (in `features.py`, owned by Coder B) gains:
  - `readout: Literal["observables","fid"] = "observables"`
  - `n_peaks: int = 653`  (fixed spectral-peak count for the FID readout)
  - existing `time_domain/wavelet/nonlinear` toggles apply as *multimodal add-ons* on the FID (Tier B).
- **Feature extraction (fid mode):** `FeatureExtractor.from_fid(fid: np.ndarray) -> (np.ndarray, list[str])`.
  Peak *bin positions* must be fixed across time steps (see §3) so the design matrix is consistent.
- **Reservoir loop** (in `evolution.py`, Coder A): when `features.cfg.readout=="fid"`, per step
  carry `rho` forward (reservoir update), then `fid = sys.fid_signal(rho)`, `feats,names = features.from_fid(fid)`.
- **SimConfig** (in `config.py`, Coder A) gains: `fid_points:int=2048`, `fid_dwell:float=3e-4`
  (s), `readout_qubits:list[int]=field(default_factory=list)` (empty → all).
- **Metric** `nmse_paper(y_true,y_pred)` = `Σ(y-ŷ)² / Σy²` (in `utils.py`/`tasks.py`, Coder B).

---

## 1. Exact molecule — 9-spin ¹³C crotonic acid (SM TABLE II)

Order: `[C1,C2,C3,C4,H1,H2,H3,H4,H5]`. H3/H4/H5 = equivalent methyl. **Readout = protons
(indices 4–8); carbons (0–3) are an inaccessible bath.** Larmor refs: ¹³C 100.6273 MHz,
¹H 400.2118 MHz. Chemical shifts are rotating-frame offsets in **Hz**.

**Chemical shifts ν (Hz):** C1 −7749.7, C2 5430.1, C3 2699.9, C4 7673.7, H1 985.9,
H2 520.3, H3=H4=H5 −1081.5

**T₁ (s):** C1 5.9, C2 4.9, C3 5.6, C4 27.5, H1 3.2, H2 3.4, methyl 2.2
**T₂\* (ms):** C1 212, C2 231, C3 208, C4 241, H1 203, H2 332, methyl 320

**J-couplings (Hz), symmetric** (methyl value applies to each of H3,H4,H5; intra-methyl
J3-4,3-5,4-5 = 0, magnetically equivalent):

| pair | J | pair | J | pair | J |
|---|---|---|---|---|---|
| C1-C2 | 40.8 | C2-C3 | 69.5 | C1-C3 | 1.6 |
| C1-C4 | 8.5 | C2-C4 | 1.4 | C3-C4 | 71.0 |
| C1-H1 | 4.0 | C2-H1 | 155.6 | C3-H1 | −1.8 |
| C4-H1 | 6.5 | C1-H2 | 6.6 | C2-H2 | −0.7 |
| C3-H2 | 162.9 | C4-H2 | 3.3 | H1-H2 | 15.8 |
| C1-methyl | 128.0 | C2-methyl | −7.1 | C3-methyl | 6.6 |
| C4-methyl | −0.9 | H1-methyl | 6.9 | H2-methyl | −1.7 |

Add as `config.crotonic_acid_paper4()` returning a `SystemConfig` (n=9). QA verifies every
number against this table.

---

## 2. FID readout algorithm (Coder A, `system.py: fid_signal`)

Paper Eq. 4: `S(t) ∝ Tr[ e^{tL}(UρU†) · O_FID ]`, `O_FID = Σ_{p∈protons}(σ_y^p + i σ_x^p)`,
readout pulse `U = Π_{p∈protons} R_x^p(π/2)`.

1. `rho_read = U rho U†` (dense matmul).
2. `r = vec(rho_read)` (column-stacking, matching the Liouvillian convention already used
   by the `action` backend).
3. Evolve `r` under the **readout Liouvillian** `L_read` and sample `⟨O_FID⟩(t)` at
   `fid_points` times spaced by `fid_dwell` → complex FID array.
   - CPU: `scipy.sparse.linalg.expm_multiply(L_read, r, start=0, stop=(fid_points-1)*dwell,
     num=fid_points)` then dot each column with `o = vec(O_FID^T)`.
   - GPU: reuse the Taylor sub-stepping stepper, sampling at the `fid_points` grid, dot with `o`.

   **Stiffness optimisation (important, and exact):** the carbon chemical-shift terms
   `π ν_C σz_C` commute with the Heisenberg evolution of the proton transverse operators
   (σz_C commutes with σ_±^H and with every σz), so they do **not** affect the proton FID.
   Build `L_read` from an `H_read` that **omits carbon chemical-shift terms** (keep all
   J-couplings, all proton chemical shifts, all T₁, and proton T₂; carbon T₂/dephasing also
   commutes and may be dropped). This lowers ‖L‖ from the carbon scale (~7750 Hz) to the
   proton scale (~1250 Hz), cutting the exponential cost ~6–8×. **QA must verify** on a small
   system that `fid_signal` with `L_read` matches the full-`L` result to ≤1e-6 (relative).

Sampling: proton offsets span ≈±1100 Hz, so `fid_dwell=0.3 ms` (Nyquist 1667 Hz) suffices;
`fid_points=2048` → window 0.61 s (≥ T₂\*, and 1024 positive-frequency bins ≥ n_peaks).

---

## 3. Feature extraction from the FID (Coder B, `features.py: from_fid`)

1. `spec = np.fft.rfft(fid)` → complex spectrum, `|spec|` magnitude.
2. **Fixed peak bins:** the peak *positions* are set by the (fixed) Hamiltonian, only their
   amplitudes vary with ρ_k. On the first call, select the `n_peaks` largest-`|spec|` bins and
   **cache those bin indices**; reuse them for every subsequent step so the feature vector is
   consistent. (If fewer than n_peaks nonzero bins, pad with zeros + stable names.)
3. Base feature vector = magnitude at the cached bins (Paper-4-style, 653). Optionally also
   real+imag (doubles features) — expose a flag; default magnitude-only for the repro.
4. **Tier-B multimodal add-ons** (only when toggled): on the FID time series add time-domain
   stats, PyWavelets decomposition, antropy entropies, and peak widths/phases → push total
   toward 1000–2000+. Degrade gracefully if libs absent (existing pattern).
5. Z-score standardisation of the design matrix happens in training (Coder B ensures the
   ridge path standardises, matching the paper's "Z-score + ridge + 10-fold CV").

Feature order/names must be deterministic and stable across steps.

---

## 4. Tasks, metrics, baselines (Coder B, `tasks.py` + `benchmarks.py`)

- **NARMA input:** superposition of sine waves (paper), not uniform random. Provide
  `narma_input_sine(n, seed)`. Target via paper Eq. 2:
  `y_{k+1} = α y_k + β y_k Σ_{i=0}^{n-1} y_{k-i} + γ s_{k-n+1} s_k + δ` (use standard
  α=.3,β=.05,γ=1.5,δ=.1; NARMA2 keep the quadratic variant already present). Keep the existing
  uniform-random NARMA too (don't break current tests).
- **Metric:** add `nmse_paper` (Σ/Σy²) and report both it and R² for NARMA. Splits: 400 train
  / 100 test (NARMA), 374 washout / 600 train / 600 test (weather).
- **Baselines:** extend `esn_baseline` usage to a sweep **ESN(500,1000,5000,10000)**; add an
  optional **RBF-SVR readout** (`sklearn.svm.SVR`, guarded) as the "QRC+RBF" comparison.
- **Reproduction runner:** `scripts/qrc_reproduce_paper4.py` — crotonic9, τ=0.01 s NARMA
  (orders 2,5,10,15,20) with FID-653 readout, GPU backend; prints NMSE table alongside the
  paper's Table I values; saves JSON. (Weather can be a follow-up hook; wire the structure.)

---

## 5. SLEEPY validation (Coder C, `app/qrc/validate_sleepy.py` + test + pyproject extra)

**Goal:** independently confirm our QuTiP Lindblad NMR physics. SLEEPY is CPU-only, so
validate on a **small homonuclear proton subsystem** (fast, and our weak-coupling Ising H is
exact there). Add `sleepy-nmr` to a new `[validation]` extra in `pyproject.toml`.

SLEEPY API (import name is `SLEEPY`, pip name `sleepy-nmr`):
```python
import SLEEPY as sl
ex = sl.ExpSys(v0H=400.2118, Nucs=['1H','1H'])   # proton Larmor MHz, homonuclear pair
ex.set_inter('CS', i=0, Hz=985.9)                # H1 offset
ex.set_inter('CS', i=1, Hz=520.3)                # H2 offset
ex.set_inter('J',  i0=0, i1=1, J=15.8)           # VERIFY key: 'J' with i0/i1/J=
L = sl.Liouvillian(ex)
L.add_relax('T1', i=0, T1=3.2); L.add_relax('T2', i=0, T2=0.203)
L.add_relax('T1', i=1, T1=3.4); L.add_relax('T2', i=1, T2=0.332)
L.add_relax('recovery')
seq = L.Sequence(Dt=fid_dwell)
rho = sl.Rho('1Hx','1Hp')      # start transverse, detect σ+  (i.e. after a 90° pulse)
rho.DetProp(seq, n=fid_points) # acquire FID
fid_sleepy = np.asarray(rho.I[0])   # VERIFY accessor for the complex signal
```
**Comparison:** build the *same* 2-proton system in our QuTiP sim (a 2-qubit `SystemConfig`
with those ν/J/T1/T2), compute `fid_signal`, and compare to SLEEPY:
- FFT both; peak **positions must match to < 1 Hz**;
- normalised FID trajectories agree to a few-percent RMS (relaxation/lineshape).
Provide `validate_sleepy.compare(...)` returning a dict of metrics, and a pytest
`test_qrc_sleepy.py` that `importorskip("SLEEPY")` and asserts the tolerances. Coder C must
**verify the exact SLEEPY API** (J key, pulse, signal accessor, whether CS 'Hz' is the
rotating-frame offset) from the tutorial/source and adjust; note any deviations in a module
docstring. If an API detail can't be confirmed, implement best-effort and mark the test
`xfail` with a clear reason rather than asserting wrong physics.

---

## 6. File ownership (no overlaps → safe parallel edits)

- **Coder A:** `config.py` (crotonic preset, SimConfig FID fields), `system.py`
  (`fid_signal`, readout ops, `L_read`), `evolution.py` (fid readout mode).
- **Coder B:** `features.py` (`FeatureConfig` fields, `from_fid`), `tasks.py`
  (sine NARMA, metric), `benchmarks.py` (ESN sweep, RBF), `scripts/qrc_reproduce_paper4.py`.
- **Coder C:** `app/qrc/validate_sleepy.py`, `tests/test_qrc_sleepy.py`,
  `pyproject.toml` `[validation]` extra only.

## 7. Acceptance criteria (checked in QA + final review)

1. `ruff` clean; existing `test_qrc.py`/`test_qrc_physics.py` still pass (no regressions).
2. `fid_signal` reduced-`L_read` matches full-`L` to ≤1e-6 on a 3–4 spin system.
3. FID spectrum of crotonic protons is physically sensible (peaks near ±ν_H ± J splittings).
4. NARMA repro (τ=0.01 s, FID-653): NMSE within ~1 order of magnitude of the paper's Table I
   (exact match not expected — different molecule realisation/noise-free sim; the goal is the
   same regime, ~1e-4–1e-6, and a large jump over the observable-only readout).
5. SLEEPY vs QuTiP: proton-pair FID peak positions < 1 Hz apart; trajectory RMS < ~5%.
6. All new deps optional/guarded; package still imports without qutip/sleepy/torch-cuda.
