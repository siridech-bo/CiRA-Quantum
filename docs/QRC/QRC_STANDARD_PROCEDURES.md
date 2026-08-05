# QRC — Standard Procedures & Conventions (canonical)

**Read this before running or reporting ANY QRC experiment.** It fixes the
conventions so results stay comparable and mistakes (like silently switching the
readout) do not recur. If a convention here proves wrong or must change, change it
*here first*, in a commit, and say so — never switch silently in a script.

---

## 0. The readout — THE STANDARD is the 653-feature FID spectrum

> **Default readout = FID → FFT → 653 largest-magnitude spectral peaks.**
> This is the Paper-4 / production readout used by weather forecasting, the
> encoding study, `qrc_memcap`, and trace-gen. **Report results on this readout
> unless explicitly stated otherwise.**

Pipeline (`FeatureConfig(readout="fid", n_peaks=653)`, `app/qrc/features.py`):
1. After the encoding pulse, evolve and record the **FID** — complex transverse
   magnetization `⟨σx⟩ + i⟨σy⟩` summed over spins — sampled densely in time
   (`fid_points`, 512 for `quick`, 2048 for `full`).
2. `spec = |FFT(fid)|`; keep the **653 largest-magnitude peak bins** (fixed once).
3. Feature vector per input step = **653** spectral magnitudes.
4. The FID's dense time-sampling **is** the time-multiplexing (each dwell sample =
   a virtual node; matches the Du-group NMR-QRC experiment, ref [35]).

### The OTHER readout (observable) — only for differentiable prototyping, MUST be labeled
`step_diff` (the autograd path) reads `⟨σx,σy,σz⟩` per qubit at `V` virtual nodes:
**`3·n_qubits × V`** features (18×V at 6 qubits), optionally + 2-body correlations
`⟨σᵢσⱼ⟩` → **63×V**. It is smaller and NOT the standard. **Any result on the
observable readout must be labeled "(observable readout, D=…)"**, because absolute
capacities/NMSE are readout-dependent (`MC ≤ D`, and 653 ≫ 180).

**Known past mistake (do not repeat):** the 2026-08 learnable-encoding /
correlation / parallel / STM-PC experiments were run on the *observable* readout
(18–180 features), not the 653 FID spectrum, without labeling it. Qualitative
findings (encoding helps; window/feedback confounds are classical; coupling helps)
are expected to survive a richer readout, but the absolute numbers are on the
smaller readout. Unify on 653 (§7) before quoting production numbers.

---

## 1. Reservoir model

- **Substrate:** weak-coupling (Ising-ZZ) NMR spin network. Hamiltonian
  `H = Σ_i π ν_i σz_i + Σ_{i<j} (π/2) J_ij σz_i σz_j` (all diagonal; entanglement
  comes from ZZ evolution of transverse states created by the pulses).
- **Dissipation:** T₁/T₂ Lindblad (physical dephasing model by default).
- **Systems:** `crotonic9_paper4` (9 spins, the paper system); `_resolve_system(n)`
  for generic n-spin. `--coupling-scale` multiplies all J (2.0 = strong coupling).
- **Evolution:** exact `exp(τL)` — `action` (CPU Krylov) / `gpu` (torch Taylor) /
  dense propagator for small systems (`ensure_diff`, dim²≤8192).

## 2. Encoding (input → pulse)

- **Default:** `θ = arcsin(√s)`, applied as `R_x(θ)`. Input `s` scaled to `[0,1]`.
- **Broadcast** (all spins same θ, Paper-4 style) vs **per-spin / frequency-
  selective** (each spin its own learned θ). Per-spin needs a learned encoder.
- **Learnable encoding:** a small MLP `s → θ(s)` (or `→ n` angles), trained by
  backprop through the differentiable step. Global vs per-spin as above.
- NOT the encoding: any scheme that injects an explicit input **window** or
  **feedback of past readouts** — those add *classical* memory (see §6 rigor).

## 3. Readout training (weights)

- **Linear ridge**, closed form: `W = (XᵀX + αI)⁻¹ Xᵀy`, α ≈ 1e-3 (paper 0.05).
- `X` is `T × D` (D = feature count). **Weights per scalar target = D+1** (incl.
  bias). Multiple targets (e.g. MC delays) → `(D+1) × n_targets`, one column each.
- **`D` for the standard readout = 653.** So the readout weight vector is length
  **654** per target. Capacity ceiling `MC ≤ D = 653`.
- **p≫n rule (critical):** need `n_train ≫ D`. For D=653 use large sets
  (paper: washout 1000 / train 1500 / test 1000). **Do not fit 653 features on a
  few-hundred-sample train split** — that overfits and inflates capacity (this is
  the Phase-1 p≫n trap). For the small observable readout (D≤180), `T≈600` is fine.

### 3.1 What is "learned" — two regimes (label every result)

There are exactly two training regimes; **state which one every result uses.**

- **(A) Standard QRC — readout only** (the Das-Giorgi-Zambrini paper and all
  canonical QRC, e.g. Fujii-Nakajima). The reservoir *and the encoding are FIXED*;
  the **only** trained object is the linear readout `W`, fit by one closed-form
  ridge solve (`XW ≈ y`, their Eq. 7; Scikit ridge, α=0.05). No gradients enter the
  quantum dynamics. This is the *whole point* of reservoir computing — cheap
  training, and no barren plateaus / variational-training pathologies. **This is our
  honest baseline.**
- **(B) Learnable encoding — our extension** (beyond the paper). Keep the *same*
  closed-form ridge readout `W`, but *additionally* train the **input encoding**
  (an MLP `s → θ`) by backpropagating through the differentiable reservoir step.
  This is strictly more than the paper does — and it is *why* we hit
  differentiability/memory costs (dense-FID backprop, the 9-spin memory wall) that
  standard QRC never sees.

| | (A) standard QRC | (B) our learnable encoding |
|---|---|---|
| readout `W` | trained (ridge) | trained (ridge, same) |
| reservoir | fixed | fixed |
| encoding | **fixed** (`arcsin√s` / paper's `β=s`) | **learned** (backprop) |

**Reporting rule:** the "trained-readout-only" number (A) is the standard baseline
to compare against the literature; the "trained-readout + trained-encoding" number
(B) is our addition and must be labeled as such. Never present a (B) number as if
it were the standard (A) regime.

### 3.2 Physics-informed reduced FID (for differentiable training)

The full 653-FID readout is memory-heavy to backprop through (regime B). "Fewer"
features is **not** guessed or top-K-by-magnitude — it is the reservoir's *actual*
resolvable line count, computed from the Hamiltonian.

Because `H` is all-`σz` (diagonal), the FID `⟨Σσx⟩` oscillates at a **finite,
analytic** set of single-quantum transition frequencies. Flipping readout spin `k`
against the other spins gives (first-order weak coupling, Hz):

> **`f_k(z) = ν_k + ½ Σ_{j≠k} J_kj z_j`,  `z_j ∈ {±1}`**

— each resonance `ν_k` split into a `2^{n-1}` J-multiplet. Every one of the 653 FFT
peaks *is* one of these lines (or an FFT skirt of one). The encoding sets only the
line **amplitudes**; the frequencies are fixed by `H`, so this basis is a constant
of the reservoir (no reference run, no peak-finding).

**Selection = enumerate the lines, then merge any closer than the decoherence
linewidth `Δf = 1/(π T₂)`** (physically unresolvable). The survivors are `D_eff`
features at *known* frequencies. Read them by a fixed **direct-DFT projection**
`X = fid_samples · Φ`, `Φ[m,k] = exp(i·2π f_k t_m)` — differentiable (`torch.matmul`),
so cheap enough to backprop through, unlike the dense 653-FID.

Measured floors (`app/qrc/spectral_lines.py` → `report_deff()`):

| system (readout) | raw transitions | linewidth | **`D_eff`** | `f_max` | dwell / M |
|---|---|---|---|---|---|
| crotonic-9 (protons 4–8) | 1280 | 1.57 Hz | **100** | 1157 Hz | ~0.17 ms / ~200 |
| generic 6-spin (all) | 192 | 1.59 Hz | **137** | 358 Hz | ~0.56 ms / ~274 |

So the standard 653 readout **oversamples crotonic-9 ~6.5×**; the physics floor is
~100. Rules: (i) keep *all* resolvable lines — don't amplitude-prune, since a line
that's dark for arcsin can carry signal for a learned encoding; (ii) `f_max` sets a
Nyquist-safe `dwell < 1/(2 f_max)`, and `M ≳ 2·D_eff` FID samples resolve the lines;
(iii) this is single-quantum only (the `Σσx` FID) — reading products/multi-quantum
observables adds their transition frequencies. **Label `D = D_eff` and the readout
("physics-informed reduced FID") on every result**, same as any non-653 readout (§0).

## 4. Splits (leakage-free)

- **3-way, time-contiguous:** train / validation / test. Fit ridge on **train**;
  if training an encoder, optimize it on **validation**; report **test**.
- **Never** optimize the encoder on the test split (label leakage).
- **Washout** first (drop initial transient; paper uses 1000).

## 5. Tasks & metrics

| Task | input | target | metric |
|------|-------|--------|--------|
| NARMA-2 | `u~U[0,0.5]` | NARMA-2 recurrence (cubic `0.6u³`) | test **NMSE** |
| NARMA-10 | `u~U[0,0.5]` | NARMA-10 (memory-bound) | test NMSE |
| STM (linear) | `u~U[0,1]` | `y=u_{t-τ}` | capacity `C(τ)=r²`; totMC=Σ_τ C |
| Parity-Check (nonlinear) | `u∈{0,1}` | `(Σ_{j=1..τ} u_{t-j}) mod 2` | Σ_τ C(τ) |
| Weather | Delhi climate | horizon-h forecast | R² (fidelity-fragile) |
| Memory capacity / IPC | `u~U[0,1]` | delayed Legendre `P_d(u_{t-k})` | Σ corr² |

- **NMSE** = `mean((ŷ−y)²)/var(y)` (1.0 = mean-predictor).
- **Capacity** = squared Pearson correlation on the test split.
- Prefer intrinsic capacity metrics (MC/IPC/NARMA) over weather-R² for *screening*
  (weather-R² needs near-full fidelity — the "fidelity wall").

## 6. Rigor rules (mandatory before claiming a quantum result)

1. **τ→0 ablation** — reset the reservoir each step (`--no-memory`). A genuine
   quantum-memory result MUST collapse to a mean-predictor without the reservoir.
   If it survives, the memory was classical.
2. **Classical control** — for anything that injects explicit history (window,
   feedback, spatial feed), fit a classical ridge (linear + quadratic) on the same
   raw window. If it matches, the win is classical, not quantum.
3. **Guardrails** — reject if the baseline ≈ mean-predictor (NMSE > 0.8 =
   under-powered); flag degenerate near-constant encoders (angle-map std < 0.1).
4. **Multi-seed** — vary data seed AND encoder init; report mean±std and win-count.
   A single-seed "beat" is not a result.
5. **Persist waveforms** — any reservoir-evolving run saves the raw FID `.npz`
   before computing any metric (see CLAUDE.md hard rule).
6. **GPU** — confirm scope + get explicit go before every GPU launch (CLAUDE.md).

## 7. Readout unification (action)

To remove the two-readout inconsistency: make the **653-FID spectral readout the
single standard** everywhere. It is differentiable (`torch.fft` + magnitude +
fixed-bin selection), so `step_diff` can emit it for the learnable encoder too.
Until done, **label every result with its readout and D**.

## 8. Reproduction pointers

- Readout construction: `app/qrc/features.py` (`_fid`, `_spectral`, `from_fid`).
- Differentiable step / observable readout: `app/qrc/system.py`
  (`ensure_diff`, `step_diff`), `qrc_correlation_readout.build_readout`.
- Encoding / tasks / benchmarks: `backend/scripts/qrc_learnable_9spin.py`,
  `qrc_memory_benchmark.py`, `qrc_memcap.py`, `qrc_judge.py`.
- Result log: `docs/QRC/QRC_Next_Stage_Status.md/.html`.
