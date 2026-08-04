# Learnable Encoding for NMR QRC — Concept, Methodology & the Parameter-Shift Rule

**Purpose.** A standing reference for the learnable-encoding direction — the idea,
the differentiable machinery we validated, the simulation methodology, and (in
detail) **how the gradient is obtained on real NMR hardware via the parameter-shift
rule (PSR)**. Written as preparation/reference for the eventual write-up.

**Status.** Concept validated in simulation (autograd through the reservoir is
correct; a learnable encoder trains end-to-end). Hardware PSR is designed here,
not yet run. See the companion conceptual note *The NMR Molecule as a Quantum
Neural Network* and the status tracker for live state.

**Date.** 2026-08 · **Related:** `QRC_Encoding_Study.md` (Phase-2 fixed-function
result), `QRC_Next_Stage_Experiments.md`.

---

## 1. The idea

A quantum reservoir computer runs data through a *fixed* quantum system and trains
only a linear readout. Paper 4 (Hou 2026) encodes each scalar input `s` as a
rotation angle `θ = arcsin(√s)` and applies a global pulse `R_x(θ)`. Our Phase-2
sweep established that, among hand-designed encodings, **`arcsin(√s)` amplitude
encoding into all spins is near-optimal** — every attempt to add structure by hand
(a phase channel; restricting to the readout protons) *reduced* capacity.

That is exactly the wall that motivates this direction. Instead of *guessing*
encodings and testing them one by one, **let the gradient search the whole space
of encodings.** The key realization:

> If the Lindblad evolution is broken into many small time steps, each step is a
> small differentiable transformation of the density matrix — mathematically a
> layer of a (residual) neural network. The molecule *is* a deep network whose
> hidden layers are fixed by physics; we learn only the **input layer** (the
> encoding) and the **output layer** (the readout).

This is the quantum analogue of a **Neural ODE** (Chen et al. 2018): a continuous,
parameterized dynamical system `dρ/dt = 𝓛(ρ; θ)` trained by differentiating
through the evolution. The molecule is a *Quantum Neural ODE*.

**Why it is worth doing even if it does not beat `arcsin(√s)`.** Two outcomes are
both publishable: (a) the gradient discovers an encoding that *beats* the
hand-designed baseline → a performance result; (b) the gradient *rediscovers*
`arcsin(√s)` → a clean, mechanistic confirmation that Paper-4's choice is
gradient-optimal, plus a reusable methodology (differentiable QRC + hardware PSR).
Our 3-spin prototype currently points toward (b) — see §5.

---

## 2. The mathematical picture

**Where the parameter enters.** One reservoir step is: encode, then evolve.

```
ρ  →  U(θ) ρ U(θ)†                 (encoding pulse, θ = encoder(s))
   →  exp(τ·𝓛) [ · ]              (fixed dissipative Lindblad evolution)
   →  ⟨O_i⟩ = Tr(O_i ρ)           (observables → features)
```

The Lindblad generator `𝓛(ρ) = −i[H,ρ] + Σ_k (L_k ρ L_k† − ½{L_k†L_k, ρ})` is
**fixed by the molecule** (chemical shifts, J-couplings, T₁/T₂). It does **not**
depend on θ. The input enters **only** through the encoding pulse `U(θ)`.

**Differentiability.** `exp(τ𝓛)` is applied as a Taylor series of superoperator
matvecs (sub-stepped so ‖hL‖≲1); every operation — the pulse `UρU†`, the matvecs,
the observable projection — is linear and smooth. So the whole map
`θ → features → loss` is differentiable, and the chain rule gives

```
∂L/∂W  =  ∂L/∂ŷ · ∂ŷ/∂f · ∂f/∂ρ · ∂ρ/∂θ · ∂θ/∂W
```

where `W` are the encoder-network weights and `∂ρ/∂θ` is the gradient *through the
quantum evolution* — the only non-trivial factor, obtained by one of the three
methods below.

---

## 3. Three ways to get `∂ρ/∂θ`

| Method | Where | Memory | Cost | Notes |
|--------|-------|--------|------|-------|
| **Backprop / autograd** (reverse-mode) | simulation | O(N·steps) — stores all intermediate states | 1 forward + 1 backward, *independent of #params* | What we use in sim. Cheapest for many parameters. |
| **Adjoint method** | simulation (long/large) | O(state) — constant | ~2× forward | Solve a backward ODE; needed when storing all states is too big (9-spin, long sequences). |
| **Parameter-shift rule (PSR)** | **real hardware** | O(state) — two forward runs | 2 experiments per pulse angle | Exact gradient from *forward-only* experiments. §6. |

The first two are simulation tools; **PSR is the hardware method** and the subject
of §6. Crucially, all three compute the *same* gradient — the simulator learns the
encoding by backprop, and the machine reproduces that gradient by the shift rule.

---

## 4. What we validated in simulation (no framework migration needed)

The existing GPU reservoir stepper is written in **native torch** (a sparse-CSR
Liouvillian, a Taylor `exp(τL)` of `torch.mv` calls, a dense pulse matmul, a sparse
observable projection). It was written for a *non-differentiable* sweep — it ends
with a `.cpu().numpy()` detach — but the ops themselves are autograd-compatible.
We proved this in three steps, all CPU, **zero GPU**:

1. **Autograd flows through a Lindblad reservoir** — `qrc_grad_smoketest.py`
   (dense 3-spin): a learnable encoder → pulse → Taylor Lindblad evolution → ridge
   → NMSE, autograd gradient matches central finite differences to **1.9×10⁻⁸**.
2. **Autograd survives the *production* op family** — `qrc_grad_prod_check.py`
   builds the real production Liouvillian `L` and observable matrix `M` and runs
   the exact production op sequence (sparse-CSR complex matvec + `torch.sparse.mm`);
   the gradient matches finite differences to **3.5×10⁻⁹ in complex128**. Sparse-CSR
   complex autograd works — no custom `autograd.Function`, no COO, **no JAX/Dynamiqs
   migration**. (The larger complex64 gap is only the fp32 finite-difference floor,
   not a gradient error.)
3. **Differentiable step wired in** — `QRCSystem.ensure_diff()` / `step_diff()`
   (device-agnostic) lift the detach behind a torch-returning path.

**Conclusion.** The document that framed this as "requires a non-trivial migration
to Dynamiqs/JAX" is wrong for our codebase. We have a differentiable Quantum Neural
ODE *today*, one lifted detach away from the production stepper.

---

## 5. Simulation methodology (the prototype)

`qrc_learnable_proto.py` — the first end-to-end learnable encoding, CPU 3-spin:

- **Encoder.** A tiny MLP `s → θ ∈ (0,π)` (random init — deliberately *not* started
  at `arcsin(√s)`), so what the gradient converges to is informative.
- **Reservoir.** The differentiable `step_diff` for a light 3-spin system (real
  Lindblad structure: shifts + J-coupling + T₂), features = ⟨σx,σy,σz⟩ per spin per
  virtual node.
- **Readout.** Closed-form **ridge** recomputed each forward (the readout is always
  the optimal linear map given the features; it is differentiable through the normal
  equations). Only the *encoder* is trained — the molecule's dynamics and the readout
  are not "weights" we learn.
- **Objective.** Train/test split; loss = **test NMSE** with ridge fit on train (so
  the encoder is pushed to *generalize*, not memorize).
- **Optimizer.** Adam on the encoder weights.

**A methodological trap we hit and fixed (record it).** The first run used a short
sequence (T=44 → 21 train points) with **37 reservoir features** → a **p≫n**
regime. The ridge overfit train (NMSE 0.10) and blew up on test (NMSE 5.6); the
"learned beats baseline" it produced was a pure artifact. **This is the same
overfitting pathology as Phase 1.** Fixed by using enough steps that
`n_train ≫ n_features` (T=240). *Any* encoding comparison must live outside the
p≫n regime.

**Honest result (valid regime).** The pipeline trains (test NMSE 0.99→0.74,
monotonic — gradients flow, Adam works). But in this short, under-converged run on
a deliberately weak reservoir it **loses to `arcsin(√s)`** (0.737 vs 0.574), and
the learned `θ(s)` converges to a **monotonic, arcsin-like shape** (Fig. 12). The
gradient is walking *back toward* Paper-4's choice — consistent with Phase-2's
"arcsin is near-optimal." Not a failure: a first, honest signal.

**Definitive simulation test (the go-gated GPU scale-up).** Needs: (i) convergence
(hundreds of steps), (ii) a **strong 9-spin reservoir** (the real system, not the
light one), (iii) **multiple seeds**, (iv) harder tasks (NARMA-10). Cost: each
9-spin forward is minutes, so this is a real GPU campaign — to be scoped and
explicitly authorized before launch.

---

## 6. The parameter-shift rule on real NMR hardware

This is the heart of the method: how the *same* gradient is obtained on a physical
molecule, where backprop is impossible.

### 6.1 Why you cannot backpropagate on hardware

On a real spectrometer you can only go **forward**: prepare a state, apply pulses,
let the spins evolve, and measure. You **cannot** (a) run the evolution backwards,
(b) read out intermediate states without collapsing them, or (c) access the
internal linear-algebra graph. So the gradient must be built from **forward
experiments only**.

### 6.2 The rule, and why it is *exact*

For an encoding pulse generated by a Pauli operator — `U(θ) = exp(−i θ σ_x/2)`,
with `σ_x² = I` (two eigenvalues ±1) — any measured expectation value

```
f(θ) = ⟨ M ⟩(θ) = Tr[ M · U(θ) ρ U(θ)† ]
```

is a **pure first-harmonic sinusoid** in θ. Writing `U(θ)=cos(θ/2)I − i sin(θ/2)σ_x`
and using `cos²(θ/2)=(1+cosθ)/2`, `sin²(θ/2)=(1−cosθ)/2`, one gets

```
f(θ) = A + B cos θ + C sin θ .
```

Its derivative `f′(θ) = −B sinθ + C cosθ` is recovered **exactly** from two
evaluations shifted by ±π/2:

```
f(θ+π/2) − f(θ−π/2) = −2B sinθ + 2C cosθ = 2 f′(θ)

⇒   ∂⟨M⟩/∂θ = ½ [ ⟨M⟩(θ + π/2) − ⟨M⟩(θ − π/2) ] .
```

Three properties matter:

- **Exact, not an approximation.** No `Δθ→0` limit. Two experiments give the true
  analytic derivative.
- **Macroscopic shift (π/2), not infinitesimal.** This is the robustness win. A
  naive finite difference `[f(θ+ε)−f(θ−ε)]/2ε` divides by a tiny ε and *amplifies*
  measurement noise; it also carries O(ε²) truncation error. PSR divides by 2 — it
  is well-conditioned against readout noise.
- **Forward-only.** Each term is a normal experiment at a different pulse angle.

### 6.3 Validity *through* dissipation and downstream evolution

A real reservoir step has dissipative evolution and a readout *after* the encoding
pulse — and later input steps after that. Does PSR still hold? **Yes**, feature by
feature, because everything after `U(θ)` is a **fixed, θ-independent, linear**
channel `Φ`:

```
⟨O_i⟩(θ) = Tr[ O_i · Φ( U(θ) ρ U(θ)† ) ]
         = Tr[ Φ†(O_i) · U(θ) ρ U(θ)† ] .
```

This is again an expectation value of a *fixed* effective observable `Φ†(O_i)` in
the state `U(θ)ρU(θ)†`, so it is still sinusoidal in θ → **PSR is exact for each
measured feature.** We never need to compute `Φ†(O_i)`: the machine applies `Φ`
(the dissipative evolution + readout) for us; we just measure `⟨O_i⟩` at the shifted
angles. The trained linear readout is applied *classically* afterward, and the loss
gradient is chained through it on a laptop.

### 6.4 Sequences and chaining to the encoder network

The input is a *sequence*; step `k` has its own angle `θ_k = NN(s_k; W)`. The angle
`θ_k` affects the state from step `k` onward, so it influences the features at every
`t ≥ k`. Shifting **only** the step-`k` pulse by ±π/2 and re-running the full
sequence yields `∂f_{t,i}/∂θ_k` for all downstream `t,i` in **one experiment pair**
(the tail is θ_k-independent, so `⟨M⟩(θ_k)` is still sinusoidal). Then:

```
∂L/∂θ_k = Σ_{t≥k, i} (∂L/∂f_{t,i}) · ∂f_{t,i}/∂θ_k        (measured, via PSR)
∂L/∂W   = Σ_k (∂L/∂θ_k) · ∂θ_k/∂W                          (classical, known network)
```

**Experiment count:** 2 experiments per distinct pulse → **2·T experiments per
gradient step** for a length-`T` sequence (not `2·#weights` — the encoder network
is differentiated classically for free). With ~100 gradient steps and `T≈200`, that
is ~4×10⁴ experiments; at ~1 min each, ~a month of pure acquisition. Expensive —
mitigations: shorter training sequences, mini-batching over sequence windows,
stochastic coordinate selection (PSR on a random subset of pulses per step), and
warm-starting from the simulation-learned encoder.

### 6.5 Global vs. per-spin pulses — the generator caveat

The clean two-term rule needs a generator with **two** eigenvalues. Two cases:

- **Per-spin angle** (`U = R_x(θ_i)` on one spin): generator `σ_x/2`, eigenvalues
  ±½ → **standard 2-term PSR, shift π/2.** Clean.
- **Global pulse** with the *same* θ on all `n` spins (Paper-4 style): generator
  `Σ_i σ_x^{(i)}/2` has eigenvalues `{−n/2,…,n/2}` → `f(θ)` is a sum of sinusoids at
  frequencies 1…n. A single 2-term rule is **not** exact. Use the **generalized
  parameter-shift rule** (Wierichs et al. 2022): `2R` shifted evaluations for `R`
  distinct frequency gaps (here `R=n`), i.e. **2n experiments per pulse**.

**Design implication:** prefer **per-spin (frequency-selective) encoding angles**.
It keeps PSR at its cheap 2-term form *and* is strictly more expressive than a
single global angle — a natural fit for a learnable encoder that outputs one angle
per addressable nucleus.

### 6.6 NMR-specific advantages and what is actually measured

- **Ensemble readout ⇒ low noise.** Unlike gate-model qubits, where each `⟨M⟩`
  needs many repeated shots and shot noise dominates, **liquid-state NMR measures a
  bulk ensemble (~10¹⁸ molecules)**: the FID *is* the expectation-value signal,
  acquired essentially in one shot. PSR gradients are therefore unusually clean on
  NMR — the π/2 robustness compounds with intrinsically low measurement noise.
- **What the observables map to.** Transverse magnetization `⟨σ_x⟩, ⟨σ_y⟩` come
  directly from the two FID quadratures; `⟨σ_z⟩` needs a 90° readout pulse to rotate
  it into the detectable plane. Per-nucleus values come from the resolved spectral
  peaks (chemical-shift-separated).
- **Virtual nodes = FID time samples.** The temporal-multiplexing "virtual nodes"
  (V samples within each τ window) map directly onto **sampling the FID at V time
  points** — native to NMR, where the FID is a continuous time signal.

### 6.7 Cost summary (hardware)

| Quantity | Scaling |
|----------|---------|
| Experiments per gradient step | `2·T` (per-spin PSR) or `2·n·T` (global-pulse generalized PSR) |
| Wall-clock per gradient step | `(experiments) × (seconds/experiment)` — minutes to hours |
| Full optimization | `~100 steps` → days–weeks; warm-start from sim to cut steps |

---

## 7. Hardware validation protocol (SPINQ, when Phase-8 sim is promising)

1. **Warm-start.** Take the simulation-learned encoder `W*` (from the 9-spin
   converged run). Do *not* start from scratch on hardware.
2. **Map observables.** Fix the readout-pulse + acquisition scheme that yields the
   feature set `⟨σ_{x,y,z}⟩` per readout nucleus from the FID/spectrum.
3. **Calibrate.** Verify pulse-angle fidelity and the effective τ; measure `f(θ)` on
   a single pulse and confirm the sinusoidal PSR form (a direct experimental check).
4. **Gradient step.** For each pulse `k`: run the sequence at `θ_k±π/2`, acquire,
   extract features, form `∂L/∂θ_k` by PSR; chain to `∂L/∂W` classically; Adam step.
5. **Alternate readout training** (least-squares `w` on measured features) with
   encoder PSR steps.
6. **Compare** hardware-learned vs. simulation-learned vs. `arcsin(√s)` encodings on
   held-out sequences; report the sim-to-hardware gap.

---

## 8. Decision gates and honest risks

- **Prior from our own data.** Phase-2 (fixed functions) + the 3-spin prototype both
  point to `arcsin(√s)` being near-optimal — the gradient trends back to it. Enter
  the expensive 9-spin/hardware program **expecting a "confirms arcsin" (methods)
  result**, and treat a genuine *beat* as upside, not the base case.
- **Proceed to the 9-spin GPU run if:** you want the converged, multi-seed answer
  and/or the methods paper regardless of the performance verdict.
- **Proceed to hardware PSR if:** the 9-spin sim shows a *reproducible* margin over
  `arcsin(√s)`, or the goal is explicitly the "gradient optimization on real NMR"
  contribution (Option C paper), for which the hardware demonstration *is* the
  result.
- **Prefer per-spin encoding** to keep PSR cheap (2-term) and expressive (§6.5).

---

## 9. References

- Chen, Rubanova, Bettencourt, Duvenaud (2018). *Neural Ordinary Differential
  Equations.* NeurIPS (Best Paper).
- Mitarai, Negoro, Kitagawa, Fujii (2018). *Quantum Circuit Learning.* Phys. Rev. A
  98, 032309. — parameter-shift gradients.
- Schuld, Bergholm, Gogolin, Izaac, Killoran (2019). *Evaluating analytic gradients
  on quantum hardware.* Phys. Rev. A 99, 032331. — the ½[f(+π/2)−f(−π/2)] rule.
- Wierichs, Izaac, Wang, Lin (2022). *General parameter-shift rules for quantum
  gradients.* Quantum 6, 677. — multi-term rule for many-eigenvalue generators.
- Khaneja, Reiss, Kehlet, Schulte-Herbrüggen, Glaser (2005). *Optimal control of
  coupled spin dynamics: GRAPE.* J. Magn. Reson. 172, 296. — pulse-shape optimization
  (the Phase-9 shaped-pulse connection).
- Hou et al. (2026). Paper 4 — the `arcsin(√s)` NMR-QRC encoding baseline.

---

## 10. Reproduction pointers

| Artifact | What |
|----------|------|
| `backend/scripts/qrc_grad_smoketest.py` | dense-CPU proof autograd flows through a Lindblad reservoir (FD match 1.9e-8) |
| `backend/scripts/qrc_grad_prod_check.py` | autograd through the **production** op family (sparse-CSR complex, FD match 3.5e-9 in complex128) |
| `backend/app/qrc/system.py` → `ensure_diff` / `step_diff` | the differentiable reservoir step (detach lifted, device-agnostic) |
| `backend/scripts/qrc_learnable_proto.py` | end-to-end learnable encoder trained by Adam (Fig. 12) |
| `docs/QRC/figures/fig12_learnable_proto.png` | training curve + the encoding the gradient discovered |
| `docs/QRC/QRC_Encoding_Study.md` | the Phase-2 fixed-function result this builds on |
