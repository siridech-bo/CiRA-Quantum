# QRC — Quantum Reservoir Computing simulator

Implements [`docs/QRC_Simulation_Plan.md`](../../../docs/QRC_Simulation_Plan.md):
an **N-qubit-generic** NMR reservoir simulator for prototyping Quantum
Reservoir Computing experiments before running them on the SPINQ Gemini
Lab — and for making the quantitative case that a **larger machine buys
more computational power**.

## Why this exists

The SPINQ Gemini Lab is a 3-qubit ¹H/³¹P/¹⁹F NMR device. Paper 4 (Hou et
al. 2026, PRL 136, 120602) shows QRC on a 9-spin system. This simulator
lets us:

1. Reproduce the QRC pipeline on the 3-qubit SPINQ system (`spinq3`).
2. **Scale the same physics to 5, 7, 9+ qubits** and measure how memory
   capacity and NARMA accuracy improve — the evidence for acquiring more
   qubits (`scaling_study`, plan §10.4).

## The one non-negotiable idea

Fading memory comes from **dissipation**, not from Hamiltonian evolution
alone (plan §5). We integrate the **Lindblad master equation** (QuTiP
`mesolve`/`propagator`) with per-spin T1 amplitude damping and T2 pure
dephasing. A pure-Schrödinger simulation would give perfect-memory-forever
and is *wrong* for QRC. The `memory` command is the gate that proves the
simulation captures this: corr² must be high at delay 0 and decay smoothly.

## Install

```bash
pip install ".[qrc]"      # QuTiP 5 + qutip-jax + pandas/sklearn/wavelet/entropy
```

Torch (the ridge readout) is already a base backend dependency. QuTiP is
imported lazily, so the package imports without the extra — only the
simulation entry points require it.

## Run

```bash
# 1) Fading-memory gate on the 3-qubit SPINQ system — run this FIRST.
python -m app.qrc.main memory --system spinq3

# 2) NARMA-2 benchmark.
python -m app.qrc.main narma --system spinq3 --order 2

# 3) The headline: scaling study 3 → 9 qubits (proof of benefit).
python -m app.qrc.main scaling --qubits 3 5 7 9 --out scaling.json

# Multi-modal features (spectral + time-domain + wavelet + entropy):
python -m app.qrc.main narma --system spinq3 --multimodal

# GPU (JAX) backend for the larger systems:
python -m app.qrc.main scaling --qubits 5 7 9 --backend jax
```

## Backend / GPU note

At ≤ ~9 qubits the Hilbert space is tiny (≤ 512×512); the **numpy**
backend on CPU is fastest and is the default. The **jax** backend
(`--backend jax`, requires `qutip-jax`) targets the larger systems and
big batched parameter sweeps, per plan §2.3. It degrades to CPU
automatically if a GPU JAX build isn't available (notably on Windows).

## Evolution modes (how the reservoir is propagated) — and the N=9 wall

`SimConfig.evolution_mode` selects the time-evolution strategy, and this
choice is what determines how many qubits you can actually reach:

| mode | how | good for | limit |
|------|-----|----------|-------|
| `propagator` | precompute one dense Liouvillian superoperator, reuse each step | N ≤ 5 (fast, seconds) | superoperator is **4ⁿ** dense: 0.27 GB @ N=6, 4.3 GB @ N=7, **550 GB @ N=9**; also slow to build (~13 min at N=6) |
| `action` | apply `exp(t·L)` to the vectorized state via sparse Krylov (`scipy.expm_multiply`), CPU | N=6–9 when no GPU | ~6.5 s/reservoir-step at N=9 (Krylov cost of a stiff 262k-dim Lindbladian) |
| `gpu` | same exact exponential, but Taylor + sub-stepping `exp(t·L)` with **sparse CUDA matvecs** (torch, complex64) | **N=6–9, fast** | needs a CUDA torch build; **~0.23 s/step at N=9 — ~28× faster than `action`** |
| `mesolve` | QuTiP adaptive ODE on the density matrix | reference/checking | **stiffness** (kHz precession vs Hz relaxation) makes it impractically slow past ~7 qubits |
| `auto` *(default)* | `propagator` for N ≤ 5, `action` for N ≥ 6 (CPU only) | everything | pick `gpu` explicitly to use the card |

All backends agree to ~3 decimals where they overlap (verified in the
tests: `action` vs `propagator`, and `gpu` vs `action`). The GPU path
makes N=9 practical: a full memory+NARMA point is **~4 min on GPU** vs
~1.75 h on CPU, so the whole N=3→9 sweep finishes in ~15 min. The
observables-only readout uses a fully-vectorized hot loop (no per-step
Qobj construction) on both the CPU (`action`) and GPU paths.

**Why the plan's `qutip-jax` GPU route doesn't apply here:** jax has no
CUDA wheels on Windows (`jax.devices()` → CPU only), so `--backend jax`
silently runs on CPU. The `gpu` evolution mode uses **torch** CUDA
instead (which does see the card), sidestepping that limitation. The
550 GB N=9 superoperator is never formed — only the ~0.001%-dense sparse
Liouvillian (a few MB) lives on the GPU, and the 512×512 state stays a
262 144-vector throughout.

Run the GPU sweep (`scripts/qrc_scaling_run.py` auto-selects `gpu` for
N ≥ 6 when CUDA is present; results append to `qrc_scaling_progress.jsonl`
as each N completes).

## Module map (plan §12)

| File | Role |
|------|------|
| `config.py` | System presets (SPINQ-3, generic-N, crotonic-9) + all knobs |
| `system.py` | N-qubit Hamiltonian, Lindblad collapse ops, propagators |
| `encoding.py` | Input→pulse: 7 encoding fns, multi-nucleus, phase-amplitude |
| `evolution.py` | Reservoir stepping + temporal multiplexing |
| `features.py` | Multi-modal features (observables/spectral/time/wavelet/entropy) |
| `training.py` | Ridge readout on GPU (torch) + CV |
| `tasks.py` | NARMA, memory capacity, weather |
| `benchmarks.py` | Orchestration, ESN baseline, **scaling study** |
| `main.py` | CLI |

## What's a faithful stand-in vs. exact

- `spinq3` uses the plan's real J-couplings, T1/T2, and Larmor freqs.
  Rotating-frame chemical shifts aren't given in the plan, so small
  distinct offsets are used (they don't change qualitative behavior).
- `crotonic9` is a **9-spin generic molecule**, not the exact Paper 4
  coupling table. Swap in the published constants when replicating Paper 4
  numerically.
- The `generic_nqubit` molecule (used by the scaling sweep) fabricates a
  physically plausible chain-coupled molecule so the *only* variable
  across the sweep is qubit count.
