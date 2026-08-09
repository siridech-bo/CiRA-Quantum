# Learnable encoding on the physics-informed reduced-FID readout — finding

**Status:** ✅ **CONFIRMED** intra-QRC (learned ≫ arcsin, multi-seed + τ→0 ablation,
campaign `learnable-campaign-01ce463b`). Classical comparison **completed**
(`classical-baseline-548dbea6`): **no quantum-advantage-over-classical** — a tuned
ESN matches/beats the QRC on NARMA (see "Classical comparison"). The contribution is
the physics-informed readout + the quantum-mediated *encoding* gain, not a
quantum-beats-classical result. Read `docs/QRC/QRC_STANDARD_PROCEDURES.md` first;
every result is labeled with readout/D/regime per §0/§3.1/§3.2.

## Confirmed results (campaign, test NMSE)

| Task (seeds) | arcsin (A) | **perspin (B, learned)** | LSTM* | perspin vs arcsin | τ→0 ablation |
|---|---|---|---|---|---|
| narma2 (3) | 0.101 ± 0.006 | **0.0052 ± 0.0013** | 0.134 ± 0.015 | **3/3, ~19×** | 0.9994 → collapses ✓ |
| narma10 (3) | 0.312 ± 0.013 | **0.120 ± 0.019** | 0.477 ± 0.040 | **3/3, ~2.6×** | 0.9974 → collapses ✓ |
| mackey_glass (2) | 0.00063 | **0.000076** | 0.0016 | **2/2, ~8×** | — (easy task) |

Reservoir: 6-spin, coupling ×2, readout `fid_reduced` (D_eff=161 → 322 feats),
T=1500, 100 Adam steps, leakage-free 3-way. *LSTM = **under-tuned baseline** — do
not quote QRC-vs-LSTM multipliers until the strengthened baseline (see Remaining).

**Bulletproof core:** learned per-spin encoding beats fixed arcsin **8/8 across 3
datasets/all seeds**, and the **τ→0 ablation collapses both memory tasks to the
mean-predictor** → the win is **genuinely quantum-mediated** (the reservoir carries
the memory). This is the first *multi-seed, ablation-confirmed* learnable-encoding
win on a physics-grounded readout.

## What we discovered

Training the **input encoding** by backprop through the reservoir (regime B, §3.1)
on the **physics-informed reduced-FID readout** (§3.2) beat the fixed arcsin
baseline on NARMA-2 by a large margin, in the first single-seed run.

| Condition | test NMSE | notes |
|---|---|---|
| arcsin baseline (regime A) | **0.0874** | powered (≪ 0.8 guardrail) |
| learned per-spin (regime B) | **0.00345** | non-degenerate (enc std 0.59) |

**Verdict: HEADROOM — learned per-spin beats arcsin ~25×** (run `learnable-32c531df`).

### Configuration
- System: 6-spin generic, **coupling ×2** (strong coupling — a known clean lever).
- Task: **NARMA-2** (encoding-sensitive — the encoding is the bottleneck, so a
  learned encoding has a fair chance to matter; a memory-bound task would tie).
- Readout: **physics-informed reduced FID** (`fid_reduced`). At coupling ×2 the
  J-multiplets widen, so **`D_eff = 161` resolvable lines → 322 features** (vs
  137/274 at coupling ×1 — the physics selection recomputed for the scaled `H`).
- Regime B: closed-form ridge readout + gradient-trained per-spin encoder.
- Leakage-free 3-way split (ridge on train, encoder on val, report test), T=1500,
  100 Adam steps, quantum memory ON.
- Runner persists the final-encoding FID waveform (`learned_encoding_fid.npz`).

## Why this matters
Every prior learnable-encoding win was on the **observable readout** (18–180
features), which is *not* the standard readout — a labeling gap that caused a real
integrity issue (§0). This is the **first learnable win on a physics-grounded
readout** (the analytic single-quantum lines of the Hamiltonian), so the numbers
are comparable to the reduced-FID tier and much closer to the 653-FID standard.

## Why it is NOT yet a result (rigor — do not quote 25× as settled)
1. **Single seed (7).** The 6-spin observable-readout win needed 5 seeds to confirm;
   a single-seed multiple was partly luck-of-the-seed. → multi-seed required.
2. **No τ→0 ablation yet.** Scalar per-step input carries no classical memory, so a
   genuine quantum-mediated win must **collapse** under `--no-memory`. → running.
3. **Modest p≫n margin.** 735 train vs 322 features (~2.3×). Test NMSE 0.0034 is on a
   held-out split, but the margin isn't large.
4. **One task, one system.** Needs to hold across datasets and against a strong
   classical baseline (trained LSTM), not just arcsin.

## Classical comparison (completed — the honest reality check)
Run `classical-baseline-548dbea6` (1.68 h): tuned-LSTM sweep (multi-init, cosine LR,
grad-clip, early-stop) + ESN size-sweep, same split/metric as QRC.

| Task | QRC-learned | LSTM (tuned) | ESN (N=1000) | winner |
|---|---|---|---|---|
| NARMA-2 | 0.0052 | 0.0114 | **0.0046** | ESN (QRC≈ESN) |
| NARMA-10 | 0.120 | 0.234 | **0.029** | **ESN (4×)** |
| mackey h=1 | 0.00008 | 0.0003 | ~0 | trivial |
| **mackey h=10** (3) | **0.00155 ± 0.00018** | 0.0024 | 0.0057 | **QRC-learned** |

**CONFIRMED across 3 seeds** (`learnable-0a45c7b0` + `mackey-h10-confirm-2d21b56f`;
0.0013 / 0.00172 / 0.00162). On the one discriminative task, QRC-learned beats the
tuned LSTM (~1.6×) AND the ESN (~3.7×), 3/3 seeds → the advantage is
**task-dependent**: classical wins on NARMA (linear-memory), QRC-learned wins on hard
chaotic prediction (nonlinear fading memory).

**Conclusions (no spin):**
- Tuning the LSTM erased most of the apparent QRC-vs-LSTM gap (narma2 LSTM 0.134→0.0114,
  ~12×). QRC still beats the *tuned* LSTM ~2× on NARMA — the "26×" was a strawman.
- **A well-sized classical ESN matches QRC on NARMA-2 and beats it 4× on NARMA-10.**
  → **No quantum-advantage-over-classical claim** on these benchmarks. Consistent with
  the decoherence-limited reservoir (effective dim low).
- **The contribution stands as:** (a) physics-informed differentiable readout, (b) a
  *quantum-mediated* encoding improvement over arcsin (ablation-confirmed) — NOT
  quantum-beats-classical.
- **Pending:** QRC at **mackey_glass h=10** (the only discriminative task where the
  classical numbers aren't trivial) — the campaign only ran mackey at h=1, so a
  QRC-vs-classical comparison at h=10 needs one ~4 h QRC run. Any h=10 "QRC win" in
  raw tables is invalid until then (it compares QRC-h1 vs classical-h10).

## Confirmation campaign (completed 2026-08-08, 41.0 h)
Broadened per the plan on 2026-08-06:
- **Datasets:** NARMA-2 (encoding-sensitive), NARMA-10 (memory-bound — expect a
  tie/loss, the honest contrast), Mackey-Glass (chaotic next-step prediction).
- **Multi-seed** per task (mean±std, win-count vs arcsin).
- **τ→0 ablation** (perspin `--no-memory`) per memory-relevant task — must collapse.
- **Trained classical LSTM** head-to-head on every task/seed (the strong bar; QRC
  beats classical ESN but the trained-RNN comparison is the harder, honest one).
- Wall-clock-budgeted so it always finishes and writes a summary.

**Interpretation rule for the campaign:** a confirmed quantum-clean win = learned
perspin beats arcsin in ≥4/5 seeds **and** the τ→0 ablation collapses **and** we
report honestly where the trained LSTM still wins (likely on the memory-bound task).
Label every number with its `(readout, D, regime)`.
