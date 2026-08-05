# Learnable encoding on the physics-informed reduced-FID readout — finding

**Status:** 🟢 single-seed HEADROOM, **not yet confirmed** (multi-seed + ablation +
classical-RNN campaign running as of 2026-08-06). Read
`docs/QRC/QRC_STANDARD_PROCEDURES.md` first — this result is labeled with its
readout and regime per §0/§3.1/§3.2.

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

## Confirmation campaign (running unattended, ~24–40 h)
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
