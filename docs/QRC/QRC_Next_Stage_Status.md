# QRC Next-Stage — Implementation Status Tracker

**Source plan:** [`QRC_Next_Stage_Experiments.md`](./QRC_Next_Stage_Experiments.md)
**Last updated:** 2026-07-28 (Feature-Lab UI shipped; trace-gen progress wired)
**Purpose:** one place that traces every phase/experiment in the plan to its real
state, so we always know *what is left*.
**Update rule:** this tracker is updated **strictly on every subtask completion**.

## Session activity log (newest first)

| When | Subtask | Result | Commit |
|------|---------|--------|--------|
| 2026-08-01 | **Decision: GRAPE deferred to the optimized-encoding stage** | 📌 GRAPE / gradient (or gradient-free) *learned* encoding is the §8.6.1/§8.8 direction (dedicated encoding paper), not now. Foundation note: the **torch GPU backend is autodiff-capable** → the differentiable-QRC path can be built on it (no Dynamiqs needed) when we get there. MC metric = ready-made objective | — |
| 2026-08-01 | **Phase-2 encoding comparison DONE via MEMORY CAPACITY** (meaningful) | ✅ `memcap-a82a02a7` (7 encodings, waveforms saved, watched live). Clean ranking: **arcsin_sqrt wins decisively** (totMC 4.93 vs 3.08; nonlinear MC ~2× others) → validates Paper-4's choice quantitatively. Well-separated (not noise, unlike weather-R²). Single-seed/quick — large gap ⇒ robust | — |
| 2026-08-01 | **LIVE FID/spectrum streaming + record-name provenance** | ✅ /fid reads live memmap (clamps over-range); frontend follows latest per encoding. Watchable + verifiable | `d2cde98`,`1115a5d` |
| 2026-08-01 | **memory-capacity metric + persist-waveform fix** | ✅ `qrc_memcap.py` (self-test MC=8.00); hard rule: never discard waveforms | `9542253`,`9449677` |
| 2026-08-01 | ~~Phase-2 2.1 weather-R² sweep~~ (superseded) | ⚠️ weather-R² too fidelity-hungry → meaningless; replaced by memory capacity above | `phase2-97d971ed` |
| 2026-08-01 | **Phase-2 2.1 sweep DONE (7 encodings) + real-GPU resume proven** | ⚠️ crash-resume proven live (stopped@73 → resumed@60, real GPU); but **numbers meaningless again** — 'screen' fidelity too small for weather-R² (fidelity wall, 3rd time). Only signal: **arcsin_sqrt best at h1** (Paper-4's choice). Recommend switching encoding metric to **memory-capacity/NARMA** (fidelity-robust) | `phase2-97d971ed` |
| 2026-07-31 | **Phase 2 launcher-integrated + UI-visible; quick subset launched** | 🔵 phase2 is now a launcher task with live progress (validated: shows in dashboard, per-encoding phase/step/ETA, stoppable). Quick subset `phase2-8d981eb8` (3 encodings @ quick fidelity, ~1.7 h) running via API. Also: **ASK-FIRST rule** added to CLAUDE.md+memory after auto-launching a long run | `7ebbd4c` |
| 2026-07-31 | **Phase 2 encoding runner built + screening sweep launched** | 🔵 `qrc_phase2.py` (re-evolves per encoding; blocked-CV; streaming/resumable; sweep-level resume). 2.1 screening (7 encodings, 9-spin) running ~5 h | `5601406` |
| 2026-07-31 | **Figure 8** — Phase-1 feature-representation plot (reproducible) | ✅ `fig8_phase1_features.png` + §5.2 in Reproduction Results; 2 panels (standalone CV bands · paired per-fold Δ) | `e9517da` |
| 2026-07-31 | **Rich multimodal (136 vs 18) + paired-fold test** | ✅ `qrc_phase1_rich.py`. 18 was too few — richer extraction ~doubles multimodal standalone (h45 0.42→0.63). But **paired per-fold test: no robust win over magnitude** (mag wins h1/h10; long-horizon = 1–2 lucky folds). No representation dominates → reinforces Phase 2 | `858e0e0` |
| 2026-07-31 | **Rigorous Phase-1 v2** (standalone/null/dim-matched/blocked-CV) | ✅ `qrc_phase1_v2.py`. **Corrects v1:** multimodal DOES carry signal (+0.003…0.005, significant vs random null); phase redundant; reps indistinguishable within CV error (±0.10–0.28); signal low-dimensional (PCA-18 ≈ full). Gate → Phase 2 (rigorously) | `2b59e1c` |
| 2026-07-31 | **Retracted the Phase-1 verdict** (methodology critique) | ⚠️ v1 underpowered/confounded (18-of-1977 swamped · appended-not-isolated · p≫n · single-seed/h=30); "phase/multimodal don't help" not supported → drove the v2 study above | — |
| 2026-07-31 | **Phase-1 sweep run (raw numbers)** | ✅ `phase1-1356c77d` produced R² per method/horizon; conclusions withdrawn (see next row) | — |
| 2026-07-31 | **Full trace-gen COMPLETED** + phase1 sweep launched | ✅ `trace-gen-a6304ab1` done clean (1474/1474 rows, valid); cached as `weather_full.npz`; phase1 `phase1-1356c77d` running | — |
| 2026-07-29 | **Re-launched full trace-gen (resumable)** | ✅ streamed to disk, completed without loss (no crash; resume path proven separately) | — |
| 2026-07-29 | **Stream FID to disk + crash-resume** (root fix for the "13h in RAM" design) | ✅ memmap streaming + resumable GPU-state checkpoints; resume is **bit-identical** (new GPU test); 70 tests pass | `5c7f43a` |
| 2026-07-29 | **Fix: telemetry must never crash the run** (WinError 5 replace race) | ✅ retry + best-effort; verified via forced-error test | `618abeb` |
| 2026-07-29 | ⚠️ Full trace-gen **FAILED** at step 1140/1474 (~13.7 h) | ❌ per-step `status.json` replace hit PermissionError → aborted the run; root-caused + fixed above; **awaiting re-launch decision** | `trace-gen-c46c804f` |
| 2026-07-28 | **Launched full weather trace-gen** (2048 fid_points, 1474 steps) | 🔵 ran to 77% then failed (see above); real ~43 s/step ⇒ ~17.7 h full | — |
| 2026-07-28 | Deployer handoff doc (`deploy/QRC_DEPLOY.md`) | ✅ in-tree deploy instructions for `quantum.cira-core.com/qrc` | `5c5bfc8`+ |
| 2026-07-28 | Feature-Lab UI: `/embedding` endpoint + scatter + Exp 1.1/1.2/1.3 charts | ✅ built, 22 tests pass, type-check clean | `fb8127b` |
| 2026-07-28 | trace-gen live progress (per-step `status.json` + event log) | ✅ verified via `/progress` parser | `34090fd` |
| 2026-07-28 | Frontend production build from `qrc-simulation` | ✅ `dist/` verified (QRC + Feature-Lab bundled) | — |
| 2026-07-28 | Preview trace (240 steps) + preview `phase1` run | ✅ `phase1-163c2c60` done; UI populated (PCA+UMAP work) | — |
| 2026-07-28 | `CLAUDE.md`: deploy architecture; dev box == prod box (`.167`) | ✅ | `721ecf4`, `34090fd` |
| 2026-07-28 | Status tracker created (this file) + Plan_2 under VCS | ✅ | `8f0d5a4` |

## Legend

| Mark | Meaning |
|------|---------|
| ✅ **Done** | Code written, wired, and exercised (or shipped). |
| 🔵 **In progress** | Being worked now; partially landed. |
| 🟡 **Code ready — NOT RUN** | Runner + code exist and pass tests, but no experiment has been executed → no results yet. Needs the trace cache and a compute pass. |
| 🟠 **Primitive only — no runner** | The physics/feature knob exists in the engine, but there is **no experiment driver** that sweeps it and no results. |
| ⬜ **Not started** | No code. |

---

## Scoreboard

| Phase | Title | State | Blocking dependency |
|-------|-------|-------|---------------------|
| **0** | Instrumentation: trace cache + FID/progress UI + run control | ✅ **Done** (both gaps closed) | — |
| **1** | Flag-level feature experiments (phase / multimodal / selection incl. UMAP) | ✅ **Done (rigorous v2)** — standalone/null/dim-matched/blocked-CV. Multimodal carries real (small, significant) signal; phase redundant; reps indistinguishable within CV error; signal low-dimensional. Gate → Phase 2 | (optional: reservoir-seed variance = extra GPU traces) |
| **2** | Encoding sweep (7 functions / phase-amp / protons-only) | 🟡 **Runner done; 2.1 run but metric inconclusive** — full 7-encoding sweep completed (crash-resume proven live), but weather-R² needs near-full fidelity (screen too small → noise). Only arcsin_sqrt-best-at-h1 signal. **Next: switch to memory-capacity/NARMA metric** | build fidelity-robust encoding metric |
| **3** | External feature libraries (tsfresh, nmrglue) | ⬜ **Not started** | new deps + integration code |
| **4** | Dimensionality-reduction benchmark (R0–R6 × Ridge/SVR) | 🟡 **Partial — reducers exist, no full grid** | best feature set from Phase 3 |
| **5** | Self-supervised representation learning (autoencoder, TS2Vec) | ⬜ **Not started** (gated) | only if Phase 4 shows headroom |
| **6** | Encoding × feature interaction + τ sweep | ⬜ **Not started** (gated) | best encoding (P2) × best features (P3) |
| **UI+** | Feature-Lab UI: Phase-1 sweep panel + UMAP/PCA embedding scatter | ✅ **Done** | — (populated by a phase1 run) |
| **Ops** | Remote access at `quantum.cira-core.com/qrc` (run/watch from browser off-box) | 🔵 **In progress** — frontend build done + deployer handoff ready; tunnel/serve pending | deployer ships branch + Cloudflare Access |

---

## Phase 0 — Instrumentation ✅ DONE

| Item | State | Where |
|------|-------|-------|
| 0.A Raw-FID trace cache (schema §1) | ✅ | `backend/scripts/qrc_gen_traces.py` |
| 0.B Offline evaluation harness | ✅ | `backend/app/qrc/feature_lab.py` |
| 0.C Backend API + run control (`/api/qrc`, 7 routes, auth-gated launch/stop, allow-list, single-job lock) | ✅ | `backend/app/routes/qrc.py`, `backend/app/qrc/launcher.py` |
| 0.D Frontend UI (dashboard, run detail, FID time chart, spectrum, new-run dialog, results) | ✅ | `frontend/src/views/Qrc*.vue`, `frontend/src/components/Qrc*.vue`, `frontend/src/stores/qrc.ts` |

**Phase 0 gaps — both now closed:**
1. ✅ **trace-gen live progress** — `qrc_gen_traces.py` now emits per-step
   `ProgressLogger.status()` (phase/step/total/ETA) + an event log into the
   run-dir, so `GET /runs/<id>/progress` drives a live bar + ETA in the UI
   (commit `34090fd`). Verified through the backend parser.
2. 🔵 **Remote access** — moved to the **Ops** row. The frontend build is done
   and the deployer handoff is ready; what remains is shipping the branch +
   Cloudflare Access so `quantum.cira-core.com/qrc` is reachable off-box.
   (Correction: the prod host **is** this dev box, `DESKTOP-1A0J7FD`/`.167`,
   and it **has** the RTX 5070 Ti — see `CLAUDE.md`.)

---

## Phase 1 — Flag-level feature experiments ✅ DONE (rigorous v2; v1 verdict corrected)

The v1 sweep (`qrc_phase1.py`, run `phase1-1356c77d`) was **underpowered/confounded**
(18-of-1977 swamped · appended-not-isolated · p≫n · single-seed/h=30-only), so its
"phase/multimodal don't help" reading was withdrawn. The rigorous study
`scripts/qrc_phase1_v2.py` (standalone eval · random-feature null · dimensionality-
matched · RidgeCV alpha · **blocked-CV mean±std, all horizons**) replaces it. Findings:

| Question | Rigorous answer |
|-----|-----|
| Do **multimodal** (18 feats) carry signal? | **Yes.** Small but statistically significant **+ve** marginal at every horizon (+0.003…+0.005, beats random-feature null); standalone R²≈0.60 @ h30. *v1's "useless" was wrong.* **Keep it.** |
| Does **phase** (re+im) help? | Structured (beats noise) but **redundant** — ~0 marginal over magnitude. |
| Does extracting **more** multimodal (136 vs 18) help? | **For multimodal itself, yes** — richer extraction (windowed TD + per-band wavelet + 10-measure complexity) ~doubles long-horizon skill (h45 CV 0.42→0.63). **But a paired per-fold test shows no robust win over magnitude** (mag wins h1/h10 in 0–1 of 5 folds; long-horizon "wins" are 1–2 lucky folds). `qrc_phase1_rich.py`. |
| Best single representation? | No robust winner — magnitude wins short-range, long-range is a fold-variance coin-flip. Differences **within CV error bars (±0.10–0.28)**. |
| Why did v1 see ±0.01 "differences"? | Noise. Blocked-CV variance is ~10–25×, and the fixed single split was optimistic. |
| Dimensionality? | **Signal is low-dimensional** — PCA-18 of magnitude ≈ full 653 (0.799 vs 0.812 @ h30); all reps converge when reduced. The p≫n dilution was the real issue. |

**Decision gate (now rigorously supported):** feature-representation choice is
**low-headroom** (all reps ≈ equivalent when reduced; multimodal adds only ~0.004)
→ prioritize **Phase 2 (encoding)**. Not because "multimodal is useless" (it isn't).
*Remaining rigor gap:* reservoir-**seed** variance needs a few extra GPU traces.

---

## Phase 2 — Encoding sweep 🔵 IN PROGRESS (runner built; screening sweep running)

Runner `scripts/qrc_phase2.py` re-evolves the reservoir per encoding setting
(holding everything else fixed) and scores each with the Phase-1 v2 rigorous
protocol (magnitude-653 readout, RidgeCV, blocked-CV mean±std, all horizons).
Reuses the streaming/resumable `StreamingTrace`; the sweep itself resumes
(cached-result settings are skipped). Fidelity presets: `screen` (rank) / `full`
(confirm) / `tiny` (smoke). Validated end-to-end.

| Exp | What | Runner | Status |
|-----|------|--------|--------|
| 2.1 | Encoding-function sweep (arcsin√, arccos, linear, sinusoidal, logarithmic, polynomial, exponential) | ✅ `qrc_phase2.py --experiment 2.1` | 🔵 **screening sweep running** (7 encodings, 9-spin, ~5 h) |
| 2.2 | Phase-amplitude encoding off/on | ✅ `--experiment 2.2` | ⬜ queued after 2.1 |
| 2.3 | Protons-only vs all-spins | ⬜ (needs `target_qubits` wiring for the weather multi-channel encoder) | ⬜ follow-up |

**Next:** read 2.1 screening ranking → confirm top encoding(s) at `--fidelity full`
→ run 2.2 → wire 2.3. (Compute: each setting is a fresh reservoir pass; screening
~40 min/setting, full ~hours.)

---

## Phase 3 — External feature libraries ⬜ NOT STARTED

| Exp | What | State |
|-----|------|-------|
| 3.1 | TSFRESH automatic feature extraction | ⬜ (dep `tsfresh` not added; no integration) |
| 3.2 | nmrglue physics-informed processing (J-couplings, cross-peaks) | ⬜ (dep `nmrglue` not added; no integration) |

---

## Phase 4 — Dimensionality-reduction benchmark 🟡 PARTIAL

The reducers themselves exist (`reduce_features`), and `qrc_phase1.py` already
runs a **subset** (multimodal set × reducers at one key horizon). The **full
R0–R6 × {Ridge, SVR} grid** benchmark harness is not built and not run.

| Row | Method | State |
|-----|--------|-------|
| R0 raw · R1 PCA · R2 KPCA · R3 UMAP · R4 LASSO · R5 MI · R6 random | reducers | ✅ exist as functions |
| Full grid × {Ridge, SVR} + heatmap deliverable | benchmark harness | ⬜ not built / not run |

---

## Phase 5 — Self-supervised representation learning ⬜ NOT STARTED (gated)

Only pursued "if Phase 4 shows headroom."

| Exp | What | State |
|-----|------|-------|
| 5.1 | Denoising autoencoder (PyTorch available) | ⬜ |
| 5.2 | Contrastive learning / TS2Vec | ⬜ (dep not added) |

---

## Phase 6 — Encoding × feature interaction ⬜ NOT STARTED (gated)

| Exp | What | State |
|-----|------|-------|
| 6.1 | Best encoding × best features cross-study | ⬜ (depends on P2 + P3) |
| 6.2 | τ sweep at best configuration | ⬜ runner not built (`tau` knob ✅ exists in `config.py`) |

---

## UI+ Feature-Lab (raised 2026-07-28) ✅ DONE — presentation approved

| Item | What | State | Where |
|------|------|-------|-------|
| Feature-Lab panel | Exp 1.1/1.2 method-comparison bars + Exp 1.3 selection-sweep chart (interactive, replaces the static PNG) | ✅ | `frontend/src/components/QrcFeatureLabPanel.vue` |
| **UMAP/PCA embedding view** | `GET /runs/<id>/embedding` → 2-D projection (PCA/UMAP) computed server-side; canvas scatter coloured by target, test points ringed | ✅ | `QrcEmbeddingScatter.vue`, `backend/app/routes/qrc.py` |

Validated on preview run `phase1-163c2c60` (23 sweep rows, 240-point embedding;
PCA + UMAP both working). **Presentation approved by user 2026-07-28.**

> Note: neither plan doc specs a "UMAP *visualization*" — UMAP is only a
> **reducer** (Exp 1.3 / 4.1 R3). This embedding scatter was a new addition.

---

## What is left, in priority order

*(Done: Phase 0, Phase 1, Feature-Lab UI — see activity log up top.)*

1. **(optional, fast) Phase-1 firm-up** — multi-seed + full-horizon re-run to
   make the "baseline near-optimal" negative result defensible (trace cached,
   no GPU, minutes). Also fix Exp-1.3 to score the horizon profile, not h=30.
2. **Phase 2 — encoding runner** (`qrc_phase2.py`): 2.1 encoding-function sweep ·
   2.2 phase-amplitude · 2.3 protons-only. **The next real lever** (per the
   gate). Re-evolves the reservoir → multi-hour GPU runs (now resumable).
3. **Ship remote access** — deployer serves the branch at
   `quantum.cira-core.com/qrc` + Cloudflare Access (build handed off). Parallel.
4. **Phase 3** (tsfresh, nmrglue) → **Phase 4 full grid** → gated **Phase 5 / 6**.

*(Done since v1 of this tracker: Phase-0 progress gap, the Feature-Lab UI, and
the UMAP/PCA embedding view — see the activity log at the top.)*
