# QRC Next-Stage — Implementation Status Tracker

**Source plan:** [`QRC_Next_Stage_Experiments.md`](./QRC_Next_Stage_Experiments.md)
**Last updated:** 2026-07-28 (Feature-Lab UI shipped; trace-gen progress wired)
**Purpose:** one place that traces every phase/experiment in the plan to its real
state, so we always know *what is left*.
**Update rule:** this tracker is updated **strictly on every subtask completion**.

## Session activity log (newest first)

| When | Subtask | Result | Commit |
|------|---------|--------|--------|
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
| **2** | Encoding sweep (7 functions / phase-amp / protons-only) | 🟠 **Primitives exist — no runner** | build `qrc_phase2.py`; needs fresh re-evolution |
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
| Best single representation? | magnitude653, but its edge is **within CV error bars (±0.10–0.28)** → representations statistically **indistinguishable**. |
| Why did v1 see ±0.01 "differences"? | Noise. Blocked-CV variance is ~10–25×, and the fixed single split was optimistic. |
| Dimensionality? | **Signal is low-dimensional** — PCA-18 of magnitude ≈ full 653 (0.799 vs 0.812 @ h30); all reps converge when reduced. The p≫n dilution was the real issue. |

**Decision gate (now rigorously supported):** feature-representation choice is
**low-headroom** (all reps ≈ equivalent when reduced; multimodal adds only ~0.004)
→ prioritize **Phase 2 (encoding)**. Not because "multimodal is useless" (it isn't).
*Remaining rigor gap:* reservoir-**seed** variance needs a few extra GPU traces.

---

## Phase 2 — Encoding sweep 🟠 PRIMITIVES EXIST, NO RUNNER

The engine already knows every knob; there is **no experiment runner** and no
results.

| Exp | What | Primitive present | Runner |
|-----|------|-------------------|--------|
| 2.1 | Encoding-function sweep (arcsin√, arccos, linear, sinusoidal, logarithmic, polynomial, exponential) | ✅ `ENCODING_FNS` in `encoding.py` | ⬜ none |
| 2.2 | Phase-amplitude encoding | ✅ `phase_amplitude` flag (`config.py` + `pulse_unitary`) | ⬜ none |
| 2.3 | Protons-only vs all-spins | ✅ `target_qubits` (`config.py`) | ⬜ none |

**To finish:** write `qrc_phase2.py` that re-evolves the reservoir per encoding
setting (unlike Phase 1, this **cannot** reuse a single trace) and records
weather R² / NARMA NMSE per setting.

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
