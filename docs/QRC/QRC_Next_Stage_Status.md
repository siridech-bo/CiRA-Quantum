# QRC Next-Stage — Implementation Status Tracker

**Source plan:** [`QRC_Next_Stage_Experiments.md`](./QRC_Next_Stage_Experiments.md)
**Last updated:** 2026-08-01 (Phase-2 encoding sweep COMPLETE — 2.1+2.2+2.3; arcsin_sqrt all-spins amplitude wins)
**Purpose:** one place that traces every phase/experiment in the plan to its real
state, so we always know *what is left*.
**Update rule:** this tracker is updated **strictly on every subtask completion**.

## Session activity log (newest first)

| When | Subtask | Result | Commit |
|------|---------|--------|--------|
| 2026-08-04 | **Reference doc: learnable-encoding concept + PSR-in-NMR methodology** | ✅ `QRC_Learnable_Encoding_Concept.md`: idea (molecule as Quantum Neural ODE), the 3 gradient routes (autograd/adjoint/**PSR**), what we validated in sim, prototype methodology (incl. the p≫n trap), and a **detailed parameter-shift-rule-on-NMR** section (exactness proof, π/2 noise-robustness, validity through dissipation, sequence chaining + 2·T cost, global-vs-per-spin generator caveat + generalized PSR, NMR ensemble-readout advantage, hardware protocol, refs) | `concept` |
| 2026-08-04 | **Learnable-encoding prototype — END-TO-END TRAINS (first learned encoding)** | ✅ `system.py` `ensure_diff`/`step_diff` (differentiable reservoir step, detach lifted, device-agnostic) + `qrc_learnable_proto.py` (CPU 3-spin, no GPU): MLP encoder → pulse angle → quantum reservoir → ridge → NMSE, trained by Adam. **Pipeline works** (loss 0.99→0.74 monotonic). Honest result: in this short/under-converged run it **loses to arcsin(√s)** (test NMSE 0.737 vs 0.574) — BUT the learned θ(s) converges to a **monotonic arcsin-like shape** (Fig 12), i.e. the gradient trends back toward Paper-4, consistent with Phase-2 (arcsin near-optimal). First p≫n run was an overfit artifact; fixed with T=240 (n_train≫features). Definitive test = convergence + strong 9-spin reservoir (GPU) + multi-seed | `proto`,`fig12` |
| 2026-08-04 | **Production-op gradient go/no-go — PASS** | ✅ `qrc_grad_prod_check.py` (CPU, no GPU): builds the **real** production Liouvillian `L` + obs-rows `M` and runs the **exact production torch op sequence** (sparse-CSR complex matvec Taylor + `torch.sparse.mm`) with a learnable encoding. Autograd matches finite differences to **3.5e-9 in complex128** (gradient is correct to machine precision); the larger complex64 gap is only the fp32 FD floor, not a gradient error (fp32 autograd value is *closer* to truth than fp32 FD). **Sparse-CSR complex autograd works — no custom Function/COO needed.** Only remaining change: lift the final `.cpu().numpy()` detach ([system.py:435]) behind `differentiable=True` + leaf θ. No framework migration | `prodcheck` |
| 2026-08-04 | **Learnable-encoding premise VALIDATED (gradient smoke test)** | ✅ `qrc_grad_smoketest.py` (CPU 3-spin, no GPU): autograd flows through the Taylor-series Lindblad reservoir to the encoding-network params and **matches finite differences to 1.9e-8**. Confirms the "NMR molecule as a Quantum Neural ODE" direction needs **NO JAX/Dynamiqs migration** — the existing torch stepper's ops are differentiable. De-risks Phase-8 (learnable encoding). Next go/no-go: wire the production GPU stepper (`differentiable=True`, lift the `.cpu().numpy()` detach) + re-run the FD check on it | `smoketest` |
| 2026-08-01 | **Phase-2.3 protons-only DONE — it HURTS (loses nonlinearity)** | ✅ `memcap-f225906f` (waveform saved). Head-to-head, identical settings: all-spins **totMC 10.81** (lin 4.25, nl 6.55) vs protons-only `[4–8]` **totMC 8.28** (lin 4.26, nl 4.02) → **−23%**. **Clean dissociation: linear memory unchanged (protons carry it), nonlinear MC −39% (carbons feed it via J-couplings).** All-spins best; carbons are an active nonlinear resource, not just a bath. **Phase-2 sweep complete** — arcsin_sqrt all-spins amplitude wins all 3 sub-questions. Fig 11 + §4.2 in encoding study (.md+.docx) | `c6cf189`,`fig11` |
| 2026-08-01 | **Phase-2.3 protons-only — wired + launched** | 🔵 `memcap protons` experiment added: arcsin_sqrt encoded only into the proton spins (`target_qubits=[4,5,6,7,8]`, derived from "H" labels) vs the all-spins baseline (reused). Validated no-GPU (identity on the 4 carbons; pulse changes, max\|Δ\|=0.20). Run `memcap-f225906f` @ quick, kmax=30, waveform-persisting, live on GPU (ETA ~31 min) | `c6cf189` |
| 2026-08-01 | **Phase-2.2 phase-amplitude DONE — it HURTS** | ✅ `memcap-36e4d1b0` (waveform saved). Head-to-head, identical settings: plain amplitude arcsin_sqrt **totMC 10.81** (linMC 4.25, nlMC 6.55) vs `R_z(2π·s)·R_x(θ)` **totMC 3.06** (linMC 1.01, nlMC 2.05) → **−72%** (linear −76%, nonlinear −69%; not a trade). Phase channel collapses FID dynamic range → scrambles states. **Plain amplitude stays best.** Fig 10 + §4.1 appended to encoding study (.md+.docx) | `0adc4e1`,`fig10` |
| 2026-08-01 | **Phase-2.2 phase-amplitude — wired + launched** | 🔵 `memcap phaseamp` experiment added: arcsin_sqrt with `R_z(2π·s)·R_x(θ)` (phase-amp ON) vs the existing arcsin_sqrt baseline (phase-amp OFF, reused). Validated no-GPU (phase-amp flag changes the pulse unitary, max\|Δ\|=0.80). Run `memcap-36e4d1b0` @ quick, kmax=30, waveform-persisting, live on GPU (~35 min) | `0adc4e1` |
| 2026-08-01 | **Manuscript addendum: encoding-optimization study** | ✅ `QRC_Encoding_Study.md` + `.docx` (Fig 9 embedded): methodology (fidelity-wall, MC/IPC/NARMA panel, provenance, readout-robustness), results, discussion (mechanism, Paper-4 validation, limitations, future/GRAPE) | `report` |
| 2026-08-01 | **Multi-metric encoding judging** (offline, from saved waveforms) | ✅ `qrc_judge.py` + fig9. arcsin_sqrt wins on ALL signals (linMC 3.66, nlIPC 1.86, NARMA NMSE 0.595); ranking **identical under magnitude653 & multimodal** → readout-independent (richer features NOT needed for the verdict). Bulletproof Phase-2.1 result | `judge` |
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
| **2** | Encoding sweep (7 functions / phase-amp / protons-only) | ✅ **COMPLETE (2.1+2.2+2.3)** — fidelity-robust MC/IPC/NARMA panel: **arcsin_sqrt all-spins amplitude wins all 3 sub-questions.** 2.1 function sweep: arcsin_sqrt decisive + readout-independent (Fig 9). 2.2 phase-amp → HURTS −72% (Fig 10). 2.3 protons-only → HURTS −23% (Fig 11; loses nonlinearity, carbons feed it). Both extensions lose nonlinearity, not memory | — |
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

## Phase 2 — Encoding sweep ✅ COMPLETE (all 3 sub-questions; the metric pivot was key)

The original runner `scripts/qrc_phase2.py` scored encodings by **weather-R²**,
which hit a **fidelity wall** three times (screen fidelity too small → the R²
gaps were pure noise). The fix was to switch to **intrinsic, fidelity-robust
metrics** — linear memory capacity (MC), nonlinear information-processing
capacity (IPC), and NARMA-10 — computed **offline from persisted waveforms**
(`scripts/qrc_memcap.py` + `scripts/qrc_judge.py`). Every encoding evolves once
(waveform saved to `artifacts/traces/…npz`); any metric is recomputed offline
forever. Head-to-head comparisons use identical MC settings on both saved
waveforms. Result: **`arcsin_sqrt` amplitude encoding into all nine spins wins
all three sub-questions.**

| Exp | What | Result | Evidence |
|-----|------|--------|----------|
| 2.1 | Encoding-function sweep (arcsin√, arccos, linear, sinusoidal, logarithmic, polynomial, exponential) | ✅ **arcsin_sqrt wins decisively** (totCap 5.51 vs 3.66 #2; linMC 3.66, nlIPC 1.86, NARMA NMSE 0.595 — only one below 1). Ranking **readout-independent** (same order under magnitude653 & multimodal) | Fig 9 · `memcap-a82a02a7` · `qrc_judge.py` |
| 2.2 | Phase-amplitude `R_z(2πs)·R_x(θ)` off/on | ✅ **HURTS −72%** (totMC 10.81→3.06; lin −76%, nl −69% — not a trade). Phase channel collapses FID dynamic range, scrambles states | Fig 10 · `memcap-36e4d1b0` · §4.1 |
| 2.3 | Protons-only vs all-spins injection | ✅ **HURTS −23%** (totMC 10.81→8.28). Clean dissociation: **linear memory unchanged** (protons carry it) but **nonlinear −39%** (carbons feed nonlinearity via J-couplings) | Fig 11 · `memcap-f225906f` · §4.2 |

**Through-line:** both ways of adding structure beyond the baseline lose
**nonlinearity, not memory**. Manuscript addendum `QRC_Encoding_Study.md/.docx`
documents methodology + all three results. **Remaining rigor gap:** single-seed /
`quick` fidelity — a multi-seed firm-up would attach formal error bars (fast; the
gaps are large enough that the ranking is already robust to noise).

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

*(Done: Phase 0, Phase 1, **Phase 2 (all 3 sub-questions)**, Feature-Lab UI —
see activity log up top.)*

1. **Ship remote access** — deployer serves the branch at
   `quantum.cira-core.com/qrc` + Cloudflare Access (build handed off). The page is
   dark until shipped; there is now a complete, figure-backed result to show.
   Independent of the science, can proceed in parallel.
2. **(optional, fast) Multi-seed firm-up** — re-run the Phase-1 negative result
   and the Phase-2 encoding rankings across a few reservoir seeds to attach formal
   error bars. Cheap (short runs); turns "single-seed, quick" into a definitive
   claim for the manuscript.
3. **Phase 3** — external feature libraries (`tsfresh` §3.1, `nmrglue` §3.2): add
   deps + integration. First unstarted science phase.
4. **Phase 4** — full R0–R6 × {Ridge, SVR} dim-reduction grid (reducers exist;
   grid harness + heatmap deliverable do not). Uses best features from Phase 3.
5. **Gated Phase 5** (self-supervised reps) — only if Phase 4 shows headroom.
6. **Gated Phase 6** — encoding×feature interaction (now has the P2 winner) + τ
   sweep (knob exists, runner doesn't).
7. **GRAPE / learned encoding** (deferred, §8.6.1/§8.8) — the differentiable-QRC
   direction; the torch GPU backend is autodiff-capable and the MC metric is a
   ready-made objective. Its own dedicated study, later.
