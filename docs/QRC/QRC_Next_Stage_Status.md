# QRC Next-Stage — Implementation Status Tracker

**Source plan:** [`QRC_Next_Stage_Experiments.md`](./QRC_Next_Stage_Experiments.md)
**Last updated:** 2026-07-28
**Purpose:** one place that traces every phase/experiment in the plan to its real
state, so we always know *what is left*.

## Legend

| Mark | Meaning |
|------|---------|
| ✅ **Done** | Code written, wired, and exercised (or shipped). |
| 🟡 **Code ready — NOT RUN** | Runner + code exist and pass tests, but no experiment has been executed → no results yet. Needs the trace cache and a compute pass. |
| 🟠 **Primitive only — no runner** | The physics/feature knob exists in the engine, but there is **no experiment driver** that sweeps it and no results. |
| ⬜ **Not started** | No code. |

---

## Scoreboard

| Phase | Title | State | Blocking dependency |
|-------|-------|-------|---------------------|
| **0** | Instrumentation: trace cache + FID/progress UI + run control | ✅ **Done** | — (two minor gaps below) |
| **1** | Flag-level feature experiments (phase / multimodal / selection incl. UMAP) | 🟡 **Code ready — NOT RUN** | trace-gen cache (~19 h), then a fast phase1 pass |
| **2** | Encoding sweep (7 functions / phase-amp / protons-only) | 🟠 **Primitives exist — no runner** | build `qrc_phase2.py`; needs fresh re-evolution |
| **3** | External feature libraries (tsfresh, nmrglue) | ⬜ **Not started** | new deps + integration code |
| **4** | Dimensionality-reduction benchmark (R0–R6 × Ridge/SVR) | 🟡 **Partial — reducers exist, no full grid** | best feature set from Phase 3 |
| **5** | Self-supervised representation learning (autoencoder, TS2Vec) | ⬜ **Not started** (gated) | only if Phase 4 shows headroom |
| **6** | Encoding × feature interaction + τ sweep | ⬜ **Not started** (gated) | best encoding (P2) × best features (P3) |
| **UI+** | Feature-Lab UI: Phase-1 sweep panel + UMAP/PCA embedding scatter | ⬜ **Not started** | Phase-1 results to visualize |
| **Ops** | Remote command via Cloudflare Tunnel (run from browser off-box) | ⬜ **Not started** | deploy + auth-gated tunnel; dev box powered |

---

## Phase 0 — Instrumentation ✅ DONE

| Item | State | Where |
|------|-------|-------|
| 0.A Raw-FID trace cache (schema §1) | ✅ | `backend/scripts/qrc_gen_traces.py` |
| 0.B Offline evaluation harness | ✅ | `backend/app/qrc/feature_lab.py` |
| 0.C Backend API + run control (`/api/qrc`, 7 routes, auth-gated launch/stop, allow-list, single-job lock) | ✅ | `backend/app/routes/qrc.py`, `backend/app/qrc/launcher.py` |
| 0.D Frontend UI (dashboard, run detail, FID time chart, spectrum, new-run dialog, results) | ✅ | `frontend/src/views/Qrc*.vue`, `frontend/src/components/Qrc*.vue`, `frontend/src/stores/qrc.ts` |

**Two open gaps in Phase 0:**
1. 🟡 **trace-gen live progress** — `qrc_gen_traces.py` does not yet emit
   `ProgressLogger.status()` step/ETA, so the UI progress bar + event log stay
   empty during a trace-gen run (results still land correctly).
2. ⬜ **Remote command** — the UI runs locally; commanding runs from a browser
   while away needs the launcher exposed via an auth-gated Cloudflare Tunnel
   (quantum.cira-core.com has no GPU; the RTX 5070 Ti is dev-box-local).

---

## Phase 1 — Flag-level feature experiments 🟡 CODE READY, NOT RUN

All feature methods + reducers are implemented in
`backend/app/qrc/feature_methods.py` and driven by
`backend/scripts/qrc_phase1.py`. **Nothing has been run** — needs a real trace.

| Exp | What | State | Where |
|-----|------|-------|-------|
| 1.1 | Add FID phase information (`magnitude653` → `phase`) | 🟡 | `build_features(method="phase")` |
| 1.2 | Enable multimodal features (+time-domain, wavelet, entropy) | 🟡 | `build_features(method="multimodal")` |
| 1.3 | Feature selection after expansion (PCA / KPCA / **UMAP** / LASSO / MI / random) + R²-vs-#features figure | 🟡 | `reduce_features(...)`, `qrc_phase1.py` |

**To finish:** run `qrc_gen_traces.py` (the ~19 h GPU pass) → run
`qrc_phase1.py --trace <name>` (fast, no re-evolution) → decision gate.

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

## UI additions (raised 2026-07-28) ⬜ NOT STARTED

Not part of the original plan text, but the natural next UI increment:

| Item | What | State |
|------|------|-------|
| Feature-Lab panel | Surface the Phase-1 sweep (R² vs #features per method) as an interactive chart instead of a static PNG | ⬜ |
| **UMAP/PCA embedding view** | Backend endpoint → 2D projection of reservoir feature vectors; frontend scatter colored by target | ⬜ |

> Note: neither plan doc actually specs a "UMAP *visualization*" — UMAP appears
> only as a **reducer** (Exp 1.3 / 4.1 R3). The embedding scatter above is a new,
> proposed addition.

---

## What is left, in priority order

1. **Run Phase 1** (unblocks the most): trace-gen cache → `qrc_phase1.py` → gate.
   Everything downstream (Phase 4, the Feature-Lab UI, UMAP viz) needs this data.
2. **Close the two Phase-0 gaps:** trace-gen live progress; remote Cloudflare Tunnel.
3. **Build the Feature-Lab UI** (Phase-1 sweep panel + UMAP/PCA embedding scatter).
4. **Build the Phase-2 runner** (`qrc_phase2.py`) — encoding / phase-amp / protons-only.
5. **Phase 3** (tsfresh, nmrglue) → **Phase 4 full grid** → gated **Phase 5 / 6**.
