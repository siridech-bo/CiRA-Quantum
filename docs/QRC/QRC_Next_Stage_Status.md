# QRC Next-Stage — Implementation Status Tracker

**Source plan:** [`QRC_Next_Stage_Experiments.md`](./QRC_Next_Stage_Experiments.md)
**Last updated:** 2026-08-01 (Phase-2 encoding sweep COMPLETE — 2.1+2.2+2.3; arcsin_sqrt all-spins amplitude wins)
**Purpose:** one place that traces every phase/experiment in the plan to its real
state, so we always know *what is left*.
**Update rule:** this tracker is updated **strictly on every subtask completion**.

## Session activity log (newest first)

| When | Subtask | Result | Commit |
|------|---------|--------|--------|
| 2026-10-21 | **Weather improvement plan, steps 1-2: readout + reservoir variants vs tuned ESN (symmetric protocol)** | 🟡 New harness `scripts/qrc_weather_bench.py`: selection on pooled validation over 4 rolling windows, alpha on val, test read once, QRC and ESN get the same readout options (raw / standardized). **Validation must be 300 days**: with 150 days, slow-leak ESNs (leak 0.02-0.05) win validation and collapse on test; with 300 days the selected ESN (1000 nodes, sr 0.5, leak 0.05, inscale 0.3) scores **0.722** pooled test R² (hindsight max 0.737). Correction to the earlier weather row: moderate leak (0.1-0.3) does help the ESN; the leak-free ESN (0.692) is what tied the QRC. **Step 1** (readout variants from the saved trace): standardizing the |FFT| features is the only lever (0.660 → **0.706**); more bins / phase / line projection: no gain. **Step 2** (new `scripts/qrc_weather_block.py`: exact block-propagator weather pass + FID readout, **48 s CPU vs 14.6 GPU-h**, matches the saved trace to 3e-4 over 1474 days; block FID = fid_signal to 2.6e-13): ¹H+¹³C dual readout 0.689, τ=10/100 ms 0.67/0.57, multi-τ 0.57-0.65, dccb7 0.669/0.586 — **adding features hurts (600 training days: data-limited, not reservoir-limited)**. Standing: QRC 0.706 vs tuned ESN 0.722, 95% CI of QRC-ESN [-0.066, +0.003] (ESN marginally ahead). Next: step 3, learned encoding on weather | `<this commit>` |
| 2026-10-21 | **Weather (the Paper-4 advantage task): QRC vs fairly tuned ESN = statistical TIE** | ⚠️ `scripts/qrc_weather_fair.py` (cira2, CPU): QRC 653-FID readout rebuilt from `weather_full.npz` reproduces every saved R² exactly; same split 374/600/500, horizons 1-45, R². ESN grid size(300-2000)×sr×leak(0.02-1)×input-scale×ridge, 3 inits, selected on the last 150 train days. **Robust baseline = val-selected no-leak ESN (2000 nodes, sr 1.1, input scale 0.1)**: main split 95% block-bootstrap CI of ΔR² includes 0 on **12/14** targets (QRC ahead only at humidity h1 and temp h5, by 0.02-0.03; long horizons all ties, e.g. temp h45 0.786 vs 0.773 [-0.065,+0.080]); across 4 windows × 14 targets QRC better on **30/56**, median ΔR² **+0.002**. The old reproduction ESN (sr 0.9, input scale fixed) gave temp h45 0.697 — the gap came from **spectral radius + input scale**, not leak. Slow-leak ESNs (0.02-0.05) win on the 150-day validation but fail on test (negative R² in places) → single-slice validation is unreliable on this seasonal series (selection lesson for SOP). **Verdict: the Paper-4 / our-reproduction weather 'QRC beats ESN at h≥10' does not survive a tuned ESN** — parity, not advantage. Data `qrc_weather_fair_esn.json`, `qrc_weather_fair_bootstrap_noleak.json` | `<this commit>` |
| 2026-10-21 | **Fully tuned ESN (data seeds 1-3) — decisively beats every QRC result** | ⚠️ `qrc_classical_fair.py --models esn` (cira2, 63 s): ESN grid size×sr×leak×input-scale×ridge, 3 inits, select on val. Test NMSE mean over seeds: **NARMA-2 1e-5** (size-matched ≤350: 3e-5), **NARMA-10 0.0107** (matched 0.0203), **Mackey-Glass h=10 ~3e-6** (matched 1e-5). The old size-only ESN (sr 0.9, leak 1, seed 1) gave 0.0046 / 0.029 / 0.0057 — tuning matters by 2-3 orders of magnitude. **Verdict: the 6-spin Mackey-Glass 'win over classical' (0.00155 vs ESN 0.0057) does NOT survive a fairly tuned ESN** (~500x worse); no QRC system so far is competitive with a tuned ESN on any task. Claims about the learned encoding vs arcsin (quantum-mediated, ablation-controlled) are unaffected. Data `docs/QRC/data/qrc_classical_fair_esn.json`. Tuned LSTM still running | `<this commit>` |
| 2026-10-21 | **Further-proof program launched: fair classical baselines (cira2) + real strongly-coupled molecule dccb7 (cira3)** | 🔵 Running. **SOP updated first** (§1: τ convention — standard 30 ms, others labeled, τ ≲ T₂, Nyquist V; new system `dccb7`; realism note: generic ×2 chain = abstract Ising reservoir, not an NMR molecule; §6.5 fair classical baselines; §6.6 J=0 ablation). (1) `scripts/qrc_classical_fair.py`: ESN full grid (size×sr×leak×input-scale×ridge, 3 inits, select on val) + size-matched ESN (≤350) + tuned LSTM, on **data seeds 1-3** (old baseline: seed 1 only, size-only ESN). (2) `dccb7` preset = 7 ¹³C of the dichlorocyclobutanone derivative (Lu/Li et al. arXiv:1601.04721 Fig. 6, ¹H decoupled): real, all 7 spins read out, 8 couplings ≥20 Hz, T₂ 0.5-1.9 s, D_eff=172 (344 feats), carrier-centred ±14.2 kHz. Powered: arcsin NARMA-2 0.586 @τ=30 ms (V=860) vs **0.094 @τ=100 ms** (V=2850, J·τ≈6). 7.2 ms/step → ~18 min/job. (3) chain on cira3: τ=100 ms main (MG h10 + NARMA-2 × 3 seeds + τ→0), τ=100 ms J=0 ablation (readout basis kept = coupled lines, `--lines-coupling-scale 1`), τ=30 ms main. Runner: `--tau`, `lines_system` | `<this commit>` |
| 2026-10-21 | **Crotonic-9 learnable-encoding campaign COMPLETE (cira3, 7.5 h)** | ✅ 12/12 jobs, real crotonic J (x1), proton reduced-FID (D_eff=100, V=200), regime B, T=1500, 100 Adam steps. **Learned per-spin beats arcsin 9/9 seeds and every tau->0 ablation collapses (~0.88-1.0) -> the encoding gain is quantum-mediated on the real 9-spin molecule too.** Test NMSE (mean ± sd, 3 seeds): **NARMA-2 0.0079±0.0035 vs arcsin 0.317 (~40x)** [6-spin: 0.0052 vs 0.101]; **NARMA-10 0.277±0.069 vs 0.506 (~1.8x)** [6-spin: 0.120 vs 0.312]; **Mackey-Glass h=10 0.0102±0.0031 vs 0.089 (~9x)** [6-spin: 0.00155]. vs classical: beats the untuned LSTM(64) on all three, but **does NOT beat the tuned refs** — NARMA-2 ≈ ESN 0.0046 (slightly worse), NARMA-10 ≫ ESN 0.029, **MG h=10 0.0102 loses to tuned LSTM 0.0024 and ESN 0.0057** -> the 6-spin MG win does not transfer to crotonic-9 under this protocol. Untested hypothesis: crotonic's real J are weak vs the 6-spin x2 coupling. Data `docs/QRC/data/qrc_crotonic9_campaign.json`; traces in `backend/artifacts/qrc_runs/crotonic9-campaign-20261020/` | `<this commit>` |
| 2026-10-20 | **Crotonic-9 learnable-encoding campaign LAUNCHED on cira3 (A100)** | 🔵 Running (`~/cira-qrc/runs/crotonic9-campaign-20261020` on cira3, pid 2386252, started ~2026-10-20). The 6-spin confirmed protocol on the real 9-spin crotonic acid: **J at real values (x1)**, reduced-FID over the **protons 4-8 (D_eff=100, V=200**, dwell 0.15 ms, Nyquist-safe per SOP 3.2), per-spin learned encoding on all 9 spins vs arcsin, T=1500, 100 Adam steps, lr 0.02, leakage-free 3-way, regime B. 12 jobs: NARMA-2 / Mackey-Glass h=10 / NARMA-10 x seeds 1-3 + one tau->0 ablation each; LSTM(64) per job; tuned LSTM/ESN references from classical-baseline-548dbea6. Measured 14.2 ms/step -> ~36 min/job, **~7.5-8 h total** (12 h budget guard). Powered: arcsin NARMA-2 @T=1500 = 0.31-0.33. Fixed: `make_fid_reduced_post` now sums the FID over the readout spins only (was all spins; 6-spin results unaffected). Runs outside the local launcher registry (remote) | `<this commit>` |
| 2026-10-20 | **Qubit wall broken: exact block-structured propagator → learnable training at 9–12 spins (lab A100, cira3)** | ✅ New `app/qrc/block_propagator.py`: for this reservoir (diagonal H + local σ₋/σz) the coherence pattern p=a⊕b and a∧p are conserved, so L splits into sub-blocks — propagator **6ⁿ entries instead of 16ⁿ** (9 spins: 0.13 GB vs 550 GB dense), built analytically (no 4ⁿ Liouvillian) + V-node readout precomputed on the k=0/1 blocks. **Exact:** generator = QuTiP Liouvillian to 1e-16; step/grad = dense path to ≤3e-12 (n=3–5, `scripts/qrc_block_check.py`); crotonic-9 vs Taylor GPU path 2.4e-13 and **~900× faster** (5.4 vs 4952 ms/step, c128). Runner: `--propagator auto|dense|block|taylor` (auto = block ≥7 spins), per-spin 2×2 pulse contractions (checkpointed), `--ckpt-every` step checkpointing. **A100 benchmark** (reduced-FID, V=D_eff, coupling ×2, c64; projected for the paper protocol T=1500 × 100 Adam steps, per condition): 9-spin 14 ms/step → **0.6 h, 3.4 GB**; 10 → **0.9 h, 13 GB**; 11 (ckpt 40) → **4.0 h, ~7 GB**; 12 (ckpt 40) → **15 h, ~33 GB**. Data `docs/QRC/data/qrc_block_bench_cira3.jsonl`. Benchmark only — no science run yet; remote runs are not in the local launcher registry | `<this commit>` |
| 2026-10-20 | **Physics simulator (/learn/simulator): re-vendored CiRA-QuantumSim to 398f1a7** | ✅ Synced `frontend/public/sim/` from c78843c to `library-reference-pdfs` @ 398f1a7 (superset of master; +56 commits). New pages: PDH lock sim, ion MS-gate verifier, Experiment-Theory validation (+guide), smooth x GBC playground; ion engine updates; Library 91 -> 113 papers (PDFs gitignored, baked into image). CDN -> vendor/ rewrites re-applied (no runtime CDN). Verified: `npm run build` OK; headless load of all 11 sim pages = 0 console errors / 0 failed / 0 external requests; library lists new refs; pdf.js reader renders a new PDF. Not QRC compute (no GPU) | `<this commit>` |
| 2026-08-09 | **Mackey-Glass h=10 QRC win CONFIRMED multi-seed (3/3) — paper finalized** | ✅ `mackey-h10-confirm-2d21b56f` (8.45 h, seeds 2,3): perspin **0.00172, 0.00162** → with seed 1, **mean 0.00155 ± 0.00018 (3 seeds)**, beats tuned-LSTM 0.0024 (3/3) AND ESN 0.0057 (3/3), ≫ arcsin 0.030. Verdict CONFIRMED. Finalized manuscript (`.tex`+`.md`+`.docx`): dropped single-seed caveat, MG h=10 = 0.00155±0.00018, ratios ~1.6×/~3.7×. Regenerated figs with mean±std error bars + **real learned-encoding map + training curve** (figP4 from saved `encoding_detail.json`). Committed + pushed | `<this commit>` |
| 2026-08-08 | **LaTeX/RevTeX manuscript — proper equations, extended physics, full references** | ✅ `QRC_Manuscript.tex` (RevTeX 4.2, PR-native): equations as proper LaTeX (H + eigenvalues, Lindblad + jump ops, FID, analytic line formula, direct-DFT projection); **extensive rigorous §II physics** (ensemble-readout advantage, rotating-frame secular derivation, dissipation, initial state, encoding/readout physics, numerics, + full crotonic-9 & 6-spin parameter tables); **26-ref bibliography** (RC/QRC/NMR-QC/PSR/tasks) embedded + `qrc_refs.bib` (2026 project refs flagged [VERIFY]). Structurally validated; compiles on Overleaf (no local TeX). Figures figP1-3 embedded. Also generated `.docx` earlier (`QRC_Manuscript.docx`) but LaTeX is the equation-quality version | `bbab417`,`93cfb56` |
| 2026-08-08 | **QRC-learned WINS on Mackey-Glass h=10 vs tuned LSTM + ESN (task-dependent picture)** | 🟢 Run `learnable-0a45c7b0` (h=10, reduced-FID, 6-spin ×2). **learned perspin 0.0013 vs arcsin 0.0235 (~18×, non-degenerate)**, and vs classical: **QRC 0.0013 < tuned-LSTM 0.0024 (~1.9×) < ESN 0.0057 (~4.4×)** — QRC-learned beats BOTH on the one discriminative task. Combined with NARMA (ESN wins), the picture is **task-dependent**: classical wins on linear-memory NARMA, QRC-learned wins on hard nonlinear chaotic prediction. ⚠️ single seed — multi-seed confirmation needed before a firm claim. Wired `--horizon` for mackey. Manuscript §VI D + abstract + discussion updated | `run 0a45c7b0`,`02a9e06` |
| 2026-08-08 | **Strengthened classical baseline — ESN beats QRC on NARMA (honest reality check)** | ✅ `classical-baseline-548dbea6` (1.68 h): tuned-LSTM sweep (multi-init, cosine LR, grad-clip, early-stop) + ESN size-sweep, same split/metric. **Tuned LSTM erased the strawman gap** (narma2 0.134→**0.0114**, ~12×); QRC still beats tuned LSTM ~2×. BUT **a well-sized ESN(N=1000) matches QRC on narma2 (0.0046 vs 0.0052) and beats it 4× on narma10 (0.029 vs 0.120)** → **NO quantum-advantage-over-classical** on these benchmarks (decoherence-limited, as expected). Contribution reframed honestly: physics-informed readout + quantum-mediated *encoding* gain, not quantum-beats-classical. mackey h=1 trivial for all; **QRC@mackey-h10 pending** (only discriminative task). Manuscript §VI + finding doc + fig20 updated | `run 548dbea6` |
| 2026-08-08 | **Reduced-FID learnable encoding CONFIRMED (multi-seed + τ→0 ablation, 3 datasets)** | ✅ Campaign `learnable-campaign-01ce463b` (41.0 h GPU, 10 jobs). Learned per-spin beats arcsin **8/8 across narma2/narma10/mackey_glass, all seeds**: narma2 **0.0052±0.0013 vs 0.101 (~19×, 3/3)**, narma10 **0.120±0.019 vs 0.312 (~2.6×, 3/3)**, mackey **7.6e-5 vs 6.3e-4 (2/2)**. **τ→0 ablation collapses both memory tasks to ≈1.0 (mean-predictor)** → genuinely quantum-mediated (the decisive control passes). Readout `fid_reduced` (D_eff=161→322 feats), regime B, leakage-free. First multi-seed+ablation-confirmed learnable win on a physics-grounded readout. ⚠️ QRC-vs-LSTM still soft (LSTM under-tuned) → strengthened baseline queued before quoting. Doc: `QRC_Reduced_FID_Learnable_Finding.md` | `run 01ce463b` |
| 2026-08-06 | **Learnable encoding on the physics-informed reduced-FID readout — HEADROOM (single-seed)** | 🟢 First learnable-encoding win on a *physics-grounded* readout (not the observable one). 6-spin, coupling ×2, NARMA-2, regime B, leakage-free 3-way, GPU ~4.1 h (run `learnable-32c531df`). **learned perspin test NMSE 0.00345 vs arcsin 0.0874 (~25×), non-degenerate (enc std 0.59), powered.** Readout `fid_reduced`, **D_eff=161 lines → 322 feats** (coupling ×2 widens multiplets, up from 137/274 at ×1 — physics selection recomputed for the scaled H). ⚠️ **NOT confirmed:** single seed, no τ→0 ablation yet, p≫n margin modest (735 train / 322 feat ≈ 2.3×). Needs multi-seed + ablation before quoting. Runner now saves the final-encoding FID trace (`learned_encoding_fid.npz`) + live per-step progress | `ef4e2ed`,`run 32c531df` |
| 2026-08-05 | **Experiment-setup + comparison web app (in the CiRA Quantum SPA)** | ✅ Built `/qrc/new` (schema-driven param panel — every knob visible, dataset dropdown, physics-informed readout+D panel, compare-against selector, Run/Stop, GPU-confirm dialog) and `/qrc/compare` (run+reference multiselect, task/readout/D badges, comparability guard, config-diff + metric-bar tables). Backend: `GET /api/qrc/schema` (launcher allow-list + UI metadata, can't drift), `references.py`+`GET/POST /references` (benchmarks/tiers seeded paper/arcsin/653-FID/reduced-FID/classical), `compare.py`+`POST /compare` (SOP §0 guard). Made `qrc_memory_benchmark` + `qrc_learnable_9spin` API-launchable. E2E verified: launch memory run via API → done → results → compare. 22 route tests + vue-tsc + vite build all pass | `4ef5eb5`,`75efa2d` |
| 2026-08-05 | **Physics-informed reduced-FID readout — analytic line selection (SOP §3.2)** | ✅ `app/qrc/spectral_lines.py`: enumerate the FID's analytic single-quantum transitions `f_k(z)=ν_k+½Σ J_kj z_j` from the all-σz Hamiltonian, merge within the decoherence linewidth `1/(πT₂)`, build the differentiable direct-DFT projection Φ. **D_eff = 100** for crotonic-9 protons (from 1280 raw — 653 oversamples ~6.5×), **137** for 6-spin. Wired as the memory benchmark's `fid_reduced` readout (validated CPU on 3-spin: richer capacity than the observable readout). Answers "how to select fewer" — it's the reservoir's resolvable line count, not a guess | `4603a78`,`022aa79` |
| 2026-08-05 | **SOP §3.1 — two training regimes recorded** | ✅ Standard QRC (paper) trains ONLY the ridge readout W (reservoir+encoding fixed); our learnable encoding adds a gradient-trained encoding. Every result must be labeled with its regime | `3bcbc9c` |
| 2026-08-05 | **Time-multiplexing (virtual nodes) STM/PC benchmark — NMR differs from qubit-boson** | ✅ `qrc_memory_benchmark.py` (Fig 19). Reproduces the Das-Giorgi-Zambrini STM(linear)/Parity-Check(nonlinear) protocol on our 6-qubit reservoir, sweeping V∈{1,2,5,10}. **Both** memories rise with V (STM totMC 1.00→**2.68**, PC totCap 0.42→**1.27**; diminishing past V=5) — **NOT** the paper's asymmetry (their linear STM stays flat). Mechanism: our spins precess at different chemical-shift frequencies, so V snapshots sample different *phases* → linearly-independent features → boosts *linear* memory too (the boson's linear content is read-once). So on an oscillatory multi-frequency reservoir, time-mux is a *general* memory booster. Gives standard memory numbers: linear MC≈2.7, nonlinear≈1.3 at V=10 | `fig19` |
| 2026-08-05 | **Stacked clean levers (encoding + coupling + correlation) — best quantum-clean config 0.058** | ✅ The 3 quantum-clean levers COMPOUND: learned per-spin (T=600, J=1, single) **0.1515** → +2× coupling +correlation readout **0.0584** (−61% further; −73% vs the 0.216 open-loop). Complementary. **Quantum-mediated** (broadcast scalar encoding, no explicit classical memory → τ→0 would collapse it), unlike the window/feedback/spatial family. Gap to classical **narrowed from ~20× to ~5×** (best classical ~0.011) but still not crossed on NARMA-2. Best honest quantum-reservoir number | `6d99a46` |
| 2026-08-05 | **Spatial-window feed (qubit i ← x_{t-i}) — helps QRC but is CLASSICAL (control)** | ✅ `qrc_spatial_feed.py` (Fig 18). Laying the input sequence across the coupled qubit register (each qubit a different lag, sliding) **beats broadcast** (0.30→**0.19**, fixed arcsin) — even beats the *learned* encoder (0.216). BUT the control is decisive: a **classical linear ridge on the same 6-window = 0.196** (matches it), and **classical quadratic = 0.012** (~15× better). So the win is the explicit classical window, and the quantum reservoir is a *worse* window-processor than a trivial classical model. **Same confound (3rd time): any explicit classical memory in front of the reservoir makes the quantum part redundant.** The only quantum-clean result stays broadcast+scalar (reservoir = sole memory) | `fig18` |
| 2026-08-05 | **Parallel QRNN (decoupled vs coupled qubits) — coupling helps, decoupling hurts** | ✅ `qrc_parallel_qrnn.py` (Fig 17). Per-qubit parallel feed, coupling sweep J∈{0,1,2}×. **Decoupled/parallel (J=0) is WORST on task** (NMSE 0.71/0.47) despite HIGHEST eff-dim (3.0); coupling monotonically **improves the task** (→0.28 at J=2) while **lowering** eff-dim (→1.4). Independent qubits = diverse-but-shallow features (no cross-qubit memory); coupling creates the task-relevant nonlinear memory. Correlations carry info only when coupled (product state factorizes at J=0). Nuance: **eff-dim is NOT a proxy for reservoir quality** (coupling lowers it but helps). Answer: parallel-independent QRNN doesn't help — the qubits must *interact* | `fig17` |
| 2026-08-04 | **Manuscript: full honest arc banked (§6 encoding · §7 correlation readout+decoherence · §8 classical comparison)** | ✅ `QRC_Encoding_Study.md` + `.docx` (Fig 6, 15, 16). §7: correlation readout helps (~44%) but eff-dim ~1 → decoherence-limited (mechanism). §8: quantum reservoir beats classical ESN on weather (fair reservoir-vs-reservoir) but loses to trained RNN/LSTM on classical tasks (with τ→0 controls proving which results are quantum-mediated). Honest bound stated: real quantum *improvements* to the reservoir; competitive vs classical reservoirs; not vs trained RNNs; advantage needs coherence>entangling regime. Complete, publishable | `report` |
| 2026-08-04 | **Correlation readout + entangling drive — why the quantum reservoir is decoherence-limited** | ✅ `qrc_correlation_readout.py` (Fig 16). Reading 2-body correlations ⟨σᵢσⱼ⟩ (uses the joint/parallel qubits) **helps NARMA-2** (0.373→**0.207**, ~44%) — a real, cheap perf lever. BUT **effective dimensionality stays ~1–2** (126 corr features → PR 1.8) → the 2ⁿ space is still unused. Key mechanism: **longer coherent evolution LOWERS eff-dim (1.4→1.1)** — decoherence (T₂) collapses the state faster than the Ising dynamics populate it. The two resources (many qubits, coherence) are in **tension**, not additive → this is *why* eff-dim~1 and the reservoir isn't competitive. Quantum advantage needs entangling dynamics faster than decoherence (long T₂ × strong couplings), which crotonic NMR lacks | `fig16` |
| 2026-08-04 | **(3) Closed-loop quantum-memory RNN — is a CLASSICAL RNN (τ→0 control decisive)** | ✅ NARMA-2: closed-loop (mem ON) 0.137 beats open-loop per-spin 0.216 — BUT closed-loop **τ→0 (quantum memory OFF) = 0.011** (near-perfect, *better* than with memory). The feature-feedback channel is a classical RNN (36-dim hidden = features) that models NARMA-2 by itself; quantum memory is redundant and even **hurts** (0.011→0.137). A tiny classical RNN beats the quantum reservoir ~20× (0.011 vs open-loop 0.216). **Confound caught by the built-in control.** Conclusion of (1)(2)(3): the *only* clean quantum result is open-loop scalar per-spin (ablation-proven); any classical memory (window/feedback) dominates → quantum reservoir not competitive with classical RNNs on these tasks | `988f17b` |
| 2026-08-04 | **Strategy-B windowed win is CLASSICAL (control), not quantum — clean quantum result is NARMA-2 per-spin** | ✅ Decisive controls: (a) **τ→0 ablation proves the NARMA-2 per-spin win IS quantum** — scalar-input per-spin collapses 0.25→**1.01** (≈mean) with quantum memory off; a scalar encoder has no classical memory, so the reservoir provably carried it. (b) NARMA-10 scalar per-spin **ties arcsin** (0.43 vs 0.38) — encoding can't help memory-bound tasks. (c) NARMA-10 **windowed** (k=12) beats arcsin (0.12 vs 0.38) BUT a **classical linear ridge on the same 12-window gets 0.14** → the win is the classical window injecting memory, **NOT the quantum reservoir** (quantum adds only ~15%, single-seed). Confound confirmed → strategy B is not a quantum result. **Lesson: injecting classical memory (window/RNN) makes the quantum reservoir redundant → the clean quantum story is scalar-input + per-spin + quantum-reservoir memory** | `0e29fe7` |
| 2026-08-04 | **Optimizer-robustness check — per-spin win is NOT an LR artifact** | ✅ Re-ran 5 seeds with a stabilized optimizer (lr 0.02 + grad-clip + best-val checkpoint). Verdict unchanged: per-spin **0.267±0.109 vs arcsin 0.423, 4/5 beat, ~37% (p≈0.043)** — matches the loose (lr 0.04) run (~42%, p≈0.036). Tightening did **not** turn 4/5→5/5 (seed 2 still ties) and did **not** fix **global** (still diverges to 1.75) → global instability is *fundamental*, the win is specifically **per-spin**. Fig 15 (both recipes). Robust, honest result → proceed to write-up | `fig15` |
| 2026-08-04 | **6-spin per-spin learnable encoding CONFIRMED beats arcsin (5 seeds)** | ✅ `qrc_6spin_aggregate.py` (Fig 15). **per-spin 0.246±0.107 vs arcsin 0.423±0.058 — beats in 4/5 seeds, ~42% mean, paired t=−3.12 (p≈0.036)**. **First confirmed positive** that learnable encoding is a real *performance* lever. Honest caveats: (i) seed-variable (1 seed tied; single-seed 2.3× was lucky-end, true effect ~42%); (ii) learned **global** is *unreliable* (training diverges, 0.28–1.75) → the win is specifically **per-spin (frequency-selective)** expressivity; (iii) needs the encoding-sensitive task + rich reservoir (3-spin/global/synthetic tied) | `fig15` |
| 2026-08-04 | **6-spin learnable encoding BEATS arcsin on NARMA-2 (single-seed, all guardrails pass)** | 🟢 `qrc_learnable_9spin.py --system 6 --task narma2` (Fig 14, leakage-free 3-way, dense-propagator ~140× speedup). arcsin baseline **0.395** (powered) vs learned **global 0.346** (−12%) vs learned **per-spin 0.175** (**−56%, 2.3× better**; std 0.98, non-degenerate; n_train 135≫37). **First positive** — win comes from **per-spin (frequency-selective) encoding** (global learned barely beats, consistent with 3-spin tie). Passes every guardrail that caught the earlier false positive. ⚠️ **single seed** — multi-seed confirmation pending before "9-spin justified" | `4b1d3ad`,`fig14` |
| 2026-08-04 | **9-spin GPU feasibility — backprop BLOCKED by memory wall (found with 0 GPU)** | ⚠️ `qrc_learnable_9spin.py` (runner + benchmark, CPU-validated). No-GPU feasibility calc: crotonic-9 needs **5,336 substeps** (stiff L) → reverse-mode autograd graph is **~201 GB per input-step** vs 16 GB card (off 100–1000×). **Naive 9-spin backprop is impossible on this hardware.** Feasible paths: (a) **PSR-in-sim** (forward-only, memory-free, reuses validated PSR; compute-heavy), (b) gradient checkpointing (~11 GB, 2–3× compute), (c) adjoint method, (d) ≤5-spin stepping stone (backprop fits). Saved a doomed GPU run | `9spin` |
| 2026-08-04 | **Learnable-encoding convergence + local-optimum test (CPU) — learned TIES arcsin** | ✅ `qrc_learnable_converge.py` (Fig 13, working regime V=4/T=240, guardrails). Random-init learned converges to **NMSE 0.570 ≈ arcsin 0.574** (tie, not a beat) via a *different-shaped* (decreasing) non-degenerate map → **performance is encoding-shape-insensitive** (reservoir is the bottleneck, echoes Phase-1). ⚠️ First run (V=3/T=110) REJECTED: under-powered (arcsin≈mean-predictor 0.99) → a **degenerate near-constant "win"** = false positive; guardrails added. Warm-start local-opt probe spoiled by too-high LR (diverged out of basin) — needs a gentle-LR re-probe to strictly prove local optimum | `e34a94e`,`fig13` |
| 2026-08-04 | **PSR simulated + validated vs autograd (closes sim↔hardware loop)** | ✅ `qrc_psr_sim.py` (CPU, no GPU): the hardware gradient recipe (forward runs at shifted angles only) reproduces the autograd reservoir-feature gradient to **machine precision** — single-spin **2-term PSR 8e-16**; global 3-spin **2-term WRONG (56%)** but **generalized 6-term (2n) exact 3e-15**. Empirically confirms the §6.5 global-vs-per-spin caveat. The gradient a spectrometer would measure = the one we backprop | `psrsim` |
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
