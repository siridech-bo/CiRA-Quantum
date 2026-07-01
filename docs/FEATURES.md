# CiRA Quantum — Feature Summary

A platform that hosts **three coordinated quantum applications** under
one account, one BYOK keyring, one auth surface, and one operator
admin view.

| App | Status | Tagline |
|-----|--------|---------|
| **Optimization** | live | Plain-English combinatorial problem → 10-solver comparison |
| **QML** | live | Variational quantum classifier vs four classical baselines on every dataset |
| **qLDPC** | preview (Sprint 0) | Quantum LDPC code-family gallery + primer |

---

## Platform layer (shared by all apps)

- **One account, one keyring.** Single login (`/login`, `/signup`),
  Fernet-encrypted BYOK keys reused across apps (your `ibm_quantum`
  key powers both `qaoa_ibmq` and `vqc_ibmq`; same for `originqc`).
- **Public landing page** (`/`) with the three-app showcase and a
  per-app capability tier strip.
- **Admin views** (`/admin`, role-gated): Overview / Users / Jobs /
  QML tabs.
- **Session-cookie auth + CSRF** (Flask sessions, secure cookies).
- **SSE event bus** (in-process default, Redis Streams when
  `USE_REDIS_QUEUE=1`).
- **Optional async queue** via Redis-RQ (`USE_REDIS_QUEUE=1`); shared
  by both apps' training/solve paths.
- **Reproducibility-by-default.** Every persisted record pins
  `code_version` (git SHA), seeds, library versions, and a 16-char
  `repro_hash`.

---

## App 1 — Optimization (`/solve`, `/templates`, `/benchmarks`)

### Solve pipeline (5 stages)

`formulate → compile → validate → solve → interpret`

- **Formulation provider layer** — Claude / OpenAI / local Ollama,
  BYOK with Fernet at rest.
- **CQM compiler** — JSON → `dimod.ConstrainedQuadraticModel`.
- **Validation harness** — Layer A oracle (small CQMs), Layer C
  constraint coverage (always), Layer B opt-in.
- **Multi-solver fan-out** — Pick any subset of 10 registered
  solvers; results land side-by-side with per-solver tier badges,
  expandable solver-comparison table, and live SSE progress.
- **LLM-driven interpretation** — Raw bitstring → human-readable
  solution with original variable names.

### Solver registry (10 solvers across 5 tiers)

| Tier | Solvers | Notes |
|------|---------|-------|
| Classical SOTA | `exact_cqm`, `cpsat`, `highs` | OR-Tools + HiGHS as dimod.Sampler adapters |
| QUBO heuristic | `gpu_sa`, `cpu_sa_neal` | CUDA simulated annealing (sm_120 verified), Neal fallback |
| Quantum-inspired | `parallel_tempering`, `simulated_bifurcation` | Hukushima-Nemoto, Goto 2019 |
| Quantum (simulator) | `qaoa_sim` | Local pyqpanda QAOA, 20-qubit cap |
| Quantum (real QPU) | `qaoa_originqc`, `qaoa_ibmq` | Origin Wukong + IBM Heron/Eagle via BYOK |

### Examples / Templates (`/templates`)

- **10 curated optimization problems** with documented optima
  (knapsack, max-cut, TSP, JSS, …).
- **Modules** grouping for structured curriculum use.
- One-click "Solve this template" → pre-fills the solve form.
- **Template-match badge** on solved jobs — compare solver's actual
  answer to textbook optimum.

### Benchmarks dashboard (`/benchmarks`, public)

- **Append-only `benchmarks/archive/` JSON store** with gzipped
  `dimod.SampleSet` companions.
- **Four views**:
  - `/benchmarks` — suite picker + recent records
  - `/benchmarks/suites/<id>` — instance × solver grid
  - `/benchmarks/solvers/<name>` — solver-profile page (version history)
  - `/benchmarks/instances/<id>` — per-instance leaderboard
- **Findings page** (`/benchmarks/findings`) — aggregated cross-suite
  match-rate + mean-time charts.
- **Citation buttons** — BibTeX entry per record.
- **Pending-cloud-jobs panel** — tracks queued Origin/IBM submissions
  until they materialize.

### Learn

- `/learn/quantum` — **Quantum 101** interactive QAOA explainer:
  4-panel view (polynomial, trained circuit, measurement histogram,
  top-K classical filter).
- `/learn/playground` — **Circuit Playground** (self-hosted Quirk-E).

---

## App 2 — QML (`/qml`, `/qml/learn`, `/qml/benchmarks`)

### Foundation

- **6 curated datasets**: Moons, Circles, Iris, Wine, MNIST 0-vs-1
  (sklearn 8×8), Breast Cancer.
- **Public dataset gallery** (`/qml`) — cards with category /
  difficulty / feature count chips.
- **Capability strip** — surfaces whether `pennylane`, `scikit-learn`,
  `qiskit-ibm-runtime` are installed.

### Study-before-train flow (`/qml/datasets/:id`)

- **Scatter preview** — native 2D for Moons/Circles; PCA-2 projection
  for higher-dim datasets.
- **What-gets-trained list** — 5-row breakdown (VQC + 4 baselines).
- **Live circuit preview** — `VqcCircuitExplainer` re-renders as the
  student touches the hyperparameter form.
- **Hyperparameter form** — qubits, layers, epochs, batch size,
  learning rate, seed, with PCA warning when `n_qubits < n_features`.

### Live training (`/qml/jobs/:id`)

- **PennyLane VQC** — AngleEmbedding (RX) → BasicEntanglerLayers
  (RY + ring CNOT) → PauliZ on qubit 0 → σ(⟨Z⟩+b).
- **Adam on binary cross-entropy**, exact gradients on the
  statevector simulator.
- **Gate-level circuit explainer** (`VqcCircuitExplainer`) — same
  visual conventions as the QAOA panel: opaque colored gate boxes,
  `|0⟩` initial state, column numbers, layer-boundary dashed marks,
  color-key footer.
- **Two expandable explainers** — "What's happening in this circuit?"
  and "Why a simulator, and what would change on a real QPU?".
- **Live training-loss chart** — dual-axis SVG (loss red, train/test
  accuracy blue/green), updates per epoch.
- **Live decision-boundary heatmap** — 20×20 probability grid, refreshes
  every 3 epochs for 2-qubit jobs. Blue → white → orange divergent
  colormap; the 0.5 ribbon is the actual decision boundary.

### Classical baselines (auto-run on the same train/test split)

- **Logistic Regression** (linear floor)
- **SVM-RBF** (kernel baseline)
- **Random Forest** (tree ensemble)
- **MLP 16-8 ReLU** (fairest head-to-head)
- **`BaselineComparison` scoreboard** — five rows, ranked by test
  accuracy, trophy on the winner, single-sentence verdict explaining
  whether the VQC matched, edged, or lost.

### Real-QPU evaluation (`QpuRunPanel` on completed jobs)

- **Multi-provider tab UI** — IBM Quantum vs Origin Quantum.
- **IBM Quantum** (`vqc_ibmq`) — Qiskit Runtime `SamplerV2` batch
  inference, one PUB per test point. Returns full test accuracy +
  confusion matrix, appended as a row in the comparison scoreboard.
- **Origin Quantum** (`vqc_originqc`) — pyqpanda3 single-point
  inference (Origin doesn't batch). User can choose
  `full_amplitude` cloud simulator or `WK_C180` real Wukong (gated
  behind `ENABLE_ORIGIN_REAL_HARDWARE=1`).
- **Live queue status** — polls every 5 s, shows cloud job ID +
  queue position + IBM/Origin status string.
- **Provider-tagged errors** — switching tabs clears stale errors;
  "no backend matches" gets a hint to try a named backend.

### Primer (`/qml/learn`)

- **5-step walkthrough**: qubit → encoding → entangler → measurement
  → learning.
- **Interactive `BlochSphereDemo`** — slider for input feature `x`
  rotates the state vector on a 2D Bloch sphere, live ⟨Z⟩ + class
  probability readouts.

### Benchmark archive (`/qml/benchmarks`, public)

- **`qml_benchmarks/archive/` append-only JSON store**.
- **Public leaderboard** sorted by test accuracy descending; columns
  for dataset / model / qubits / wall time / contributor / QPU chip.
- **Filters** by dataset and model with live facet counts.
- **Record detail page** (`/qml/benchmarks/:id`):
  - Identity strip (contributor, repro hash, code version, hardware ID)
  - Headline metric tiles
  - Full training history + baseline comparison + QPU runs
  - **Copy-BibTeX button**
- **Admin-curated** — only admins can archive a run, contributor
  credit goes to the original runner.
- **Weight-stripping** — trained weights / scatter / decision grid
  removed before write (model state, not citation material).

---

## App 3 — qLDPC (`/qldpc`, `/qldpc/learn`) — Sprint 0 preview

### Code-family registry (4 families today)

| Family | Regime | Sprint |
|--------|--------|--------|
| Surface code | Topological | 0 |
| Toric code | Topological | 0 |
| Bicycle code | CSS classical | 0 |
| Hypergraph-product code | CSS product | 0 |

### Endpoints (live today)

- `GET /api/qldpc/health` — capability flags
- `GET /api/qldpc/code-families` — registry list
- `GET /api/qldpc/code-families/<id>` — family detail
- `GET /api/qldpc/code-families/<id>/matrix` — parity-check matrix
- `GET /api/qldpc/code-families/<id>/distance` — code distance
- `GET /api/qldpc/code-families/<id>/tanner-graph` — Tanner graph JSON

### Frontend pages

- `/qldpc` — code-family gallery + Sprint roadmap
- `/qldpc/learn` — primer (LDPC fundamentals)
- `/qldpc/codes/:id` — single-family detail with matrix + Tanner graph
  visualization

### Roadmap (sprints)

- **Sprint 3** — `stim` Monte Carlo threshold benchmarks
- **Sprint 4** — `qiskit-qec` syndrome circuits on IBMQ hardware

---

## Admin (`/admin`, role-gated)

Four tabs reading from `GET /api/admin/*`:

- **Overview** — headline counters for both live apps:
  - Optimization: total jobs, jobs-by-status, last 24 h, top LLM
    providers, jobs in flight, pending cloud jobs.
  - QML: training jobs, QPU runs across both providers, archived
    records, QML in flight.
- **Users** — every registered user, role, last-login, BYOK
  providers on file (names only — never ciphertext), total job count.
- **Jobs** — paginated cross-user optimization job list with status
  filter.
- **QML** — paginated cross-user QML training jobs (filter by status)
  + real-QPU runs table (filter by provider IBM/Origin). Rows are
  clickable into the user's job detail page.

---

## Optional dependency groups (`pip install ".[<name>]"`)

| Group | Pulls in | Unlocks |
|-------|----------|---------|
| `[dev]` | pytest, ruff, fakeredis, pytest-httpx | Tests + lint |
| `[queue]` | rq, redis | Async-queue launcher (both apps) |
| `[quantum]` | pyqpanda3, pyqpanda_alg, sympy | `qaoa_sim`, `qaoa_originqc`, `vqc_originqc` |
| `[ibm-quantum]` | qiskit, qiskit-ibm-runtime, scipy | `qaoa_ibmq`, `vqc_ibmq` |
| `[qml]` | pennylane, scikit-learn, matplotlib | All QML training paths |
| `[qldpc]` | (empty Sprint 0; matrix/stim/qiskit-qec deps in later sprints) | qLDPC matrix gen + benchmarks + hardware |

---

## Phase status reference

### Optimization

| Phase | Ships |
|-------|-------|
| 0 | Flask app factory + SQLite + auth + frontend shell |
| 1 | GPU simulated-annealing sampler |
| 2 | Validation harness + Benchmarks foundation |
| 3 | Formulation provider layer + BYOK encryption |
| 4 | Solve API endpoint (5-stage pipeline) |
| 5 | Frontend Solve UI (Vue 3 + Vuetify, live SSE) |
| 5B | Problem-template / Modules library |
| 5C | Public Benchmarks dashboard |
| 5D | Multi-solver fan-out + comparison |
| 6 | Async job queue (Redis-RQ, opt-in) |
| 7 | Admin read-only views (Users / Jobs / Overview / QML) |
| 8 | Classical solver tiers (CP-SAT, HiGHS) |
| 9A | Local OriginQC pyqpanda QAOA |
| 9B | Origin Quantum cloud (Wukong) |
| 9C | Parallel Tempering + Simulated Bifurcation |
| 10 | Interactive QAOA explainer + Circuit Playground |
| 11 | IBM Quantum cloud QAOA (`qaoa_ibmq`) |

### QML

| Phase | Ships |
|-------|-------|
| QML-1 | Foundation — schema, dataset registry, blueprint shell, landing |
| QML-2 | Local VQC + live loss + confusion matrix + circuit explainer |
| QML-3 | Four classical baselines + side-by-side comparison |
| QML-4 | Live decision-boundary heatmap + `/qml/learn` primer + Bloch demo |
| QML-5 | IBM Quantum cloud (`vqc_ibmq`) |
| QML-6 | Origin Quantum cloud (`vqc_originqc`) — single-point inference |
| QML-7 | Public benchmark archive + BibTeX citation |
| QML-8 | Admin QML tab + Redis-RQ async-queue path |

### qLDPC

| Sprint | Ships |
|--------|-------|
| 0 | Code-family registry (4 families) + gallery + primer + matrix/Tanner endpoints |
| 1 (planned) | Matrix-generation deps + interactive constructors |
| 3 (planned) | `stim` Monte Carlo threshold benchmarks |
| 4 (planned) | `qiskit-qec` syndrome circuits on IBMQ hardware |

---

## Backend test surface

319+ pytest tests covering:

- All optimization solver adapters (with mocked cloud paths for IBM /
  Origin)
- QML data loader / VQC trainer / baselines / cloud inference (IBM +
  Origin mocked via `_FakeSamplerV2` and a synthetic `pyqpanda3`)
- QML benchmark archive (build / write / load / cite / admin
  gating / delete)
- QML async-queue launcher (threaded / fakeredis / fallback)
- Admin endpoints across all three tabs
- Auth, BYOK encryption, template gallery, benchmark routes

Frontend: Vite production build clean; ~990 KB JS gzipped to ~315 KB.

---

## Security model

- **BYOK encryption at rest** with Fernet under `KEY_ENCRYPTION_SECRET`.
- **Shared keyring across apps** — your `ibm_quantum` BYOK key works
  for both `qaoa_ibmq` and `vqc_ibmq`.
- **Stored keys never echoed** back over the API.
- **Cross-user job access returns 404, not 403** to avoid existence
  leaks.
- **Real-QPU paths gated** behind `ENABLE_ORIGIN_REAL_HARDWARE=1` /
  explicit IBM backend names to prevent accidental quota burns.
- **Archived TrainRecords strip** trained weights, raw scatter points,
  decision grids, and API tokens before write.
- **Admin endpoints role-gated** via `@admin_required`; non-admins get
  bounced from `/admin` by the router guard.
- **Default seeded admin** password must be changed on first boot.
