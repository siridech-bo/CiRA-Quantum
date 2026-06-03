# CiRA Quantum

An academic platform that hosts two coordinated quantum applications
under one account, one BYOK keyring, and one auth surface:

- **Optimization** — natural-language problem statements → LLM-driven
  CQM formulation → 10 solver tiers (classical SOTA, QUBO heuristic,
  quantum-inspired, simulator, real superconducting QPU) → solution.
- **QML** — variational quantum classifiers trained locally on a
  PennyLane statevector simulator, with classical baselines on the
  same train/test split and optional real-QPU evaluation on IBM
  Quantum or Origin Wukong.

The full specification lives at the repo root in
[`PROJECT_TEMPLATE v2.md`](./PROJECT_TEMPLATE%20v2.md). The v1
specification is preserved at [`PROJECT_TEMPLATE.md`](./PROJECT_TEMPLATE.md)
for audit. Every non-obvious choice made during construction is logged
in [`DECISIONS.md`](./DECISIONS.md) with a date and a one-line rationale.

## Status

### Optimization app

| Phase | Status | What it ships |
|-------|--------|---------------|
| 0  | ✅ shipped | Flask app factory + SQLite + auth + frontend shell |
| 1  | ✅ shipped | GPU simulated-annealing sampler (`app.optimization.gpu_sa`) |
| 2  | ✅ shipped | Validation harness + Benchmarks foundation (registry, run records, citation, suite runner) |
| 3  | ✅ shipped | Formulation provider layer (Claude / OpenAI / local LLM) + BYOK encryption |
| 4  | ✅ shipped | Solve API endpoint — synchronous pipeline wires Phase 0+1+2+3 end-to-end |
| 5  | ✅ shipped | Frontend Solve UI — Vue 3 + Vuetify; live SSE progress, results, history, BYOK manager |
| 5B | ✅ shipped | Problem-template / Modules library — 10 curated problems with documented optima |
| 5C | ✅ shipped | Public Benchmarks dashboard — `/benchmarks` (no auth); 4 views over the append-only archive |
| 5D | ✅ shipped | Multi-solver fan-out with side-by-side per-solver comparison |
| 6  | ✅ shipped | Async job queue (Redis-RQ, opt-in via `USE_REDIS_QUEUE=1`) |
| 7  | ✅ shipped | Admin read-only views — users / jobs / overview / QML |
| 8  | ✅ shipped | Classical solver tiers — OR-Tools CP-SAT + HiGHS as `dimod.Sampler` adapters |
| 9A | ✅ shipped | Quantum tier (local simulator) — OriginQC pyqpanda QAOA, optional dep, 20-qubit cap |
| 9B | ✅ shipped | Quantum tier (Origin Quantum cloud BYOK) — `full_amplitude` cloud simulator + `WK_C180` real Wukong QPU |
| 9C | ✅ shipped | Quantum-inspired classical tiers — Parallel Tempering + Simulated Bifurcation |
| 10 | ✅ shipped | Interactive QAOA explainer (`/learn/quantum`) + circuit playground |
| 11 | ✅ shipped | IBM Quantum cloud QAOA (`qaoa_ibmq`) via Qiskit Runtime SamplerV2 |

### QML app (sister application)

| Phase | Status | What it ships |
|-------|--------|---------------|
| QML-1 | ✅ shipped | Foundation — `qml_jobs` schema, dataset registry (6 datasets), `/api/qml/*` blueprint, landing page tile, `/qml` route |
| QML-2 | ✅ shipped | Local VQC training — PennyLane statevector simulator, live training loss + accuracy via SSE, confusion matrix, gate-level circuit explainer |
| QML-3 | ✅ shipped | Classical baselines — LogReg, SVM-RBF, RandomForest, MLP on the same train/test split; side-by-side comparison table with quantum-vs-classical verdict |
| QML-4 | ✅ shipped | Educational explainers — live decision-boundary heatmap during training, `/qml/learn` primer with interactive Bloch sphere demo |
| QML-5 | ✅ shipped | IBM Quantum cloud (`vqc_ibmq`) — Qiskit Runtime `SamplerV2` batch inference on the trained weights, queue/shot status polling, reuses `ibm_quantum` BYOK |
| QML-6 | ✅ shipped | Origin Quantum cloud (`vqc_originqc`) — pyqpanda3 single-point inference per submission, `full_amplitude` cloud simulator + `WK_C180` real Wukong QPU, reuses `originqc` BYOK |
| QML-7 | ✅ shipped | Public benchmark archive — `qml_benchmarks/archive/*.json` TrainRecord files, `/qml/benchmarks` leaderboard, BibTeX citation, admin-curated |
| QML-8 | ✅ shipped | Admin tab for QML jobs + QPU runs + archive count, Redis-RQ async-queue path for training (shared `USE_REDIS_QUEUE=1` flag) |

## Architecture at a glance

```
                           ┌─────────────────────────────┐
                           │  Frontend (Vue 3 + Vuetify) │
                           │  port 3070                  │
                           └──────────────┬──────────────┘
                                          │
                                          │ /api/*  (cookie session + CSRF)
                                          ▼
                           ┌─────────────────────────────┐
                           │  Flask app factory          │
                           │  port 5009                  │
                           │                             │
                           │  /api/solve  /api/qml/train │
                           │  /api/benchmarks  /api/qml/benchmarks
                           │  /api/admin                 │
                           └──┬───────────────────────┬──┘
                              │                       │
                  Optimization│                       │QML
                              ▼                       ▼
              ┌────────────────────────┐  ┌──────────────────────────┐
              │ Orchestrator           │  │ Trainer                  │
              │  formulate → compile → │  │  load dataset → VQC →    │
              │  validate → solve →    │  │  Adam → baselines →      │
              │  interpret             │  │  (optional) QPU inference│
              └───────────┬────────────┘  └──────────┬───────────────┘
                          │                          │
                          ▼                          ▼
                ┌──────────────────┐       ┌──────────────────────┐
                │ Solver registry  │       │ Dataset registry     │
                │  10 tiers,       │       │  6 datasets, VQC +   │
                │  classical→QPU   │       │  4 sklearn baselines │
                └──────────────────┘       └──────────────────────┘
```

Both apps share: the Flask factory, SQLite store, BYOK keyring,
auth/session layer, SSE event bus, optional Redis-RQ launcher
(`USE_REDIS_QUEUE=1`), and admin views. Each app keeps its own
job table (`jobs` vs `qml_jobs`), its own routes prefix
(`/api/...` vs `/api/qml/...`), and its own frontend area (`/solve`,
`/templates`, `/benchmarks` vs `/qml`, `/qml/learn`, `/qml/benchmarks`).

## Getting started

The full setup lives in [`backend/README.md`](./backend/README.md).
Short form:

```bash
# Backend
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1            # Windows PowerShell
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.10.0
pip install -r requirements.txt
pip install ".[qml]"                     # PennyLane + scikit-learn for QML
# Optional extras: ".[ibm-quantum]" for vqc_ibmq, ".[quantum]" for Origin
python run.py                            # port 5009

# Frontend (separate terminal)
cd frontend
npm install                              # first time only
npm run dev                              # port 3070
```

Open <http://localhost:3070>. The default admin (`admin` / `admin123`)
is seeded on first backend boot — **change it immediately**.

### Picking an app

- The landing page (`/`) shows both apps side-by-side. Pick **Optimization**
  or **QML** based on what you want to do.
- **First-time QML users** should start at `/qml/learn` — a 5-step primer
  (qubit → encoding → entangler → measurement → learning) with an
  interactive Bloch sphere demo, ~5 min read.

### Optional dependency groups

| Extra | Brings in | Used by |
|-------|-----------|---------|
| `[quantum]` | pyqpanda3, pyqpanda_alg, sympy | Local OriginQC QAOA, Origin Wukong cloud, `vqc_originqc` |
| `[ibm-quantum]` | qiskit, qiskit-ibm-runtime, scipy | `qaoa_ibmq`, `vqc_ibmq` |
| `[qml]` | pennylane, scikit-learn, matplotlib | All QML training paths |
| `[queue]` | rq, redis | Redis-RQ async-queue launcher (both apps) |
| `[dev]` | pytest, ruff, fakeredis, pytest-httpx | Test + lint |

## Requirements

- Python 3.12+
- NVIDIA GPU with CUDA 12.8 (Blackwell sm_120 verified on RTX 5070 Ti)
  — required by the optimization-side GPU SA tier; QML training is
  CPU-only.
- PyTorch 2.10.0 (cu128 wheel)
- Node 18+ for the frontend

## Tests

```bash
cd backend
pytest tests/ -v
ruff check .
```

GPU SA tests require a working CUDA device. Formulation provider tests
mock HTTP via `pytest-httpx`. QML cloud tests use mocked
`qiskit-ibm-runtime` / `pyqpanda3` modules so they never touch the
real cloud. The manual integration test (real LLM calls, real API
tokens) is intentionally not invoked by CI; see
[`backend/tests/manual_integration_test.py`](./backend/tests/manual_integration_test.py)
for the command line.

## Security

- BYOK (bring-your-own-key). Cloud-LLM and cloud-QPU keys are encrypted
  at rest with Fernet under `KEY_ENCRYPTION_SECRET`. The optimization
  side and the QML side share the same keyring — your `ibm_quantum` key
  works for both `qaoa_ibmq` and `vqc_ibmq`; your `originqc` key for
  both `qaoa_originqc` and `vqc_originqc`.
- Stored API keys are never echoed back over the API. Archived
  TrainRecords strip trained weights + raw scatter points + API tokens
  before write.
- Cross-user job access returns `404`, not `403`, to avoid existence
  leaks.
- Real-QPU paths (Wukong, IBM QPU backends) are gated behind
  `ENABLE_ORIGIN_REAL_HARDWARE=1` / explicit IBM backend names
  respectively, so accidental quota burns are unlikely.
- The seeded admin password must be changed on first boot — the
  default is for local development only.

## License

MIT — see [`LICENSE`](./LICENSE).
