# CiRA Quantum

An academic platform for **quantum-inspired optimization**: take a
natural-language problem statement, formulate it into a Constrained
Quadratic Model with an LLM (Claude / OpenAI / local Ollama), validate
the formulation, lower it to a BQM/Ising and solve it on a GPU with
simulated annealing, and present the interpreted solution end-to-end.

The full specification lives at the repo root in
[`PROJECT_TEMPLATE v2.md`](./PROJECT_TEMPLATE%20v2.md). The v1
specification is preserved at [`PROJECT_TEMPLATE.md`](./PROJECT_TEMPLATE.md)
for audit. Every non-obvious choice made during construction is logged
in [`DECISIONS.md`](./DECISIONS.md) with a date and a one-line rationale.

## Status

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
| 8  | ✅ shipped | Classical solver tiers — OR-Tools CP-SAT + HiGHS as `dimod.Sampler` adapters |
| 9A | ✅ shipped | Quantum tier (local simulator) — OriginQC pyqpanda QAOA, optional dep, 20-qubit cap |
| 9B | ⏳ next   | Quantum tier (Origin Quantum cloud BYOK + real Wukong QPU) |

## Architecture at a glance

```
┌──────────────────────────┐   POST /api/solve     ┌────────────────────────┐
│  Frontend (Vue 3 + Vita) │ ────────────────────▶ │  Flask app factory     │
│  port 3070               │ ◀──── SSE stream ──── │  port 5009             │
└──────────────────────────┘                       └─────────┬──────────────┘
                                                             │
                                                             ▼
                                    ┌─────────────────────────────────────────────┐
                                    │ Orchestrator (5-stage async pipeline)       │
                                    │                                             │
                                    │ formulate → compile → validate → solve → interpret
                                    │   │           │          │         │       │
                                    │   ▼           ▼          ▼         ▼       ▼
                                    │  LLM      dimod      Layer A/C   GPU SA   user-units
                                    │ provider   CQM       harness    sampler   output
                                    └─────────────────────────────────────────────┘
```

Provider layer (Phase 3), GPU SA sampler (Phase 1), and validation
harness (Phase 2) are independent modules — the orchestrator (Phase 4)
is the glue. Each phase ships with its own tests and is exercised
independently.

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
python run.py                            # port 5009

# Frontend (separate terminal)
cd frontend
npm install                              # first time only
npm run dev                              # port 3070
```

Open <http://localhost:3070>. The default admin (`admin` / `admin123`)
is seeded on first backend boot — **change it immediately**.

## Requirements

- Python 3.12+
- NVIDIA GPU with CUDA 12.8 (Blackwell sm_120 verified on RTX 5070 Ti)
- PyTorch 2.10.0 (cu128 wheel)
- Node 18+ for the frontend

## Tests

```bash
cd backend
pytest tests/ -v
ruff check .
```

GPU SA tests require a working CUDA device. Formulation provider tests
mock HTTP via `pytest-httpx`. The manual integration test (real LLM
calls, real API tokens) is intentionally not invoked by CI; see
[`backend/tests/manual_integration_test.py`](./backend/tests/manual_integration_test.py)
for the command line.

## Security

- BYOK (bring-your-own-key). Cloud-LLM keys are encrypted at rest with
  Fernet under `KEY_ENCRYPTION_SECRET`.
- Stored API keys are never echoed back over the API.
- Cross-user job access returns `404`, not `403`, to avoid existence
  leaks.
- The seeded admin password must be changed on first boot — the
  default is for local development only.

## License

MIT — see [`LICENSE`](./LICENSE).
