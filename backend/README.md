# CiRA Quantum — Backend

This is the backend for CiRA Quantum. It is being built one phase at a time
per `PROJECT_TEMPLATE v2.md` at the repository root. (The v1 specification
is preserved alongside as `PROJECT_TEMPLATE.md` for audit trail.)

**Phases shipped:**

| Phase | Status | What it ships |
|-------|--------|---------------|
| 0 | ✅ shipped | Flask app factory + SQLite + auth + frontend shell (built late, see DECISIONS.md) |
| 1 | ✅ shipped | GPU simulated-annealing sampler (`app.optimization.gpu_sa`) |
| 2 | ✅ shipped | Validation harness + Benchmarks foundation (registry, run records, citation, suite runner) |
| 3 | ✅ shipped | Formulation provider layer (Claude / OpenAI / local LLM) + BYOK encryption |
| 4 | ✅ shipped | Solve API endpoint — synchronous pipeline wires Phase 0+1+2+3 end-to-end (`/api/solve`, `/api/jobs`, `/api/jobs/<id>/stream`, `/api/keys`) |
| 5 | ✅ shipped | Frontend Solve UI — Vue 3 + Vuetify; live SSE progress, results, history, BYOK manager |
| 5B | ✅ shipped | Problem-template / Modules library — 10 curated problems with documented optima, match badge, one-click "Try this example" launcher |
| 5C | ✅ shipped | Public Benchmarks dashboard — `/benchmarks` (no auth); 4 views (suite grid, solver profile, instance leaderboard, dashboard) reading from `benchmarks/archive/` |
| 8 | ✅ shipped | Classical solver tiers — OR-Tools CP-SAT + HiGHS adapters as `dimod.Sampler` (CQM-native via `_CQM_NATIVE` marker); classical-beats-QUBO honesty check passes |
| 9A | ✅ shipped | Quantum tier (local simulator) — OriginQC pyqpanda QAOA wrapped as a `dimod.Sampler`; optional dep (`pip install ".[quantum]"`); 20-qubit cap on CPU sim path |
| 9B | ✅ shipped | Quantum tier (Origin Quantum cloud BYOK) — `qaoa_originqc` registered when `QPANDA_API_KEY_FILE` env var points at a credential file; hybrid local-train + cloud-execute; real-QPU backends gated behind `ENABLE_ORIGIN_REAL_HARDWARE=1`. Auto-caps at n=7 qubits on real-QPU backends per the empirical transpilation wall (see `BENCHMARK_REPORT_002.md` Phase 9B follow-up section). |
| 9B+ | ✅ shipped | Non-blocking cloud-job capture flow — pending-jobs panel auto-polls and materializes results when Wukong returns, no long-running Python process required |
| 9C | ✅ shipped | Quantum-inspired classical tiers — Parallel Tempering (custom Hukushima-Nemoto) + Simulated Bifurcation (open algorithm behind Fujitsu's SQBM+). See [BENCHMARK_REPORT_002.md](../BENCHMARK_REPORT_002.md) for findings. |
| 6 | ⏳ next | Async job queue + GPU contention |

The Flask web layer is live: `python run.py` brings up the auth shell
on port 5009; `npm run dev` in `../frontend/` brings up the Vue login
flow on port 3070. The solve route, pipeline orchestrator, and SSE
event stream arrive in Phase 4.

## Requirements

- Python 3.12+
- NVIDIA GPU with CUDA 12.8 (Blackwell sm_120 verified on RTX 5070 Ti)
- PyTorch 2.10.0 built against cu128

## Install

PyTorch's cu128 wheel is not on PyPI's default index, so it is installed
separately from the rest of the dependencies.

```bash
# 1. Create a virtual environment
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# Linux / macOS
source .venv/bin/activate

# 2. Install PyTorch from the cu128 index
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.10.0

# 3. Install everything else
pip install -r requirements.txt
```

`uv` is also supported if you prefer it:

```bash
uv venv
uv pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.10.0
uv pip install -e ".[dev]"
```

`triton` is platform-conditional in `requirements.txt` and `pyproject.toml`:
the upstream `triton` package on Linux, the community-maintained
`triton-windows` build on Windows. Both resolve cleanly in a fresh venv.

## Verify GPU is visible

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Run the tests

```bash
pytest tests/ -v
```

GPU SA tests require a working CUDA device. The Phase 2 & Phase 3 tests
do not — formulation provider tests mock HTTP via `pytest-httpx`.

## GPU SA — single sample (Phase 1)

```bash
python -m app.optimization.gpu_sa --bqm tests/instances/tiny_5var.json
python tests/benchmark_gpu_vs_cpu.py        # GPU vs CPU SA, dense Ising
```

## Benchmark suite runner (Phase 2)

The suite runner executes a registered solver over every instance in a
registered suite and writes one `RunRecord` per instance to
`benchmarks/archive/`.

```bash
python -m app.benchmarking.run_suite \
    --solver gpu_sa --instances knapsack/small \
    --num-reads 1000 --num-sweeps 1000

# Cite an archived record:
python -m app.benchmarking.cite <record_id>            # BibTeX
python -m app.benchmarking.cite <record_id> --string   # short citation
```

Registered solvers: `exact_cqm`, `cpu_sa_neal`, `gpu_sa` (CUDA-conditional),
`cpsat` (OR-Tools), `highs` (HiGHS MIP), `qaoa_sim` (pyqpanda QAOA on CPU
statevector, optional `[quantum]` extra), `qaoa_originqc` (env-gated by
`QPANDA_API_KEY_FILE`), `parallel_tempering` (custom Hukushima-Nemoto PT),
`simulated_bifurcation` (Goto et al. 2019 via the `simulated-bifurcation`
PyPI package).
Registered suites: `knapsack/small`, `setcover/small`, `jss/small`,
`maxcut/gset_subset`, `graph_coloring/small`. Run records carry an
honest convergence flag — `converged_to_expected: null` when the instance
has no ground truth, `true`/`false` when it does.

## Formulation providers (Phase 3)

Three providers convert a natural-language problem statement into
validated `cqm_v1` JSON: `claude`, `openai`, `local`. Each is a
separate module that registers itself on import; the public API is
`get_provider("claude")` (or `"openai"`, `"local"`).

**BYOK.** All cloud providers require user-supplied API keys
(`api_key=...` per call). Stored keys are encrypted at rest with
Fernet under `KEY_ENCRYPTION_SECRET` — see DECISIONS.md for the
threat model and rotation behavior.

**Default models** (overridable via the provider constructor):

- Claude → `claude-sonnet-4-6` (Opus is one constructor arg away)
- OpenAI → `gpt-5-mini` (full GPT-5 is one constructor arg away)
- Local → `qwen3:8b` against an Ollama endpoint at `LOCAL_LLM_ENDPOINT`
  (default `http://localhost:11434`)

The defaults are picked for cost-conservatism, not headline quality.
The Phase-5C public Benchmark dashboard is where the platform will
honestly compare them; until then, see `DECISIONS.md` for full
rationale (Sonnet vs Opus, GPT-5-mini vs GPT-5, 8B vs 70B).

### Local LLM — known limitations

**The local provider is best-effort, not production-grade.** Model
choice within the local tier matters a lot — measurably more than at
the cloud tier — so the default is set based on empirical evidence
from this codebase, not name recognition.

The defaults are **`qwen3:8b`** for interactive use and **`qwen3:14b`**
for the gate / quality-sensitive runs. Switch between them at run
time with the `LOCAL_LLM_MODEL` environment variable:

```bash
# Default — fastest local option, fine for beginner-tier Modules.
python -m app.benchmarking.run_suite ...

# More reliable on JSS-class problems, ~7× slower per call.
LOCAL_LLM_MODEL=qwen3:14b python -m app.benchmarking.run_suite ...
```

**Empirical numbers from this codebase** (16 GB RTX 5070 Ti, JSS-3-job-2-machine
prompt, disjunctive big-M encoding, multiple trials):

| Model | Latency | JSS-3×2 outcome | Notes |
|-------|---------|-----------------|-------|
| `qwen3:8b` (default) | ~90 s | 1/2 trials correct (10), 1/2 dropped a disjunctive helper | Best UX for the common case |
| `qwen3:14b` (env) | ~470–520 s | 2/2 trials correct (10), schema-valid, compiles | Best reliability inside 16 GB |
| `llama3.1:8b` | timed out at 300 s | — | Same parameter class as `qwen3:8b`, can't keep up here |
| `phi4:14b` | ~320–490 s | 0/2 — compiles but wrong optimum (11 ≠ 10) | A worse failure mode: looks fine, isn't |
| `qwen3-coder:30b`, `gpt-oss:20b`, `devstral:latest` | — | HTTP 400 on `/api/chat` | Need a `/api/generate` adapter (Phase-10 candidate) |
| `deepseek-coder:6.7b` | ~125 s | under-specifies (4 constraints, opt=8 ≠ 10) | Fast but wrong |

What that translates to in tier terms:

- ✅ **`qwen3:8b` (default)** works on every Phase-2 instance class we
  tested in the easy tier: knapsack, set-cover, max-cut, graph-coloring
  (small). On JSS-3×2 specifically, it succeeds about half the time.
- ✅ **`qwen3:14b` (env override)** consistently lands the correct
  optimum on JSS-3×2 in our trials. Use it when correctness matters
  more than latency.
- ⚠️ **JSS at 3×3 and beyond** is still risky territory even at 14B —
  the disjunctive big-M constraint count grows quadratically, and the
  model starts dropping or mistyping variable references.
- ❌ **Don't expect** any local 8B/14B model to produce a correct
  formulation for full Phase-2 fixtures like `jss_5job_5machine`
  (76 vars, 125 constraints) — the prompt + emitted JSON together
  push past the model's effective working memory.

If your formulation has to be correct, route through Claude or OpenAI.
If you're learning, on a budget, or running offline, the local
provider with `qwen3:8b` is good enough for the introductory tier of
the Modules library — and `LOCAL_LLM_MODEL=qwen3:14b` is one env
variable away when you need it.

This is the cost the platform pays for the BYOK trade. A larger local
model (33B or 70B) would close most of the gap, but doesn't fit into
16 GB VRAM alongside the GPU SA solver. A future phase may add a
"large local LLM" tier for workstations with more VRAM — explicitly
opt-in, not the default.

### Run the manual integration test (gate before Phase 4)

```bash
ANTHROPIC_API_KEY=sk-ant-... \
OPENAI_API_KEY=sk-... \
LOCAL_LLM_ENDPOINT=http://localhost:11434 \
LOCAL_LLM_MODEL=qwen3:14b \
python tests/manual_integration_test.py
```

Exit code 0 when ≥ 2 of 3 providers produce a CQM that compiles and
passes the validation harness. This script spends real API tokens —
not invoked from CI. The per-provider timeout is 10 minutes; the cloud
arms typically return in well under 30 s, the local arm at
`qwen3:14b` takes ~8 minutes.

## Run the dev servers (Phase 0)

```bash
# Backend (Flask, port 5009)
cd backend
python run.py

# Frontend (Vite + Vue, port 3070) — separate terminal
cd ../frontend
npm install              # first time only
npm run dev
```

Open <http://localhost:3070> in a browser: an unauthenticated visit
redirects to `/login`. The default admin (`admin` / `admin123`) is
seeded on the first backend boot; change it immediately. Signup,
login, logout, and the placeholder MainApp shell all work end-to-end
through the Vite `/api` proxy.

## Lint

```bash
ruff check .
```
