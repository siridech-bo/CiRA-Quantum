# CiRA Quantum — Project Template & Specification

**Spec version: v2.0** (revised from v1.0 after strategic re-positioning)
**Last revised:** May 2026, between Phase 1 delivery and Phase 2 kickoff

CiRA Quantum is an academic platform for learning and benchmarking quantum and quantum-inspired optimization. Users describe combinatorial optimization problems in natural language; the system formulates them as CQM/QUBO models, runs them across multiple solver tiers (classical, quantum-inspired, quantum), and produces honest, reproducible, citable performance comparisons. Three audiences are served — students, researchers, and practitioners — through three modes: **Learn**, **Benchmark**, and **Solve**.

This document is both the **architecture template** (following the CiRA Oculus company conventions) and the **build specification with phased deliverables** that an AI coder works from, one phase at a time.

---

## Strategic Vision (v2)

### What changed between v1 and v2

The v1 spec positioned CiRA Quantum as a "hybrid quantum-inspired optimization web app" — a tool for solving QUBO problems on GPU and SQBM+. After the strategic review conducted between Phase 1 and Phase 2, the project's positioning has been deliberately re-scoped.

The v1 framing had two structural weaknesses: it overlapped with commercial offerings (SQBM+, VeloxQ, Hexaly) without a defensible differentiator, and it required quantum advantage on industrial problems to deliver value — a bet the empirical evidence in 2026 does not support. The v1 architecture was technically sound; the positioning was strategically fragile.

The v2 framing repositions the project as an **academic benchmarking and curriculum platform** for quantum optimization. This change leverages the project's actual asset — being led by a university faculty member with access to students, academic networks, and grant funding — instead of competing in commercial markets where bigger players have structural advantages.

The technical architecture is mostly preserved. The validation harness, formulation pipeline, GPU SA solver, and `dimod.Sampler`-based dispatch all remain. What changes is what the platform optimizes for, the order of solver-tier additions, and the user-facing surfaces.

### Three audiences

| Audience | Primary mode | What they need |
|---|---|---|
| **Students** | Learn | Structured Modules, hands-on exercises, automatic feedback, clear explanations of why solvers behave as they do |
| **Researchers** | Benchmark | Reproducible comparisons, citable results, longitudinal performance tracking, ability to add new solvers and instances |
| **Practitioners** | Solve | The original v1 use case — describe a problem, get a solution with honest comparison across paradigms |

The three audiences share infrastructure (formulation, validation, solving) but differ in how the platform presents itself. The same codebase serves all three through deliberately different surfaces.

### Three modes

**Learn mode** — Curriculum-led. Structured Modules cover QUBO formulation, classical solvers, quantum annealing concepts, gate-based algorithms, and quantum machine learning (in approximate progression). Each module includes readings, hands-on exercises that run in the platform's solver infrastructure, and assessment via the validation harness. Cohort management for course use.

**Benchmark mode** — Comparison-led. Public dashboard showing every solver tier in the platform run against canonical instance suites (Gset for Max-Cut, Taillard for JSS, OR-Library for Knapsack, etc.). Results are reproducible, versioned, and citable. Researchers contribute new instances, new solvers, or new methodology improvements via a contribution pipeline.

**Solve mode** — Practitioner-led. The v1 vision: bring a real problem in plain language, get it formulated, solved, and explained, with honest comparison across all available solver tiers. This is now positioned as the "applied" face of the educational platform rather than the platform's core identity.

### Why this strategy is defensible

Three asymmetric advantages compound under this positioning:

**Academic credibility cannot be bought.** Commercial quantum-software companies (Strangeworks, Classiq, vendor-aligned platforms) have inherent conflicts of interest when reporting comparisons. A university-led platform that publishes peer-reviewable Benchmarks has authority no commercial vendor can match.

**Student labor is structurally aligned.** Graduate courses, senior projects, and master's theses can produce platform contributions as legitimate educational deliverables. Each cohort adds durable value. Commercial platforms must pay engineers; academic platforms grow through teaching.

**Time horizon matches the technology.** Quantum advantage on industrial optimization, if it materializes, will likely emerge over 5–10 years. Academic platforms naturally operate on this timescale; commercial platforms must show ROI faster. Patience becomes a competitive advantage.

### What v2 explicitly does NOT pretend

To preserve credibility, the platform must be honest about what it is and isn't:

- It does not claim quantum advantage on industrial optimization problems exists today
- It does not promise commercial-grade performance — the GPU SA tier is excellent for learning and benchmarking, not for production scheduling at scale
- It does not compete with Gurobi, CPLEX, OR-Tools, or other production-grade classical solvers; it integrates them as Benchmark baselines
- It does not duplicate Qiskit, Cirq, PennyLane, or other quantum SDKs; it integrates them as solver tiers when gate-based algorithms are added
- It does not market itself as a startup or a commercial product; it is, and is presented as, an academic research platform

This honesty is the platform's differentiator. Most quantum-software offerings overpromise; CiRA Quantum's pitch is precisely the opposite — *here is what we actually know, here is how to compare it, here is how to learn from it.*

### What changes in the technical phases

The phase numbers from v1 are preserved (Phase 0 through Phase 8) for continuity with shipped work. Three new phases are added (Phase 5C, Phase 8 expansion, Phase 9–11) and several existing phases get revision notes. Phase 0 (skeleton/auth) and Phase 1 (GPU SA library) shipped under v1 and remain valid under v2 with no rework needed. Phase 2 onward is revised; Phase 2 in particular gets a substantial scope expansion because the validation harness is now the foundation of the Benchmark mode and must be built with reproducibility, citation, and longevity from the start.

The full revised phase table appears in the [Phased Deliverables Specification](#phased-deliverables-specification) section. Where a phase has v2 revision notes, they appear inline with the original v1 content preserved for audit trail.

### Academic-track scope (roadmap, not committed)

Several capabilities make sense only when specific academic conditions are met (a research grant funded, a course cohort confirmed, a publication target identified). These are listed in [Appendix A: Academic Roadmap](#appendix-a-academic-roadmap) as a roadmap, not committed phases. Examples include FERPA-compliant student data handling, multi-tenant cohort deployment, and CiRA Quantum Reports as a publication track. They become real phases when the underlying need is real.

---

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Tech Stack Overview](#tech-stack-overview)
3. [Project Structure](#project-structure)
4. [Frontend Architecture](#frontend-architecture)
5. [Backend Architecture](#backend-architecture)
6. [Authentication System](#authentication-system)
7. [Multi-User & Job Ownership](#multi-user--job-ownership)
8. [Optimization Pipeline](#optimization-pipeline)
9. [Configuration](#configuration)
10. [API Endpoints Reference](#api-endpoints-reference)
11. [Phased Deliverables Specification](#phased-deliverables-specification)
12. [Getting Started](#getting-started)
13. [Typography & Fonts](#typography--fonts)
14. [AI Coder Workflow Notes](#ai-coder-workflow-notes)

---

## Project Overview

### Purpose

CiRA Quantum is an academic platform for learning, benchmarking, and experimenting with quantum and quantum-inspired optimization. Three audiences — students, researchers, and practitioners — interact with the same infrastructure through three deliberately different modes (Learn, Benchmark, Solve).

The platform's job, in priority order:

1. **Teach** combinatorial optimization formulation honestly, including when classical methods are the right answer
2. **Benchmark** solver performance reproducibly across paradigms, with citation-grade output
3. **Formulate** prose problems into validated CQM/QUBO models via LLM providers
4. **Solve** problems across multiple solver tiers — classical, quantum-inspired, and (eventually) quantum
5. **Interpret** raw solver output back into the user's domain terms
6. **Track** the evolution of the quantum optimization landscape over time

### Target Users

| User Type | Primary Mode | Use Case |
|-----------|--------------|----------|
| **Undergraduate students** | Learn | First exposure to QUBO formulation through guided Modules |
| **Graduate students** | Learn + Benchmark | Hands-on coursework, thesis Benchmarks, contribution to platform |
| **Researchers** | Benchmark | Reproducible cross-solver comparisons for publications |
| **Educators** | Learn (cohort admin) | Course delivery using the platform's Modules |
| **Practitioners** | Solve | Real optimization problems with honest paradigm comparison |
| **Internal (team)** | All three | Platform development, curriculum design, benchmark curation |

### Scope — Public Web App, V2

- Anyone can sign up
- BYOK (Bring Your Own Key) for LLM providers — users supply their own Claude/OpenAI keys
- Local GPU SA on the team's RTX 5070 Ti as the foundational solver tier
- Classical solver tiers (CPU SA, ExactCQMSolver) registered for benchmarking from Phase 2 onward
- Job history per user; admin sees all jobs
- **Problem template library** — curated examples organized as a curriculum seed
- **Public benchmark dashboard** — every registered solver run on canonical instance suites
- Quantum-inspired and quantum solver tiers added incrementally as solver adapters

### Non-Goals (V2)

- Direct D-Wave QPU access in early phases (deferred to Phase 12+, opt-in BYOK)
- Hosted LLM keys (cost exposure too high for a free public service)
- Multi-GPU clusters
- Custom solver tuning UI for non-experts (hidden behind advanced flags)
- Production-grade SLAs — this is a research platform, not a commercial service
- Competing with Gurobi/CPLEX/OR-Tools as a primary solver — they are integrated as Benchmark baselines, not as the platform's main offering

### v2 revision notes

The v1 scope listed the template library as "curated example problems users can browse and run with one click." Under v2, the template library is reframed as **the curriculum seed for Learn mode** — same JSON files, but with explicit learning objectives, prerequisite chains, and assessment hooks. See revised Phase 5B.

The v1 scope was silent on benchmarking. Under v2, the public benchmark dashboard is added as Phase 5C, with reproducibility infrastructure built into Phase 2 to support it.

---

## Tech Stack Overview

### Frontend

| Technology | Version | Purpose |
|------------|---------|---------|
| **Vue.js 3** | ^3.4.0 | Reactive UI framework (Composition API) |
| **Vuetify 3** | ^3.5.0 | Material Design component library |
| **Vue Router** | ^4.3.0 | Client-side routing with navigation guards |
| **Pinia** | ^2.1.7 | State management |
| **Axios** | ^1.6.0 | HTTP client for API requests |
| **Vite** | ^5.0.0 | Build tool and dev server |
| **TypeScript** | ^5.3.0 | Type-safe JavaScript |
| **@mdi/font** | ^7.4.0 | Material Design Icons |
| **EventSource (native)** | — | Server-Sent Events for live solve status |

### Backend

| Technology | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.12+ | Programming language |
| **Flask** | Latest | Web framework |
| **Flask-CORS** | Latest | CORS handling |
| **SQLite** | Built-in | Database for users and job history |
| **Werkzeug** | Latest | Password hashing & security |
| **cryptography** | Latest | Encrypting BYOK API keys at rest |

### Optimization Stack

| Technology | Version | Purpose |
|------------|---------|---------|
| **PyTorch** | 2.10.0+cu128 | GPU simulated annealing on Blackwell (sm_120) |
| **dimod** | Latest | BQM/CQM data structures, ExactCQMSolver oracle |
| **dwave-samplers** | Latest | Reference CPU SimulatedAnnealingSampler |
| **anthropic** | Latest | Claude formulation provider |
| **openai** | Latest | OpenAI formulation provider |
| **httpx** | Latest | Local LLM provider (Ollama/vLLM endpoints) |
| **jsonschema** | Latest | Validate LLM-emitted CQM JSON |

### Target Hardware

- **Solver host:** Workstation with NVIDIA RTX 5070 Ti (16 GB VRAM, sm_120) running Ubuntu 22.04 or Windows 11
- **Frontend/auth host:** Same machine in V1; can be split later

---

## Project Structure

```
project-root/
├── frontend/                            # Vue.js frontend application
│   ├── src/
│   │   ├── components/
│   │   │   ├── ProblemInput.vue        # Problem statement textarea + provider picker
│   │   │   ├── SolveStatus.vue         # SSE-driven live status display
│   │   │   ├── ResultDisplay.vue       # Solution viewer + CQM inspector
│   │   │   ├── JobHistory.vue          # User's past jobs
│   │   │   ├── ApiKeyManager.vue       # BYOK key entry & storage
│   │   │   ├── PhaseStatusBadge.vue    # Visual indicator per pipeline stage
│   │   │   ├── TemplateCard.vue        # One template tile in the gallery
│   │   │   ├── TemplateDetailModal.vue # Full template view with 'Try this' button
│   │   │   └── TemplateMatchBadge.vue  # ✓/✗ vs expected optimum
│   │   ├── views/
│   │   │   ├── LoginPage.vue
│   │   │   ├── SignupPage.vue
│   │   │   ├── MainApp.vue             # Solve interface
│   │   │   ├── JobDetailPage.vue       # Single job, full detail
│   │   │   ├── TemplateGalleryPage.vue # Browse example problems
│   │   │   └── AdminPage.vue           # User & job management
│   │   ├── stores/
│   │   │   ├── auth.ts
│   │   │   ├── solve.ts                # Solve flow state
│   │   │   └── templates.ts            # Template gallery state
│   │   ├── router/
│   │   │   └── index.ts
│   │   ├── App.vue
│   │   └── main.ts
│   ├── package.json
│   ├── vite.config.ts
│   └── tsconfig.json
│
├── backend/                             # Flask backend application
│   ├── app/
│   │   ├── __init__.py                 # Flask app factory
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── auth.py                 # /api/auth/*
│   │   │   ├── solve.py                # /api/solve, /api/jobs/*
│   │   │   ├── templates.py            # /api/templates/*
│   │   │   ├── admin.py                # /api/admin/*
│   │   │   └── health.py               # /api/health
│   │   ├── models.py                   # User, Job, ApiKey tables
│   │   ├── auth.py                     # @login_required, @admin_required
│   │   ├── config.py
│   │   ├── crypto.py                   # API key encryption
│   │   ├── formulation/
│   │   │   ├── __init__.py
│   │   │   ├── base.py                 # FormulationProvider ABC
│   │   │   ├── claude.py
│   │   │   ├── openai.py
│   │   │   ├── local.py                # Ollama / vLLM / llama.cpp
│   │   │   ├── prompts/
│   │   │   │   ├── system.txt
│   │   │   │   └── examples.json
│   │   │   └── schemas/
│   │   │       └── cqm_v1.json
│   │   ├── optimization/
│   │   │   ├── __init__.py
│   │   │   ├── gpu_sa.py               # GPUSimulatedAnnealingSampler
│   │   │   ├── validation.py           # Oracle agreement, constraint checks
│   │   │   ├── compiler.py             # CQM JSON → dimod.CQM
│   │   │   ├── interpreter.py          # SampleSet → human-readable
│   │   │   └── tiers.py                # Solver tier dispatch
│   │   ├── templates/
│   │   │   ├── __init__.py
│   │   │   ├── registry.py             # loads & validates template JSONs
│   │   │   ├── library/                # curated example problems (10+ JSON files)
│   │   │   └── schemas/
│   │   │       └── template_v1.json
│   │   └── pipeline/
│   │       ├── __init__.py
│   │       ├── orchestrator.py         # End-to-end solve pipeline
│   │       └── events.py               # SSE event emitter
│   ├── tests/
│   │   ├── conftest.py                 # Fixtures: tiny instances, oracles
│   │   ├── instances/                  # Canonical test problems (JSON)
│   │   ├── test_gpu_sa.py
│   │   ├── test_validation.py
│   │   ├── test_formulation_*.py
│   │   ├── test_pipeline.py
│   │   ├── test_routes_solve.py
│   │   └── test_oracle_match.py        # Layer 4: gold standard
│   ├── data/                           # SQLite database
│   ├── run.py                          # Application entry point
│   └── requirements.txt
│
├── docs/
│   ├── PROJECT_TEMPLATE.md             # This document
│   ├── FORMULATION_PROMPT_GUIDE.md     # Prompt engineering notes
│   └── DEPLOYMENT.md                   # Production deployment runbook
│
└── README.md
```

---

## Frontend Architecture

The frontend follows the same patterns as CiRA Oculus. The router and auth store are reused with minor adjustments. The main differences are the addition of `solve.ts` Pinia store and SSE-driven status components.

### Main Entry Point (`main.ts`)

```typescript
import { createApp } from 'vue'
import { createPinia } from 'pinia'
import App from './App.vue'
import router from './router'

import 'vuetify/styles'
import { createVuetify } from 'vuetify'
import * as components from 'vuetify/components'
import * as directives from 'vuetify/directives'
import '@mdi/font/css/materialdesignicons.css'

const vuetify = createVuetify({
  components,
  directives,
  theme: {
    defaultTheme: 'dark',
    themes: {
      dark: {
        dark: true,
        colors: {
          primary: '#7C4DFF',      // Quantum purple
          secondary: '#424242',
          accent: '#00E5FF',       // Annealing cyan
          error: '#FF5252',
          info: '#2196F3',
          success: '#4CAF50',
          warning: '#FB8C00',
          background: '#121212',
          surface: '#1E1E1E'
        }
      }
    }
  }
})

const pinia = createPinia()
const app = createApp(App)

app.use(pinia)
app.use(router)
app.use(vuetify)
app.mount('#app')
```

### Router Configuration (`router/index.ts`)

Same pattern as CiRA Oculus. Route table:

```typescript
[
  { path: '/login',     component: LoginPage,    meta: { requiresGuest: true } },
  { path: '/signup',    component: SignupPage,   meta: { requiresGuest: true } },
  { path: '/',          component: MainApp,      meta: { requiresAuth: true  } },
  { path: '/jobs/:id',  component: JobDetailPage,meta: { requiresAuth: true  } },
  { path: '/admin',     component: AdminPage,    meta: { requiresAuth: true, requiresAdmin: true } }
]
```

### Solve Store (`stores/solve.ts`)

```typescript
import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import axios from 'axios'

interface Job {
  id: string
  problem_statement: string
  provider: 'claude' | 'openai' | 'local'
  status: 'queued' | 'formulating' | 'validating' | 'solving' | 'complete' | 'error'
  cqm_json?: object
  variable_registry?: Record<string, string>
  solution?: object
  interpreted_solution?: string
  error?: string
  created_at: string
  completed_at?: string
}

export const useSolveStore = defineStore('solve', () => {
  const currentJob = ref<Job | null>(null)
  const history = ref<Job[]>([])
  const eventSource = ref<EventSource | null>(null)

  async function submitProblem(problemStatement: string, provider: string, apiKey: string) {
    const response = await axios.post('/api/solve',
      { problem_statement: problemStatement, provider, api_key: apiKey },
      { withCredentials: true }
    )
    currentJob.value = response.data.job
    streamStatus(response.data.job.id)
    return response.data
  }

  function streamStatus(jobId: string) {
    if (eventSource.value) eventSource.value.close()
    eventSource.value = new EventSource(`/api/jobs/${jobId}/stream`, { withCredentials: true })
    eventSource.value.onmessage = (event) => {
      const update = JSON.parse(event.data)
      if (currentJob.value?.id === jobId) {
        Object.assign(currentJob.value, update)
        if (['complete', 'error'].includes(update.status)) {
          eventSource.value?.close()
        }
      }
    }
  }

  async function loadHistory() {
    const response = await axios.get('/api/jobs', { withCredentials: true })
    history.value = response.data.jobs
  }

  return { currentJob, history, submitProblem, streamStatus, loadHistory }
})
```

### Vite Configuration (`vite.config.ts`)

```typescript
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import vuetify from 'vite-plugin-vuetify'
import path from 'path'

export default defineConfig({
  plugins: [vue(), vuetify({ autoImport: true })],
  resolve: { alias: { '@': path.resolve(__dirname, 'src') } },
  server: {
    port: 3011,
    proxy: {
      '/api': {
        target: 'http://localhost:5009',
        changeOrigin: true
      }
    }
  }
})
```

---

## Backend Architecture

The backend follows the CiRA Oculus pattern (Flask app factory, blueprints, SQLite, session auth) and adds the optimization pipeline modules.

### Flask App Factory (`__init__.py`)

```python
from flask import Flask
from flask_cors import CORS
from datetime import timedelta
import os

def create_app():
    app = Flask(__name__)

    CORS(app, supports_credentials=True,
         origins=['http://localhost:3011', 'http://localhost:3012'])

    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'change-in-production')
    app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=8)
    app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
    app.config['SESSION_COOKIE_SECURE'] = False  # True in production with HTTPS

    from app.models import init_db
    with app.app_context():
        init_db()

    from app.routes.auth import auth_bp
    from app.routes.solve import solve_bp
    from app.routes.admin import admin_bp
    from app.routes.health import health_bp
    app.register_blueprint(auth_bp, url_prefix='/api/auth')
    app.register_blueprint(solve_bp, url_prefix='/api')
    app.register_blueprint(admin_bp, url_prefix='/api/admin')
    app.register_blueprint(health_bp, url_prefix='/api')

    return app
```

### Configuration (`config.py`)

```python
import os
from datetime import timedelta

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Database
DATABASE_PATH = os.path.join(BASE_DIR, 'data', 'app.db')

# Session
SECRET_KEY = os.environ.get('SECRET_KEY', 'change-in-production')
SESSION_LIFETIME = timedelta(hours=8)

# Encryption for stored API keys (BYOK at-rest)
KEY_ENCRYPTION_SECRET = os.environ.get('KEY_ENCRYPTION_SECRET', 'change-this-32-byte-secret-now!!!')

# Roles
ROLE_ADMIN = 'admin'
ROLE_USER = 'user'
DEFAULT_ADMIN_USERNAME = 'admin'
DEFAULT_ADMIN_PASSWORD = 'admin123'

# Optimization
GPU_DEVICE = os.environ.get('GPU_DEVICE', 'cuda:0')
DEFAULT_NUM_READS = 1000
DEFAULT_NUM_SWEEPS = 1000
MAX_PROBLEM_VARIABLES = 50000   # Safety cap
SOLVE_TIMEOUT_SECONDS = 300

# Rate limits (per user)
MAX_CONCURRENT_JOBS = 1
MAX_JOBS_PER_HOUR = 10
MAX_PROMPT_LENGTH = 8000

# Local LLM endpoint
LOCAL_LLM_ENDPOINT = os.environ.get('LOCAL_LLM_ENDPOINT', 'http://localhost:11434')
```

### Database Models (`models.py`)

Three tables: `users` (same as CiRA Oculus pattern minus `private_folder`), `api_keys` (encrypted BYOK), and `jobs` (the optimization run history).

```python
import sqlite3
import os
import uuid
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from app.config import DATABASE_PATH, DEFAULT_ADMIN_USERNAME, DEFAULT_ADMIN_PASSWORD, ROLE_ADMIN

def get_db_connection():
    os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE,
            password_hash TEXT NOT NULL,
            display_name TEXT,
            role TEXT NOT NULL DEFAULT 'user',
            is_active INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            last_login TEXT
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS api_keys (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            provider TEXT NOT NULL,
            encrypted_key BLOB NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            UNIQUE (user_id, provider)
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            problem_statement TEXT NOT NULL,
            provider TEXT NOT NULL,
            status TEXT NOT NULL,
            cqm_json TEXT,
            variable_registry TEXT,
            solution TEXT,
            interpreted_solution TEXT,
            error TEXT,
            num_variables INTEGER,
            num_constraints INTEGER,
            solve_time_ms INTEGER,
            created_at TEXT NOT NULL,
            completed_at TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    ''')

    conn.commit()

    cursor.execute('SELECT COUNT(*) FROM users')
    if cursor.fetchone()[0] == 0:
        create_user(DEFAULT_ADMIN_USERNAME, DEFAULT_ADMIN_PASSWORD, 'Administrator', ROLE_ADMIN)

    conn.close()

def create_user(username, password, display_name=None, role='user', email=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO users (username, email, password_hash, display_name, role, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (username, email, generate_password_hash(password),
          display_name or username, role, datetime.utcnow().isoformat()))
    conn.commit()
    user_id = cursor.lastrowid
    conn.close()
    return {'id': user_id, 'username': username, 'display_name': display_name or username, 'role': role}

def verify_user(username, password):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
    user = cursor.fetchone()
    conn.close()
    if user and user['is_active'] and check_password_hash(user['password_hash'], password):
        return dict(user)
    return None

def create_job(user_id, problem_statement, provider):
    conn = get_db_connection()
    cursor = conn.cursor()
    job_id = str(uuid.uuid4())
    cursor.execute('''
        INSERT INTO jobs (id, user_id, problem_statement, provider, status, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (job_id, user_id, problem_statement, provider, 'queued', datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()
    return job_id

def update_job(job_id, **fields):
    if not fields:
        return
    conn = get_db_connection()
    cursor = conn.cursor()
    keys = ', '.join(f'{k} = ?' for k in fields.keys())
    cursor.execute(f'UPDATE jobs SET {keys} WHERE id = ?', list(fields.values()) + [job_id])
    conn.commit()
    conn.close()

def get_job(job_id, user_id=None, is_admin=False):
    conn = get_db_connection()
    cursor = conn.cursor()
    if is_admin:
        cursor.execute('SELECT * FROM jobs WHERE id = ?', (job_id,))
    else:
        cursor.execute('SELECT * FROM jobs WHERE id = ? AND user_id = ?', (job_id, user_id))
    job = cursor.fetchone()
    conn.close()
    return dict(job) if job else None
```

### Authentication Utilities (`auth.py`)

Identical pattern to CiRA Oculus — `@login_required`, `@admin_required`, `get_current_user()`. Removed are `is_path_allowed_for_user` and folder-related helpers (replaced with job-ownership checks inline).

```python
from functools import wraps
from flask import session, jsonify
from app.config import ROLE_ADMIN

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'error': 'Authentication required', 'code': 'AUTH_REQUIRED'}), 401
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'error': 'Authentication required', 'code': 'AUTH_REQUIRED'}), 401
        if session.get('user_role') != ROLE_ADMIN:
            return jsonify({'error': 'Admin access required', 'code': 'ADMIN_REQUIRED'}), 403
        return f(*args, **kwargs)
    return decorated_function

def get_current_user():
    if 'user_id' not in session:
        return None
    return {
        'id': session.get('user_id'),
        'username': session.get('username'),
        'display_name': session.get('display_name'),
        'role': session.get('user_role')
    }
```

---

## Authentication System

Same flow as CiRA Oculus — POST credentials to `/api/auth/login`, server stores session, frontend uses `withCredentials: true`. The only addition for V1 is a `/api/auth/signup` endpoint open to public.

```python
session['user_id'] = user['id']
session['username'] = user['username']
session['display_name'] = user['display_name']
session['user_role'] = user['role']
session.permanent = True
```

Sessions: 8-hour expiry, secure cookies in production.

Frontend: `requiresAuth`, `requiresGuest`, `requiresAdmin` route guards (same pattern).

Backend: `@login_required`, `@admin_required` decorators (same pattern).

---

## Multi-User & Job Ownership

### User Roles

| Role | Access |
|------|--------|
| **admin** | Full access; sees all jobs, manages users, can delete any job |
| **user** | Sees only own jobs; cannot see other users' problem statements or solutions |

### Ownership Rules

- Every `job` record has a `user_id` foreign key
- All `/api/jobs/*` endpoints filter by `user_id = current_user.id` unless the caller is admin
- API keys are stored per-user, encrypted with `KEY_ENCRYPTION_SECRET`, never returned via the API in plaintext

### Rate Limiting

Per user, enforced in the request handler before queueing:

- Max 1 concurrent in-flight job
- Max 10 jobs per hour
- Max 8000 characters in a problem statement

Exceeded limits return HTTP 429 with `code: 'RATE_LIMITED'`.

---

## Optimization Pipeline

This is the core domain-specific architecture. The pipeline runs in five stages, each emitting a status event to the SSE stream.

```
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│   POST /api/solve                                            │
│       │                                                      │
│       ▼                                                      │
│  ┌────────────────────────┐                                  │
│  │ 1. Formulation         │  status: 'formulating'           │
│  │    LLM provider        │                                  │
│  │    → CQM JSON          │                                  │
│  │    → variable_registry │                                  │
│  └──────────┬─────────────┘                                  │
│             │                                                │
│             ▼                                                │
│  ┌────────────────────────┐                                  │
│  │ 2. Compilation         │  status: 'compiling'             │
│  │    JSON → dimod.CQM    │                                  │
│  │    Schema validation   │                                  │
│  └──────────┬─────────────┘                                  │
│             │                                                │
│             ▼                                                │
│  ┌────────────────────────┐                                  │
│  │ 3. Validation          │  status: 'validating'            │
│  │    a. ExactCQMSolver   │                                  │
│  │       on tiny instance │                                  │
│  │    b. CPU SA sanity    │                                  │
│  │    c. Constraint scan  │                                  │
│  └──────────┬─────────────┘                                  │
│             │                                                │
│             ▼                                                │
│  ┌────────────────────────┐                                  │
│  │ 4. Solve               │  status: 'solving'               │
│  │    GPU SA on 5070 Ti   │                                  │
│  │    (or SQBM+ later)    │                                  │
│  └──────────┬─────────────┘                                  │
│             │                                                │
│             ▼                                                │
│  ┌────────────────────────┐                                  │
│  │ 5. Interpretation      │  status: 'complete'              │
│  │    SampleSet → prose   │                                  │
│  └────────────────────────┘                                  │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

Each stage may fail, transitioning the job to `status: 'error'` with `error: <stage>: <message>`.

### CQM JSON Schema (`schemas/cqm_v1.json`)

The contract between the LLM and the backend. The LLM must emit exactly this structure:

```json
{
  "version": "1",
  "variables": [
    { "name": "x_job1_machine_A", "type": "binary",  "description": "Job 1 runs on Machine A" },
    { "name": "start_job1",       "type": "integer", "lower_bound": 0, "upper_bound": 100,
      "description": "Start time of Job 1" }
  ],
  "objective": {
    "sense": "minimize",
    "linear": { "start_job1": 1.0 },
    "quadratic": {}
  },
  "constraints": [
    {
      "label": "job1_assigned_once",
      "type": "equality",
      "linear": { "x_job1_machine_A": 1, "x_job1_machine_B": 1 },
      "quadratic": {},
      "rhs": 1
    }
  ],
  "test_instance": {
    "description": "Tiny case used for ExactCQMSolver validation",
    "expected_optimum": 5.0
  }
}
```

### Solver Tiers (`tiers.py`)

```python
class SolverTier(Enum):
    GPU_SA = 'gpu_sa'           # V1 default
    CPU_SA = 'cpu_sa'           # Validation only
    SQBM_PLUS = 'sqbm_plus'     # Phase 8
    EXACT = 'exact'             # Validation only
```

In V1 the user-visible solver is always GPU_SA. CPU_SA and EXACT are used inside the validation stage.

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SECRET_KEY` | `change-in-production` | Flask session secret |
| `KEY_ENCRYPTION_SECRET` | `change-this...` | Fernet key for BYOK encryption |
| `GPU_DEVICE` | `cuda:0` | PyTorch device for GPU SA |
| `LOCAL_LLM_ENDPOINT` | `http://localhost:11434` | Ollama / vLLM URL |
| `MAX_PROBLEM_VARIABLES` | `50000` | Safety cap on problem size |
| `SOLVE_TIMEOUT_SECONDS` | `300` | Per-solve timeout |

### Ports

| Service | Port | Purpose |
|---------|------|---------|
| Frontend dev | 3011 | Vite dev server |
| Backend dev | 5009 | Flask dev server |
| Local LLM | 11434 | Ollama default |

---

## API Endpoints Reference

### Authentication

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| POST | `/api/auth/signup` | Create new user account | No |
| POST | `/api/auth/login` | Log in | No |
| POST | `/api/auth/logout` | Log out | Yes |
| GET | `/api/auth/me` | Current user info | Yes |
| POST | `/api/auth/change-password` | Change password | Yes |

### API Keys (BYOK)

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| GET | `/api/keys` | List user's stored providers (no values) | Yes |
| PUT | `/api/keys/:provider` | Store/update key for provider | Yes |
| DELETE | `/api/keys/:provider` | Remove stored key | Yes |

### Solve & Jobs

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| POST | `/api/solve` | Submit a problem; returns `job_id` | Yes |
| GET | `/api/jobs` | List user's jobs (paginated) | Yes |
| GET | `/api/jobs/:id` | Get one job (full detail) | Yes |
| GET | `/api/jobs/:id/stream` | SSE stream of status updates | Yes |
| DELETE | `/api/jobs/:id` | Delete a job from history | Yes |

### Problem Templates

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| GET | `/api/templates` | List all templates (id, title, category, difficulty, tags) | Yes |
| GET | `/api/templates/:id` | Full template detail | Yes |
| GET | `/api/templates/categories` | List of categories with counts | Yes |
| POST | `/api/solve/from-template/:id` | Submit a job pre-filled from a template | Yes |

### Admin

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| GET | `/api/admin/users` | List all users | Admin |
| POST | `/api/admin/users` | Create user | Admin |
| PUT | `/api/admin/users/:id` | Update user | Admin |
| DELETE | `/api/admin/users/:id` | Delete user | Admin |
| GET | `/api/admin/jobs` | List all jobs (any user) | Admin |
| GET | `/api/admin/stats` | Usage stats | Admin |

### Health

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| GET | `/api/health` | Liveness check | No |
| GET | `/api/health/gpu` | GPU availability check | Yes |

---

# Phased Deliverables Specification

This section is the **build spec for the AI coder**. The project is divided into phases, each with a single goal, explicit scope, file list, acceptance criteria, and verification steps. **Hand the AI coder one phase at a time. Do not proceed to the next phase until the current phase's Definition of Done passes.**

### v2 phase plan (revised)

| Phase | Goal | Status | Approx. Effort |
|-------|------|--------|----------------|
| **0** | Project skeleton, auth, DB, app shell | Shipped under v1 ✅ | (delivered) |
| **1** | GPU Simulated Annealing library (no web) | Shipped under v1 ✅ | (delivered) |
| **2** | **Validation harness + Benchmarks foundation** (revised) | Next | 2–3 days |
| **3** | Formulation provider layer (Claude / OpenAI / local) | Mostly intact | 2 days |
| **4** | Solve API endpoint (synchronous pipeline) | Mostly intact | 1–2 days |
| **5** | Frontend solve UI with SSE | Adds Compare and Learn shells | 2 days |
| **5B** | **Modules library (revised: curriculum seed, not just examples)** | Substantially revised | 2–3 days |
| **5C** | **Benchmark dashboard (NEW)** | New under v2 | 2 days |
| **6** | Async job queue + GPU contention | Unchanged | 1 day |
| **7** | Public web app hardening | Mostly intact | 2–3 days |
| **8** | **Classical solver tiers (NEW: CP-SAT, HiGHS) — was previously SQBM+ slot** | Reordered under v2 | 2 days |
| **9** | Quantum-inspired solver tiers (PT, PQQA, SQBM+) | Reordered under v2 | 2–3 days each |
| **10** | Contribution pipeline (NEW: solver tiers, instances, modules) | Roadmap | TBD |
| **11** | Multi-tenant academic deployment (NEW) | Roadmap | TBD |
| **12+** | Quantum hardware tiers (D-Wave, Qiskit/QAOA, neutral atoms) | Specialty tiers | Per-vendor |

Phases 10 and 11 are sketched in [Appendix A: Academic Roadmap](#appendix-a-academic-roadmap) and become committed phases when academic conditions warrant (research grant, course cohort, publication target).

### v2 reordering rationale

Three changes from the v1 phase ordering deserve explicit justification:

**Phase 2 expanded from "validation harness" to "validation + Benchmarks foundation."** Under v1, validation existed to prevent silently-wrong CQMs. Under v2, the same code also produces the platform's Benchmark output. Designing for reproducibility, citation, and longevity from the start is dramatically cheaper than retrofitting later.

**Classical solver tiers (CP-SAT, HiGHS) moved earlier.** Under v1, CP-SAT and Gurobi-equivalents weren't planned at all. Under v2, honest Benchmarks require classical baselines from early in the platform's life. Phase 8 now holds these baselines; the QUBO-specialty tiers (PQQA, SQBM+) move to Phase 9.

**Phase 5B reframed from "templates" to "curriculum seed."** Same JSON files, but with explicit learning objectives, prerequisite chains, and assessment hooks. The work is mostly additive — Phase 5B's v1 specification is preserved; the curriculum metadata is layered on top.

---

## Phase 0 — Project Skeleton & Authentication

### Goal
A running Vue + Flask shell where users can sign up, log in, see an empty home page, and log out.

### Scope — In
- Repository structure per the **Project Structure** section above
- Vue + Vuetify + Pinia + Router with the dark theme
- Flask app factory, blueprints, SQLite DB initialization
- Login, signup, logout, change-password endpoints and pages
- `@login_required` and `@admin_required` decorators
- Default admin (`admin` / `admin123`) created on first run
- Auth store with `checkAuth()` on app boot
- Health check endpoint

### Scope — Out
- Anything optimization-related
- API key management
- Admin user-management UI

### Files to Create

```
backend/
├── app/__init__.py
├── app/config.py
├── app/auth.py
├── app/models.py                  # users + jobs + api_keys tables (table defs ready, jobs unused this phase)
├── app/routes/__init__.py
├── app/routes/auth.py
├── app/routes/health.py
├── run.py
├── requirements.txt

frontend/
├── src/main.ts
├── src/App.vue
├── src/router/index.ts
├── src/stores/auth.ts
├── src/views/LoginPage.vue
├── src/views/SignupPage.vue
├── src/views/MainApp.vue          # placeholder "Welcome, {{ user.display_name }}"
├── package.json
├── vite.config.ts
├── tsconfig.json
```

### API Contracts

**POST /api/auth/signup**
```json
Request:  { "username": "alice", "email": "a@b.com", "password": "...", "display_name": "Alice" }
Response: { "success": true, "user": { "id": 2, "username": "alice", "display_name": "Alice", "role": "user" } }
Errors:   400 (validation), 409 (username taken)
```

**POST /api/auth/login**
```json
Request:  { "username": "alice", "password": "..." }
Response: { "success": true, "user": { ... } }
Errors:   401 (bad credentials)
```

**GET /api/auth/me, POST /api/auth/logout, POST /api/auth/change-password**
— same pattern as CiRA Oculus.

**GET /api/health**
```json
Response: { "status": "ok", "version": "0.1.0" }
```

### Tests Required

```
backend/tests/test_auth.py:
  - test_signup_creates_user
  - test_signup_rejects_duplicate_username
  - test_login_with_valid_credentials_succeeds
  - test_login_with_invalid_credentials_fails
  - test_me_returns_user_when_authenticated
  - test_me_returns_401_when_not_authenticated
  - test_logout_clears_session
  - test_default_admin_created_on_init
```

### Definition of Done

- [ ] `cd backend && python run.py` starts on port 5009 with no errors
- [ ] `cd frontend && npm run dev` starts on port 3011
- [ ] Visiting `http://localhost:3011` redirects to `/login`
- [ ] Signing up creates a new user and logs them in
- [ ] Logging in as `admin` / `admin123` works
- [ ] Logging out returns to login page
- [ ] All pytest tests in `tests/test_auth.py` pass
- [ ] No hard-coded secrets in committed code (use env vars with placeholder defaults)

---

## Phase 1 — GPU Simulated Annealing Library

### Goal
A `GPUSimulatedAnnealingSampler` Python class that implements `dimod.Sampler`, runs on the local RTX 5070 Ti, and beats CPU SA by ≥10× on a 1000-variable benchmark. **No web integration in this phase.**

### Scope — In
- `backend/app/optimization/gpu_sa.py` containing the sampler class
- A CLI harness `python -m app.optimization.gpu_sa --bqm path/to/bqm.json` for manual testing
- Pytest suite covering correctness against `ExactSolver` on tiny instances and scaling vs CPU SA on medium ones
- Optional: `@torch.compile` wrapping of the inner sweep for the 2–4× extra speedup

### Scope — Out
- Frontend
- API endpoints
- Provider/formulation layer

### Files to Create

```
backend/
├── app/optimization/__init__.py
├── app/optimization/gpu_sa.py
└── tests/
    ├── conftest.py                # fixtures for tiny BQMs
    ├── instances/
    │   ├── tiny_5var.json
    │   ├── medium_100var.json
    │   └── benchmark_1000var.json
    ├── test_gpu_sa.py
    └── benchmark_gpu_vs_cpu.py    # not a test, a runnable benchmark
```

### Class Contract

```python
class GPUSimulatedAnnealingSampler(dimod.Sampler):
    def __init__(self, device: str = "cuda:0"):
        """Verify CUDA is available and the device exists."""

    @property
    def parameters(self) -> dict:
        return {"num_reads": [], "num_sweeps": [], "beta_range": [], "seed": []}

    @property
    def properties(self) -> dict:
        return {"device": str, "compute_capability": tuple, "vram_total_gb": float}

    def sample(self, bqm: dimod.BinaryQuadraticModel,
               num_reads: int = 1000,
               num_sweeps: int = 1000,
               beta_range: tuple = (0.1, 5.0),
               seed: int | None = None) -> dimod.SampleSet:
        """Returns a SampleSet sorted by energy (lowest first)."""
```

### Tests Required

```
test_gpu_sa.py:
  - test_gpu_available
        Verifies torch.cuda.is_available() and device sm_120

  - test_sampler_implements_dimod_interface
        Confirms .parameters, .properties, .sample(bqm) signatures

  - test_two_variable_problem_finds_optimum
        BQM = {0: -1, 1: 1, (0,1): 2}; expected optimum at (1, 0) energy = -1

  - test_agrees_with_exact_solver_5var
        Random 5-var BQM; ExactSolver gives the truth; GPU SA at num_reads=2000 must match

  - test_agrees_with_exact_solver_10var
        Same but 10 variables; loosen tolerance, tighten num_reads if needed

  - test_returns_correct_num_reads
        sample(bqm, num_reads=500) returns 500 records

  - test_seed_is_reproducible
        Same seed → identical SampleSet

  - test_handles_empty_bqm
        Returns valid (empty-ish) SampleSet, doesn't crash

  - test_scales_to_1000_variables
        100-edge sparse 1000-var BQM completes in <30 seconds
```

### Benchmark Harness

`benchmark_gpu_vs_cpu.py` reports a table:

```
N        CPU SA (ms)    GPU SA (ms)    Speedup    Both find same optimum?
100         220            45             4.9x        yes
500        4800           260            18.5x        yes
1000      28000           890            31.5x        yes
5000     780000          5400           144x          yes
```

### Definition of Done

- [ ] All tests in `test_gpu_sa.py` pass
- [ ] `python -m app.optimization.gpu_sa --bqm tests/instances/tiny_5var.json` prints a valid SampleSet
- [ ] Benchmark shows ≥10× speedup over `dwave-samplers.SimulatedAnnealingSampler` at N=1000
- [ ] No CUDA out-of-memory crashes at N=10000 dense BQM
- [ ] Code lints clean (`ruff check`, `mypy`)

---

## Phase 2 — Validation Harness + Benchmarks Foundation

> **v2 status:** This is the next phase. **Substantially revised** from v1 — the v1 Phase 2 specification is preserved below the v2 revision for audit trail.

### v2 Goal

A reusable library that does two things with the same primitives:

1. **Validates that a CQM encodes the intended problem** — proves correctness of LLM-generated formulations through oracle agreement, multi-solver agreement, and constraint coverage. (This was the v1 goal in full.)
2. **Produces reproducible, citable Benchmark records** — every solver run on every instance produces a versioned, hash-pinned record that can be cited in publications and re-executed by anyone with the same code version.

The same harness serves both purposes because they share the same primitives: run multiple solvers, compare results, record what happened. v2 adds the *recording and reproducibility* infrastructure that turns validation runs into Benchmark assets.

### v2 Why this matters

Under v1, the validation harness was internal infrastructure — the safety net that kept Claude-Code-generated CQMs from being silently wrong. Under v2, the **same infrastructure is the public face of the platform**:

- Every Module exercise is graded by the validation harness
- Every entry in the Benchmark dashboard comes from the same run-record schema
- Every research paper citing the platform points to specific Benchmark records produced by this code

If Phase 2 ships only the v1 scope (correctness checking), Phase 5C (the Benchmark dashboard) has nothing to display, and Phase 5B's Modules cannot reliably grade student work. Building the Benchmarks foundation now is dramatically cheaper than retrofitting it across three later phases.

### v2 Scope — In

All v1 Phase 2 scope items, plus:

- **Solver registry** (`app/benchmarking/registry.py`) — every solver tier registers its name, version, source library, hardware identifier, parameter schema. The registry is the source of truth for what solvers exist and what counts as "the same solver" across runs.
- **Run records** (`app/benchmarking/records.py`) — every solver invocation produces a JSON record containing: solver identity, full parameters, instance identity, hardware identity, timestamps, results, sample-set archive path, reproducibility hash. Records are append-only and preserved indefinitely.
- **Reproducibility hash** — deterministic hash over (code version + instance + solver + parameters). Two records with the same hash should produce identical results modulo seeded randomness.
- **CLI Benchmark runner** — `python -m app.benchmarking.run_suite --solver <name> --instances <suite>` executes a Benchmark suite from the command line and writes records to `benchmarks/archive/`. Phase 5C's UI reads from the same data this CLI produces.
- **Three registered solver tiers, minimum:** `ExactCQMSolver`, `dwave-samplers.SimulatedAnnealingSampler`, and `GPUSimulatedAnnealingSampler` (from Phase 1). Every Benchmark and every multi-solver agreement test runs against all three so comparisons are real, not self-checks.
- **Citation generator** — `python -m app.benchmarking.cite <record_id>` produces a BibTeX entry and a citation string for any preserved record.
- **Instance suite registry** — canonical instance suites organized by problem class. Phase 2 ships at least: `knapsack/small`, `setcover/small`, `jss/small`, `maxcut/gset_subset`, `graph_coloring/small`. Each instance has known optima or best-known values for oracle comparison.

### v2 Scope — Out

Same v1 outs (no frontend, no LLM provider layer), plus:

- The Benchmark dashboard UI (Phase 5C consumes the records this phase produces)
- Cohort-based grading (Phase 5B / Phase 11 territory)
- Public contribution of new solvers or instances (Phase 10)

### v2 Files to Create

```
backend/
├── app/optimization/validation.py       (v1 scope — preserved)
├── app/optimization/compiler.py         (v1 scope — preserved)
├── app/benchmarking/__init__.py         (NEW under v2)
├── app/benchmarking/registry.py         (NEW under v2)
├── app/benchmarking/records.py          (NEW under v2)
├── app/benchmarking/run_suite.py        (NEW under v2 — CLI entry point)
├── app/benchmarking/cite.py             (NEW under v2 — citation generator)
├── app/benchmarking/schemas/
│   ├── solver_v1.json                   (NEW — solver registry schema)
│   ├── record_v1.json                   (NEW — run record schema)
│   └── instance_v1.json                 (NEW — instance metadata schema)
└── tests/
    ├── instances/                       (v1 — expanded with size variants)
    │   ├── knapsack_5item.json          (preserved)
    │   ├── knapsack_20item.json         (NEW)
    │   ├── setcover_4item.json          (preserved)
    │   ├── jss_3job_3machine.json       (preserved)
    │   ├── jss_5job_5machine.json       (NEW)
    │   ├── maxcut_6node.json            (preserved)
    │   ├── maxcut_50node.json           (NEW)
    │   └── graphcoloring_4node.json     (preserved)
    ├── test_compiler.py                 (v1 scope)
    ├── test_validation.py               (v1 scope)
    ├── test_oracle_match.py             (v1 scope)
    ├── test_registry.py                 (NEW under v2)
    ├── test_records.py                  (NEW under v2)
    ├── test_reproducibility.py          (NEW under v2)
    └── test_cite.py                     (NEW under v2)

benchmarks/
├── archive/                             (NEW — long-term record storage)
└── README.md                            (NEW — explains schema and citation model)
```

### v2 API additions

```python
# app/benchmarking/registry.py
@dataclass
class SolverIdentity:
    name: str                    # 'gpu_sa', 'cpu_sa_neal', 'exact_cqm'
    version: str                 # semver of the solver's implementation
    source: str                  # 'cira-quantum', 'dwave-samplers', 'dimod'
    hardware: str | None         # 'cuda:0:rtx5070ti', 'cpu', None for exact
    parameter_schema: dict       # JSON-schema of accepted sample() kwargs

def register_solver(identity: SolverIdentity, sampler_cls: type) -> None: ...
def get_solver(name: str) -> tuple[SolverIdentity, type]: ...
def list_solvers() -> list[SolverIdentity]: ...

# app/benchmarking/records.py
@dataclass
class RunRecord:
    record_id: str               # unique, sortable, citable
    code_version: str            # git SHA at time of run
    solver: SolverIdentity
    parameters: dict             # full sample() kwargs
    instance_id: str             # references instance registry
    hardware_id: str             # actual hardware (may be more specific than solver.hardware)
    started_at: datetime
    completed_at: datetime
    repro_hash: str              # deterministic hash over the above
    results: dict                # energies, samples, feasibility
    sample_set_path: str         # archived dimod.SampleSet (gzipped JSON)

def record_run(...) -> RunRecord: ...
def replay_record(record_id: str) -> RunRecord: ...  # re-execute, compare
def archive_path(record_id: str) -> Path: ...
```

### v2 Tests Required

All v1 test cases remain. Plus:

```
test_registry.py:
  - test_register_solver_succeeds
  - test_register_duplicate_name_fails
  - test_get_solver_returns_identity_and_class
  - test_phase_1_gpu_sa_registers_correctly
  - test_dwave_neal_registers_correctly
  - test_exact_cqm_registers_correctly

test_records.py:
  - test_record_includes_all_required_fields
  - test_record_id_is_sortable_and_unique
  - test_repro_hash_is_deterministic
  - test_repro_hash_differs_on_parameter_change
  - test_sample_set_round_trips_through_archive

test_reproducibility.py:
  - test_replay_with_same_seed_produces_identical_record       # for seeded solvers
  - test_replay_with_different_code_version_warns               # detects spec drift
  - test_three_solver_agreement_recorded_correctly

test_cite.py:
  - test_bibtex_entry_well_formed
  - test_citation_string_includes_record_id_and_date
  - test_cite_unknown_record_returns_clear_error
```

### v2 Definition of Done

All v1 DoD items remain (in the preserved v1 section below). v2 adds:

- [ ] Three solver tiers (`ExactCQMSolver`, `dwave-samplers.SA`, `GPUSimulatedAnnealingSampler`) registered and discoverable via `list_solvers()`
- [ ] Every Benchmark run produces a RunRecord written to `benchmarks/archive/`
- [ ] Running the same RunRecord twice on the same code version with the same seed produces identical results (verified by `replay_record`)
- [ ] CLI tool `python -m app.benchmarking.run_suite --solver gpu_sa --instances knapsack/small` runs and writes records
- [ ] CLI tool `python -m app.benchmarking.cite <record_id>` produces a valid BibTeX entry
- [ ] All tests pass (v1 set + v2 additions)
- [ ] `benchmarks/README.md` documents the record schema, the citation model, and the long-term preservation policy

### v2 Note to the AI coder

The Phase 1 GPU SA shipped with three kernel modes (`jit`, `compile`, `eager`). For Phase 2 Benchmark runs, use `kernel="compile"` — the warmup cost is amortized across the suite and the comparison numbers should reflect production performance. For the validation harness's correctness tests, continue using `kernel="jit"` to keep CI fast. This distinction must be encoded in the record's `parameters` field so the kernel mode is reproducible.

The Phase 1 DECISIONS.md notes that `triton-windows` is required for `torch.compile` on Windows. Verify this is platform-conditional in `pyproject.toml` (`platform_system == 'Windows'`); if not, fix it before Phase 2 ships, otherwise Linux contributors will see install failures.

The v1 Phase 2 spec referenced `dimod.cqm_to_bqm(lagrange_multiplier=10.0)` — this is fine for early Benchmarks but the Lagrange multiplier becomes a tuning parameter that affects results. Document the chosen value in DECISIONS.md and surface it in the RunRecord parameters so future tuning is traceable.

---

### v2 supersedes v1 here (post-approval reconciliation, 2026-05-07)

Three deltas between the v1 API documented further down and the v1 baseline as actually shipped under v2. The v1 content is preserved verbatim for audit trail; the **operative Phase 2 contract is the version below**.

**1. `compile_cqm_json` returns a 3-tuple, not a 2-tuple.**

```python
# v2 operative signature
def compile_cqm_json(
    cqm_json: dict,
) -> tuple[dimod.ConstrainedQuadraticModel, dict[str, str], str]:
    """Returns (cqm, variable_registry, sense)."""
```

The third element is the JSON's `objective.sense` value (`"minimize"` or `"maximize"`). dimod CQMs always minimize internally; for maximize-sense problems the compiler negates linear and quadratic coefficients before adding them. The validation harness needs the original sense to convert internal-minimization energies back into the user-facing units that `expected_optimum` is stated in. Putting the sense in a dedicated tuple element (rather than smuggling it through the variable registry) keeps the boundary clean.

**2. `ValidationReport.oracle_skipped` field.**

The v1 ValidationReport dataclass distinguishes "oracle agreed" from "oracle disagreed", but not "oracle didn't run." When the CQM has more variables than `max_oracle_vars` (default 12), `ExactCQMSolver` is bypassed entirely. v2 adds an explicit `oracle_skipped: bool` field; without it, `oracle_agreement = True` is ambiguous between "vacuously passed" and "actually verified". The shipped dataclass:

```python
@dataclass
class ValidationReport:
    oracle_agreement: bool
    oracle_skipped: bool                  # NEW under v2
    solver_agreement: bool
    constraints_active: dict[str, bool]
    energy_oracle: float | None
    energy_gpu_sa: float | None
    energy_cpu_sa: float | None
    warnings: list[str]
    # passed is now a property, not a stored bool — derived from the
    # three layer flags so it can never drift out of sync.
```

**3. JSS instance designations.**

The v2 file list specifies both `tests/instances/jss_3job_3machine.json` (preserved) and `tests/instances/jss_5job_5machine.json` (NEW). Phase 2 also ships an additional `jss_2job_2machine.json` for one specific reason:

- **`jss_2job_2machine.json`** (7 variables, 8 constraints, optimum makespan = 4) is the *oracle-target* instance. It's brute-forceable in 31104 states, so the `test_jss_3x3_matches_brute_force` test exercises the actual Layer-A oracle agreement against this instance.
- **`jss_3job_3machine.json`** (19 variables, 27 constraints, expected makespan = 11) is *feasibility / Layer-C only*. The 19-variable disjunctive big-M encoding makes `ExactCQMSolver` attempt a 115 TiB cartesian-product allocation (verified empirically) and is genuinely beyond vanilla SA at any tested Lagrange multiplier (`5..100`, 2000 reads × 5000 sweeps). It exists to demonstrate Layer A graceful skip and Layer C constraint coverage on a realistically-sized JSS encoding; parallel tempering / PT-SQA (Phase 9) is the right tool for Layer-B feasibility recovery on it.
- **`jss_5job_5machine.json`** (76 variables, 125 constraints, expected_optimum = `null`) is a *no-ground-truth* instance. The harness handles it gracefully — Layer A skipped, Layer C executes (all 125 constraints active), Layer B runs and reports "no feasible found" as a warning, RunRecord written with `converged_to_expected = null`.

**4. `RunRecord.results` includes a convergence flag.**

Added during Phase 2 review for the Phase 5C dashboard's honesty. The `results` dict gains:

- `expected_optimum: float | null` — copied from the instance manifest.
- `gap_to_expected: float | null` — `|best_user_energy - expected_optimum|`.
- `converged_to_expected: bool | null` — `True` iff the gap is within `max(1e-3 * |expected|, 1e-9)`. **`null` when the instance has no ground truth** — never inferred. The dashboard reads `null` as "heuristic estimate, no ground truth available," not as success.

These three fields are also reflected in [`backend/app/benchmarking/schemas/record_v1.json`](backend/app/benchmarking/schemas/record_v1.json).

---

### v1 Phase 2 specification (preserved for audit trail)

The original v1 Phase 2 specification follows. v2 supersedes it where the two conflict. v2 expands scope but does not remove anything; all v1 deliverables remain in the v2 phase.

### Goal
A reusable validation library that proves a CQM encodes the intended problem. This is **the safety net for LLM-generated formulations** and the most strategically important phase. Without it, the system silently returns wrong answers.

### Scope — In
- `backend/app/optimization/validation.py` with three layers:
  - **Layer A — Oracle agreement:** run `ExactCQMSolver` on a tiny instance and compare to a known answer
  - **Layer B — Solver agreement:** run `ExactCQMSolver`, `SimulatedAnnealingSampler`, and `GPUSimulatedAnnealingSampler` on the same small CQM; assert agreement within tolerance
  - **Layer C — Constraint coverage:** for each constraint in a CQM, verify there exists at least one violating assignment that registers a violation, and one satisfying assignment that doesn't
- `backend/app/optimization/compiler.py` to convert CQM-JSON → `dimod.ConstrainedQuadraticModel`
- Test fixtures for canonical problems (knapsack, set cover, JSS, max-cut, graph coloring)

### Scope — Out
- Frontend
- LLM provider layer

### Files to Create

```
backend/
├── app/optimization/validation.py
├── app/optimization/compiler.py
└── tests/
    ├── instances/
    │   ├── knapsack_5item.json
    │   ├── setcover_4item.json
    │   ├── jss_3job_3machine.json
    │   ├── maxcut_6node.json
    │   └── graphcoloring_4node.json
    ├── test_compiler.py
    ├── test_validation.py
    └── test_oracle_match.py        # gold standard
```

### API

```python
# compiler.py
def compile_cqm_json(cqm_json: dict) -> tuple[dimod.CQM, dict]:
    """Returns (cqm, variable_registry)."""

# validation.py
@dataclass
class ValidationReport:
    oracle_agreement: bool       # Layer A
    solver_agreement: bool       # Layer B
    constraints_active: dict     # Layer C: { constraint_label: bool }
    energy_oracle: float | None
    energy_gpu_sa: float | None
    energy_cpu_sa: float | None
    warnings: list[str]
    passed: bool                 # all three layers green

def validate_cqm(cqm: dimod.CQM, expected_optimum: float | None = None,
                 max_oracle_vars: int = 12) -> ValidationReport:
    """Run all three validation layers."""
```

### Tests Required

```
test_compiler.py:
  - test_compile_minimal_cqm
  - test_compile_with_binary_and_integer_vars
  - test_compile_with_inequality_constraints
  - test_compile_rejects_invalid_schema
  - test_variable_registry_preserves_descriptions

test_validation.py:
  - test_layer_a_oracle_agreement_passes_for_correct_cqm
  - test_layer_a_oracle_agreement_fails_for_wrong_objective_sign
  - test_layer_b_solver_agreement
  - test_layer_c_detects_inactive_constraint   # Wrong-operator constraints don't activate
  - test_layer_c_detects_missing_constraint
  - test_validation_report_aggregates_correctly

test_oracle_match.py:
  - test_knapsack_5item_matches_oracle
  - test_setcover_4item_matches_oracle
  - test_jss_3x3_matches_brute_force
  - test_maxcut_6node_matches_oracle
```

### Definition of Done

- [ ] All tests pass
- [ ] Compiling a malformed CQM JSON raises `ValueError` with a clear message
- [ ] `ValidationReport.passed` is `True` for all canonical instances and `False` when objective sign is intentionally flipped
- [ ] Layer C catches a deliberately-broken constraint (e.g., `<=` flipped to `>=`)

---

## Phase 3 — Formulation Provider Layer

### Goal
A pluggable provider system that converts a natural-language problem statement into a CQM JSON via Claude, OpenAI, or a local LLM. Each provider is a separate file implementing the same abstract interface.

### Scope — In
- `backend/app/formulation/base.py` — `FormulationProvider` abstract class
- Three concrete providers: `claude.py`, `openai.py`, `local.py`
- Prompt templates in `prompts/` (system prompt + few-shot examples)
- JSON schema validation against `schemas/cqm_v1.json`
- BYOK key storage with Fernet encryption (`crypto.py`)
- Per-provider unit tests using **mocked** API responses (no real spend in CI)

### Scope — Out
- Frontend
- Pipeline orchestration (Phase 4)
- Real API calls in CI (only manual integration tests)

### Files to Create

```
backend/
├── app/crypto.py
├── app/formulation/__init__.py
├── app/formulation/base.py
├── app/formulation/claude.py
├── app/formulation/openai.py
├── app/formulation/local.py
├── app/formulation/prompts/
│   ├── system.txt
│   └── examples.json
├── app/formulation/schemas/
│   └── cqm_v1.json
└── tests/
    ├── test_crypto.py
    ├── test_formulation_base.py
    ├── test_formulation_claude.py
    ├── test_formulation_openai.py
    ├── test_formulation_local.py
    └── fixtures/
        ├── mock_claude_response.json
        └── mock_openai_response.json
```

### Provider Contract

```python
# base.py
@dataclass
class FormulationResult:
    cqm_json: dict                 # validated against cqm_v1.json
    variable_registry: dict        # var_name → human description
    raw_llm_output: str            # for debugging
    tokens_used: int               # prompt + completion
    model: str                     # e.g. "claude-opus-4-7"

class FormulationProvider(ABC):
    name: str

    @abstractmethod
    async def formulate(self, problem_statement: str, api_key: str,
                        timeout: int = 60) -> FormulationResult:
        """Raise FormulationError on any failure."""

    @abstractmethod
    def estimate_cost(self, problem_statement: str) -> float:
        """USD estimate based on token count."""
```

### Prompt Strategy

- **System prompt** (`prompts/system.txt`) tells the LLM to:
  - Output only valid JSON conforming to the cqm_v1 schema
  - Use descriptive variable names (`x_job1_machine_A`, not `x0`)
  - Always populate `variable_registry` descriptions
  - Always include a `test_instance` section with `expected_optimum` for tiny cases
  - Refuse to formulate non-optimization requests
- **Few-shot examples** (`prompts/examples.json`) — three solved problems: knapsack, set cover, JSS

### Tests Required

```
test_crypto.py:
  - test_encrypt_decrypt_roundtrip
  - test_decrypt_with_wrong_key_fails
  - test_keys_stored_as_bytes_not_str

test_formulation_base.py:
  - test_validate_against_schema_accepts_valid_json
  - test_validate_rejects_missing_required_fields
  - test_formulation_result_dataclass

test_formulation_claude.py:    # uses httpx_mock or aioresponses
  - test_claude_provider_calls_correct_endpoint
  - test_claude_provider_extracts_json_from_response
  - test_claude_provider_raises_on_invalid_json
  - test_claude_provider_passes_api_key_in_header
  - test_estimate_cost

test_formulation_openai.py: similar pattern

test_formulation_local.py:
  - test_local_provider_calls_ollama_endpoint
  - test_local_provider_handles_streaming_response
```

### Manual Integration Tests (NOT in CI)

A separate `manual_integration_test.py` script that requires real API keys via env var. Run it once at the end of the phase to confirm:
- Each provider, given the prompt "schedule 3 jobs on 2 machines minimizing makespan", produces a valid CQM JSON
- Compiling that JSON produces a `dimod.CQM`
- Running validation (Phase 2) returns `passed: True` for at least 2 of 3 providers

### Definition of Done

- [ ] All unit tests pass
- [ ] Manual integration test passes for Claude and OpenAI (local LLM may fail; that's known and acceptable)
- [ ] BYOK keys are encrypted in the DB (verify by inspecting the `api_keys` table directly)
- [ ] No API keys appear in any log output
- [ ] Schema validation rejects 5 deliberately-malformed CQM JSONs

---

## Phase 4 — Solve API Endpoint

### Goal
A working POST `/api/solve` endpoint that ties together formulation → compilation → validation → solving → interpretation, exposed via REST + SSE. **Synchronous pipeline only** in this phase — async queueing comes in Phase 6.

### Scope — In
- `app/routes/solve.py` with `/api/solve`, `/api/jobs`, `/api/jobs/:id`, `/api/jobs/:id/stream`
- `app/pipeline/orchestrator.py` — runs the five-stage pipeline
- `app/pipeline/events.py` — emits SSE events to a per-job queue
- `app/optimization/interpreter.py` — converts SampleSet to human-readable solution using the variable registry
- BYOK key endpoints: `/api/keys` GET/PUT/DELETE
- Job ownership enforcement: `user_id` filter on all reads

### Scope — Out
- Frontend (Phase 5)
- Async queue (Phase 6) — pipeline runs in the request handler thread for now
- Rate limiting (Phase 7)

### Files to Create / Modify

```
backend/
├── app/routes/solve.py             # NEW
├── app/pipeline/__init__.py        # NEW
├── app/pipeline/orchestrator.py    # NEW
├── app/pipeline/events.py          # NEW
├── app/optimization/interpreter.py # NEW
└── tests/
    ├── test_pipeline.py            # end-to-end with mocked LLM
    ├── test_routes_solve.py        # HTTP-level tests
    └── test_routes_keys.py
```

### API Contracts

**POST /api/solve**
```json
Request:
{
  "problem_statement": "Schedule 3 jobs on 2 machines minimizing makespan...",
  "provider": "claude",
  "use_stored_key": true               // OR provide "api_key": "sk-..."
}

Response (immediate):
{
  "success": true,
  "job": {
    "id": "9f2e...",
    "status": "queued",
    "created_at": "2026-05-06T12:34:56Z"
  }
}

Errors:
  400 - missing fields, problem_statement too long
  401 - not authenticated
  402 - no API key on file for provider and none supplied
  429 - rate limited (Phase 7)
```

**GET /api/jobs/:id**
```json
Response:
{
  "job": {
    "id": "9f2e...",
    "status": "complete",
    "problem_statement": "...",
    "provider": "claude",
    "cqm_json": { ... },
    "variable_registry": { ... },
    "validation_report": { "passed": true, "warnings": [] },
    "solution": { "x_job1_A": 1, ... },
    "interpreted_solution": "Job 1 runs on Machine A starting at t=0...",
    "num_variables": 12,
    "num_constraints": 5,
    "solve_time_ms": 1240,
    "created_at": "...",
    "completed_at": "..."
  }
}
```

**GET /api/jobs/:id/stream** — Server-Sent Events
```
event: status
data: {"status": "formulating"}

event: status
data: {"status": "compiling", "num_variables": 12, "num_constraints": 5}

event: status
data: {"status": "validating"}

event: status
data: {"status": "solving"}

event: status
data: {"status": "complete", "solve_time_ms": 1240}
```

### Pipeline Orchestrator Pseudocode

```python
async def run_pipeline(job_id: str, user_id: int, problem_statement: str,
                       provider_name: str, api_key: str, emit_event):
    try:
        emit_event(job_id, 'formulating')
        provider = get_provider(provider_name)
        result = await provider.formulate(problem_statement, api_key)
        update_job(job_id, cqm_json=json.dumps(result.cqm_json),
                   variable_registry=json.dumps(result.variable_registry))

        emit_event(job_id, 'compiling')
        cqm, registry = compile_cqm_json(result.cqm_json)

        emit_event(job_id, 'validating')
        report = validate_cqm(cqm, expected_optimum=result.cqm_json
                              .get('test_instance', {}).get('expected_optimum'))
        if not report.passed:
            raise PipelineError(f"Validation failed: {report.warnings}")

        emit_event(job_id, 'solving')
        bqm, invert = dimod.cqm_to_bqm(cqm, lagrange_multiplier=10.0)
        sampler = GPUSimulatedAnnealingSampler()
        sampleset = sampler.sample(bqm, num_reads=1000)
        solution = sampleset.first.sample

        interpreted = interpret_solution(solution, registry, cqm)
        update_job(job_id, status='complete',
                   solution=json.dumps(solution),
                   interpreted_solution=interpreted,
                   completed_at=now())
        emit_event(job_id, 'complete')

    except Exception as e:
        update_job(job_id, status='error', error=str(e))
        emit_event(job_id, 'error', error=str(e))
```

### Tests Required

```
test_pipeline.py:
  - test_pipeline_end_to_end_with_mocked_llm   # uses a stub provider
  - test_pipeline_handles_formulation_error
  - test_pipeline_handles_validation_failure
  - test_pipeline_handles_solver_timeout

test_routes_solve.py:
  - test_solve_requires_auth
  - test_solve_with_invalid_provider_returns_400
  - test_solve_creates_job_in_db
  - test_get_job_owner_only
  - test_get_job_admin_can_see_any
  - test_delete_job_removes_from_db
  - test_jobs_list_paginated_and_filtered_by_user

test_routes_keys.py:
  - test_put_key_stores_encrypted
  - test_get_keys_lists_providers_no_values
  - test_delete_key_removes
```

### Definition of Done

- [ ] All tests pass
- [ ] `curl -X POST http://localhost:5009/api/solve -H "Cookie: session=..." -d '{"problem_statement": "knapsack...", "provider": "claude", "use_stored_key": true}'` returns a job_id
- [ ] Polling `GET /api/jobs/:id` shows the job advancing through statuses
- [ ] Connecting to `/api/jobs/:id/stream` with `curl -N` shows SSE events in real time
- [ ] Final `interpreted_solution` is human-readable and references the original problem terms
- [ ] User A cannot see User B's job via direct ID access (returns 404)

---

## Phase 5 — Frontend Solve UI

### Goal
A complete user-facing UI for submitting problems, watching progress, and viewing results.

### Scope — In
- `views/MainApp.vue` becomes the solve interface
- All components in the project structure (`ProblemInput`, `SolveStatus`, `ResultDisplay`, `JobHistory`, `ApiKeyManager`, `PhaseStatusBadge`)
- `views/JobDetailPage.vue` for re-viewing a past job
- Pinia store `stores/solve.ts`
- SSE-driven status updates
- API key management UI (BYOK input form, list of stored providers)

### Scope — Out
- Admin UI (Phase 7)
- Mobile-specific layouts (basic responsive only)

### Components Specification

**`ProblemInput.vue`**
- Large textarea with character counter (max 8000)
- Provider dropdown: Claude / OpenAI / Local
- "Use stored key" checkbox (default on if user has a key for selected provider)
- "API key" password input (visible only when checkbox unchecked)
- "Solve" button (disabled while a job is running for this user)
- Cost estimate displayed before submission ("≈ $0.03 for Claude")

**`SolveStatus.vue`**
- Vertical timeline with 5 stages: Formulating → Compiling → Validating → Solving → Done
- Active stage shows a spinner; completed stages show a green check; future stages are dimmed
- Live elapsed time since job started
- Cancel button (Phase 6 — stub for now)

**`ResultDisplay.vue`**
- Top section: `interpreted_solution` rendered as Markdown
- Tabs: "Solution" / "CQM" / "Validation Report" / "Raw LLM Output"
- "CQM" tab uses a JSON viewer (collapsible nodes)
- "Validation Report" tab shows per-layer pass/fail with expandable details
- Copy-to-clipboard buttons on each section

**`JobHistory.vue`**
- Table with columns: Created, Problem (truncated), Provider, Status, Variables, Solve Time
- Click row → navigates to `/jobs/:id`
- Status filter chip
- "Delete" action per row with confirmation

**`ApiKeyManager.vue`**
- Section in user settings showing stored providers (no key values displayed)
- Form to add/update a key per provider
- Delete button per provider

### Definition of Done

- [ ] Submit a problem, watch the timeline animate through 5 stages, see the final solution rendered
- [ ] Submit a malformed problem ("write me a poem"), see a clear error
- [ ] View an old job from history, see the same UI populate from stored data
- [ ] Add an OpenAI key, submit using "Use stored key", confirm it works
- [ ] Mobile (≥360px width) doesn't break layout
- [ ] No console errors during the full flow
- [ ] Lighthouse accessibility score ≥90 on the main view

---

## Phase 5B — Problem Template Library

> **v2 status:** Reframed under v2 as the **curriculum seed for Learn mode**. The v1 specification below is preserved — the template JSON files, the gallery UI, the match-badge feature all remain. v2 adds curriculum metadata (learning objectives, prerequisite chains, assessment hooks) layered on top of the v1 schema.

### v2 additions

The v1 template schema is extended with optional fields that mark templates as learning resources:

```json
{
  "id": "knapsack_classic",
  // ... all v1 fields preserved ...
  "module": {
    "module_id": "qubo_foundations",
    "lesson_id": "binary_selection_with_capacity",
    "order": 1,
    "learning_objectives": [
      "Recognize the binary-selection-under-budget pattern",
      "Translate verbal capacity constraints into Lagrange penalty terms",
      "Compare exact and heuristic solver behavior on the same problem"
    ],
    "prerequisites": ["intro_qubo"],
    "assessment_hooks": {
      "expected_optimum": 21,
      "feasibility_required": true,
      "minimum_solvers_must_agree": 2
    }
  }
}
```

Templates without a `module` field continue to function exactly as in v1 — they appear in the gallery and can be run with one click, but they're not part of any structured Module. Templates with a `module` field additionally appear in the Modules library (the Learn mode UI, sketched in Phase 5C-adjacent work) and can be assigned as exercises in cohort deployments.

### v2 Definition of Done — additions

All v1 DoD items remain. Plus:

- [ ] At least 5 of the 10 required templates have `module` metadata populated, forming a coherent introductory Module ("QUBO Foundations" or similar)
- [ ] The template registry validates the v2-extended schema without rejecting v1-only templates
- [ ] The Modules-library frontend view (a sibling of the v1 gallery) displays templates grouped by module_id with prerequisite chains visualized

### v2 Note: Modules vs Templates terminology

Per the v2 terminology decision, **"Modules"** refers to structured learning units (groups of related templates with learning objectives). **"Templates"** continues to mean individual problem JSON files. A Module is a collection of Templates plus pedagogical metadata. This distinction matters in code (`module.py` vs `template.py`) and in UI (Modules library page vs Template gallery page).

---

### v1 Phase 5B specification (preserved for audit trail)

### Goal

A curated library of 10+ example optimization problems that users can browse by category, view in detail, and run with one click. The system compares the user's result to the template's expected optimum and shows a ✓ / ✗ correctness badge — turning each template into a self-validating tutorial.

### Why This Phase Exists

A new user landing on a blank textarea has no idea what to type. Examples are the difference between "looks impressive" and "I just got it to work." They also serve as **regression tests for the formulation pipeline** — if a known-good template's expected optimum stops being reproducible, something in the LLM prompt or solver pipeline has degraded. Templates are simultaneously onboarding, marketing, education, and CI signal.

### Scope — In

- JSON-file-based template registry (no DB table — keeps templates version-controlled in the repo)
- 10+ curated templates spanning beginner → advanced and multiple problem domains
- Backend endpoints for listing and fetching templates
- Frontend gallery page with category and difficulty filters
- Detail modal with "Try this example" one-click submit
- Match badge on result that compares the solver's output to the template's expected optimum
- A small "New here? Start with an example" banner on the main solve page for users with no jobs yet

### Scope — Out

- User-submitted public templates (admin-only template editing comes in Phase 7)
- Template versioning, comments, ratings
- Per-tier or per-provider template variants
- Public (unauthenticated) access to the gallery (deferred to Phase 7 hardening)

### Files to Create

```
backend/
├── app/templates/__init__.py
├── app/templates/registry.py              # loads + validates JSONs at startup
├── app/templates/schemas/template_v1.json
├── app/templates/library/
│   ├── knapsack_classic.json
│   ├── number_partitioning.json
│   ├── set_cover_simple.json
│   ├── max_cut_6node.json
│   ├── tsp_5cities.json
│   ├── graph_coloring_planar.json
│   ├── bin_packing_basic.json
│   ├── jss_3job_3machine.json
│   ├── nurse_rostering_small.json
│   └── portfolio_3asset.json
├── app/routes/templates.py
└── tests/
    ├── test_template_registry.py
    └── test_routes_templates.py

frontend/
├── src/views/TemplateGalleryPage.vue
├── src/components/TemplateCard.vue
├── src/components/TemplateDetailModal.vue
├── src/components/TemplateMatchBadge.vue
└── src/stores/templates.ts
```

### Template Schema (`schemas/template_v1.json`)

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": [
    "id", "title", "category", "difficulty",
    "summary", "problem_statement", "expected_optimum"
  ],
  "properties": {
    "id":                { "type": "string", "pattern": "^[a-z0-9_]+$" },
    "title":             { "type": "string" },
    "category":          { "enum": ["allocation", "scheduling", "routing", "graph", "finance", "logic"] },
    "difficulty":        { "enum": ["beginner", "intermediate", "advanced"] },
    "tags":              { "type": "array", "items": { "type": "string" } },
    "summary":           { "type": "string", "maxLength": 200 },
    "problem_statement": { "type": "string" },
    "real_world_example":{ "type": "string" },
    "expected_pattern": {
      "type": "object",
      "properties": {
        "variables":   { "type": "string" },
        "objective":   { "type": "string" },
        "constraints": { "type": "string" }
      }
    },
    "expected_optimum":          { "type": ["number", "null"] },
    "expected_solution_summary": { "type": "string" },
    "learning_notes":            { "type": "string" },
    "estimated_solve_time_seconds": { "type": "integer" },
    "min_solver_tier":           { "enum": ["cpu_sa", "gpu_sa", "sqbm_plus"] }
  }
}
```

### Required Templates (10 minimum)

The library must cover this breadth at first ship:

| ID | Category | Difficulty | Pattern Demonstrated |
|----|----------|------------|----------------------|
| `knapsack_classic` | allocation | beginner | Binary selection under linear capacity |
| `number_partitioning` | logic | beginner | Spin variables, balance objective |
| `set_cover_simple` | logic | beginner | Set covering with binary indicators |
| `max_cut_6node` | graph | beginner | Quadratic objective, no constraints |
| `tsp_5cities` | routing | intermediate | Permutation via one-hot encoding |
| `graph_coloring_planar` | graph | intermediate | Multi-color discrete with conflict constraints |
| `bin_packing_basic` | allocation | intermediate | Multiple capacity constraints |
| `jss_3job_3machine` | scheduling | advanced | Time-indexed precedence + machine conflicts |
| `nurse_rostering_small` | scheduling | advanced | Multi-shift soft + hard constraints |
| `portfolio_3asset` | finance | advanced | Markowitz quadratic with budget |

### Example Template — `knapsack_classic.json`

```json
{
  "id": "knapsack_classic",
  "title": "0/1 Knapsack Problem",
  "category": "allocation",
  "difficulty": "beginner",
  "tags": ["binary", "linear", "single-constraint"],
  "summary": "Pack items into a knapsack without exceeding capacity, maximizing total value.",
  "problem_statement": "I have 5 items with weights [2, 3, 4, 5, 9] and values [3, 4, 5, 8, 10]. My knapsack can hold at most 20 units of weight. Which items should I pack to maximize total value?",
  "real_world_example": "A delivery driver deciding which packages to load given a vehicle weight limit, or a hiker choosing supplies for a trip with a backpack capacity.",
  "expected_pattern": {
    "variables": "x_i in {0, 1} — whether item i is selected",
    "objective": "maximize sum(value_i * x_i)",
    "constraints": "sum(weight_i * x_i) <= 20"
  },
  "expected_optimum": 21,
  "expected_solution_summary": "Items 0, 1, 2, 3 are packed (total weight 14, total value 21). Item 4 is too heavy.",
  "learning_notes": "The canonical 'select subset under budget' QUBO. The capacity constraint becomes a Lagrange penalty in the BQM. Recognize this pattern in: portfolio selection, feature selection, resource allocation.",
  "estimated_solve_time_seconds": 1,
  "min_solver_tier": "gpu_sa"
}
```

### Example Template — `max_cut_6node.json`

```json
{
  "id": "max_cut_6node",
  "title": "Max-Cut on a 6-Node Graph",
  "category": "graph",
  "difficulty": "beginner",
  "tags": ["spin", "quadratic", "unconstrained"],
  "summary": "Partition a graph's nodes into two groups to maximize the number of edges crossing between them.",
  "problem_statement": "I have a graph with 6 nodes labeled 0 through 5 and edges: (0,1), (0,2), (1,3), (2,3), (2,4), (3,5), (4,5). Partition the nodes into two groups so that the number of edges crossing between the groups is maximized. Tell me which nodes go in each group.",
  "real_world_example": "Designing a two-team tournament so that as many rivalries as possible are between teams (not within). Or partitioning a circuit into two chips to minimize cross-chip wires (the dual problem).",
  "expected_pattern": {
    "variables": "s_i in {-1, +1} — which group node i belongs to",
    "objective": "maximize sum over edges (i,j) of (1 - s_i * s_j) / 2",
    "constraints": "none — pure quadratic objective"
  },
  "expected_optimum": 6,
  "expected_solution_summary": "Group A = {0, 3, 4} and Group B = {1, 2, 5} (or its complement). 6 of the 7 edges cross.",
  "learning_notes": "Max-Cut is the textbook example of a quadratic-objective, constraint-free QUBO. It maps directly to Ising form. The 'group A vs group B' symmetry means there are always at least two equivalent optima.",
  "estimated_solve_time_seconds": 1,
  "min_solver_tier": "gpu_sa"
}
```

### Example Template — `jss_3job_3machine.json`

```json
{
  "id": "jss_3job_3machine",
  "title": "Job-Shop Scheduling: 3 Jobs on 3 Machines",
  "category": "scheduling",
  "difficulty": "advanced",
  "tags": ["integer", "precedence", "no-overlap", "makespan"],
  "summary": "Schedule 3 jobs through 3 machines in a fixed processing order, minimizing makespan.",
  "problem_statement": "I have 3 jobs that must each be processed on Machines A, B, and C in that fixed order. Processing times are: Job 1 = (3, 2, 2), Job 2 = (2, 1, 4), Job 3 = (4, 3, 1) — meaning Job 1 takes 3 time units on Machine A, then 2 on B, then 2 on C. Each machine can only process one job at a time. Find the schedule (start times) that minimizes the time at which all jobs finish (the makespan).",
  "real_world_example": "A small machine shop with three workstations and three pending orders, each of which must visit all three stations in sequence.",
  "expected_pattern": {
    "variables": "start_jm — integer start time of job j on machine m",
    "objective": "minimize makespan (max over j of finish time)",
    "constraints": "(1) precedence: start_j(B) >= start_j(A) + duration_jA, etc. (2) no-overlap: for each machine, no two jobs occupy it simultaneously"
  },
  "expected_optimum": 11,
  "expected_solution_summary": "One optimal schedule: Job 2 starts first, then Job 1, then Job 3. Makespan = 11.",
  "learning_notes": "Job-shop scheduling exposes the friction between CQM and dwave-optimization NL formulations: in CQM, no-overlap is many quadratic constraints; in NL, ListVariable encodes the permutation structurally. This template should solve correctly via either path, which is itself a useful invariant to verify.",
  "estimated_solve_time_seconds": 5,
  "min_solver_tier": "gpu_sa"
}
```

### API Contracts

**GET /api/templates**

```json
Response:
{
  "templates": [
    {
      "id": "knapsack_classic",
      "title": "0/1 Knapsack Problem",
      "category": "allocation",
      "difficulty": "beginner",
      "tags": ["binary", "linear", "single-constraint"],
      "summary": "Pack items into a knapsack without exceeding capacity, maximizing total value.",
      "estimated_solve_time_seconds": 1
    }
  ],
  "categories": [
    { "name": "allocation", "count": 2 },
    { "name": "scheduling", "count": 2 },
    { "name": "routing", "count": 1 },
    { "name": "graph", "count": 2 },
    { "name": "finance", "count": 1 },
    { "name": "logic", "count": 2 }
  ]
}
```

**GET /api/templates/:id** — returns the full template JSON object.

**POST /api/solve/from-template/:id**

```json
Request:
{ "provider": "claude", "use_stored_key": true }

Response:
{
  "success": true,
  "job": {
    "id": "9f2e...",
    "status": "queued",
    "template_id": "knapsack_classic",
    "expected_optimum": 21
  }
}

Errors:
  404 — unknown template id
  402 — no API key for the chosen provider
  429 — rate limited
```

The `jobs` table gains optional `template_id` and `expected_optimum` columns. When set, the result page automatically renders the match badge.

### UI Specifications

**TemplateGalleryPage.vue**
- Route: `/templates`, requires auth
- Header: "Example Problems" with subtitle "Browse curated optimization problems and try them with one click"
- Filter row: category chips (all + per-category), difficulty toggle group, search box for title/tag
- Card grid: 3 columns desktop, 2 tablet, 1 mobile (Vuetify responsive grid)
- Empty-state when no results match: "No examples match your filters. Try clearing them."

**TemplateCard.vue**
- Vuetify v-card with hover lift
- Top row: difficulty chip (color-coded — green=beginner, amber=intermediate, red=advanced) + category chip
- Title (h6)
- Summary (2 lines max, ellipsis if longer)
- Bottom row: tag chips (max 3 visible, "+N more" for the rest)
- Click anywhere on card → opens TemplateDetailModal

**TemplateDetailModal.vue**
- Vuetify v-dialog, max-width 800px
- Header: title + difficulty badge + category
- Body sections (each in a v-card variant="tonal"):
  - Summary
  - Problem Statement (monospace block, copyable)
  - Real-World Example
  - Expected Pattern (Variables, Objective, Constraints — each in monospace)
  - Learning Notes
- Footer:
  - "Copy problem statement" v-btn variant="outlined"
  - "Try this example" v-btn color="primary" — calls `POST /api/solve/from-template/:id`, closes modal, navigates to the new job's detail page

**TemplateMatchBadge.vue**
- Three states:
  - **Match:** green chip "✓ Matches expected optimum: 21"
  - **Mismatch:** amber chip "✗ Got 18 (expected 21)" with tooltip explaining possible causes (LLM picked a different valid formulation, solver didn't reach optimum, expected value is wrong)
  - **No expected:** hidden
- Renders only on jobs with non-null `template_id`

### Frontend Routing & Navigation Updates

- Add `/templates` route, `requiresAuth: true`
- Add "Examples" link in app bar (visible to all authenticated users) — navigates to `/templates`
- On main page (`/`), if user has no jobs, show banner: "New here? Start with an example →" linking to `/templates`

### Tests Required

```
test_template_registry.py:
  - test_loads_all_template_files_at_startup
  - test_validates_each_template_against_schema
  - test_unique_ids
  - test_required_templates_present              # The 10 from the table above
  - test_get_by_id_returns_full_object
  - test_get_by_id_unknown_returns_none
  - test_categories_aggregated_correctly

test_routes_templates.py:
  - test_get_templates_requires_auth
  - test_get_templates_returns_summary_list
  - test_get_template_by_id_returns_full
  - test_get_template_by_id_404
  - test_solve_from_template_creates_job_with_template_id
  - test_solve_from_template_unknown_id_returns_404
  - test_solve_from_template_sets_expected_optimum
  - test_categories_endpoint_returns_counts
```

### Definition of Done

- [ ] All 10 required templates present in `app/templates/library/` and validated against the schema at app startup
- [ ] `GET /api/templates` returns the full list with category counts
- [ ] `GET /api/templates/knapsack_classic` returns a complete template object
- [ ] `POST /api/solve/from-template/knapsack_classic` creates a job with `template_id` and `expected_optimum` set
- [ ] Frontend `/templates` page renders the gallery with working category/difficulty/search filters
- [ ] Clicking a card opens the detail modal with all six sections populated
- [ ] "Try this example" button creates a job and navigates to its detail page
- [ ] Result page on a template-derived job shows TemplateMatchBadge with the correct comparison
- [ ] App bar has "Examples" link that navigates to `/templates`
- [ ] Empty-state banner appears on `/` when user has zero jobs
- [ ] All tests pass

### Manual Walkthrough Script (for QA)

A new user's first session should look like this end-to-end:

1. Sign up for a fresh account
2. See banner on home page: "New here? Start with an example →"
3. Click banner → arrive at `/templates`
4. Filter to "beginner" difficulty → see 4 cards
5. Click the Knapsack card → modal opens
6. Read summary, problem statement, expected pattern, learning notes
7. Click "Try this example" → modal closes, job submits, page navigates to job detail
8. Watch the 5-stage timeline animate through formulating → compiling → validating → solving
9. See result with green "✓ Matches expected optimum: 21" badge
10. Read interpreted solution: "Pack items 0, 1, 2, 3 for total value 21"

If steps 1–10 work end-to-end without console errors and without manual recovery, the phase is done.

### Why Templates Are Also Regression Tests

Once Phase 5B ships, every template's `expected_optimum` becomes an automated check:

- An admin endpoint (`POST /api/admin/templates/regression-check`) runs every template through the pipeline and reports which ones still match expected
- This catches regressions in the LLM prompt, the formulation schema, the validation harness, or the solver — any of which could silently break correctness on real user problems
- Run it as a scheduled job (daily) and as a CI gate before deploying changes to the formulation prompts
- Treat any template that flips from match to mismatch as a P0 bug

This is a free, durable correctness signal that pays dividends across every later phase.

---

## Phase 5C — Benchmark Dashboard

> **v2 status:** New under v2. Has no v1 equivalent. Depends on the Benchmarks foundation built in Phase 2.

### Goal

A public dashboard view that displays Benchmark results across solver tiers, instance suites, and time. The dashboard reads from the `benchmarks/archive/` directory populated by Phase 2's CLI runner — the UI is a presentation layer, not a benchmark execution engine.

### Why this phase exists

Phase 2 produces RunRecords; without a way to display them, the records are inert. Phase 5C is what turns the platform from "a tool that solves QUBOs" into "a public scoreboard for QUBO solvers." This is the deliverable that distinguishes CiRA Quantum from every other quantum-software platform — none of them publish honest, longitudinal, cross-vendor Benchmarks.

### Scope — In

- **Suite view:** for each instance suite (e.g., `maxcut/gset_subset`), a table with rows = instances, columns = solvers, cells = best-energy / time-to-best.
- **Solver view:** for each registered solver, a profile page showing its performance across all suites it's been run on, with version history.
- **Instance view:** for each instance, a leaderboard of every solver that's ever run on it, sorted by best energy and then by time.
- **Time-series view:** for each (solver, suite) pair, a line chart of best energy over time as the solver's version advances. This is the longitudinal asset that compounds.
- **Reproducibility metadata:** every cell in every table links to the underlying RunRecord with full reproducibility hash and citation generator.
- **Public access:** Benchmarks are public-readable without login; submitting new runs requires authentication.

### Scope — Out

- Running new Benchmarks from the UI (use the CLI from Phase 2)
- Real-time leaderboard updates (refresh on a schedule, e.g., every 6 hours)
- Editing or deleting historical records (records are append-only)

### Files to Create

```
backend/
├── app/routes/benchmarks.py             # /api/benchmarks/*
└── tests/
    ├── test_routes_benchmarks.py
    └── fixtures/benchmark_records/       # synthetic records for UI testing

frontend/
├── src/views/BenchmarkDashboardPage.vue  # main dashboard
├── src/views/BenchmarkSuitePage.vue       # one suite, all solvers
├── src/views/BenchmarkSolverPage.vue      # one solver, all suites
├── src/views/BenchmarkInstancePage.vue    # one instance, leaderboard
├── src/components/SolverComparisonTable.vue
├── src/components/PerformanceTimeSeries.vue
├── src/components/ReproducibilityBadge.vue
├── src/components/CitationButton.vue
└── src/stores/benchmarks.ts
```

### API Contracts

**GET /api/benchmarks/suites** — list of instance suites with metadata
**GET /api/benchmarks/suites/:suite_id** — full suite with all RunRecords
**GET /api/benchmarks/solvers/:solver_id** — solver profile + record history
**GET /api/benchmarks/instances/:instance_id** — leaderboard for one instance
**GET /api/benchmarks/records/:record_id** — full RunRecord
**GET /api/benchmarks/records/:record_id/cite** — BibTeX/citation string

All endpoints are public (no auth required for read).

### Definition of Done

- [ ] Dashboard at `/benchmarks` displays at least three suites with at least three solvers run on each
- [ ] Clicking a cell navigates to the underlying RunRecord with full metadata
- [ ] "Cite this record" button produces a working BibTeX entry
- [ ] Time-series chart renders for at least one (solver, suite) pair with multiple historical runs
- [ ] Public URL is reachable without login
- [ ] Page load (cold) under 3 seconds for the suite view; uncached but pre-aggregated

### Why this phase pays off long-term

Each Benchmark run added makes the dashboard more valuable; each new solver registered makes existing Benchmarks more interesting; each year of operation builds longitudinal data nobody else has. **This is the asset that compounds and is the platform's primary public face.** Treat it accordingly — design for 10-year preservation.

---

## Phase 6 — Async Job Queue & GPU Contention

### Goal
Two users can submit jobs simultaneously without crashing the GPU. The second job queues, the first completes, the second runs. The frontend reflects "queued" state cleanly.

### Scope — In
- A worker process that consumes a queue (Redis + RQ, or a simple SQLite-backed queue for V1)
- The Flask request handler enqueues jobs and returns immediately
- A single GPU lock — only one solve runs on the GPU at a time
- Frontend SSE handles `status: 'queued'` and shows queue position

### Scope — Out
- Distributed queueing (single worker is fine)
- Priority queues, fair scheduling beyond FIFO

### Implementation Notes

For V1, use **RQ + Redis** (Redis on `localhost:6379`, deployed via Docker for ease). Worker process is `python -m app.worker` running on the same machine as the Flask app.

```python
# app/worker.py
from rq import Worker, Queue, Connection
from redis import Redis

def gpu_solve_job(job_id, user_id, problem_statement, provider, api_key):
    # acquire GPU lock (file lock or Redis lock)
    with gpu_lock():
        run_pipeline(job_id, user_id, problem_statement, provider, api_key, emit_event)

if __name__ == '__main__':
    redis_conn = Redis()
    with Connection(redis_conn):
        worker = Worker([Queue('solve')])
        worker.work()
```

### Definition of Done

- [ ] Submit two jobs back-to-back from two different sessions; both complete without GPU OOM
- [ ] Second job shows "queued (position 1)" in UI before transitioning to "formulating"
- [ ] Killing the worker process cleanly leaves jobs in `queued` state (no corruption)
- [ ] `docker-compose up` brings up Redis + worker + backend + frontend
- [ ] Tests: submit 5 jobs, verify all 5 complete in order

---

## Phase 7 — Public Web App Hardening

### Goal
The app is safe to expose to the public internet.

### Scope — In
- **Rate limiting:** per-user (1 concurrent job, 10/hour) and per-IP (signup limits)
- **API key encryption verified:** keys never appear in logs, never returned plaintext
- **Abuse detection:** flag suspicious problem statements (length, prompt-injection patterns)
- **Email verification on signup:** magic-link via Resend or Postmark
- **Admin dashboard:** view all users, jobs, ban/unban users, see usage stats
- **Privacy policy & ToS** pages (markdown-rendered)
- **HTTPS in production:** secure cookies, `SESSION_COOKIE_SECURE=True`
- **Cloudflare Tunnel** setup for exposing the GPU host
- **Error tracking:** Sentry or similar
- **Rate-limit middleware** for unauthenticated endpoints (signup, login)

### Scope — Out
- Payment / billing
- Multi-region deployment

### Files to Create

```
backend/
├── app/middleware/
│   ├── rate_limit.py
│   └── abuse_detection.py
├── app/routes/admin.py             # full admin endpoints
└── app/email.py                    # magic-link sending

frontend/
├── src/views/AdminPage.vue
├── src/views/PrivacyPolicyPage.vue
├── src/views/TermsOfServicePage.vue
├── src/views/EmailVerifyPage.vue
└── src/components/AdminUserTable.vue
└── src/components/AdminJobTable.vue

deployment/
├── docker-compose.yml
├── Dockerfile.backend
├── Dockerfile.worker
├── nginx.conf
└── cloudflared-config.yml
```

### Abuse Detection Heuristics (V1, simple)

A problem statement is flagged for admin review (job runs but flagged in DB) if any of:
- Contains "ignore previous instructions" or known prompt-injection markers
- Is shorter than 30 characters
- Has fewer than 3 distinct words
- Repeats the same word ≥10 times
- Has been submitted >3 times by the same user in 1 hour (likely abuse retry)

### Definition of Done

- [ ] Sign up requires email verification before first solve
- [ ] Submitting >10 jobs/hour returns 429
- [ ] Pen-test by team: try SQL injection, XSS, CSRF, prompt-injection — all fail safely
- [ ] Logs grep for known API key prefixes (`sk-`, `claude-`) returns zero hits
- [ ] Public URL (Cloudflare Tunnel) is reachable from outside the network
- [ ] HTTPS is enforced (HTTP redirects to HTTPS)
- [ ] Admin can view, search, and ban users from the dashboard

---

## Phase 8 — Classical Solver Tiers (v2 reordered)

> **v2 status:** New under v2, replacing the slot v1 had given to SQBM+ integration. The v1 Phase 8 (SQBM+) is renumbered as Phase 9 with its specification preserved below.

### Goal

Add **OR-Tools CP-SAT** and **HiGHS** (both free, open-source) as registered solver tiers in the Benchmark registry. These are the classical baselines that make CiRA Quantum's Benchmarks honest — without them, every comparison is "QUBO solver vs QUBO solver" rather than "QUBO solver vs the actual SOTA classical alternative."

### Why this phase came first under v2

For the v2 educational and benchmarking positioning, **classical baselines are not optional.** A Benchmark that doesn't include CP-SAT for scheduling problems or HiGHS for MIP problems is not a defensible Benchmark — it tells students and researchers that QUBO solvers won when they didn't. Phase 8 is the truth-in-advertising phase.

This was the most consequential v1 → v2 change. v1 ordered Phase 8 as SQBM+ specifically because the v1 narrative was about quantum-inspired solving. v2's narrative is about honest comparison, which requires the classical SOTA to be in the picture.

### Scope — In

- **CP-SAT adapter** wrapping OR-Tools' CP-SAT solver as a `dimod.Sampler`-compatible class. Translates CQM constraints to CP-SAT model variables and constraints. Returns sample sets in the dimod format.
- **HiGHS adapter** wrapping the HiGHS open-source MIP solver similarly.
- Both adapters register with the Phase 2 solver registry.
- Documentation page explaining when each classical tier is the right answer (CP-SAT for scheduling/routing, HiGHS for MIP-shaped problems, GPU SA / quantum-inspired for unstructured QUBOs).
- At minimum, both classical tiers run on every existing Benchmark instance suite from Phase 2, with results in the dashboard.

### Scope — Out

- Gurobi or CPLEX (commercial; deferred to opt-in BYO-license in a later phase)
- LKH or Concorde (specialized for TSP; deferred)
- Customizing CP-SAT or HiGHS internals — use them as drop-in solvers via their official Python APIs

### Files to Create

```
backend/
├── app/optimization/cpsat_sampler.py     # OR-Tools CP-SAT adapter
├── app/optimization/highs_sampler.py     # HiGHS adapter
└── tests/
    ├── test_cpsat_sampler.py
    ├── test_highs_sampler.py
    └── test_cpsat_beats_gpu_sa_on_jss.py  # honesty check
```

### Definition of Done

- [ ] `register_solver` lists both `cpsat` and `highs` after import
- [ ] Both adapters pass the same tests as Phase 1's GPU SA (correctness on `tiny_5var`, agreement with `ExactCQMSolver`)
- [ ] Benchmark suite re-runs and the dashboard displays CP-SAT and HiGHS results
- [ ] At least one Benchmark explicitly demonstrates classical-beats-QUBO (e.g., CP-SAT solving a 3x3 JSS in <100ms while GPU SA takes >1s)
- [ ] Documentation page explains "when each tier is the right answer"

### Why this matters for the Learn mode

Phase 5B's Modules can now include lessons titled "When QUBO is wrong" — exercises where the student is shown the same problem solved by CP-SAT and GPU SA, and asked to identify why the classical solver wins. This is genuine education, only possible because classical baselines are first-class Benchmark tiers.

---

## Phase 9 — Quantum-Inspired Solver Tiers (v2 reordered, was Phase 8)

> **v2 status:** Renumbered from v1 Phase 8. The original SQBM+ specification is preserved below; v2 expands the scope to include Parallel Tempering and PQQA as solver tiers, building toward the full quantum-inspired roadmap discussed in earlier conversations.

### v2 Goal

Add three quantum-inspired solver tiers, in priority order:

1. **GPU Parallel Tempering** (Phase 9a) — extends the Phase 1 GPU SA with a temperature axis and replica-exchange step. Same memory layout, ~50× upgrade on hard problems. Cheapest expansion.
2. **PQQA via QQA4CO** (Phase 9b) — wraps the ICLR 2025 library as a `dimod.Sampler`. Pip-installable, PyTorch-native, complementary to GPU SA on large sparse problems.
3. **SQBM+ (Phase 9c, v1 Phase 8 content)** — the v1 specification preserved below. Commercial cloud-based bSB/dSB tier via REST adapter.

Each becomes a registered solver tier in the Benchmark registry. Each is opt-in (users select the tier; the platform doesn't dispatch automatically based on problem). All three appear in the dashboard alongside classical and GPU SA tiers.

### v2 Note on ordering within Phase 9

These three sub-phases can be done in any order based on user demand. PT is the cheapest if internal benchmarking shows GPU SA hitting limits. PQQA is the most aligned with future-hardware bets discussed in earlier conversations. SQBM+ is the most credible commercial benchmark partner. Pick based on which adds the most Benchmark value at the moment Phase 9 starts.

---

### v1 Phase 8 specification (preserved for audit trail) — SQBM+ Integration

### Goal
Add SQBM+ as a second solver tier, selectable per-job. Users with quality-sensitive problems can opt into the higher-cost solver.

### Scope — In
- `app/optimization/sqbm_plus.py` — SQBM+ REST adapter implementing the same `dimod.Sampler` interface
- BQM serialization to SQBM+ text/HDF5 format
- Solver tier selector in UI (`GPU SA` default, `SQBM+` opt-in)
- Per-tier cost display
- Admin can configure SQBM+ endpoint URL

### Files to Create

```
backend/
├── app/optimization/sqbm_plus.py
├── app/optimization/serializers.py    # BQM → SQBM+ format
└── tests/
    ├── test_sqbm_plus.py              # mocked HTTP
    └── test_serializers.py
```

### Definition of Done

- [ ] Same problem solved via GPU SA and SQBM+ produces solutions within 5% of each other on a benchmark suite
- [ ] UI shows "Recommended for ≥1000 variable problems" hint when user is on SQBM+ tier
- [ ] Failure of SQBM+ endpoint falls back to GPU SA with a warning, not a hard error

---

## Getting Started

### Prerequisites

- Node.js 18+
- Python 3.12+
- NVIDIA GPU with CUDA ≥12.8 (Blackwell sm_120 supported)
- PyTorch 2.10.0 with cu128
- Redis (for Phase 6+)

### Installation

```bash
# Backend
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Frontend
cd ../frontend
npm install
```

### Development

```bash
# Terminal 1: Backend
cd backend && python run.py
# → http://localhost:5009

# Terminal 2: Frontend
cd frontend && npm run dev
# → http://localhost:3011

# Terminal 3 (Phase 6+): Worker
cd backend && python -m app.worker

# Terminal 4 (Phase 6+): Redis
docker run -d -p 6379:6379 redis:7
```

### Production Build

```bash
cd frontend && npm run build
# Output in dist/

# Backend served via gunicorn
cd backend && gunicorn -w 4 -b 0.0.0.0:5009 'app:create_app()'
```

### Default Credentials

- Username: `admin`
- Password: `admin123`

**Change immediately after first login.**

---

## Typography & Fonts

Vuetify 3 default Roboto. To customize, follow the same pattern as CiRA Oculus (Google Fonts link in `index.html` + `defaults.global.font.family` in `main.ts`).

For this project, recommended: **Inter** for UI, **JetBrains Mono** for the CQM JSON viewer and any code blocks.

---

## AI Coder Workflow Notes

This template is designed to be handed to an AI coder (Claude Code, Cursor, etc.) one phase at a time. Some patterns that work well:

### Per-Phase Kickoff Prompt

```
Read PROJECT_TEMPLATE.md, focusing on Phase {N}. Implement the
deliverables listed under that phase exactly as specified.

Constraints:
- Follow the company conventions in the upper sections of the template
  (Vue 3 + Vuetify, Flask + SQLite, session auth)
- Do not implement scope items from any other phase
- Write the listed tests first; make them fail; then make them pass
- Do not skip the validation harness — it is the safety net for the
  whole system

When the phase's Definition of Done checklist is complete, stop and
report back.
```

### Per-Phase Verification Prompt

```
Phase {N} claims to be complete. Walk through each item in the
"Definition of Done" checklist. For each item, run the relevant
test or command and report the result. If any item fails, identify
why and propose a fix. Do not move to Phase {N+1}.
```

### Hard Rules for the AI Coder

1. **The validation harness (Phase 2) is non-negotiable.** Never skip it. Every CQM produced by an LLM must pass `validate_cqm()` before being solved.
2. **Default to classical solvers in tests.** Never put a real GPU solve in CI; use `dimod.ExactSolver` for ground truth and a small `SimulatedAnnealingSampler` run for sanity.
3. **No real LLM API calls in CI.** Mock all HTTP calls. Real-API integration tests run manually before phase sign-off.
4. **No hard-coded secrets.** All keys come from env vars with placeholder defaults that are obviously not real.
5. **Tests first.** For each new file with logic, write the test before the implementation.
6. **Prefer composition over inheritance** unless the abstract class is genuinely useful (the FormulationProvider is).
7. **No bullet points in user-facing text** unless the request specifically calls for a list.
8. **One PR per phase.** Don't bundle phases. The phase boundaries are deliberate review checkpoints.

### What to Do When the Spec Is Ambiguous

If a phase doesn't specify a detail (e.g., "what happens if the LLM returns invalid JSON twice in a row?"), make the simplest reasonable choice and **document it in a `DECISIONS.md` file at the project root**. Do not silently invent behavior. Examples:

```
Phase 3: When the LLM returns invalid JSON, the provider raises
FormulationError immediately with the raw output included in the
exception. No retry. The pipeline catches this and marks the job
as 'error' with status_code=422.
```

---

## Key Patterns Used

1. **Composition API** — Vue 3 `<script setup>` syntax everywhere
2. **Pinia stores** — `auth.ts` and `solve.ts`, no Vuex
3. **Route guards** — `requiresAuth`, `requiresGuest`, `requiresAdmin`
4. **Decorators** — `@login_required`, `@admin_required`
5. **Session-based auth** — Server-side sessions with secure cookies
6. **BYOK pattern** — User-supplied API keys, encrypted at rest
7. **Provider abstraction** — Pluggable LLM backends via abstract class
8. **Validation harness** — Multi-layer correctness checks for LLM-generated formulations
9. **Server-Sent Events** — Live pipeline status without WebSocket complexity
10. **Phase-gated delivery** — Each phase has explicit DoD; no parallel phase work

---

---

# Appendix A: Academic Roadmap (Roadmap, Not Committed)

This appendix lists capabilities that make sense only when specific academic conditions are met. They are documented as a roadmap so the project's direction is clear, but they are **not committed phases** — each becomes a real phase only when the underlying need materializes.

The trigger for each capability is documented so the team knows when to start scoping it as an actual phase.

## A.1 — Cohort Management (Phase 11 candidate)

**Trigger:** A faculty member (the project lead or a collaborator) confirms they will use CiRA Quantum in a course with at least 10 enrolled students within the next academic term.

**Scope when committed:**

- Cohort objects in the data model (course_id, term, instructor_id, student_ids)
- Instructor admin UI for assigning Modules to a cohort
- Student dashboard showing assigned Modules and progress
- Auto-grading of Module exercises via the validation harness
- Gradebook export to CSV / LMS-compatible formats
- FERPA compliance (US) or equivalent regional student-data protection — depends on which institutions use the platform first

**Why deferred:** Multi-tenant cohort infrastructure is heavy work that's wasted if no one uses it. Build it when a real cohort is committed.

## A.2 — Contribution Pipeline (Phase 10 candidate)

**Trigger:** Either (a) the project lead has graduate students whose theses require platform contributions, or (b) at least three external researchers have requested the ability to contribute solvers, instances, or Modules.

**Scope when committed:**

- GitHub-based contribution workflow (PR templates, contributor guide, CLA)
- Automated validation of contributed solver tiers (must register correctly, must pass canonical Benchmarks)
- Automated validation of contributed instances (schema check, optimum or best-known value verifiable)
- Automated validation of contributed Modules (curriculum schema check, exercise auto-grading)
- Public credit and citation generation for contributors

**Why deferred:** A contribution pipeline without contributors is waste. Build it when contributors arrive.

## A.3 — CiRA Quantum Reports (Publication Track)

**Trigger:** The team has produced at least three Benchmark studies worth publishing, or a research grant explicitly funds publication infrastructure.

**Scope when committed:**

- Periodic publication of curated Benchmark studies, methodology improvements, case studies
- Editorial process (the team or a small editorial board)
- Reproducibility audit — every published claim must be re-executable from the platform's archive
- DOI assignment for each publication
- Integration with Zenodo or institutional repository for long-term preservation

**Why deferred:** Publication infrastructure without publications is waste. Build it when the substance warrants it.

## A.4 — Grant-Funded Solver Tiers

**Trigger:** A grant or industry partnership explicitly funds the integration of a specific solver (e.g., "industry partner X funds a six-month engagement to add their proprietary solver to CiRA Quantum's Benchmark dashboard").

**Scope when committed:** Per-engagement, scoped to the specific solver and its Benchmark integration. May or may not include UI work depending on the partnership.

**Why this matters strategically:** Grant- and partnership-funded work is how the platform sustains itself in the long run without commercial ROI pressure. Each such engagement adds Benchmark value and funds platform maintenance. The architecture's plug-in nature (every solver is a `dimod.Sampler` adapter) makes this pattern feasible.

## A.5 — Multi-Institution Federation

**Trigger:** Three or more universities are running CiRA Quantum instances and want to share Benchmark data without sharing student data.

**Scope when committed:** Federation protocol for sharing Benchmark records across instances while keeping student/cohort data local. Aggregated dashboards across federated instances.

**Why deferred:** Federation is a problem that only exists at scale. Solve it when scale arrives.

---

# Appendix B: v1 → v2 Change Summary

For readers familiar with v1 of this spec, here is a concise summary of what changed.

## Positioning

| Aspect | v1 | v2 |
|---|---|---|
| Project framing | "Hybrid quantum-inspired optimization web app" | "Academic platform for learning and benchmarking quantum optimization" |
| Primary audience | Practitioners with optimization problems | Students, researchers, educators, with practitioners as one of three audiences |
| Primary value | Solve problems faster than alternatives | Teach honestly, benchmark reproducibly, position for future quantum impact |
| Differentiation | "We solve QUBOs" | "We are the honest scoreboard for quantum and classical optimization" |
| Time horizon | 1–2 year ROI | 5–10 year academic platform |

## Architecture

| Aspect | v1 | v2 |
|---|---|---|
| Validation harness | Internal correctness check | Foundation of public Benchmarks (same code, public surface) |
| Solver tier ordering | QUBO solvers first, classical never | Classical baselines (CP-SAT, HiGHS) at Phase 8, quantum-inspired at Phase 9 |
| Template library | Onboarding feature | Curriculum seed for Learn mode |
| Public surface | "Solve" mode | "Learn", "Benchmark", and "Solve" modes — three coordinated surfaces |
| Reproducibility | Not specified | Citation-grade RunRecords from Phase 2 onward |

## Phase plan

| Phase | v1 | v2 |
|---|---|---|
| 0 | Skeleton + auth | Same — shipped |
| 1 | GPU SA library | Same — shipped |
| 2 | Validation harness | Validation harness + Benchmarks foundation (substantially expanded) |
| 3 | Formulation providers | Same |
| 4 | Solve API | Same |
| 5 | Frontend solve UI | Same + Compare and Learn shell additions |
| 5B | Template library | Curriculum seed (renamed Modules in v2 terminology) |
| 5C | (did not exist) | Benchmark dashboard (NEW) |
| 6 | Async queue | Same |
| 7 | Public hardening | Same |
| 8 | SQBM+ | **Reordered: Classical solver tiers (CP-SAT, HiGHS)** |
| 9 | (did not exist) | Quantum-inspired tiers (PT, PQQA, SQBM+) |
| 10–11+ | (did not exist) | Roadmap appendix items (cohort management, contribution pipeline, etc.) |

## Terminology

| v1 term | v2 term | Notes |
|---|---|---|
| CiRA Optima | **CiRA Quantum** | Renamed throughout |
| Templates | Templates + **Modules** | Templates remain; Modules are structured groups of Templates with curriculum metadata |
| Comparison | **Benchmark** | Standardized to academic-literature terminology |
| Lessons / Tutorials | **Modules** | Standardized to a single term |

---

*Originally adapted from the CiRA Oculus project template — A SAM Dataset Annotation Platform — and re-positioned in v2 as the architecture and build specification for CiRA Quantum, an academic platform for learning, benchmarking, and experimenting with quantum and quantum-inspired optimization.*

*v2 spec produced May 2026 between Phase 1 delivery and Phase 2 kickoff. v1 content preserved inline within each phase for audit trail. v2 supersedes v1 where the two conflict.*
