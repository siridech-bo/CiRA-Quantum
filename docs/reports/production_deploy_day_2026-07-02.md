# Production Deploy Day — CiRA Quantum

**Date:** 2026-07-02
**Scope:** One long afternoon shipping the platform from a working dev environment on `.167` to a public production URL at `quantum.cira-core.com`, plus a self-contained offline demo on `.141` for a customer visit.
**Commits landed:** `f907575` → `11f5db4` (9 total)
**End state:** `https://quantum.cira-core.com/` serving optimization + QML + qLDPC modules; `http://192.168.1.141:5209/` running as a portable demo.

---

## 0 — Where we started

Before today, the platform was working end-to-end on the dev backend but had never been exposed publicly:

- **Number Partitioning `[4, 3, 2, 3, 1]` case that motivated hardcoded formulators** was resolved yesterday — Claude was consistently emitting the right structure (5 vars, 0 constraints, offset present) but drifted 17–32% on linear coefficients, producing an optimum of 5 or −39 instead of 1. The classifier→hardcoded routing layer (four families: max_cut, number_partitioning, max_independent_set, portfolio_selection) shipped in `f907575` and closed that whole class of bug by construction.
- **Approval gate**, **plain-English solution**, **real-QPU (Wukong 180-2)**, **template gallery split**, and **login on landing page** all shipped in `5d7a438` / `1759a31` / `f907575`.
- **Dev deploy** at `localhost:5009` was healthy but not user-facing.

Today's mission had three chapters: (1) get the last two solver bugs cleaned up, (2) figure out how to actually ship it to a public URL under `cira-core.com`, (3) prep a copy on a laptop so the customer meeting could happen offline.

---

## 1 — Solver bug fixes

Two "the pipeline isn't wrong, but the UI makes it look broken" bugs surfaced during the deploy shakedown.

### 1.1 `simulated_bifurcation → error` should have been `→ skipped`

**Symptom:** the solver comparison table had a red `error` chip on the `simulated_bifurcation` row for every single solve, even when the solve otherwise completed successfully. Users were confused.

**Root cause:** dropping the `simulated-bifurcation` PyPI package to shrink the Docker image (see §2.2 below) meant `SimulatedBifurcationSampler` never got registered in the solver registry. When the orchestrator's fan-out reached the `simulated_bifurcation` entry, `get_solver(name)` raised `KeyError`, which was caught and classified as `status="error"`. But that's the wrong shape: the pipeline correctly *cannot* run a solver whose dep isn't installed, exactly like it correctly *cannot* run a QAOA tier whose lowered qubit count exceeds the cap — and the qubit-cap case already returned `status="skipped"`.

**Fix** ([`backend/app/pipeline/orchestrator.py`](../../backend/app/pipeline/orchestrator.py:252-269)): the `KeyError` branch now returns `status="skipped"` with a clear "optional dependency not installed" message. This same pattern also cleans up `gpu_sa` (torch not installed in prod image) automatically.

**Also applies to:** any future optional-dep solver — the pattern generalizes.

### 1.2 QML real-QPU inference was hard-gated to 2-qubit jobs

**Symptom:** users training a VQC on Breast Cancer (Digits, or any high-dim dataset) were told *"Real-QPU inference is enabled for 2-qubit jobs — this run used more qubits — train a 2-qubit job to try the real-hardware path."* Even though Origin's Wukong-C180 has 12 qubits sitting right there.

**Root cause:** the QML training pipeline persisted a `scatter_points` field in the job's `metrics` JSON for the decision-boundary heatmap. Because heatmaps are inherently 2D, `scatter_points` had a hardcoded `{x, y, label, split}` shape. The real-QPU submit endpoint (`/api/qml/jobs/<id>/qpu/*`) then reused `scatter_points` as its X_test data source — so any job with `n_qubits > 2` had no reconstructable test split and the endpoint hard-rejected with HTTP 400.

**Fix** ([`backend/app/qml/trainer.py`](../../backend/app/qml/trainer.py) + [`backend/app/routes/qml.py`](../../backend/app/routes/qml.py)):
- **Trainer** now also persists a `test_split = {"X_test": [[...N-dim...]], "y_test": [...]}` field alongside `scatter_points`. Independent — `scatter_points` still exists for 2-qubit jobs so the decision-boundary heatmap keeps working.
- **`_load_test_split_from_metrics`** prefers the new N-dim field; falls back to `scatter_points` for jobs trained before today so old 2-qubit runs still have working QPU submits.
- **QPU cap preflight** added for the Origin path: `full_amplitude` cap 7, `WK_C180 / WK_C180_2 / HanYuan_01` cap 12. Rejects with `QUBIT_CAP_EXCEEDED` (400) if `n_features > cap` before submitting, instead of letting pyqpanda3 emit a cryptic circuit-compile failure downstream.
- **Frontend rename**: `has2dSplit` → `hasTestSplit`, gate on the more general condition. Alert copy rewritten for jobs that genuinely have no test split (old runs pre-2026-07-02): *"This job doesn't have a persisted test split. Re-train to enable real-QPU inference."*

**Test coverage:** `test_run_training_job_persists_metrics` now asserts `test_split` is present and matches `n_qubits × len(y_test)` shape. Passes inside the production Docker image.

---

## 2 — Docker image: from 15.4 GB to 3.62 GB in six rebuilds

The image had to fit within Cloudflare Containers' 2 GB compressed cap even though we ended up not using that path. Each rebuild taught us something.

### 2.1 First cut — `deploy/Dockerfile` (commit `23040c8`)

Multi-stage Python 3.12 slim build:
- **Stage 1 (builder):** full toolchain, `pip wheel` all deps into `/wheels`
- **Stage 2 (runtime):** slim base, `pip install --no-index --find-links=/wheels` from wheel cache
- **Extras included:** `[quantum,ibm-quantum]`
- **CMD:** `gunicorn --config /app/gunicorn_conf.py app:create_app()`

**Size:** **15.4 GB**. Culprit: `simulated-bifurcation` pulled `torch` which pulled the full CUDA toolkit (~2 GB of `nvidia-*` wheels).

### 2.2 Trim pass 1 — move optional/heavy solvers out of base

**`backend/pyproject.toml`:**
- `simulated-bifurcation` → new `[classical-extras]` extra
- `triton` (Windows + Linux markers) → new `[gpu]` extra (was carrying weight for a GPU SA path we'd already dropped from prod)

**Also fixed a latent bug:** `pyqpanda_alg`'s package `__init__` eagerly imports its QSVR submodule which needs `sklearn` at module load. Added `scikit-learn` to `[quantum]` extra to prevent import-time `ModuleNotFoundError` on the Origin QAOA path.

**Size:** **3.7 GB**.

### 2.3 Trim pass 2 — strip pyqpanda_alg's over-declared dev tooling

`pyqpanda_alg` transitively pulls in `sphinx-autoapi`, `mypy`, `astroid`, `ast-serialize`, and the full sphinx docs chain as runtime install_requires (upstream bug — these should be `dev_requires`). Added a `pip uninstall -y ...` step in the runtime stage to remove them.

**Size:** **2.47 GB uncompressed / 727 MB compressed**. Well under Cloudflare's 2 GB compressed cap.

### 2.4 Frontend build stage (commit `5a8eff4`)

The Vue SPA wasn't yet in the image — the runtime stage only copied `backend/`. Added a stage 1 (`frontend-builder`, Node 20) that runs `npx vite build` and produces `dist/`. Runtime stage copies `dist/` in and sets `CIRA_SPA_DIR=/app/frontend/dist` so Flask's `_register_spa()` mounts the SPA at `/` with fallback for Vue Router deep-links.

**Two gotchas along the way:**
- **Node 22** — vue-tsc 1.8's TypeScript-internals monkey-patch breaks against Node 22.23+'s bundled TS. Downgraded the build stage to Node 20 (LTS).
- **`vue-tsc` still crashed on Node 20** because our `package.json` allows `typescript: ^5.3.0`, which resolves to TS 5.4+ in fresh installs. Bypassed by running `npx vite build` directly instead of `npm run build` (which chains `vue-tsc --noEmit && vite build`). Type-check happens at dev commit time; the Docker build only needs to produce `dist/`.

### 2.5 🔐 Security — dev DB was being baked into every image (commit `6b5900a`)

**Discovered during smoke test.** The smoke container showed:
- Two users: `admin` + `phase5` (dev-only account)
- Four encrypted BYOK key rows from May/June (dev-era)

Impossible if the image were clean.

**Root cause:** `.dockerignore` was at `deploy/.dockerignore`. Docker only reads `.dockerignore` from the **build context root** — which is the repo root when we run `docker build -f deploy/Dockerfile .`. A `.dockerignore` anywhere else is silently ignored, no error, no warning.

So every image built to date had bundled `backend/data/app.db` including encrypted BYOK keys. The Fernet secret was the config.py default (`change-this-32-byte-secret-now!!!`), meaning anyone who obtained both the image and that string could decrypt real API keys.

**Blast radius:** local Docker daemon on `.167` only. Image never got pushed anywhere. Still — this is the kind of leak that would go unnoticed indefinitely.

**Fix:** moved `.dockerignore` to the repo root with a prominent warning comment. Verified after rebuild: fresh container has only the seeded default admin, zero `api_keys` rows.

**Recommendation:** rotate the four dev BYOK keys (Anthropic, OpenAI, IBM Quantum, Origin QC) as defense-in-depth. Not urgent (image never left the box) but appropriate hygiene.

### 2.6 Include `[qml,qldpc]` extras (commit `03bc76e`)

The QML page showed *"Training stack not installed — pip install .[qml]"* in prod. The extras were being deliberately excluded to keep the image slim, but QML is core product not optional add-on.

Added `[qml,qldpc]` to both wheel-build and runtime install lines. Image grew from **2.48 GB → 3.62 GB** (+1.14 GB, mostly PennyLane's autograd stack + matplotlib).

**Latent bug caught:** the earlier "strip pyqpanda_alg tooling" step had included `dill` in the uninstall list because it looked like dev tooling. But `qiskit.passmanager` imports `dill` at module load time — removing it silently broke `qiskit_ibm_runtime` imports (the IBMQ tier appeared as unavailable in `/api/qml/health`). Only caught after the QML rollout because `qiskit_ibm_runtime` isn't exercised on the plain optimization path. Kept `dill` installed, added a comment.

---

## 3 — Cloudflare Tunnel + NSSM deployment

### 3.1 Pivot: no Cloudflare Containers

Initial plan (commit `23040c8`) targeted Cloudflare Containers + D1 + Workers dispatcher. Wrote a full `wrangler.toml`, `DEPLOY.md` runbook, D1 migrations SQL. Then handed off the question to the operator who runs `cira-core.com`:

> "Which subdomain? What's the dispatcher pattern? How do you handle SSE?"

**Response:** *"There is no unified dispatcher, no Cloudflare Containers, no D1. Each subdomain on `cira-core.com` is a self-contained setup. Oculus runs on a Windows dev box via Cloudflare Tunnel → localhost, NSSM Windows service. Do the same."*

The Cloudflare Containers scaffold stays in the repo as future-migration material. All actual deployment work switched to the NSSM + Docker Desktop pattern.

### 3.2 First NSSM attempt — native Python + waitress (commit `590a0d6`)

Following the Oculus pattern, initial plan was `python -m waitress` under NSSM (waitress on Windows, gunicorn on Linux for the docker container). Added:
- `waitress` to base deps
- `CIRA_DB_PATH` env var override so SQLite could live on a host volume outside the repo
- `CIRA_SPA_DIR` env var + `_register_spa()` factory hook so Flask serves the built Vue SPA at `/` with Vue Router fallback

### 3.3 Path B pivot: NSSM wraps `docker run` (commit `5a8eff4`)

User pointed out we already had the Docker image built — why go through `venv + pip install + npm build` on `.167` again? Switched to Path B: NSSM wraps `docker run` for the pre-built image. Frontend build stage added to the Dockerfile (§2.4) so the SPA is baked in.

Rewrote `install_quantum_service.ps1` to wrap `docker run --rm --name cira-quantum -p 5209:5209 --env-file ... -v ...:/app/data cira-quantum-backend:local` under NSSM. All the operator muscle-memory (`Restart-Service`, log rotation, restart policies) stays; the container is just the process NSSM watches.

**Three PowerShell 5.1 gotchas fought during install:**

1. **`2>&1 | Out-Null` on native execs.** In Windows PowerShell 5.1, redirecting a native command's stderr inside PowerShell wraps each line as an `ErrorRecord` and sets `$?` to `$false` even when the exe returned exit code 0. This tripped `$ErrorActionPreference = 'Stop'` on the idempotent `nssm stop` / `nssm remove` calls (which are supposed to fail on the first install). Wrapped in `try/catch` with EAP briefly relaxed.

2. **Backticks in `throw` messages broke the parser.** Original script had `` `winget install NSSM.NSSM` `` (double-backticks for markdown-style inline code) inside double-quoted strings. Somewhere the parser interpreted this as an escape sequence and failed with *"Missing closing '}'"* at seemingly unrelated lines. Removed backticks from all error strings.

3. **NSSM AppParameters + Windows arg parsing + spaces in paths.** The install script built `docker run --rm ... --env-file D:\CiRA Quantum\.env -v D:\data\cira-quantum:/app/data cira-quantum-backend:local` and passed it to NSSM. NSSM stores AppParameters as one string on docker.exe's command line. Windows argument parsing splits unquoted whitespace, so `docker` saw `--env-file D:\CiRA` as the arg, then treated `Quantum\.env` as a stray argument. Fought embedded-quote escaping through PowerShell → NSSM → cmd.exe for a while before doing the simple thing: **moved `.env` to `D:\data\cira-quantum\.env`** (no space in path). Cleaner anyway — secrets colocated with the DB, and one directory to protect.

### 3.4 The Cloudflared `--config` gotcha (documented in commit `7190278`)

DNS was registered (`cloudflared tunnel route dns oculus-prod quantum.cira-core.com`), config file was edited, Restart-Service Cloudflared succeeded. But `https://quantum.cira-core.com/api/health` returned **404** for over an hour.

**Root cause:** the Cloudflared Windows service runs as `LocalSystem`. `LocalSystem`'s `%USERPROFILE%` resolves to `C:\Windows\System32\config\systemprofile\` — where there's no `.cloudflared\config.yml`. Cloudflared silently fell through to a fetch of any remotely-managed config (empty in our case), and the tunnel came up serving nothing but the `http_status:404` catch-all. The `C:\Users\viset\.cloudflared\config.yml` we'd edited was never loaded.

**Fix:** update the service binary path with `sc.exe` to include an explicit `--config` flag:

```powershell
sc.exe config Cloudflared binPath= '"C:\Program Files (x86)\cloudflared\cloudflared.exe" --config C:\Users\viset\.cloudflared\config.yml tunnel run oculus-prod'
Restart-Service Cloudflared -Force
```

After that: `https://quantum.cira-core.com/api/health` → **200 in 145ms**. Oculus still worked because our config keeps its ingress rule too.

### 3.5 End-to-end live verification

| Test | Result |
|------|--------|
| `curl https://quantum.cira-core.com/api/health` | **200** with `{"status":"ok","version":"0.1.0"}` |
| `curl https://quantum.cira-core.com/` | **200**, SPA HTML with CiRA Quantum branding |
| `curl https://quantum.cira-core.com/settings` (Vue Router deep-link) | **200**, falls back to `index.html` cleanly |
| `curl https://oculus.cira-core.com/api/system-status` (regression) | **200**, unaffected |

BYOK keys were preserved by seeding `D:\data\cira-quantum\app.db` from the dev DB (same `KEY_ENCRYPTION_SECRET`), so no re-entry required.

**Real end-to-end proof:** submitted a solve on `https://quantum.cira-core.com/`, `qaoa_originqc` completed on real Wukong hardware with energy 3.0. Screenshot in the solver comparison table showed all six solvers running: cpsat / cpu_sa_neal / parallel_tempering / qaoa_sim / **qaoa_originqc (real QPU)** / simulated_bifurcation (skipped, per §1.1).

---

## 4 — Origin QC key debugging

Between test solves, `qaoa_originqc` started failing with:

> `RuntimeError: Cloud submit failed after 4 attempts: RuntimeError: Unauthorized`

Direct test with `pyqpanda3.qcloud.QCloudService(api_key=..., url=...)` returned `Unauthorized` cleanly — Origin was rejecting the key at auth time, not a network issue. The container's outbound connectivity to `qcloud.originqc.com.cn` was fine (307 redirect to `/zh` on a plain GET).

Root cause was on Origin's side — the previously-stored key had been revoked or rotated on their dashboard. User pasted a new key via Settings → API Keys, `QCloudService init: OK (auth successful)` on the new key, next solve worked, real QPU returned a valid result.

**Lesson for the runbook:** the *"Unauthorized after 4 attempts"* error means the key made it through the decrypt path and reached Origin — always check Origin's dashboard before assuming it's a code bug.

---

## 5 — QML training UX misfire

During the first customer-shape workload (10-qubit VQC on Breast Cancer, 50 epochs, 426 samples), user reported *"Queued — the trainer is spinning up"* stayed on screen for minutes.

**Not stuck.** Container CPU pegged at 103% (one core), and DB inspection showed `status='training'` with 20 of 50 epochs already in `training_history`. PennyLane's `default.qubit` CPU statevector for 10 qubits is `2^10 = 1024` complex amplitudes per forward pass, ×2600 forward+backward passes per epoch → **~15-30 sec/epoch on one CPU core** → total ~15-25 min for the job.

**Two things went wrong in the UI, though:**
1. The frontend showed *"Queued"* copy even though `status='training'` was set in the DB minutes earlier. Either the SSE stream isn't delivering epoch events, or the frontend isn't consuming them.
2. The SSE endpoint returned 1260 bytes over 48s (about the size of 5-6 epoch events), so events *are* being emitted but not enough. Worth investigating whether events emit only at epoch boundary.

**Also worth flagging to the user:** 10 qubits on Breast Cancer is likely over-parameterized. 31 trainable params vs 426 samples → barren-plateau-adjacent landscape. Loss dropped 0.66 → 0.64 over 20 epochs. Recommended `n_qubits=4, n_layers=2, ansatz=su2_brick, momentum=adam, epochs=30` — proven to converge fast on breast cancer.

Neither issue is a blocker for shipping; documented for a follow-up UX pass.

---

## 6 — Portable offline demo on `.141` (`cira-nb3`)

Customer visit was on the calendar; user wanted a laptop-portable copy of the whole stack for demo. **Not** cold-standby, **not** live replication — just a self-contained demo box.

### 6.1 Reconnaissance

- Target: `192.168.1.141` (`cira-nb3`, Ubuntu 20.04.5 LTS, 5.15 kernel, 162 GB disk with 72 GB free)
- SSH creds in a text file (`cira` / weak default password)
- Docker: **not installed**
- Also has Ollama running (unrelated)

Set up SSH key auth first — Windows OpenSSH client on `.167` didn't have a scriptable way to pass a password to the remote. Installed **paramiko** in host Python to bootstrap:
- Read the credentials file
- SSH in once with password
- Append our public key to `~/.ssh/authorized_keys`
- Add the host key to `~/.ssh/known_hosts`

From that point on: pure key auth, no password re-entry.

### 6.2 Docker install

Wrote a `install_docker.sh` that follows Docker's official Ubuntu 20.04 recipe. **Two shell-escaping bugs** fought before it worked:

1. First pass had `$(dpkg --print-architecture)` in a Python string passed via paramiko `exec_command`. The `$()` was evaluated locally in the Windows Bash shell before the string was sent, so the remote command received an empty `arch=`. Fixed by writing the script to a file locally, scp'ing it, and running it as `bash /tmp/install_docker.sh`.

2. Even in the file-based script, the `_sudo` wrapper (`echo "$SUDO_PASS" | sudo -S "$@"`) was fine for arg-taking commands but broke when we piped external stdin to it (e.g. `echo "..." | _sudo tee /etc/apt/sources.list.d/docker.list`). The inner echo of the password overrode the outer echo of the deb line. Fixed by writing the deb line to a temp file first, then `_sudo cp` it into place.

Ended up with **Docker 28.1.1** installed and `cira` in the `docker` group. Fresh SSH session inherited the group so we could drop `sudo` for image ops.

### 6.3 Image transfer

- `docker save cira-quantum-backend:local | gzip -1 > /d/tmp/cira-quantum-transfer/cira-quantum-backend.tar.gz` → **1012 MB gz**
- `scp` to `.141:/tmp/` in background
- `ssh cira@.141 docker load < /tmp/cira-quantum-backend.tar.gz` → **2.43 GB in Docker storage**

### 6.4 Data + config

- `/var/lib/cira-quantum/` created + `cira`-owned
- `app.db` from prod (`D:\data\cira-quantum\app.db`) → same location on `.141`
- `.env` written with **the same `KEY_ENCRYPTION_SECRET`** as prod so seeded BYOK keys decrypt
- `SESSION_COOKIE_SECURE=false` on this box because localhost:5209 isn't HTTPS

Verified inside the running container: `admin` + `phase5` users present, **all 4 BYOK keys decrypt cleanly** (Claude x2, IBM Q, Origin QC).

### 6.5 Start/stop scripts + demo cheat sheet

Three files shipped to `~cira/`:

- **`start-cira-quantum.sh`** — idempotent: `docker rm -f` any prior container, then `docker run -d --restart unless-stopped ...`, then polls `/api/health` for up to 45s. Prints the URL + admin creds when healthy.
- **`stop-cira-quantum.sh`** — clean shutdown.
- **`DEMO_CHEATSHEET.md`** — full runbook with:
  - Where things live (paths, container name, image, port)
  - What works with vs without internet at the customer site
  - Log tailing during a demo (`docker logs -f cira-quantum-demo`)
  - How to reset the DB between customers
  - How to pull a fresher image from `.167` later
  - Security notes (rotate the weak `cira` password before handing this to anyone outside the team)

### 6.6 End-to-end verification from `.167`

| Test | Result |
|------|--------|
| `curl http://192.168.1.141:5209/api/health` | **200** with `{"status":"ok","version":"0.1.0"}` |
| `curl http://192.168.1.141:5209/` | **200**, SPA served |
| BYOK decrypt (all 4 keys inside container) | ✅ all pass |

Container is up, `--restart unless-stopped` so a laptop reboot auto-recovers. Ready to close the lid and walk out with it.

---

## 7 — What shipped today

### Commits (9 total, chronological)

| SHA | Title |
|-----|-------|
| `f907575` | Classifier-routed hardcoded formulators + landing-page auth + docs |
| `23040c8` | Cloudflare Containers deploy scaffold + Docker image trimmed to 2.47 GB |
| `590a0d6` | NSSM deploy scaffold for quantum.cira-core.com on .110 |
| `3c6480d` | NSSM deploy: use current repo location, drop the git-clone step |
| `5a8eff4` | Path B: NSSM-wraps-docker deploy with SPA baked into the image |
| `6b5900a` | 🔐 SECURITY: move .dockerignore to build context root |
| `7190278` | Land quantum.cira-core.com deploy + capture the real-run learnings |
| `03bc76e` | Include [qml,qldpc] extras + keep dill in production image |
| `11f5db4` | QML real-QPU inference: lift the 2-qubit-only restriction |

### Live surfaces

| URL | Purpose |
|-----|---------|
| **`https://quantum.cira-core.com/`** | Public production, Cloudflare Tunnel → NSSM `CiraQuantumSvc` → Docker container on `.167` |
| **`http://192.168.1.141:5209/`** | Portable offline demo (laptop) |

### Architecture at end of day

```
┌────────────────────────────────────────────────────────────────────┐
│   Cloudflare edge                                                  │
│   cira-core.com zone                                               │
└──────────────────┬─────────────────────────────────────────────────┘
                   │ Cloudflare Tunnel (oculus-prod, --config-flagged)
                   │
┌──────────────────▼──────────────────────────────────────────────────┐
│  .167 (DESKTOP-1A0J7FD, Windows)                                    │
│   ┌────────────────────────────┐   ┌───────────────────────────┐    │
│   │ OculusBackendSvc (NSSM)    │   │ CiraQuantumSvc (NSSM)     │    │
│   │  → python backend\run.py   │   │  → docker run cira-quantum│    │
│   │  → localhost:5008          │   │     → localhost:5209      │    │
│   └────────────────────────────┘   └───────────────────────────┘    │
│                                              │                       │
│                                     ┌────────▼────────┐              │
│                                     │ D:\data\        │              │
│                                     │  cira-quantum\  │              │
│                                     │   app.db + .env │              │
│                                     └─────────────────┘              │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  .141 (cira-nb3, Ubuntu 20.04 laptop)  ← portable demo             │
│   Docker → cira-quantum-demo container → localhost:5209            │
│   /var/lib/cira-quantum/{app.db, .env}                              │
│   (same KEY_ENCRYPTION_SECRET as prod → seeded BYOK keys work)      │
└─────────────────────────────────────────────────────────────────────┘
```

### Image details

- **`cira-quantum-backend:local`** — 3.62 GB uncompressed, ~1 GB compressed
- Stages: `frontend-builder` (Node 20 + Vite) → `builder` (Python 3.12 slim + wheel build) → `runtime` (Python 3.12 slim + wheels + SPA)
- Extras baked in: `[quantum,ibm-quantum,qml,qldpc]`
- Excluded: `[classical-extras]` (torch + CUDA), `[gpu]` (triton), `[dev]` (pytest/ruff)
- Entrypoint: `gunicorn --config /app/gunicorn_conf.py app:create_app()` on `PORT=5209`
- Non-root user, healthcheck at `/api/health`

### Non-obvious deploy-time truths worth remembering

1. **`.dockerignore` must be at build-context root** — not next to the Dockerfile. Ours is at repo root with a warning comment. Do not move it back.
2. **Cloudflared as LocalSystem needs an explicit `--config` flag** — its default lookup finds nothing under `%USERPROFILE%`. Set via `sc.exe config` on the binPath.
3. **PowerShell 5.1 `2>&1` on native execs wraps stderr as ErrorRecord** — fights `$ErrorActionPreference = 'Stop'`. Use `try/catch` or relax EAP briefly.
4. **NSSM AppParameters can't handle unquoted spaces in paths** — keep everything under space-free directories (`D:\data\cira-quantum\` not `D:\CiRA Quantum\`).
5. **Fernet keys survive DB copy** — seeding `.141`'s app.db from prod works because we shipped the same `KEY_ENCRYPTION_SECRET`.
6. **Docker Desktop's WSL2 backend is fine on Windows** — no CUDA image means no GPU passthrough concerns, image runs the same as it would on Linux.
7. **`vue-tsc 1.8` is incompatible with TypeScript 5.4+** — bypass by running `npx vite build` directly instead of `npm run build`. Upgrade to `vue-tsc 2.x` is a follow-up.
8. **`dill` is a load-time dep of qiskit's passmanager** — don't add it to any "strip dev tooling" uninstall list.

---

## 8 — Follow-ups (known, tracked, not-yet-done)

Not blocking anything, in priority order:

1. **HiGHS glibc symbol mismatch** — `highspy` throws `undefined symbol: _ZN5Highs13releaseMemoryEv` on Debian trixie's slim base at every boot. Registry try/except catches it; CPSAT is the actual exact classical solver. Fix: pin base to `python:3.12-slim-bookworm` OR use `highspy`'s manylinux2014 wheel. ~30 min.
2. **UptimeRobot monitor** on `https://quantum.cira-core.com/api/health` at 5-min interval, same phone target as Oculus. ~3 min in the UptimeRobot dashboard.
3. **QML SSE-vs-UI mismatch** — frontend shows "Queued" during `status='training'` even when events are flowing. Either the stream isn't buffered properly, or the SPA store isn't reducing epoch events. ~1-2 hour investigation.
4. **`vue-tsc` upgrade to 2.x** — will let us restore `npm run build` in the frontend-builder Docker stage (type-check at image build). ~15 min.
5. **Rotate the four dev BYOK keys** — defense-in-depth after the `.dockerignore` incident. ~5 min per provider.
6. **Rotate `cira` password on `.141`** — currently the weak default `aaaa`. ~30 sec.
7. **Cloudflared `--config` fix applied for Oculus too** — currently working because our config keeps its ingress, but worth verifying the ops person doesn't rely on a different mechanism that we've now shadowed.

---

## 9 — Wall-clock summary

Full deploy from *"nothing public"* to *"customer-ready portable demo"* in one afternoon:

| Phase | Wall time | Delivery |
|-------|-----------|----------|
| Bug fixes (SB skipped, QML 2-qubit lift) | ~1 hr | 2 commits |
| Docker image evolution (6 rebuilds) | ~1.5 hr | 3 commits |
| NSSM deployment + tunnel debug | ~2 hr | 3 commits |
| Origin key + QML training debugging | ~30 min | conversation only |
| `.141` portable demo setup | ~40 min | scripts + cheat sheet on the laptop |
| This report | ~20 min | you're reading it |

Total: **~6 hours** of active work.

**End state:** the platform runs on three surfaces — public production (`quantum.cira-core.com`), dev on `.167` (same box), and a portable demo on `.141`. All three exercise the same Docker image. All three preserve BYOK keys via the same `KEY_ENCRYPTION_SECRET`. Adding a fourth deploy (another laptop, another VM, a customer-hosted install) is now a scripted 15-min operation, not a multi-hour effort.

Nice piece of shipping.

---

*Written 2026-07-02, by the person(s) who did the work + Claude Opus 4.7. Screenshots from the smoke tests archived alongside the commits. Any specific code location cited in this document traces back to a commit SHA in the table above.*
