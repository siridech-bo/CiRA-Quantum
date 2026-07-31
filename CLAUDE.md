# CiRA Quantum — project context for Claude

Instructions and durable facts. Read this before reasoning about deployment,
architecture, or where things run. If something here proves wrong, fix it here.

## Running GPU jobs — ASK + CONFIRM FIRST, EVERY TIME (hard rule)

**GPU time is a real, billable cost to the user. Confirm before EVERY GPU
launch** — including **short validation / smoke-test runs**, not just multi-hour
sweeps. This is a shared single-GPU dev box; the user decides. Before launching:

1. **STOP and ask for an explicit go-ahead.** State the **expected wall-clock**,
   the **scope/fidelity**, and **what it consumes**. Offer a faster/cheaper
   option. Wait for a clear "go" — do not launch in the same breath as asking.
2. **This applies to *every* GPU run, however brief.** A 2-minute validation run
   still costs GPU time and still needs a heads-up first. Never slip one in.
3. **Re-confirm if a number you quoted changes.** If you said "~1.5 h" and it's
   really "~1.7 h", pause and re-confirm — don't just launch the corrected one.
4. **A prior "go" does NOT authorize the next run.** Each GPU launch needs its
   own confirmation, even mid-task.
5. **Launch via the launcher API, not a raw background process**, so the run is
   visible + stoppable in the QRC UI (`registry.json`). Raw `python …` background
   jobs are invisible to the dashboard — don't use them for real runs.
6. Don't start a second GPU job while one is running (single-GPU lock + cost).

Estimating: quote a **real** number (measure a few steps if unsure), and correct
it honestly if slower — don't hand-wave "~5h" when it's ~10h.

## Deployment architecture (get this right)

- **`quantum.cira-core.com` is self-hosted**, NOT a Cloudflare-hosted /
  GPU-less container. It runs on a **Windows host (`.167`, DESKTOP-1A0J7FD)**
  and is exposed through a **Cloudflare Tunnel** (`oculus-prod`, uuid
  `94fc52c7-e33b-4f0d-b174-ae18cf2888ff`). Cloudflare is only the tunnel +
  DNS + edge; the app runs on the box.
- **That host HAS a GPU.** QRC's GPU compute (trace-gen / phase1, `torch`
  CUDA) runs there. Do **not** assume "prod = no GPU" — that was a stale note
  and is wrong.
- **`CiraQuantumSvc` runs as a Docker service via NSSM** (sibling to
  `OculusBackendSvc`, which is native `python run.py`). A host GPU is **not**
  automatically visible inside a container — GPU-in-container needs the NVIDIA
  runtime (`--gpus all`, WSL2 backend, NVIDIA Container Toolkit). If that's not
  configured, run the service native (NSSM → python) instead.
- Adding a service = add an NSSM service + a `cloudflared` ingress rule
  (hostname and/or `path:` regex) + a DNS route (`cloudflared tunnel route dns
  oculus-prod <host>`) + `Restart-Service Cloudflared`. One tunnel serves every
  subdomain. Runbook: `deploy/nssm/ADD_NEW_SERVICE.md`; ingress example:
  `deploy/nssm/cloudflared_ingress.yml`.
- Deploy-host identity is `.167` (an earlier `.110` was stale DHCP drift, fixed
  in commit `5e0d430`).
- **This dev box IS the prod box.** Hostname `DESKTOP-1A0J7FD`, LAN IP
  `192.168.1.167`. The machine Claude develops on == the machine serving
  `quantum.cira-core.com` == the machine with the RTX 5070 Ti. So QRC
  trace-gen / phase1 runs done here are *already on the prod host*; deploying
  QRC is "build the SPA + serve it + tunnel already points here", not "move
  compute to another box". Do not ask whether they're the same machine — they
  are.

## QRC is PART of the CiRA Quantum app, not a separate product

- `qrc_bp` is registered in the same `create_app()` as production, at
  **`/api/qrc`**, right next to `qml_bp` (see `backend/app/__init__.py`). The
  Vue SPA already has **`/qrc`** and **`/qrc/runs/:id`** routes.
- Therefore QRC's home is **`quantum.cira-core.com/qrc`** — a page in the one
  CiRA Quantum app. **No subdomain, no separate frontend, no split instance.**
  Deploying QRC = build the SPA from the QRC branch + run the app on the
  GPU host; `/api/qrc` and the GPU compute are the same instance, so login and
  launch/stop share one session (no cross-instance auth problem).

## Ports / dev

- Backend dev: `python backend/run.py` → **:5009**. Prod container → **:5209**.
- Frontend dev: Vite **:3070**, proxies `/api/*` → `:5009`. Prod serves the
  built SPA. Frontend needs no `base` change for `/qrc` (it's a normal route).
- Local dev SQLite (`data/app.db`) is a **different DB from production** — the
  local admin password (reset to `admin123`) is not the production password.

## Branching

- QRC is developed on branch **`qrc-simulation`** (well ahead of `main`; carries
  the entire QRC sub-project). Deploy docs/artifacts live under `deploy/`.
- Commit on the feature branch for QRC work; only merge to `main` once a stage
  is validated + reviewed.

## Progress tracking (strict)

- Keep **`docs/QRC/QRC_Next_Stage_Status.md` + `.html`** current. Update them
  **on every subtask completion** — prepend a row to the "Session activity log"
  and flip any affected phase/scoreboard state. This is the user's standing
  rule for knowing what Claude is up to; do not batch it or skip it.

## Conventions

- Windows host; primary shell is PowerShell, Bash tool also available. Write
  files UTF-8; avoid non-ASCII in console prints (cp1252 `UnicodeEncodeError`).
- Backend blueprints gate mutating routes with `@login_required`; QRC read-only
  routes (list/progress/fid/results/embedding) are public by design.
