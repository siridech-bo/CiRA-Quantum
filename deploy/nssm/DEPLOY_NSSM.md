# CiRA Quantum — Deploy to `.110` via NSSM-wrapped Docker

Target: `quantum.cira-core.com` served from `.110` via Docker Desktop
+ NSSM Windows service + Cloudflare Tunnel ingress rule added to the
existing `oculus-prod` tunnel. **All Python deps, the built Vue SPA,
and the pyqpanda3 / qiskit toolchain are baked into a single Docker
image** — the host just runs `docker run` via NSSM.

Estimated total time: **~30 min** end-to-end (image build if not
already done + service install + tunnel config).

## Paths (single source of truth)

| Role           | Path                                        |
|----------------|---------------------------------------------|
| Code (repo)    | `D:\CiRA Quantum` (same as dev)             |
| Image name     | `cira-quantum-backend:local`                |
| Container name | `cira-quantum`                              |
| SQLite DB      | `D:\data\cira-quantum\app.db` (mounted at `/app/data/` inside container) |
| Log files      | `D:\logs\cira-quantum\svc.out.log` / `.err.log` |
| Service `.env` | `D:\CiRA Quantum\.env` (loaded via `--env-file`) |
| Port           | `5209` (host-side; container listens on same) |
| Service name   | `CiraQuantumSvc`                            |
| Tunnel         | reuses existing `oculus-prod` (single tunnel, two ingress rules) |

The repo stays where it lives at `D:\CiRA Quantum\` (matches Oculus's
`D:\CiRA Oculus\` pattern). Nothing gets moved to `D:\services\...` —
`git pull` is the redeploy trigger.

## 1. Confirm the repo is up to date

```powershell
cd "D:\CiRA Quantum"
git checkout main
git pull
```

## 2. Confirm Docker Desktop is running

Docker Desktop's systray icon should be green (running). If not, start
it and wait for the WSL2 backend to initialize (~30 s on cold boot).

```powershell
docker info --format '{{.ServerVersion}}'
```

Should print a version number, no errors. If it errors with "cannot
connect to Docker daemon," Docker Desktop isn't running yet.

## 3. Build (or refresh) the container image

Only needs to run when `pyproject.toml`, the frontend, or the
`deploy/Dockerfile` itself changes. Otherwise skip — the image on
disk from the last build is still valid.

```powershell
cd "D:\CiRA Quantum"
docker build -f deploy/Dockerfile -t cira-quantum-backend:local .
```

First build: ~10 min (pyqpanda3 wheel compile + Node build stage).
Subsequent builds cache aggressively — usually ~30 s if only backend
Python changed, ~2 min if the frontend changed.

Verify:

```powershell
docker images cira-quantum-backend:local
```

Should show a single image around 2.5 GB uncompressed.

## 4. Create data + log directories

```powershell
New-Item -ItemType Directory -Path 'D:\data\cira-quantum' -Force
New-Item -ItemType Directory -Path 'D:\logs\cira-quantum' -Force
```

Data and logs live outside the container so `docker rm` / image
rebuilds never touch user state (BYOK-encrypted keys, job history,
admin password).

## 5. Write the `.env` file

```powershell
Copy-Item "D:\CiRA Quantum\deploy\nssm\.env.example" `
          "D:\CiRA Quantum\.env"
notepad "D:\CiRA Quantum\.env"
```

Fill in the three `REPLACE_WITH_*` secrets. Generate values with:

```powershell
# SECRET_KEY (Flask session signing):
python -c "import secrets; print(secrets.token_urlsafe(64))"

# KEY_ENCRYPTION_SECRET (Fernet BYOK key — 32 bytes base64):
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

# DEFAULT_ADMIN_PASSWORD: pick something strong; the admin rotates it
# at first login.
```

**Important:** in `.env`, `CIRA_DB_PATH` must be **`/app/data/app.db`**
(Linux path — that's inside the container where the host volume is
mounted). The template already has this. Do NOT change it to
`D:\data\cira-quantum\app.db` — that's a host path and won't work
inside the container.

⚠️ Do NOT commit the populated `.env`. It stays only on `.110`.

## 6. Manual smoke test (before touching NSSM)

Run the container directly to confirm it comes up:

```powershell
docker run --rm --name cira-quantum-smoke `
    -p 5209:5209 `
    --env-file "D:\CiRA Quantum\.env" `
    -v "D:\data\cira-quantum:/app/data" `
    cira-quantum-backend:local
```

Wait ~30 s for boot (pyqpanda3 + qiskit imports take a moment on
first run), then from a second PowerShell window:

```powershell
curl http://127.0.0.1:5209/api/health
curl http://127.0.0.1:5209/       # should return the SPA index.html
```

Expected first response:
`{"fts5":"available","sqlite_vec":"loaded","status":"ok","version":"0.1.0"}`

Ctrl+C the manual run before moving on. `--rm` removes the container
automatically on exit.

## 7. Install the NSSM service

Elevated PowerShell (Run as Administrator):

```powershell
cd "D:\CiRA Quantum"
.\deploy\nssm\install_quantum_service.ps1
```

The script:

- Preflight-checks NSSM, Docker CLI, Docker daemon, and the image.
- Stops + removes any prior `CiraQuantumSvc` (idempotent).
- Removes any dangling container by name (defensive).
- Installs the service to run `docker run --rm ... cira-quantum-backend:local`.
- Configures NSSM restart policy (5 s delay, 60 s throttle) matching Oculus.
- Configures NSSM log rotation (10 MB per file, online rotation).
- Sets `AppStopMethod* = 15 s` so `docker stop`'s SIGTERM has time to
  drain in-flight SSE streams before the SIGKILL.
- Depends on `com.docker.service` so NSSM waits for Docker Desktop
  before starting the container at boot.
- Starts the service and polls `/api/health` for up to 45 s.

Expected end of script output:

```
----- Service status -----
Name              Status  StartType
----              ------  ---------
CiraQuantumSvc    Running Automatic

----- Health probe -----
Health check: HTTP 200
{"fts5":"available","sqlite_vec":"loaded","status":"ok","version":"0.1.0"}
```

## 8. Add the Cloudflare Tunnel ingress rule

Edit `C:\Users\viset\.cloudflared\config.yml` per the instructions in
[`cloudflared_ingress.yml`](cloudflared_ingress.yml) — merge the
`quantum.cira-core.com` block into the existing `ingress:` list
BEFORE the `service: http_status:404` catch-all.

Then register the DNS route (only once):

```powershell
cloudflared tunnel route dns oculus-prod quantum.cira-core.com
```

Restart the tunnel service so the new rule takes effect:

```powershell
Restart-Service Cloudflared
```

## 9. Verify from off-LAN

From a phone hotspot or any non-`.110` machine:

```powershell
curl -s https://quantum.cira-core.com/api/health
```

Expected:
`{"fts5":"available","sqlite_vec":"loaded","status":"ok","version":"0.1.0"}`

Also browse to `https://quantum.cira-core.com/` — should render the
Vue SPA (login/signup landing page).

## 10. Add UptimeRobot monitor

Log into UptimeRobot, add a new monitor:

- Type: **HTTP(S)**
- Friendly name: `CiRA Quantum backend`
- URL: `https://quantum.cira-core.com/api/health`
- Monitoring interval: **5 minutes**
- Alert contacts: same phone push target as the Oculus monitor.

## Operator cheat sheet

| Action                    | Command                                             |
|---------------------------|-----------------------------------------------------|
| View service status       | `Get-Service CiraQuantumSvc`                        |
| Start / stop / restart    | `Start-Service CiraQuantumSvc` / `Stop-…` / `Restart-…` |
| Tail service log          | `Get-Content -Wait D:\logs\cira-quantum\svc.out.log` |
| Tail container log        | `docker logs -f cira-quantum` |
| Enter running container   | `docker exec -it cira-quantum bash` |
| Redeploy after code change (see below for details) | rebuild image + `Restart-Service CiraQuantumSvc` |
| Rotate secrets            | edit `.env` → `Restart-Service CiraQuantumSvc`      |
| Manual DB backup          | `Copy-Item D:\data\cira-quantum\app.db D:\backups\cira-quantum\app.db.$(Get-Date -Format 'yyyyMMdd-HHmmss').bak` |

## Redeploy on code change

Backend Python only:

```powershell
cd "D:\CiRA Quantum"
git pull
docker build -f deploy/Dockerfile -t cira-quantum-backend:local .
Restart-Service CiraQuantumSvc
```

Frontend only (rebuild is still needed because the SPA is baked into
the image — that's the tradeoff of the "one deployable" pattern):

```powershell
cd "D:\CiRA Quantum"
git pull
docker build -f deploy/Dockerfile -t cira-quantum-backend:local .
Restart-Service CiraQuantumSvc
```

The Docker builder caches aggressively — a frontend-only change
finishes rebuilding in ~2 min, backend-only change in ~30 s.

## Rollback

Rollback = check out the prior commit + rebuild + restart:

```powershell
cd "D:\CiRA Quantum"
git checkout <prior_commit_sha>
docker build -f deploy/Dockerfile -t cira-quantum-backend:local .
Restart-Service CiraQuantumSvc
```

Or, if you want the old image back without a rebuild, tag the current
one before every deploy:

```powershell
docker tag cira-quantum-backend:local cira-quantum-backend:$(git rev-parse --short HEAD)
```

Then rollback becomes:

```powershell
docker tag cira-quantum-backend:<prior_sha> cira-quantum-backend:local
Restart-Service CiraQuantumSvc
```

## Known constraints

- **HiGHS solver skips registration at boot** — glibc symbol mismatch
  in the `highspy` wheel on Debian trixie's slim base. Non-blocker:
  CPSAT remains as the exact classical solver, and the benchmarking
  registry's try/except catches the failure gracefully. Watch
  `docker logs cira-quantum` for `Failed to register HiGHSSampler` —
  expected line, not an incident.

- **Docker Desktop dependency.** The NSSM service depends on
  `com.docker.service`. If Docker Desktop is not running, the service
  will fail to start at boot until Docker is up. On `.110` where
  Docker Desktop autostarts, this is fine.

- **No warm standby.** `.110` is a single failure domain. If it goes
  down, we lose availability until it comes back. Acceptable at
  research-customer scale; revisit when incident volume forces it.

- **Home upstream bandwidth is the cap on concurrent SSE streams.**
  Cloudflare Tunnel holds one long-lived TCP connection per streaming
  client back through the home upload link. Not a bottleneck now.

## Path A (native Python + NSSM) reference

The pre-Docker deploy path — venv + waitress + NSSM without Docker —
is still viable but retired in favor of the containerized approach
above. If for any reason Docker Desktop stops working on `.110`, the
fallback path is:

1. Create `D:\CiRA Quantum\venv`, `pip install -e ".[quantum,ibm-quantum]"`.
2. `cd frontend; npm ci; npm run build`.
3. Rewrite `install_quantum_service.ps1` to invoke
   `& "D:\CiRA Quantum\venv\Scripts\python.exe" -m waitress --host=0.0.0.0 --port=5209 --threads=16 --call app:create_app`.
4. Same NSSM `set` commands otherwise.

Deferred — no reason to prepare for a Docker-Desktop failure that
hasn't happened yet.
