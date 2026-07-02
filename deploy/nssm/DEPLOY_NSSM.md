# CiRA Quantum — Deploy to `.110` via NSSM + Cloudflare Tunnel

Target: `quantum.cira-core.com` served from `.110` (Windows dev box),
same pattern as `oculus.cira-core.com`. Single-box deploy, no Docker,
native Python 3.12 + `waitress` WSGI server + NSSM Windows service +
Cloudflare Tunnel ingress rule added to the existing `oculus-prod`
tunnel.

Everything in this runbook is self-contained — a fresh operator with
admin PowerShell on `.110` can go from `git clone` to
`curl https://quantum.cira-core.com/api/health` returning `200` by
following the ten steps below in order. Estimated total time: **~1 hour**
end-to-end, most of which is the first `pip install` and the frontend
build.

## Paths (single source of truth)

| Role           | Path                                        |
|----------------|---------------------------------------------|
| Code (repo)    | `D:\services\cira-quantum`                  |
| SQLite DB      | `D:\data\cira-quantum\app.db`               |
| Log files      | `D:\logs\cira-quantum\svc.out.log` / `.err.log` |
| Service `.env` | `D:\services\cira-quantum\.env`             |
| Port           | `5209` (deliberately non-adjacent to Oculus's 5008) |
| Service name   | `CiraQuantumSvc`                            |
| Tunnel         | reuses existing `oculus-prod` (single tunnel, two ingress rules) |

## 1. Clone the repo

```powershell
mkdir D:\services -Force
cd D:\services
git clone https://github.com/siridech-bo/CiRA-Quantum.git cira-quantum
cd cira-quantum
```

## 2. Create data + log directories

```powershell
mkdir D:\data\cira-quantum -Force
mkdir D:\logs\cira-quantum -Force
```

These live outside the repo so `git pull` and service reinstalls never
touch user state (BYOK-encrypted API keys, job history, admin
password).

## 3. Create the Python venv and install deps

```powershell
cd D:\services\cira-quantum\backend
python -m venv D:\services\cira-quantum\venv
D:\services\cira-quantum\venv\Scripts\Activate.ps1

# Base app + the two extras we need in prod. Skip [classical-extras]
# and [gpu] — those pull torch + CUDA and we don't ship GPU SA.
pip install -e ".[quantum,ibm-quantum]"

# Verify:
python -c "from app import create_app; app = create_app(); print('OK, routes:', len(list(app.url_map.iter_rules())))"
```

Expected output: `OK, routes: 63`.

## 4. Build the frontend

```powershell
cd D:\services\cira-quantum\frontend
npm ci
npm run build
```

Output lands at `D:\services\cira-quantum\frontend\dist\`. Verify:

```powershell
Test-Path D:\services\cira-quantum\frontend\dist\index.html
```

Should print `True`.

## 5. Write the `.env` file

```powershell
Copy-Item D:\services\cira-quantum\deploy\nssm\.env.example `
          D:\services\cira-quantum\.env
notepad D:\services\cira-quantum\.env
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

⚠️ Do NOT commit the populated `.env`. It stays only on `.110`.

## 6. Manual smoke test (before touching NSSM)

```powershell
cd D:\services\cira-quantum\backend
D:\services\cira-quantum\venv\Scripts\python.exe -m waitress `
    --host=0.0.0.0 --port=5209 --threads=16 --call app:create_app
```

From a second PowerShell window:

```powershell
curl http://127.0.0.1:5209/api/health
curl http://127.0.0.1:5209/           # should return the SPA index.html
```

Expected first response:
`{"fts5":"available","sqlite_vec":"loaded","status":"ok","version":"0.1.0"}`

Ctrl+C the manual waitress process before moving on.

## 7. Install the NSSM service

Elevated PowerShell (Run as Administrator):

```powershell
cd D:\services\cira-quantum
.\deploy\nssm\install_quantum_service.ps1
```

The script:

- Stops + removes any prior `CiraQuantumSvc` (idempotent).
- Installs the service pointing at the venv's `waitress` module.
- Configures NSSM restart policy (5 s delay, 60 s throttle) matching Oculus.
- Configures NSSM log rotation (10 MB per file, online rotation).
- Sets `AppStopMethod* = 15 s` so in-flight SSE streams drain cleanly.
- Starts the service and prints its status + a health-probe result.

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

And restart the tunnel service so the new rule takes effect:

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
| Tail stdout log           | `Get-Content -Wait D:\logs\cira-quantum\svc.out.log` |
| Tail stderr log           | `Get-Content -Wait D:\logs\cira-quantum\svc.err.log` |
| Redeploy after `git pull` | `Restart-Service CiraQuantumSvc`                    |
| Rebuild frontend          | `cd D:\services\cira-quantum\frontend; npm ci; npm run build; Restart-Service CiraQuantumSvc` |
| Rotate secrets            | edit `.env` → `Restart-Service CiraQuantumSvc`      |
| Manual DB backup          | `Copy-Item D:\data\cira-quantum\app.db D:\backups\cira-quantum\app.db.$(Get-Date -Format 'yyyyMMdd-HHmmss').bak` |

## Redeploy on code change

Small-app-only change (no dep changes, no schema change):

```powershell
cd D:\services\cira-quantum
git pull
Restart-Service CiraQuantumSvc
```

Deps changed (`pyproject.toml` touched):

```powershell
cd D:\services\cira-quantum
git pull
D:\services\cira-quantum\venv\Scripts\Activate.ps1
pip install -e ".[quantum,ibm-quantum]"
Restart-Service CiraQuantumSvc
```

Frontend changed:

```powershell
cd D:\services\cira-quantum\frontend
git pull   # if not already pulled above
npm ci     # only when package-lock.json changes
npm run build
Restart-Service CiraQuantumSvc
```

## Rollback

Rollback = `git checkout <prior_commit>` + rebuild + restart. There's
no automated rollback — one-box deploys and SQLite make that
inherently manual. If a deploy corrupts the DB (unlikely — schema is
idempotent), the recovery path is:

```powershell
Stop-Service CiraQuantumSvc
Copy-Item D:\backups\cira-quantum\app.db.<latest>.bak D:\data\cira-quantum\app.db -Force
Start-Service CiraQuantumSvc
```

Set up an hourly scheduled task that runs the manual-backup command
above once the service is stable enough that we care about data loss.

## Known constraints

- **HiGHS solver skips registration at boot** — glibc symbol mismatch
  in the `highspy` wheel on some Python 3.12 installations. Non-blocker:
  CPSAT remains as the exact classical solver, and the benchmarking
  registry's try/except catches the failure gracefully. Watch
  `svc.err.log` for `Failed to register HiGHSSampler` — expected line,
  not an incident.

- **No warm standby.** `.110` is a single failure domain. If it goes
  down, we lose availability until it comes back. Acceptable at
  research-customer scale; revisit when incident volume forces it.

- **Home upstream bandwidth is the cap on concurrent SSE streams.**
  Cloudflare Tunnel holds one long-lived TCP connection per streaming
  client back through the home upload link. Not a bottleneck now;
  worth watching if user growth spikes.

## Where the other deploy files live

The `deploy/` folder in the repo also contains:

- `Dockerfile`, `wrangler.toml`, `gunicorn_conf.py`, `DEPLOY.md`,
  `migrations/0001_initial_schema.sql`

Those are the Cloudflare Containers + D1 deploy path — **not used by
this NSSM deploy**. They stay in the repo as future-migration
scaffolding for if/when a Cloudflare-hosted redeploy makes sense.
