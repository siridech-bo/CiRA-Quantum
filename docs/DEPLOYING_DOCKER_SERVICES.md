# Deploying a Docker Service to `cira-core.com`

> **Last revision:** 2026-07-04
> **Audience:** anyone standing up a new Docker-based service under
> the `cira-core.com` umbrella (CiRA Quantum, CiRA ME, CiRA Vision,
> future modules).
> **Companion runbook:** [`deploy/nssm/ADD_NEW_SERVICE.md`](../deploy/nssm/ADD_NEW_SERVICE.md)
> is the step-by-step commands. This document explains **why**
> the pattern is shaped the way it is, and gives you enough context
> to make the parameter decisions yourself before you follow the
> runbook.

This is the go-to guide for the pattern that shipped 2026-06 (Oculus)
and 2026-07 (CiRA Quantum). It's not the only way to deploy a service —
Cloudflare Pages + Workers + D1 exist and work — but it's the one
we've paid for in production and can operate confidently.

---

## 1 · The pattern in one picture

```
                       cira-core.com  (Cloudflare DNS zone)
                              │
                              │
                     ┌────────┴─────────┐
                     │  Cloudflare      │
                     │  Tunnel:         │ single tunnel, one UUID
                     │  oculus-prod     │
                     └────────┬─────────┘
                              │
      ┌────────────┬──────────┼──────────┬─────────────┐
      │            │          │          │             │
   oculus.       quantum.   me.         (…future       404 catch-all
   :5008         :5209      :NEW_PORT    subdomains)
      │            │          │
      ▼            ▼          ▼
   ┌──────────────────────────────────────────────────────────┐
   │  .167 (DESKTOP-1A0J7FD, Windows) — the deploy host       │
   │                                                          │
   │  ┌──────────────┐  ┌────────────────┐  ┌──────────────┐  │
   │  │ NSSM svc:    │  │ NSSM svc:      │  │ NSSM svc:    │  │
   │  │ OculusBackend│  │ CiraQuantumSvc │  │ NEW_NSSM_SVC │  │
   │  │              │  │                │  │              │  │
   │  │ → python.exe │  │ → docker.exe   │  │ → docker.exe │  │
   │  │   backend\   │  │   run --rm     │  │   run --rm   │  │
   │  │   run.py     │  │   ...          │  │   ...        │  │
   │  │              │  │   quantum:local│  │   NEW_IMAGE  │  │
   │  └──────────────┘  └────────┬───────┘  └──────┬───────┘  │
   │                             │                 │          │
   │                    ┌────────▼─────┐  ┌────────▼───────┐  │
   │                    │ D:\data\     │  │ D:\data\       │  │
   │                    │  cira-quantum│  │  NEW_SUB\      │  │
   │                    │  app.db+env  │  │  data + env    │  │
   │                    └──────────────┘  └────────────────┘  │
   └──────────────────────────────────────────────────────────┘
```

**Three layers, three responsibilities:**

- **Cloudflare Tunnel** owns the public-facing HTTPS and the DNS. One
  tunnel serves every subdomain via ingress rules in a config file.
- **NSSM** owns the Windows service lifecycle — restart-on-crash, log
  rotation, start-at-boot. Wraps `docker.exe` (or a native `python.exe`
  for legacy services like Oculus).
- **Docker** owns the runtime — image, port mapping, volume mount,
  env vars.

**One tunnel, many services** is the load-bearing insight. You don't
provision a new tunnel per subdomain (which would require a new
credentials file and a new NSSM `Cloudflared*` service each time). You
add one ingress rule to the existing tunnel's YAML, and one NSSM
service on the host.

---

## 2 · When to use this pattern

**Use this when your new service:**

- Runs as a Docker container (any language, any framework).
- Serves HTTP on a single port.
- Needs a subdomain under `cira-core.com`.
- Needs to persist state between restarts (SQLite, uploaded files,
  cached model weights) but doesn't need distributed replication.
- Is OK with running on a single Windows host (`.167` currently).
- Is OK with home-ISP upstream bandwidth (~50 concurrent SSE streams
  is the soft ceiling before Cloudflared saturates).

**Don't use this when your new service:**

- Needs multi-region low-latency reads → use Cloudflare Workers +
  D1 + R2. Scaffolding for that path already exists at
  `deploy/wrangler.toml` but wasn't validated end-to-end.
- Needs autoscaling under bursty traffic → use Cloudflare
  Containers or a managed cloud (Fly.io, Render, Cloud Run).
- Handles regulated data (patient records, financial PII) that
  can't sit on a home Windows box → use a cloud host with
  compliance certifications.
- Runs a GPU workload > 1 h → put it on a real GPU box behind a
  short-lived tunnel, don't NSSM-wrap it.

---

## 3 · Prerequisites

Tick every one before starting. If any fails, fix it before continuing.

- [ ] `.167` (Windows deploy host) is up and reachable via RDP or on-console.
- [ ] Docker Desktop is running on `.167` (systray icon green).
- [ ] NSSM is installed at
      `C:\Users\viset\AppData\Local\Microsoft\WinGet\Packages\NSSM.NSSM_Microsoft.Winget.Source_8wekyb3d8bbwe\nssm-2.24-101-g897c7ad\win64\nssm.exe`.
- [ ] Cloudflared is running as a Windows service and reachable via
      `Get-Service Cloudflared`.
- [ ] Cloudflared service `binPath` includes the `--config` flag
      pointing at `C:\Users\viset\.cloudflared\config.yml`. See §7 in the
      runbook if it doesn't — this is a one-time fix that applies
      globally.
- [ ] You have admin PowerShell access on `.167`.
- [ ] Your Docker image builds cleanly and is tagged locally
      (`docker images` shows it).
- [ ] Your image exposes a `/api/health` (or equivalent) endpoint that
      returns HTTP 200 within ~30 s of container boot.

---

## 4 · The six decisions you make up front

Every service gets its own instance of these six parameters. Write
them down. The runbook, the NSSM install script, and the Cloudflared
config all reference them literally.

| # | Parameter | Rules | Quantum example | CiRA ME example |
|---|-----------|-------|-----------------|-----------------|
| 1 | **Subdomain** | short, all lowercase, matches product name | `quantum` | `me` |
| 2 | **Port** | 52xx range convention, non-adjacent to existing ports | `5209` | `5310` |
| 3 | **NSSM service name** | PascalCase, ends with `Svc` | `CiraQuantumSvc` | `CiraMeSvc` |
| 4 | **Container name** | kebab-case | `cira-quantum` | `cira-me` |
| 5 | **Image reference** | `name:tag` | `cira-quantum-backend:local` | `cira-me:local` |
| 6 | **Data directory** | `D:\data\<subdomain>\`, no spaces | `D:\data\cira-quantum` | `D:\data\cira-me` |

### 4.1 · Currently-used ports on `.167`

Never conflict with these:

| Port | Service |
|------|---------|
| 5008 | Oculus (native Python via NSSM) |
| 5209 | CiRA Quantum (Docker via NSSM) |

**Suggested next allocations:** 5310, 5411, 5512, 5613. The
non-adjacent 5x1x spacing makes each service memorable and
port-scan-collision-resistant.

### 4.2 · Why `D:\data\<subdomain>\` and not next to the repo

Three reasons the data directory lives outside the repo:

1. **`git pull` doesn't touch state.** No risk of a rebase silently
   wiping BYOK keys, user accounts, or job history.
2. **Container UID mismatches are localized.** If your image runs as
   UID 10001, only the data dir needs a chown, not your dev checkout.
3. **NSSM AppParameters can't handle spaces in paths.** `D:\CiRA
   Quantum\.env` gets shredded by Windows argument parsing when NSSM
   passes it to `docker run`. `D:\data\cira-quantum\.env` doesn't.

Contents inside the data dir:

- Your service's persistence file (SQLite, JSON blob, whatever)
- `.env` with all secrets — **never committed**
- Any user-uploaded artifacts (model weights, images, uploads)

---

## 5 · High-level workflow

```
1. Build Docker image locally      →   `docker build ...`
2. Provision paths on .167         →   `New-Item -Path D:\data\...`
3. Write .env with secrets         →   editor
4. Fork the install script         →   copy install_quantum_service.ps1
5. Run the install script          →   admin PowerShell
6. Edit Cloudflared config         →   editor
7. Register DNS route              →   `cloudflared tunnel route dns ...`
8. Restart Cloudflared             →   `Restart-Service Cloudflared -Force`
9. Verify from off-LAN             →   `curl https://NEW_SUB.cira-core.com/api/health`
10. Add UptimeRobot monitor        →   UptimeRobot dashboard
```

Steps 1-5 provision the backend. Steps 6-9 expose it publicly. Step 10
is monitoring. If you follow them in order you have a live service
in about 30 min. Full commands are in
[`deploy/nssm/ADD_NEW_SERVICE.md`](../deploy/nssm/ADD_NEW_SERVICE.md).

---

## 6 · Worked example — CiRA ME (as if you were doing it now)

Concrete run-through so you can see how the six parameters flow
through every step. Actual commands to copy-paste.

### 6.1 · Parameters chosen for CiRA ME

| # | Parameter | Value |
|---|-----------|-------|
| 1 | Subdomain | `me` |
| 2 | Port | `5310` |
| 3 | NSSM service name | `CiraMeSvc` |
| 4 | Container name | `cira-me` |
| 5 | Image reference | `cira-me:local` |
| 6 | Data directory | `D:\data\cira-me` |

### 6.2 · Step 1: Build the image

Assumes you have a `Dockerfile` in the CiRA ME repo. If it's not built
yet:

```powershell
cd "D:\path\to\cira-me-repo"
docker build -t cira-me:local .
docker images cira-me:local        # confirm 1 line of output
```

Image must expose `/api/health` (or whatever health endpoint you use)
and listen on `PORT` from env.

### 6.3 · Step 2: Provision paths + logs

```powershell
New-Item -ItemType Directory -Path 'D:\data\cira-me' -Force
New-Item -ItemType Directory -Path 'D:\logs\cira-me' -Force
```

### 6.4 · Step 3: Write `.env`

Create `D:\data\cira-me\.env` with something like:

```dotenv
# Flask session signing. Generate fresh with:
# python -c "import secrets; print(secrets.token_urlsafe(64))"
SECRET_KEY=REPLACE_WITH_FRESH_TOKEN

# If your service does BYOK Fernet encryption:
# python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
KEY_ENCRYPTION_SECRET=REPLACE_WITH_FERNET_KEY

# Sessions travel over HTTPS via Cloudflare Tunnel; keep this on.
SESSION_COOKIE_SECURE=true

# Your service-specific vars. For CiRA ME:
# CIRA_ME_MODEL_PATH=/app/data/model-weights
# CIRA_ME_MAX_UPLOAD_MB=25

PORT=5310
```

### 6.5 · Step 4: Fork the install script

```powershell
Copy-Item deploy/nssm/install_quantum_service.ps1 `
          deploy/nssm/install_me_service.ps1
```

Then edit only the top parameter block:

```powershell
$svc         = 'CiraMeSvc'
$image       = 'cira-me:local'
$container   = 'cira-me'
$port        = 5310
$datadir     = 'D:\data\cira-me'
$logdir      = 'D:\logs\cira-me'
$env_file    = 'D:\data\cira-me\.env'
```

Leave everything below the parameter block untouched — those are the
NSSM commands, sudo password handling, and the health-poll logic that
have already been debugged for the three PowerShell 5.1 gotchas we
hit during the Quantum deploy.

### 6.6 · Step 5: Install the service

**Elevated PowerShell:**

```powershell
cd "D:\CiRA Quantum"           # or wherever you cloned this repo
.\deploy\nssm\install_me_service.ps1
```

Expected tail of the output:

```
----- Service status -----
Name        Status  StartType
----        ------  ---------
CiraMeSvc   Running Automatic

----- Health probe -----
Health check: HTTP 200
```

### 6.7 · Step 6: Add the Cloudflared ingress rule

**Backup first:**

```powershell
Copy-Item 'C:\Users\viset\.cloudflared\config.yml' `
          "C:\Users\viset\.cloudflared\config.yml.bak-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
```

Edit `C:\Users\viset\.cloudflared\config.yml` to add the new ingress
rule **BEFORE** the `service: http_status:404` catch-all:

```yaml
tunnel: 94fc52c7-e33b-4f0d-b174-ae18cf2888ff
credentials-file: C:\Users\viset\.cloudflared\94fc52c7-e33b-4f0d-b174-ae18cf2888ff.json

ingress:
  - hostname: oculus.cira-core.com
    service: http://localhost:5008
    originRequest:
      noTLSVerify: true
      connectTimeout: 30s

  - hostname: quantum.cira-core.com
    service: http://localhost:5209
    originRequest:
      noTLSVerify: true
      connectTimeout: 300s

  # ---- CiRA ME (added 2026-07-04) ----
  - hostname: me.cira-core.com
    service: http://localhost:5310
    originRequest:
      noTLSVerify: true
      connectTimeout: 300s

  - service: http_status:404
```

Validate the config before restarting the tunnel:

```powershell
cloudflared --config 'C:\Users\viset\.cloudflared\config.yml' tunnel ingress validate
cloudflared --config 'C:\Users\viset\.cloudflared\config.yml' tunnel ingress rule https://me.cira-core.com/api/health
```

Both should return OK / a matched rule.

### 6.8 · Step 7: Register DNS

```powershell
cloudflared tunnel route dns oculus-prod me.cira-core.com
```

Runs once. Creates a CNAME `me.cira-core.com` → `oculus-prod` tunnel.
Cloudflare propagates in ~1 min.

### 6.9 · Step 8: Restart Cloudflared

**Elevated PowerShell.** Brief interruption of Oculus + Quantum
(5-10 s while cloudflared re-establishes its four edge tunnels).

```powershell
Restart-Service Cloudflared -Force
```

### 6.10 · Step 9: Verify end-to-end

```powershell
Start-Sleep -Seconds 15   # let Cloudflare edge pick up the rule

# The new service:
curl -s https://me.cira-core.com/api/health

# Regression check the two existing services didn't break:
curl -sI https://oculus.cira-core.com/api/system-status | Select-Object -First 1
curl -sI https://quantum.cira-core.com/api/health | Select-Object -First 1
```

All three should return HTTP 200.

### 6.11 · Step 10: UptimeRobot monitor

Log into UptimeRobot manually:

- Type: HTTP(S)
- Friendly name: `CiRA ME`
- URL: `https://me.cira-core.com/api/health`
- Interval: 5 minutes
- Alert contacts: same phone push target as Oculus / Quantum.

---

## 7 · Worked example — CiRA Quantum (for reference)

Same procedure, different parameters. Already shipped; commands in
[`deploy/nssm/DEPLOY_NSSM.md`](../deploy/nssm/DEPLOY_NSSM.md).

| Parameter | Value |
|-----------|-------|
| Subdomain | `quantum` |
| Port | `5209` |
| NSSM service name | `CiraQuantumSvc` |
| Container name | `cira-quantum` |
| Image reference | `cira-quantum-backend:local` |
| Data directory | `D:\data\cira-quantum` |

Additional context in the full deploy report at
[`docs/reports/production_deploy_day_2026-07-02.md`](reports/production_deploy_day_2026-07-02.md).

---

## 8 · The eight gotchas — learned the hard way

Every one of these cost time during the Quantum deploy. Written here
so you don't rediscover them.

### 8.1 · `.dockerignore` MUST live at the build context root

Docker only reads `.dockerignore` from the directory passed as the
final `docker build ... .` argument (the "build context root").
Anywhere else — including next to your Dockerfile in a `deploy/`
subfolder — is silently ignored. No warning, no error.

**How this bites you:** your dev SQLite DB (with real secrets or
encrypted BYOK keys) gets baked into every image layer.

**Fix:** always `.dockerignore` at the repo root, never elsewhere.

### 8.2 · Cloudflared as LocalSystem needs `--config` on the binPath

The Cloudflared Windows service runs as `LocalSystem`, whose
`%USERPROFILE%` resolves to `C:\Windows\System32\config\systemprofile\` —
which has no `.cloudflared\config.yml`. Without an explicit `--config`
flag on the service's binary path, cloudflared falls through to
Cloudflare-managed config (empty by default) and every request 404s.

**Fix (one-time):**

```powershell
sc.exe config Cloudflared binPath= '"C:\Program Files (x86)\cloudflared\cloudflared.exe" --config C:\Users\viset\.cloudflared\config.yml tunnel run oculus-prod'
Restart-Service Cloudflared -Force
```

Already applied on `.167` as of 2026-07-02. You don't need to redo it,
but verify with `Get-CimInstance Win32_Service -Filter "Name='Cloudflared'"`.

### 8.3 · PowerShell 5.1 `2>&1` on native execs breaks `$?`

Windows PowerShell 5.1 wraps native command stderr as an `ErrorRecord`
and sets `$?` to `$false` **even when the exe returned exit code 0**.
This trips `$ErrorActionPreference = 'Stop'` on idempotent
`nssm stop` / `nssm remove` calls (which legitimately fail on the first
install).

**Fix:** wrap in `try/catch` with `$ErrorActionPreference` briefly
relaxed. Already done in `install_quantum_service.ps1` — inherit that
by copying the script.

### 8.4 · Backticks in `throw` messages break the parser

Original Quantum install script had `` `winget install NSSM.NSSM` ``
(double-backticks for markdown-style inline code) inside double-quoted
strings. The parser sometimes interpreted this as an escape sequence
and failed with *"Missing closing '}'"* at seemingly unrelated lines.

**Fix:** use single quotes (`'winget install NSSM.NSSM'`) in
`throw` and `Write-Warning` messages. Already applied in the Quantum
script.

### 8.5 · NSSM AppParameters + Windows arg parsing hates unquoted spaces

NSSM stores docker-run arguments as one string on `docker.exe`'s
command line. Windows argument parsing splits unquoted whitespace, so
`--env-file D:\CiRA Quantum\.env` becomes `--env-file D:\CiRA` plus
`Quantum\.env` (stray argument). Embedded double-quotes get stripped
by various layers of PowerShell → NSSM → cmd.exe escaping.

**Fix:** keep all NSSM-visible paths space-free. Naming convention:
`D:\data\<subdomain>\` not `D:\Some Product\`.

### 8.6 · `dill` is a runtime dep of qiskit's passmanager

If your Docker image tries to slim itself by uninstalling `pyqpanda_alg`'s
over-declared docs tooling, resist the temptation to also remove
`dill`. `qiskit.passmanager` imports it at module load time; removing
it silently breaks `qiskit_ibm_runtime`. Not everyone notices until
someone tries the IBMQ tier.

**Fix:** don't add `dill` to any `pip uninstall` list.

### 8.7 · `vue-tsc 1.8` is incompatible with TypeScript 5.4+

If your frontend has `"vue-tsc": "^1.8.27"` in `package.json` AND
allows `"typescript": "^5.3.0"`, a fresh `npm ci` in a Docker build
context resolves TypeScript 5.4+. vue-tsc's monkey-patch of TS
internals fails with:

```
Search string not found: "/supportedTSExtensions = .*(?=;)/"
```

**Two fixes:**

1. Upgrade to `"vue-tsc": "^2.x"` (proper long-term).
2. Bypass in the Dockerfile: run `npx vite build` directly instead of
   `npm run build` (which chains `vue-tsc --noEmit && vite build`).

### 8.8 · Container UID vs host mount UID

If your Dockerfile uses `USER nonroot` with UID 10001 (or any non-root
UID), the host mount at your data directory must be chowned to
match. Otherwise the container can read but not write — SQLite writes
fail with `attempt to write a readonly database`.

**Fix on Linux hosts:** `sudo chown -R 10001:10001 /var/lib/NEW_SUB`.
On Windows under Docker Desktop, WSL2 usually handles UID mapping
transparently — but test that your service can write to persistent
files before assuming.

---

## 9 · Ongoing operations

Once your service is live, the operator commands are uniform across
every NSSM+Docker deployment.

| Action | Command |
|--------|---------|
| Check status | `Get-Service NEW_NSSM_SVC` |
| Start / stop / restart | `Start-Service …` / `Stop-…` / `Restart-…` |
| Tail service log | `Get-Content -Wait D:\logs\NEW_SUB\svc.out.log` |
| Tail container log | `docker logs -f NEW_CONTAINER` |
| Enter container shell | `docker exec -it NEW_CONTAINER bash` |
| Rebuild after code change | `docker build -f Dockerfile -t NEW_IMAGE .; Restart-Service NEW_NSSM_SVC` |
| Rotate secrets | edit `NEW_DATA_DIR\.env`, `Restart-Service NEW_NSSM_SVC` |
| Backup DB | `Copy-Item NEW_DATA_DIR\app.db D:\backups\NEW_SUB\app.db.$(Get-Date -Format 'yyyyMMdd-HHmmss').bak` |

### 9.1 · Redeploy pattern

Small code change (no dep bumps, no schema migrations):

```powershell
cd "D:\path\to\repo"
git pull
docker build -f Dockerfile -t NEW_IMAGE .
Restart-Service NEW_NSSM_SVC
```

Dep or schema change: same as above, plus a manual verification the
service starts cleanly (`Get-Content -Wait D:\logs\NEW_SUB\svc.err.log`
in another window).

### 9.2 · Rollback

Rollback = check out the prior commit + rebuild + restart. There's no
automated rollback — one-box deploys and SQLite make that inherently
manual. If a deploy corrupts state, restore from the latest backup:

```powershell
Stop-Service NEW_NSSM_SVC
Copy-Item D:\backups\NEW_SUB\app.db.<latest>.bak NEW_DATA_DIR\app.db -Force
Start-Service NEW_NSSM_SVC
```

Set up an hourly scheduled task doing the backup command once the
service is stable enough that you care about data loss.

---

## 10 · Removing a service

To take a service off `cira-core.com` entirely:

```powershell
# Elevated PowerShell.
Stop-Service NEW_NSSM_SVC -Force
& 'C:\Users\viset\AppData\Local\Microsoft\WinGet\Packages\NSSM.NSSM_Microsoft.Winget.Source_8wekyb3d8bbwe\nssm-2.24-101-g897c7ad\win64\nssm.exe' remove NEW_NSSM_SVC confirm
docker rm -f NEW_CONTAINER
docker rmi NEW_IMAGE
```

Then remove the ingress rule from
`C:\Users\viset\.cloudflared\config.yml` and
`Restart-Service Cloudflared -Force`.

DNS: `cloudflared tunnel route dns` cannot be undone via CLI. The CNAME
can be deleted from Cloudflare's dashboard, but it's harmless to leave
pointing at the tunnel — with no matching ingress rule the request
falls through to the 404 catch-all.

Data (`NEW_DATA_DIR`) and logs (`D:\logs\NEW_SUB`) are yours to keep or
delete.

---

## 11 · Related documents

| File | Purpose |
|------|---------|
| [`deploy/nssm/ADD_NEW_SERVICE.md`](../deploy/nssm/ADD_NEW_SERVICE.md) | Step-by-step commands, no explanation. Follow when you're mid-deploy. |
| [`deploy/nssm/DEPLOY_NSSM.md`](../deploy/nssm/DEPLOY_NSSM.md) | Quantum-specific runbook. Historical reference. |
| [`deploy/nssm/install_quantum_service.ps1`](../deploy/nssm/install_quantum_service.ps1) | The install script to fork for your new service. |
| [`deploy/nssm/cloudflared_ingress.yml`](../deploy/nssm/cloudflared_ingress.yml) | Reference ingress rule format. |
| [`deploy/Dockerfile`](../deploy/Dockerfile) | Worked example of a multi-stage image with frontend build. |
| [`.dockerignore`](../.dockerignore) (repo root!) | Worked example, note the warning at the top. |
| [`docs/reports/production_deploy_day_2026-07-02.md`](reports/production_deploy_day_2026-07-02.md) | Narrative of the Quantum deploy day — every gotcha in context. |
| [`deploy/DEPLOY.md`](../deploy/DEPLOY.md) | Cloudflare Containers + D1 alternative path. Scaffolding exists; not currently used. |

---

## 12 · The five-line summary

If you only remember five things from this document:

1. **One tunnel, many ingress rules.** Never provision a new tunnel per
   service.
2. **`.dockerignore` at the repo root, never anywhere else.**
3. **No spaces in NSSM-visible paths.** `D:\data\<subdomain>\` is the
   convention.
4. **The Cloudflared service needs `--config` on its binPath.** Already
   applied on `.167`; verify before touching config.
5. **Ports live in the 52xx range, non-adjacent.** Currently taken:
   5008, 5209. Suggested next: 5310, 5411, 5512.

Everything else is elaboration.

---

*Written 2026-07-04. Any specific command in this document was
executed at least once against the live `.167` deploy host during the
2026-07-02 Quantum roll-out or the 2026-06 Oculus roll-out. If you
find a step that doesn't work, that's a regression — file it against
this doc.*
