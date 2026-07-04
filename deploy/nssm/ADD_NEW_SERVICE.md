# Adding a New Service to `cira-core.com`

**Audience:** anyone (you, teammate, ops person, future-you) preparing to
add a fresh service to `cira-core.com` following the same NSSM + Docker
+ Cloudflare Tunnel pattern we shipped for `oculus.cira-core.com`
(2026-06) and `quantum.cira-core.com` (2026-07-02).

**Scope:** generic runbook — pick any subdomain, any container image,
any port. Every step is parameterized. A worked example (CiRA Quantum)
appears at the end so you can see it in action.

**Deploy target:** Windows host running Docker Desktop + NSSM +
Cloudflare Tunnel (`oculus-prod`). All three services already exist on
`.167` (DESKTOP-1A0J7FD). Adding a new service means adding a new
NSSM service, a new Cloudflared ingress rule, a new DNS record —
never a whole new machine or a new tunnel.

**Estimated time:** ~30 min end-to-end if the container image is ready
and you have admin PowerShell + the operator password to `cira-core.com`.

---

## 0. Architecture — what you're building

```
                  cira-core.com  (Cloudflare zone)
                        │
                        │ Tunnel: oculus-prod
                        │ (uuid 94fc52c7-e33b-4f0d-b174-ae18cf2888ff)
                        │
     ┌──────────────────┼──────────────────┬──────────────────┐
     │                  │                  │                  │
oculus.cira-core   quantum.cira-core   NEW_SUB.cira-core   http_status:404
     :5008              :5209              :NEW_PORT      (catch-all)
     │                  │                  │
     ▼                  ▼                  ▼
NSSM: OculusBackendSvc  CiraQuantumSvc  NEW_NSSM_SVC
  → python              → docker         → docker
    backend\run.py        (image)          (image)
```

**Key facts about this shape (learned the hard way, don't fight them):**

- **One Cloudflared service** on `.167` handles every subdomain — you
  add an ingress rule, not a new tunnel.
- **NSSM wraps each backend** so operator ops (`Get-Service`,
  `Restart-Service`, log rotation) work uniformly no matter whether the
  backend is native Python (Oculus) or a Docker container (Quantum).
- **The Cloudflared service runs as `LocalSystem`** and needs an
  explicit `--config` flag pointing at `viset`'s user profile. This is
  already set — you inherit it.
- **DNS is registered on Cloudflare** via `cloudflared tunnel route
  dns …` — no manual A/AAAA/CNAME records needed.

---

## 1. Pick names + parameters

Before you touch anything, decide on these six values. Write them down
somewhere. The rest of the runbook references them literally.

| Placeholder | Example (Quantum) | Notes |
|-------------|-------------------|-------|
| `NEW_SUB` | `quantum` | The subdomain — becomes `NEW_SUB.cira-core.com`. |
| `NEW_PORT` | `5209` | Host-side port. See §1.1 for how to pick. |
| `NEW_NSSM_SVC` | `CiraQuantumSvc` | Windows service name. PascalCase. |
| `NEW_CONTAINER` | `cira-quantum` | Docker container name. kebab-case. |
| `NEW_IMAGE` | `cira-quantum-backend:local` | Docker image reference. |
| `NEW_DATA_DIR` | `D:\data\cira-quantum` | Persistent SQLite / secrets. |

### 1.1 Picking `NEW_PORT`

Rule: **non-adjacent to existing ports** so a config drift or port
scanner doesn't accidentally collide.

Currently in use on `.167`:

| Port | Service |
|------|---------|
| 5008 | Oculus (native Python) |
| 5209 | CiRA Quantum (Docker) |

Suggested next slots (52xx range = CiRA services convention):
- **5310, 5411, 5512, 5613** … — non-adjacent spacing lets each
  service pick a memorable number.
- **Avoid** ports below 5000 (system reserved), and **avoid** 5008 /
  5209 (already in use). Sequential picks (5210, 5211) are also OK
  but the 52xx-range spacing is easier for muscle memory.

### 1.2 Picking `NEW_DATA_DIR`

Rule: **outside your repo checkout**, on a persistent disk, ideally
under `D:\data\<service>\` for consistency with the two existing
services.

Contents:
- `app.db` (or your service's persistence — SQLite by convention)
- `.env` (secrets — never committed)
- Any user-uploaded artifacts

Rule: **no spaces in the path.** NSSM stores docker-run arguments as
one string on the service command line; unquoted whitespace makes
Windows argument parsing split it. `D:\data\my-service\` = fine.
`D:\My Service\` = the docker run silently fails when NSSM starts
the service.

---

## 2. Prerequisites checklist

Before you start, tick every one of these. If any fails, stop and fix
before continuing.

- [ ] You have admin PowerShell access to `.167` (needed for NSSM
      install + Restart-Service Cloudflared).
- [ ] Docker Desktop is running on `.167` (systray icon green).
- [ ] NSSM is installed at `C:\Users\viset\AppData\Local\Microsoft\WinGet\Packages\NSSM.NSSM_Microsoft.Winget.Source_8wekyb3d8bbwe\nssm-2.24-101-g897c7ad\win64\nssm.exe`.
- [ ] Cloudflared is running as a Windows service and reachable via
      `Get-Service Cloudflared` returning `Running`.
- [ ] The Cloudflared service `binPath` includes the `--config` flag
      (see §7.1 for the exact fix if it doesn't).
- [ ] Your new Docker image builds cleanly and you have it tagged
      locally (see §3.1).
- [ ] Your image includes a `/api/health` endpoint (or an equivalent)
      that returns 200 within ~30 s of container boot.
- [ ] You've decided on the six parameters from §1 and written
      them down.

---

## 3. Prepare the Docker image

### 3.1 Build it locally (on `.167` or on your dev box)

Your `Dockerfile` doesn't need to look like ours — but the shape below
proved out with 3+ months of production use, so borrow what fits:

```dockerfile
# Multi-stage: keep the runtime image small
FROM python:3.12-slim AS builder
# ... install deps into /wheels ...

FROM node:20-slim AS frontend-builder   # only if you have a SPA
# ... npm ci + build ...

FROM python:3.12-slim AS runtime
# ... install wheels, copy backend, copy dist ...
ENV PORT=NEW_PORT
EXPOSE NEW_PORT
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl --fail --silent "http://127.0.0.1:${PORT}/api/health" || exit 1
CMD ["gunicorn", "--config", "/app/gunicorn_conf.py", "app:create_app()"]
```

Then:

```powershell
docker build -f deploy/Dockerfile -t NEW_IMAGE .
docker images NEW_IMAGE   # confirm it's there
```

### 3.2 ⚠️ Put `.dockerignore` at the REPO ROOT, not next to the Dockerfile

**Critical.** Docker only reads `.dockerignore` from the build context
root (the directory passed as the final `docker build ... .` argument).
A `.dockerignore` anywhere else — including next to your Dockerfile in
a `deploy/` subfolder — is silently ignored. No error, no warning.

The 2026-07-02 CiRA Quantum deploy shipped an image with the dev
SQLite DB (encrypted BYOK keys and all) baked into every image layer
for hours before we noticed, because our `.dockerignore` was in the
wrong place. Don't repeat that.

**Contents** should exclude at minimum:

```
.git
.gitignore
.env
.env.*
!.env.example
node_modules
frontend/node_modules
frontend/dist
backend/data
backend/**/*.db
backend/**/*.log
```

### 3.3 Test the image locally before shipping

```powershell
docker run --rm -p NEW_PORT:NEW_PORT `
    -e SECRET_KEY="local-dev" `
    -e SESSION_COOKIE_SECURE=false `
    NEW_IMAGE
# then in another shell:
curl http://127.0.0.1:NEW_PORT/api/health
```

Should return 200 within ~30 s. Ctrl+C when done.

---

## 4. Provision host resources

```powershell
# From any admin PowerShell.
New-Item -ItemType Directory -Path 'NEW_DATA_DIR' -Force
New-Item -ItemType Directory -Path 'D:\logs\NEW_SUB' -Force
```

Then create the `.env` file at `NEW_DATA_DIR\.env`:

```dotenv
# ---- Secrets ----------------------------------------------------------
# Generate fresh values with:
#   python -c "import secrets; print(secrets.token_urlsafe(64))"
SECRET_KEY=<generate a fresh 64-byte urlsafe token>

# Only if your service does BYOK Fernet encryption:
# python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
KEY_ENCRYPTION_SECRET=<generate a 32-byte fernet key>

# ---- Runtime toggles --------------------------------------------------
# Cookie hardening. On because Cloudflare Tunnel gives us HTTPS.
SESSION_COOKIE_SECURE=true

# Your service-specific env vars go here.

PORT=NEW_PORT
```

**Never commit this file.** It stays only on `.167`.

**Path convention:** `.env` lives in `NEW_DATA_DIR` (co-located with
persistent data), NOT next to the repo. This avoids two problems at
once: (1) `git pull` never touches secrets, (2) NSSM's docker-run
argument parsing doesn't have to deal with spaces in a path like
`D:\CiRA Whatever\.env`.

---

## 5. Write the NSSM install script

Copy `deploy/nssm/install_quantum_service.ps1` as a starting point and
adapt the top parameter block. The rest of the script is fine as-is;
it's already been debugged for the three PowerShell 5.1 gotchas we
discovered on the Quantum deploy (see §7).

Minimum edits at the top of the script:

```powershell
$svc         = 'NEW_NSSM_SVC'          # was 'CiraQuantumSvc'
$image       = 'NEW_IMAGE'             # was 'cira-quantum-backend:local'
$container   = 'NEW_CONTAINER'         # was 'cira-quantum'
$port        = NEW_PORT                # was 5209
$datadir     = 'NEW_DATA_DIR'          # was 'D:\data\cira-quantum'
$logdir      = 'D:\logs\NEW_SUB'       # was 'D:\logs\cira-quantum'
$env_file    = 'NEW_DATA_DIR\.env'     # was 'D:\data\cira-quantum\.env'
```

**Do NOT change:**
- `$nssm` path (winget-installed location, same on every machine)
- `$docker` path (Docker Desktop default install location)
- The `_sudo`-avoidance patterns for stop/remove (they exist to work
  around a PowerShell 5.1 stderr wrapping bug)
- The `AppStopMethod*` values (needed for graceful SSE drain if your
  service uses streaming; harmless otherwise)
- The `com.docker.service` dependency (ensures Docker Desktop is up
  before your container tries to start on boot)

Save the file as `deploy/nssm/install_NEW_SUB_service.ps1`. Commit it
to git.

---

## 6. Install the NSSM service

**Elevated PowerShell (Run as Administrator).**

```powershell
cd "D:\your-repo-path"
.\deploy\nssm\install_NEW_SUB_service.ps1
```

Expected output at the end:

```
----- Service status -----
Name              Status  StartType
----              ------  ---------
NEW_NSSM_SVC      Running Automatic

----- Health probe -----
Health check: HTTP 200
```

**If the script fails** with `nssm.exe : Administrator access is
needed to install a service` — you're not elevated. Right-click
PowerShell → Run as Administrator → retry.

**If the service comes up in `Paused` state** — check
`D:\logs\NEW_SUB\svc.err.log`. The most common cause: NSSM's
AppParameters got mangled by a path-with-space. Confirm your
`NEW_DATA_DIR` and `$env_file` have no spaces.

---

## 7. Add the Cloudflared ingress rule

### 7.1 Verify the `--config` flag on the Cloudflared service

**Do this once** (it's already correct as of 2026-07-02, but worth
verifying before touching config):

```powershell
Get-CimInstance Win32_Service -Filter "Name='Cloudflared'" | Select-Object PathName
```

Expected output:

```
"C:\Program Files (x86)\cloudflared\cloudflared.exe" --config C:\Users\viset\.cloudflared\config.yml tunnel run oculus-prod
```

If `--config` is **missing**, the tunnel silently ignores our local
config file and falls through to Cloudflare-managed config (which is
empty). Every request 404s until you fix this:

```powershell
# Elevated PowerShell:
sc.exe config Cloudflared binPath= '"C:\Program Files (x86)\cloudflared\cloudflared.exe" --config C:\Users\viset\.cloudflared\config.yml tunnel run oculus-prod'
Restart-Service Cloudflared -Force
```

### 7.2 Edit the Cloudflared config

Backup first:

```powershell
Copy-Item 'C:\Users\viset\.cloudflared\config.yml' `
          "C:\Users\viset\.cloudflared\config.yml.bak-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
```

Then edit `C:\Users\viset\.cloudflared\config.yml` to add your ingress
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

  # ---- NEW SERVICE ----
  - hostname: NEW_SUB.cira-core.com
    service: http://localhost:NEW_PORT
    originRequest:
      noTLSVerify: true
      connectTimeout: 300s     # generous for cold-boot import time

  - service: http_status:404
```

**Why `connectTimeout: 300s`?** Cloudflared defaults to 30 s for the
TCP connect from the tunnel to your local origin. On a cold-booted
container, Python imports (pyqpanda3, qiskit, PennyLane, ...) can
take 15-30 s before your Flask app is ready to accept connections.
30 s risks 502 during that window. 300 s is generous. It does NOT
affect long-lived streams — that's `keepAliveTimeout` (defaults to
1h30m, plenty for SSE).

### 7.3 Validate the config before restarting

```powershell
cloudflared --config 'C:\Users\viset\.cloudflared\config.yml' tunnel ingress validate
cloudflared --config 'C:\Users\viset\.cloudflared\config.yml' tunnel ingress rule https://NEW_SUB.cira-core.com/api/health
```

Expected:
- `Validating rules from ... OK`
- `Matched rule #N ... hostname: NEW_SUB.cira-core.com, service: http://localhost:NEW_PORT`

If validation fails, fix the YAML before restarting Cloudflared —
otherwise Oculus + Quantum will go down with your bad rule.

### 7.4 Register the DNS route

```powershell
cloudflared tunnel route dns oculus-prod NEW_SUB.cira-core.com
```

**Only needs to run once** per hostname. Creates a CNAME in Cloudflare
pointing `NEW_SUB.cira-core.com` at the tunnel. Cloudflare propagates
globally in ~1 min.

### 7.5 Restart Cloudflared

```powershell
# Elevated PowerShell:
Restart-Service Cloudflared -Force
```

**⚠️ This briefly interrupts Oculus + Quantum** — cloudflared takes
5-10 s to reconnect its four edge tunnels. Do this during a low-usage
window if possible.

---

## 8. Verify end-to-end

```powershell
# Wait ~10 s for Cloudflare edge to pick up the new rule.
Start-Sleep -Seconds 10

# Health check via the tunnel:
curl -s https://NEW_SUB.cira-core.com/api/health

# Should return your service's health response, HTTP 200.

# Regression check the existing services didn't break:
curl -sI https://oculus.cira-core.com/api/system-status | Select-Object -First 1
curl -sI https://quantum.cira-core.com/api/health | Select-Object -First 1
# Both should return HTTP 200.
```

If any of those fail, see **§10 Troubleshooting**.

---

## 9. Add an UptimeRobot monitor

Manual — log in to UptimeRobot:

- Type: **HTTP(S)**
- Friendly name: `<Service name>`
- URL: `https://NEW_SUB.cira-core.com/api/health`
- Monitoring interval: **5 minutes**
- Alert contacts: same phone push target as Oculus + Quantum.

Confirmation: after ~5 min the monitor shows green in the UptimeRobot
dashboard.

---

## 10. Troubleshooting

### 10.1 Service is `Paused` (not `Running`)

The most common cause on the Quantum deploy was `docker run` failing
before NSSM could establish that the service was healthy. Check:

```powershell
Get-Content -Tail 30 'D:\logs\NEW_SUB\svc.err.log'
```

Likely errors:
- `docker: open D:\… : The system cannot find the file specified` —
  path-with-space got split. Move your `.env` under `NEW_DATA_DIR`
  (§4) and re-run the install script.
- `Cannot connect to the Docker daemon` — Docker Desktop isn't
  running. Start it, wait for the systray icon to go green, then
  `Restart-Service NEW_NSSM_SVC`.
- `image not found` — you haven't built or pulled `NEW_IMAGE` yet.

### 10.2 `curl https://NEW_SUB.cira-core.com/api/health` returns 404

The 404 with an empty body is Cloudflare's `http_status:404` fallback
firing because no ingress rule matched. Two candidates:

1. **DNS didn't propagate yet.** Wait 60 s, retry.
2. **Cloudflared didn't reload the config.** Check its uptime:
   ```powershell
   Get-Service Cloudflared | Select-Object Name, Status
   ```
   If it says `Running` but the change was made after boot, the config
   is stale. `Restart-Service Cloudflared -Force`.
3. **The `--config` flag is missing on the service** (§7.1). Symptoms:
   `cloudflared tunnel ingress validate` on the CLI says OK, but the
   running service is loading a different config or Cloudflare-managed
   config. Fix per §7.1 and restart.

### 10.3 Service starts, health probe fails with `Connection refused`

The container is running but not listening. Check:

```powershell
# Inside the container:
docker exec NEW_CONTAINER curl -v http://127.0.0.1:NEW_PORT/api/health
```

If that also fails, the container's app isn't binding to `0.0.0.0` or
isn't binding to `NEW_PORT`. Check your app's config for `HOST=0.0.0.0`
and `PORT=NEW_PORT` in the `.env`.

### 10.4 Health probe returns 200, but the SPA at `/` returns 404

Your Flask factory isn't mounting the SPA. If you're following our
pattern:

- Set `CIRA_SPA_DIR=/app/frontend/dist` in the container's env (baked
  into the image via `ENV` in the Dockerfile).
- Ensure your `create_app()` factory has a `_register_spa()` block
  (see `backend/app/__init__.py` in this repo for the pattern).

### 10.5 Container starts, but Fernet-encrypted secrets fail to decrypt

The `KEY_ENCRYPTION_SECRET` in your `.env` doesn't match what encrypted
the secrets in the SQLite DB. Two paths:

1. **Fresh service, no data to preserve** — start clean, re-enter
   secrets via UI.
2. **Migrated from another environment** — set `KEY_ENCRYPTION_SECRET`
   to the value that was in effect when the DB was written. See the
   Quantum → `.141` demo migration for the pattern.

---

## 11. Ongoing operations

### 11.1 Restart

```powershell
Restart-Service NEW_NSSM_SVC
```

Downtime: ~10-30 s while the container recreates and boots.

### 11.2 Redeploy code changes

Assuming your redeploy loop is `git pull` → rebuild image →
restart service:

```powershell
cd "D:\your-repo-path"
git pull
docker build -f deploy/Dockerfile -t NEW_IMAGE .
Restart-Service NEW_NSSM_SVC
```

### 11.3 Tail logs

```powershell
# NSSM-captured stdout (persistent, rotates at 10 MB):
Get-Content -Wait 'D:\logs\NEW_SUB\svc.out.log'

# Container's own stdout (ephemeral, only current run):
docker logs -f NEW_CONTAINER
```

### 11.4 Enter the running container

```powershell
docker exec -it NEW_CONTAINER bash
```

Useful for debugging DB state, testing outbound network, tailing
in-container files.

### 11.5 Rotate secrets

Edit `NEW_DATA_DIR\.env` → `Restart-Service NEW_NSSM_SVC`. Container
picks up the new env at boot.

### 11.6 Manual DB backup

```powershell
Copy-Item "NEW_DATA_DIR\app.db" `
          "D:\backups\NEW_SUB\app.db.$(Get-Date -Format 'yyyyMMdd-HHmmss').bak"
```

Consider a Task Scheduler entry doing this hourly.

---

## 12. Rollback / removal

To remove the service entirely:

```powershell
# Elevated PowerShell:
Stop-Service NEW_NSSM_SVC -Force
& 'C:\Users\viset\AppData\Local\Microsoft\WinGet\Packages\NSSM.NSSM_Microsoft.Winget.Source_8wekyb3d8bbwe\nssm-2.24-101-g897c7ad\win64\nssm.exe' remove NEW_NSSM_SVC confirm
docker rm -f NEW_CONTAINER
docker rmi NEW_IMAGE
```

Then remove the ingress rule from `C:\Users\viset\.cloudflared\config.yml`
and `Restart-Service Cloudflared -Force`.

DNS: `cloudflared tunnel route dns oculus-prod NEW_SUB.cira-core.com`
CANNOT be undone via CLI (Cloudflare's UI can delete the CNAME, but
it's harmless to leave it pointing at the tunnel — if there's no
matching ingress rule, requests just fall through to the 404 catch-all).

Data (`NEW_DATA_DIR`) and logs (`D:\logs\NEW_SUB\`) are yours to keep
or delete.

---

## 13. Gotchas — the eight non-obvious truths

Every one of these caught us during the Quantum deploy on 2026-07-02.
Written here so you don't have to rediscover them.

### 13.1 `.dockerignore` MUST live at the build context root

Not next to the Dockerfile. Not in a `deploy/` subfolder. Docker
silently ignores it anywhere else — no warning, no error. Your dev
SQLite DB will end up baked into every image layer.

### 13.2 Cloudflared as LocalSystem needs `--config` on the binPath

`LocalSystem`'s `%USERPROFILE%` is `C:\Windows\System32\config\systemprofile\`
which has no `.cloudflared\config.yml`. Without the explicit `--config`
flag, Cloudflared falls through to Cloudflare-managed config (empty)
and every request returns 404. Fix via `sc.exe config Cloudflared binPath= ...`.

### 13.3 PowerShell 5.1 `2>&1` on native execs breaks $?

Windows PowerShell 5.1 wraps native command stderr as an ErrorRecord
and sets `$?` to `$false` **even when the exe returned exit code 0**.
This trips `$ErrorActionPreference = 'Stop'` on idempotent
`nssm stop` / `nssm remove` calls (which legitimately fail on the
first install). Wrap in `try/catch` with EAP briefly relaxed.

### 13.4 Backticks in `throw` messages can break the parser

Original Quantum install script had `` `winget install NSSM.NSSM` ``
(double-backticks for markdown-style inline code) inside double-quoted
strings. The parser sometimes interpreted this as an escape sequence
and failed with *"Missing closing '}'"* at seemingly unrelated lines.
Solution: remove backticks from all error strings. Use single quotes
(`'winget install NSSM.NSSM'`) instead.

### 13.5 NSSM AppParameters + Windows arg parsing + unquoted spaces

NSSM stores docker-run arguments as one string on `docker.exe`'s
command line. Windows argument parsing splits unquoted whitespace.
`--env-file D:\CiRA Quantum\.env` becomes `--env-file D:\CiRA` +
`Quantum\.env` (stray argument). Embedded double-quotes get stripped
by various layers of the escape chain. **Solution: keep all NSSM-visible
paths space-free.** Not a fix in code — a naming convention.

### 13.6 `dill` is a runtime dep of qiskit's passmanager

If you build a slim Docker image and strip pyqpanda_alg's over-declared
dev tooling (sphinx, mypy, astroid, etc.), do NOT include `dill` in the
uninstall list. `qiskit.passmanager` imports it at module load time.
Removing it silently breaks `qiskit_ibm_runtime` — you won't see the
error until someone tries the IBMQ tier.

### 13.7 `vue-tsc 1.8` is incompatible with TypeScript 5.4+

If your frontend has `"vue-tsc": "^1.8.27"` in `package.json` AND
allows `"typescript": "^5.3.0"`, a fresh `npm ci` in a Docker build
context resolves TypeScript 5.4+, and vue-tsc's monkey-patch of TS
internals fails with *"Search string not found:
/supportedTSExtensions = .*(?=;)/"*.

Fixes:
1. Upgrade to `vue-tsc: ^2.x` (proper long-term fix).
2. Bypass type-check in the Docker build by running `npx vite build`
   directly instead of `npm run build` (which chains
   `vue-tsc --noEmit && vite build`).

### 13.8 Container UID vs host mount UID

If your Dockerfile uses `USER cira` with UID 10001 (or any non-root
UID), the host mount at `NEW_DATA_DIR` must be chowned to match.
Otherwise the container can read but not write — SQLite writes fail
with `attempt to write a readonly database`.

Fix on Linux hosts:

```bash
sudo chown -R 10001:10001 /var/lib/NEW_SUB
```

On Windows hosts under Docker Desktop, this usually just works (WSL2
handles UID mapping) — but check if your service can write to
`app.db` before assuming.

---

## 14. Worked example — CiRA Quantum

The full Quantum deploy, side-by-side with the placeholder table above:

| Placeholder | Value |
|-------------|-------|
| `NEW_SUB` | `quantum` |
| `NEW_PORT` | `5209` |
| `NEW_NSSM_SVC` | `CiraQuantumSvc` |
| `NEW_CONTAINER` | `cira-quantum` |
| `NEW_IMAGE` | `cira-quantum-backend:local` |
| `NEW_DATA_DIR` | `D:\data\cira-quantum` |

Artifacts in this repo:

- **`deploy/nssm/install_quantum_service.ps1`** — the real install
  script. Copy this to `install_NEW_SUB_service.ps1` and adapt the
  top parameter block.
- **`deploy/nssm/cloudflared_ingress.yml`** — reference ingress rule
  format.
- **`deploy/nssm/DEPLOY_NSSM.md`** — the original Quantum-specific
  runbook. Now that you have this generic guide, DEPLOY_NSSM.md
  serves as a reference for the specific Quantum choices.
- **`deploy/Dockerfile`** — worked example of a multi-stage image
  with a frontend build stage baked in.
- **`.dockerignore`** (repo root!) — worked example, note the
  prominent warning at the top.

---

## 15. Operator cheat sheet

Print this. Stick it on the monitor.

| Action | Command |
|--------|---------|
| Check service status | `Get-Service NEW_NSSM_SVC` |
| Start / stop / restart | `Start-Service …` / `Stop-Service …` / `Restart-Service …` |
| Tail NSSM stdout | `Get-Content -Wait D:\logs\NEW_SUB\svc.out.log` |
| Tail NSSM stderr | `Get-Content -Wait D:\logs\NEW_SUB\svc.err.log` |
| Tail container stdout | `docker logs -f NEW_CONTAINER` |
| Container shell | `docker exec -it NEW_CONTAINER bash` |
| Redeploy code | `git pull; docker build …; Restart-Service NEW_NSSM_SVC` |
| Rotate secrets | edit `NEW_DATA_DIR\.env`, `Restart-Service NEW_NSSM_SVC` |
| Validate cloudflared config | `cloudflared --config C:\Users\viset\.cloudflared\config.yml tunnel ingress validate` |
| Restart cloudflared | `Restart-Service Cloudflared -Force` (elevated) |
| Manual DB backup | `Copy-Item NEW_DATA_DIR\app.db D:\backups\NEW_SUB\app.db.$(Get-Date -Format 'yyyyMMdd-HHmmss').bak` |

---

*Written 2026-07-04, based on the 2026-07-02 CiRA Quantum deploy.
Every rule, gotcha, and workaround here has been paid for at least
once with real debugging time. Trust the runbook; the shortcuts
are usually mistakes.*
