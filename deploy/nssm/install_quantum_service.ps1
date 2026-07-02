# install_quantum_service.ps1 — install CiRA Quantum as an NSSM
# Windows service on .110, wrapping the pre-built Docker image.
#
# Path B (Docker-in-NSSM). All Python deps + the built Vue SPA + the
# pyqpanda3/qiskit/dimod toolchain are baked into the container image
# ``cira-quantum-backend:local`` (see deploy/Dockerfile). This script
# creates a Windows service that, when started, runs the container in
# the foreground; NSSM's restart policy handles container crashes the
# same way it would handle a python.exe crash.
#
# Why NSSM + Docker (vs docker-desktop autostart) — because Docker
# Desktop's own service model doesn't give us NSSM's rotate/log/exit
# policies. Wrapping ``docker run`` in NSSM keeps all the operator
# muscle memory (``Get-Service`` / ``Restart-Service`` / log files at
# a known path) while getting the "everything's in the container"
# simplicity.
#
# Run from an elevated PowerShell (Administrator) on .110 after the
# preflight steps in deploy/nssm/DEPLOY_NSSM.md are complete:
#   1. Docker Desktop is running (systray icon healthy).
#   2. Image ``cira-quantum-backend:local`` exists (``docker images``).
#   3. .env file written at D:\CiRA Quantum\.env with real secrets.
#   4. Data + log directories exist.
#
# Idempotent: any prior CiraQuantumSvc gets stopped + removed before
# install, and any prior container gets removed too — so ``iwr | iex``
# re-runs work cleanly during ops iterations.

$ErrorActionPreference = 'Stop'

# ---- Path configuration -----------------------------------------------
# All configurable in one block so an operator can retarget to a
# different drive / image / port without hunting through the script.

$svc         = 'CiraQuantumSvc'
$image       = 'cira-quantum-backend:local'
$container   = 'cira-quantum'
$port        = 5209
$datadir     = 'D:\data\cira-quantum'
$logdir      = 'D:\logs\cira-quantum'
# .env lives at D:\data\cira-quantum\.env — alongside the SQLite DB
# rather than in the git checkout. Two wins over the earlier
# ``D:\CiRA Quantum\.env`` convention:
#   * No space in the path, so NSSM's AppParameters can pass it to
#     docker.exe without embedded-quote escaping.
#   * The data + secrets live together outside the repo, so a
#     ``git clone`` on a fresh box (or ``rm -rf`` of the repo) can't
#     accidentally touch either.
$env_file    = 'D:\data\cira-quantum\.env'

# NSSM location — matches Oculus's install path. Winget places it here.
$nssm = 'C:\Users\viset\AppData\Local\Microsoft\WinGet\Packages\NSSM.NSSM_Microsoft.Winget.Source_8wekyb3d8bbwe\nssm-2.24-101-g897c7ad\win64\nssm.exe'

# Docker CLI. Docker Desktop installs it here on Windows; if the
# operator installed via a different route (Chocolatey, standalone
# CLI), override this line.
$docker = 'C:\Program Files\Docker\Docker\resources\bin\docker.exe'

# Docker run arguments. NSSM will pass these to docker.exe verbatim.
#   --rm                : remove the container on exit — NSSM restart
#                         recreates it anyway, so we don't want a pile
#                         of "Exited" containers accumulating.
#   --name              : predictable name so ``docker exec`` / ``docker logs``
#                         work without having to look up the id.
#   -p 5209:5209        : bind container's 5209 to host's 5209.
#   --env-file          : load prod secrets from D:\CiRA Quantum\.env
#                         (never baked into the image).
#   -v host:container   : mount the host data dir at /app/data so the
#                         SQLite file, BYOK-encrypted keys, and job
#                         history survive container/image rebuilds.
#                         The CIRA_DB_PATH env var (set inside .env)
#                         must point at /app/data/app.db for this to
#                         work.
$appargs = @(
    'run',
    '--rm',
    '--name', $container,
    '-p', "$($port):$($port)",
    '--env-file', $env_file,
    '-v', "$($datadir):/app/data",
    $image
) -join ' '

$logout = "$logdir\svc.out.log"
$logerr = "$logdir\svc.err.log"

# ---- Preflight checks -------------------------------------------------

if (-not (Test-Path $nssm)) {
    throw "NSSM not found at $nssm. Install via 'winget install NSSM.NSSM' first."
}
if (-not (Test-Path $docker)) {
    throw "Docker CLI not found at $docker. Is Docker Desktop installed and running?"
}

# Docker daemon must be up. If the daemon isn't reachable, the
# service would install fine but fail immediately on start. We use
# ``docker version`` (no --format flag) because certain Go-template
# braces in --format make the outer PowerShell parser choke.
& $docker version 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Docker daemon not reachable. Start Docker Desktop and wait for the systray icon to go green, then re-run this script."
}

# Image must exist. Building it here would blow past the "1 hour
# ops-only" deploy budget, so we require the operator to have built
# it separately (see DEPLOY_NSSM.md step 3).
& $docker image inspect $image 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Docker image '$image' not found. Build it with: docker build -f deploy/Dockerfile -t $image .  from the repo root, then re-run."
}

if (-not (Test-Path $env_file)) {
    Write-Warning ".env not found at $env_file - service will start but the pipeline will fail at first LLM call. Create it before proceeding (see DEPLOY_NSSM.md step 4)."
}

# ---- Ensure host paths exist -----------------------------------------

New-Item -ItemType Directory -Path $datadir -Force | Out-Null
New-Item -ItemType Directory -Path $logdir  -Force | Out-Null

# ---- Idempotent removal ---------------------------------------------
# Stop + remove any existing service AND any dangling container so a
# re-run cleanly upgrades. We expect these to fail on the first
# install ("Can't open service!" from nssm, "No such container" from
# docker) — Windows PowerShell 5.1 wraps native stderr in an
# ErrorRecord regardless of redirection, so we wrap in try/catch
# with $ErrorActionPreference briefly relaxed instead of fighting it.

$saved_eap = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
try { & $nssm stop   $svc 2>&1 | Out-Null } catch { }
Start-Sleep -Seconds 2
try { & $nssm remove $svc confirm 2>&1 | Out-Null } catch { }

# If the container is still up from a prior run (e.g. NSSM was
# force-killed without stopping the container), remove it too.
try { & $docker rm -f $container 2>&1 | Out-Null } catch { }
$ErrorActionPreference = $saved_eap

# ---- Install ----------------------------------------------------------

& $nssm install $svc $docker $appargs
& $nssm set $svc AppDirectory     'C:\'
& $nssm set $svc DisplayName      'CiRA Quantum Backend (Docker)'
& $nssm set $svc Description      'CiRA Quantum optimization pipeline. Wraps docker run for cira-quantum-backend:local; serves quantum.cira-core.com via Cloudflare Tunnel.'

# I/O
& $nssm set $svc AppStdout        $logout
& $nssm set $svc AppStderr        $logerr

# NSSM's built-in log rotation — 10 MB per file, online rotation
# (no service downtime), files enabled. Matches Oculus.
& $nssm set $svc AppRotateFiles   1
& $nssm set $svc AppRotateOnline  1
& $nssm set $svc AppRotateBytes   10000000

# Restart policy — any exit → restart after 5 s, back off if crashing
# more than once per 60 s. Matches Oculus.
& $nssm set $svc AppExit Default  Restart
& $nssm set $svc AppRestartDelay  5000
& $nssm set $svc AppThrottle      60000

# Graceful drain: give the container 15 s to receive SIGTERM (docker
# stop sends it) and drain in-flight SSE streams before docker kill
# (SIGKILL) fires. NSSM's own AppStop* controls how it signals the
# docker.exe process, which forwards SIGTERM to the container's PID 1.
& $nssm set $svc AppStopMethodConsole 15000
& $nssm set $svc AppStopMethodWindow  15000
& $nssm set $svc AppStopMethodThreads 15000

# Start-at-boot + Tcpip + Docker service dependencies. Docker Desktop
# on Windows registers its daemon as ``com.docker.service``; we depend
# on it so NSSM waits until docker is up before starting the container.
& $nssm set $svc Start            SERVICE_AUTO_START
& $nssm set $svc DependOnService  Tcpip com.docker.service

# ---- Start + verify ---------------------------------------------------

& $nssm start $svc

# Give the container time to boot — ~30 s for pyqpanda3 + qiskit + the
# hardcoded formulator registrations to complete on first run.
Write-Host "Waiting up to 45s for the container to come healthy..." -ForegroundColor Cyan
$ready = $false
for ($i = 0; $i -lt 15; $i++) {
    Start-Sleep -Seconds 3
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:$port/api/health" -UseBasicParsing -TimeoutSec 3
        if ($r.StatusCode -eq 200) {
            $ready = $true
            break
        }
    } catch {
        # Not up yet — normal during the boot window.
    }
}

Write-Host ""
Write-Host "----- Service status -----" -ForegroundColor Cyan
Get-Service $svc | Format-Table Name, Status, StartType -AutoSize

Write-Host "----- Health probe -----" -ForegroundColor Cyan
if ($ready) {
    $r = Invoke-WebRequest -Uri "http://127.0.0.1:$port/api/health" -UseBasicParsing -TimeoutSec 3
    Write-Host ("Health check: HTTP {0}" -f $r.StatusCode) -ForegroundColor Green
    Write-Host $r.Content
} else {
    Write-Warning "Health probe never returned 200 in 45s. Check container logs:"
    Write-Warning "  docker logs $container"
    Write-Warning "Service log at $logerr"
}

Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Add the quantum.cira-core.com ingress rule to cloudflared config"
Write-Host "     (see deploy\nssm\cloudflared_ingress.yml)"
Write-Host "  2. Register the DNS route:"
Write-Host "       cloudflared tunnel route dns oculus-prod quantum.cira-core.com"
Write-Host "  3. Restart the cloudflared service"
Write-Host "  4. Verify from off-LAN:"
Write-Host "       curl -s https://quantum.cira-core.com/api/health"
Write-Host "  5. Add UptimeRobot monitor for that URL at 5-min interval"
