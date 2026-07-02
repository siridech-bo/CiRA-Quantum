# install_quantum_service.ps1 — install CiRA Quantum as an NSSM
# Windows service on .110, matching the Oculus pattern.
#
# Adapted from the operator's D:\tmp\install_oculus_service_v2.ps1
# with two changes:
#   1. Waitress instead of Flask-dev-server / gunicorn (gunicorn is
#      Unix-only; waitress is Windows-native, pure-Python, thread-pool
#      based — safe for our SSE streams).
#   2. AppStopMethod* bumped to 15 s so in-flight SSE streams get a
#      chance to close cleanly on ``nssm stop`` / ``sc stop``.
#
# Run from an elevated PowerShell (Administrator) on .110 after the
# preflight steps in deploy/nssm/DEPLOY_NSSM.md are complete:
#   1. repo cloned to $workdir
#   2. venv created under $workdir\venv and deps installed
#   3. frontend built (``npm run build`` in frontend/)
#   4. .env file written under $workdir with real secrets
#   5. data + log directories exist ($datadir, log dir)
#
# Idempotent: stopping + removing an existing service before install
# so ``iwr | iex`` re-runs work cleanly during ops iterations.

$ErrorActionPreference = 'Stop'

# ---- Path configuration -----------------------------------------------
# All configurable in one block so an operator can retarget to a
# different drive / venv / port without hunting through the script.

$svc      = 'CiraQuantumSvc'
$workdir  = 'D:\services\cira-quantum'
$datadir  = 'D:\data\cira-quantum'
$logdir   = 'D:\logs\cira-quantum'
$port     = 5209                            # deliberately non-adjacent to Oculus (5008) to prevent collisions
$spa_dir  = "$workdir\frontend\dist"
$env_file = "$workdir\.env"

# NSSM location — matches Oculus's install path. Winget places it here.
$nssm = 'C:\Users\viset\AppData\Local\Microsoft\WinGet\Packages\NSSM.NSSM_Microsoft.Winget.Source_8wekyb3d8bbwe\nssm-2.24-101-g897c7ad\win64\nssm.exe'

# Python interpreter: venv under the service directory (cleaner than
# Oculus's system-Python pattern; each service owns its own dep tree).
$python = "$workdir\venv\Scripts\python.exe"

# Waitress invocation — bind on all interfaces, thread-pool sized for
# SSE + long LLM/QPU calls, expose the WSGI callable via --call so
# our create_app() factory is invoked at boot.
$appargs = "-m waitress --host=0.0.0.0 --port=$port --threads=16 --call app:create_app"

$logout = "$logdir\svc.out.log"
$logerr = "$logdir\svc.err.log"

# ---- Preflight checks -------------------------------------------------

if (-not (Test-Path $nssm)) {
    throw "NSSM not found at $nssm. Install via ``winget install NSSM.NSSM`` first."
}
if (-not (Test-Path $python)) {
    throw "Python interpreter not found at $python. Create the venv per DEPLOY_NSSM.md step 3 first."
}
if (-not (Test-Path $env_file)) {
    Write-Warning ".env not found at $env_file — service will start but the pipeline will fail at first LLM call. Create it before starting."
}
if (-not (Test-Path $spa_dir)) {
    Write-Warning "Frontend build not found at $spa_dir — the / route will fail. Run ``npm run build`` in frontend/ before starting."
}

# ---- Ensure host paths exist -----------------------------------------

New-Item -ItemType Directory -Path $datadir -Force | Out-Null
New-Item -ItemType Directory -Path $logdir  -Force | Out-Null

# ---- Idempotent removal ---------------------------------------------
# Stop + remove an existing service if present so a re-run of this
# script upgrades cleanly. ``2>&1 | Out-Null`` swallows the "service
# not found" noise on a fresh install.

& $nssm stop   $svc 2>&1 | Out-Null
Start-Sleep -Seconds 2
& $nssm remove $svc confirm 2>&1 | Out-Null

# ---- Install ----------------------------------------------------------

& $nssm install $svc $python $appargs
& $nssm set $svc AppDirectory     $workdir
& $nssm set $svc DisplayName      'CiRA Quantum Backend'
& $nssm set $svc Description      'CiRA Quantum optimization pipeline. Serves quantum.cira-core.com via Cloudflare Tunnel.'

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

# Graceful drain for SSE — give in-flight streams up to 15 s to close
# cleanly on service stop. NEW vs Oculus; Oculus doesn't have SSE.
& $nssm set $svc AppStopMethodConsole 15000
& $nssm set $svc AppStopMethodWindow  15000
& $nssm set $svc AppStopMethodThreads 15000

# Start-at-boot + Tcpip dependency (matches Oculus).
& $nssm set $svc Start            SERVICE_AUTO_START
& $nssm set $svc DependOnService  Tcpip

# ---- Start + verify ---------------------------------------------------

& $nssm start $svc
Start-Sleep -Seconds 6

Write-Host ""
Write-Host "----- Service status -----" -ForegroundColor Cyan
Get-Service $svc | Format-Table Name, Status, StartType -AutoSize

Write-Host "----- Health probe -----" -ForegroundColor Cyan
try {
    $r = Invoke-WebRequest -Uri "http://127.0.0.1:$port/api/health" -UseBasicParsing -TimeoutSec 5
    Write-Host ("Health check: HTTP {0}" -f $r.StatusCode) -ForegroundColor Green
    Write-Host $r.Content
} catch {
    Write-Warning "Health probe failed: $_"
    Write-Warning "Check the service log at $logerr for the failure reason."
}

Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Add the cira-quantum ingress rule to cloudflared config"
Write-Host "     (see deploy\nssm\cloudflared_ingress.yml)"
Write-Host "  2. Register the DNS route:"
Write-Host "       cloudflared tunnel route dns oculus-prod quantum.cira-core.com"
Write-Host "  3. Restart the cloudflared service"
Write-Host "  4. Verify from off-LAN:"
Write-Host "       curl -s https://quantum.cira-core.com/api/health"
Write-Host "  5. Add UptimeRobot monitor for that URL at 5-min interval"
