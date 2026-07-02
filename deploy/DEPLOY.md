# CiRA Quantum — Cloudflare deployment guide

Target architecture:

```
                    cira-core.com  (Cloudflare zone)
                         │
             ┌───────────┴───────────┐
             │                       │
        R2 (SPA)              Worker dispatcher
                                     │
                                     ├── /api/*      ──► Container (Flask backend)
                                     ├── everything  ──► R2 SPA bucket
                                     └── /health     ──► handled at edge
                                                       (avoids cold-starting the container)
                     Container ────────► D1 (users / jobs / api_keys)
                                └────► outbound: Anthropic, OpenAI, Origin QC, IBM Q
```

## 1 · One-time setup

### 1.1 Cloudflare account prerequisites

- A paid Workers plan (Containers is a paid-tier feature; the free
  Workers plan doesn't include it).
- `cira-core.com` added to the account as a zone with active DNS.
- Wrangler installed locally: `npm i -g wrangler` (or `pnpm add -g`).
- Signed in: `wrangler login`.

### 1.2 Provision resources

```powershell
# From the repo root — deploy/ artifacts are what wrangler reads.

# D1 database — creates it and prints a UUID.
wrangler d1 create cira-quantum-db
# Paste the returned UUID into deploy/wrangler.toml under
# [[d1_databases]] → database_id. Commit that change.

# R2 buckets — one for the built SPA, one for training artifacts.
wrangler r2 bucket create cira-core-frontend
wrangler r2 bucket create cira-core-artifacts

# Apply the initial schema.
wrangler d1 migrations apply cira-quantum-db --remote
```

### 1.3 Secrets

Never commit these — set them once per environment via `wrangler secret put`:

| Secret                       | Purpose                                          |
|------------------------------|--------------------------------------------------|
| `SECRET_KEY`                 | Flask session signing. Use `python -c "import secrets; print(secrets.token_urlsafe(64))"`. |
| `KEY_ENCRYPTION_SECRET`      | Fernet key for BYOK encryption at rest. **32 bytes**, base64. `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`. |
| `DEFAULT_ADMIN_PASSWORD`     | Admin bootstrap password on the very first DB init. Change immediately after first login. |

```powershell
wrangler secret put SECRET_KEY
wrangler secret put KEY_ENCRYPTION_SECRET
wrangler secret put DEFAULT_ADMIN_PASSWORD
```

BYOK provider keys (Anthropic, OpenAI, Origin, IBM Quantum) are **not**
in this list — they're per-user and stored in the D1 `api_keys` table,
encrypted with `KEY_ENCRYPTION_SECRET`.

### 1.4 Environment (non-secret) values

Set in [`wrangler.toml`](wrangler.toml) under `[containers.env]` — see
that file for the full list. Highlights:

- `ENABLE_HARDCODED_ROUTING=1` — classifier→hardcoded stage 1.
- `ENABLE_ORIGIN_REAL_HARDWARE=1` — real Wukong-180 submissions
  (needed once BYOK Origin keys are in play).
- `SESSION_COOKIE_SECURE=true` — cookie only sent over HTTPS.
- `USE_REDIS_QUEUE=0` — single-container deploy; flip to `1` if you
  later add Cloudflare Queues + Redis.

## 2 · First deploy

```powershell
# From the repo root.
wrangler deploy --config deploy/wrangler.toml
```

Wrangler will:

1. Build the container image using `deploy/Dockerfile` (linux/amd64).
2. Push it to Cloudflare's registry.
3. Register the D1, R2, and container bindings on the dispatcher.
4. Roll out the Worker + container.

First build takes ~10 min because `pyqpanda3` doesn't ship prebuilt
wheels for slim images and has to compile in the builder stage.
Subsequent builds cache that layer and complete in 1–2 min.

## 3 · Frontend publish

```powershell
cd frontend
npm ci
npm run build

# Upload the built SPA to R2. --recursive keeps folder structure.
wrangler r2 object put cira-core-frontend --file dist/index.html --path /
wrangler r2 object put cira-core-frontend --file dist/assets --recursive --path assets/
```

Point the dispatcher at the SPA bucket via bindings — the Worker in
`worker/dispatcher.ts` handles GET requests by streaming from the
`SPA` R2 binding.

## 4 · Post-deploy verification

Run these against the live host:

```powershell
# Health check — should be 200 with a JSON body.
curl https://cira-core.com/api/health

# CORS preflight — should return 204 with Access-Control-Allow-* headers.
curl -X OPTIONS https://cira-core.com/api/solve \
     -H "Origin: https://cira-core.com" \
     -H "Access-Control-Request-Method: POST"

# Session cookie flow — sign in as admin, confirm the cookie is
# Secure + SameSite=Lax.
curl -i -X POST https://cira-core.com/api/auth/login \
     -H "Content-Type: application/json" \
     -d '{"username":"admin","password":"<the DEFAULT_ADMIN_PASSWORD you set>"}'
```

Expected headers on the login response:
```
Set-Cookie: session=...; Secure; HttpOnly; SameSite=Lax; Path=/
```

If `Secure` is missing, `SESSION_COOKIE_SECURE` didn't land — recheck
the `[containers.env]` block.

## 5 · Origin QC connectivity check

`qcloud.originqc.com.cn` sometimes rate-limits by outbound region.
From a container shell (via `wrangler tail --format=pretty` while a
job runs, or via a one-off `wrangler containers exec`), the pyqpanda3
submission should succeed to `WK_C180_2` — if you see
`ConnectionRefused` or a 30-second hang, override the placement hint
in `wrangler.toml` to APAC:

```toml
[placement]
mode = "smart"
hint = "apac"
```

Redeploy and retest. Origin's docs confirm APAC POPs work; US POPs
are unreliable for their cloud endpoints.

## 6 · Migrating existing dev data (optional)

Only relevant if you want your dev SQLite users/jobs to appear on the
live host — for a clean production start, skip this and re-register.

```powershell
# Export dev DB tables to CSV.
python scripts/export_dev_db.py --out ./dev_export

# Import each table into D1. Use --file for each CSV.
wrangler d1 execute cira-quantum-db --remote --file ./dev_export/users.sql
wrangler d1 execute cira-quantum-db --remote --file ./dev_export/api_keys.sql
# ... etc.
```

⚠️  **API keys will only decrypt if `KEY_ENCRYPTION_SECRET` in prod is
identical to your dev value.** If you rotated the secret (recommended),
re-enter keys via Settings → API Keys after logging in.

## 7 · What lives where — cheat sheet

| Concern                          | Where                            |
|----------------------------------|----------------------------------|
| Static SPA                       | R2 `cira-core-frontend` bucket   |
| Backend Flask app                | Container (defined in Dockerfile)|
| Long-lived SSE streams           | Container                        |
| Users / api_keys / jobs / QML    | D1 `cira-quantum-db`             |
| BYOK secrets at rest             | D1 `api_keys.encrypted_key`      |
| Session signing                  | `SECRET_KEY` (wrangler secret)   |
| BYOK Fernet key                  | `KEY_ENCRYPTION_SECRET` (wrangler secret) |
| Rate limiting + edge CORS        | Worker dispatcher                |
| Deployment source of truth       | `deploy/wrangler.toml`           |
| Schema changes                   | `deploy/migrations/`             |

## 8 · Known follow-ups (not blockers for first deploy)

- **`models.py` D1 shim.** The current models.py opens a
  `sqlite3.Connection` and runs `PRAGMA table_info` for the idempotent
  ALTERs. That needs a small D1-flavored shim (`app/db_d1.py`) that
  routes queries through the D1 binding when running on Cloudflare
  and falls back to real SQLite for local dev. Track as a follow-up
  PR — the current image will fail at first request against D1 until
  that shim lands.
- **Frontend build hooks.** Once the initial deploy is verified, wire
  a GitHub Action that runs `wrangler deploy --config deploy/wrangler.toml`
  on merge to `main`, and a second job that runs
  `npm run build && wrangler r2 object put --recursive` for the SPA.
- **Observability dashboard.** Cloudflare's per-container Logs tab is
  enough at first-user scale; add a Grafana / Sentry integration only
  when incident volume justifies it.
- **Removing GPU SA.** The `gpu_sa` solver is excluded from this image
  (no torch, no CUDA), but the code still imports it in
  `bootstrap_default_solvers()`. Guarding the import so its absence
  doesn't spam warnings on container boot is a five-line cleanup.

## 9 · Rollback

```powershell
# List recent deployments — each has an ID.
wrangler deployments list --config deploy/wrangler.toml

# Roll back to a previous ID.
wrangler rollback <deployment-id> --config deploy/wrangler.toml
```

D1 rollback is not automatic — migrations are one-way. If a bad
schema migration lands, hand-craft a forward migration that reverses
the change (drop columns, re-add old defaults, etc.) and apply it as
`000N_revert_<change>.sql`. Never try to `.rollback` D1 by editing
migration history.
