# Deploy QRC to `quantum.cira-core.com/qrc`

**Goal:** make the QRC control plane + Feature-Lab UI reachable at
`quantum.cira-core.com/qrc` so runs can be launched and watched from a browser
off-box.

**Source:** branch **`qrc-simulation`**, HEAD **`5c5bfc8`** (or later on that
branch). Deploy the CiRA Quantum service from this branch.

---

## Key fact — it's the same app, not a new service

`qrc_bp` is registered in the existing `create_app()` at `/api/qrc` (next to
`qml_bp`), and `/qrc` is a route in the same Vue SPA. The Dockerfile's frontend
stage already runs the vite build → `/app/frontend/dist`, so the QRC pages +
Feature-Lab ship automatically.

- **No subdomain, no path-split ingress, no second instance.**
  `quantum.cira-core.com/qrc` and `/api/qrc` are served by the one container at
  `:5209`, which the existing `oculus-prod` tunnel already routes to.
- One instance on one origin → **login/session cookies just work**; there is no
  cross-instance auth to solve.

The host serving `quantum.cira-core.com` is `DESKTOP-1A0J7FD` /
`192.168.1.167`, and it **has** the RTX 5070 Ti. QRC's GPU compute runs there.

---

## The one real decision — GPU visibility

QRC's trace-gen / phase1 need the GPU on the host. A host GPU is **not** visible
inside a container by default. Choose one:

- **(a) Container with GPU passthrough** — NVIDIA Container Toolkit + `--gpus
  all`, WSL2 backend, and a **CUDA-enabled torch** in the image.
- **(b) Run this one service native (NSSM → python)** like `OculusBackendSvc`,
  where the host's already-working CUDA torch is visible with zero extra config.
  **Recommended** — the GPU path is known-good on this box natively;
  containerizing CUDA + torch + passthrough is the harder route.

---

## Two must-not-miss requirements (either option)

1. **Install extras `[qrc]` and `[featurelab]`** — `pip install ".[qrc,featurelab]"`.
   - `[qrc]` → the multimodal feature stack.
   - `[featurelab]` → `umap-learn` for the UMAP embedding view.
   - Ensure **torch is a CUDA build** (not CPU-only) or GPU evolution mode won't
     engage. Verify the wheels/requirements the image installs actually include
     these — the current prod image may not.
2. **Persistent volume for `artifacts/`.** The launcher writes the run registry
   + per-run dirs + the trace cache (the expensive ~19 h `.npz`) under
   `backend/artifacts/{qrc_runs,traces}`. Mount a volume there, or set env
   `QRC_RUNS_DIR` / `QRC_TRACES_DIR` to a mounted path, so traces + run history
   survive restarts. **Losing the trace cache = re-running 19 h.**

---

## Gate it

Put **Cloudflare Access** (email OTP) in front of `/qrc` and `/api/qrc`. The QRC
control endpoints are `@login_required`, but the read-only ones
(list/progress/fid/results/embedding) are public by design — Access is the
outer gate.

## Also (approved separately)

Cherry-pick `5e0d430` (`.110 → .167` deploy-artifact fix) onto `main`.

---

## Verification after deploy

1. `curl -s https://quantum.cira-core.com/api/qrc/runs` → `200`, JSON array.
2. Open `https://quantum.cira-core.com/qrc` → dashboard renders.
3. GPU visible: exec into the running service →
   `python -c "import torch; print(torch.cuda.is_available())"` → `True`.
4. Log in, launch a **tiny** trace-gen (e.g. `subtask=weather, fid_points=64,
   splits=[3,6,3]`) → the run detail page shows a **moving progress bar + ETA +
   event log** (the live-progress wiring), and finishes in seconds.
5. Open that run → **Feature Lab** panel: the PCA/UMAP embedding scatter + sweep
   charts render.

Report back the `torch.cuda.is_available()` result and the tiny-trace outcome.

---

## Context / cross-refs

- Same-machine + architecture facts: repo `CLAUDE.md`.
- Add-a-service runbook (NSSM + tunnel pattern): `deploy/nssm/ADD_NEW_SERVICE.md`.
- Tunnel ingress example: `deploy/nssm/cloudflared_ingress.yml`.
- QRC implementation status: `docs/QRC/QRC_Next_Stage_Status.md`.
