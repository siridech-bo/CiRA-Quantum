# QRC Stage A — Shared Contracts (coders code to THIS, not to each other's live code)

Stage A = Phase 0 (trace cache + eval harness + control API + UI) + Phase 1 experiments, per
`QRC_Next_Stage_Experiments.md`. Corrected API + interface contracts below. Environment: Python
3.12, QuTiP 5.3, torch+CUDA, scikit-learn, sleepy; frontend Vue 3 + Vuetify + Pinia + vue-router +
axios. Run backend from `d:\CiRA Quantum\backend` with `PYTHONIOENCODING=utf-8`.

## 0. Corrected code API (the recommendation doc guessed wrong — use these)
- `FeatureConfig(readout="fid", n_peaks=653, fid_complex=False, time_domain=False, wavelet=False,
  nonlinear=False, observables=("x","y","z"))`  — field is **`readout`** (not `mode`).
- `EncodingConfig(fn="arcsin_sqrt", axis="x", target_qubits=[], phase_amplitude=False)` — field is
  **`fn`** (not `function`); `target_qubits` is a **`list[int]`** (protons = `[4,5,6,7,8]`, empty = all).
- `QRCSystem.fid_signal(rho) -> np.ndarray`  (complex128, shape `(sim.fid_points,)`, index 0 = t=0).
- `SimConfig(..., fid_points=2048, fid_dwell=3e-4, evolution_mode="gpu")`.
- Weather eval: `benchmarks.forecast_from_X(X, weather_norm, horizons, tr_cfg, use_rbf=, tune_rbf=)`.
- NARMA data: `tasks.narma_sequence_sine(n_steps, order, seed)`; `tasks.nmse_paper(y,ŷ)`.
- Reservoir: `benchmarks.run_weather_reservoir(...)`, `benchmarks.run_narma_multitask(...)` (single
  pass); reservoir loop is `evolution.Reservoir.run(seq, progress_cb=)`.

## 1. Trace-cache `.npz` schema  (Coder A writes; Coder B + Coder C read)
Path: `backend/artifacts/traces/<name>.npz`. Keys:
- `fids` : complex64, shape `[n_steps, fid_points]` — per-step raw FID.
- `fid_dwell` : float (s).
- `task` : str, `"weather"` or `"narma"`.
- `split` : int[3] `[washout, n_train, n_test]`.
- `seed` : int.
- weather only: `weather_norm` float[n_days, 2] (temp,humidity in [0,1]); `horizons` int[].
- narma only: `narma_input` float[n_steps]; `orders` int[].
- `meta` : JSON string (full config for provenance).

## 2. Feature builder + eval signatures  (Coder A: eval; Coder B: builders)
- Builder: `build_features(fids: np.ndarray, method: str, **params) -> tuple[np.ndarray, list[str]]`
  returns `(X[n_steps, n_feat], names)`. Methods: `"magnitude653"` (baseline — MUST reproduce v2),
  `"phase"` (mag+re+im), `"multimodal"` (mag + time/wavelet/entropy). Deterministic, stable names.
- Reducer: `reduce_features(Xtr, ytr, Xte, method: str, k: int|None) -> tuple[Xtr2, Xte2]`.
  Methods: `"none","pca","kpca","umap","lasso","mi","random"`.
- Eval: `evaluate_features(X: np.ndarray, npz: dict, method_name: str) -> dict` — uses the npz's
  task/targets/split to return the Evaluation Protocol dict:
  `{experiment, feature_count, weather_r2:{h1,h10,h20,h30,h45}, narma10_nmse, wall_clock_s,
    delta_vs_baseline:{...}}`. (weather trace → weather_r2 populated; narma trace → narma10_nmse.)

## 3. `/api/qrc` endpoint contract  (Coder C serves; Coder D consumes)
Base `/api/qrc`. JSON everywhere. Read-only endpoints are unauthenticated; **control endpoints
require auth** (reuse `auth_bp`; mirror how `qml`/`solve` gate mutating routes).
- `GET  /runs` → `[{id, task, status, progress_pct, eta_s, created_utc}]`.
- `GET  /runs/<id>/progress` → `{phase, step, total, eta_s, status, events:[{t,elapsed_s,kind,message}],
  results:{...}}` (read from the run's `events.jsonl`).
- `GET  /runs/<id>/fid?step=k` → `{step, n_steps, t:[…], real:[…], imag:[…], mag:[…],
  freq_hz:[…], spectrum_mag:[…], peaks_hz:[…]}` (from the trace `.npz`; compute FFT server-side).
- `GET  /runs/<id>/results` → the results JSON.
- `POST /runs`  *(auth)* → body `{task, config}` (task ∈ trace-gen|narma|weather|phase1); launches
  the runner as a managed subprocess (run registry + PID + run-dir with `events.jsonl`); returns
  `{id}`. Enforce **single active heavy job** (409 if one is running) + a config **allow-list**.
- `POST /runs/<id>/stop` *(auth)* → terminate; `{ok}`.
- `GET  /runs/<id>/status` → `{status, exit_code}`.
Run registry: a JSON/dir under `backend/artifacts/qrc_runs/` mapping id → {task, run_dir, pid,
status, config}. Do NOT expose the tunnel/secrets — internet exposure is handled separately in deploy.

## 4. File ownership (disjoint — safe parallel edits)
- **Coder A:** `scripts/qrc_gen_traces.py`, `app/qrc/feature_lab.py`, minimal hook in
  `evolution.py`/`benchmarks.py` to surface the per-step FID for caching.
- **Coder B:** `app/qrc/feature_methods.py`, `scripts/qrc_phase1.py`, `[featurelab]` pyproject extra
  (`umap-learn`, guarded).
- **Coder C:** `app/routes/qrc.py`, one `register_blueprint` line in `app/__init__.py`,
  `app/qrc/launcher.py` (managed-subprocess launcher + registry), `tests/test_routes_qrc.py`.
- **Coder D:** `frontend/src/stores/qrc.ts`, `frontend/src/router/index.ts` (add routes),
  `frontend/src/views/QrcDashboardPage.vue`, `QrcRunDetailPage.vue`, small components,
  `frontend/package.json` (chart lib, e.g. `uplot`).

## 5. Acceptance (QA + lead)
1. ruff clean; existing 16 qrc tests + backend route tests pass; frontend `npm run build` succeeds.
2. **Baseline `magnitude653` reproduces v2** from a cached trace (weather R²@45 ≈ 0.786 ± tol;
   NARMA-10 ≈ 2.46e-5). This is the make-or-break gate.
3. `phase`/`multimodal` produce finite, stable-count features; reducers all run; Phase-1 figure
   (R² vs #features per method) generated.
4. `/api/qrc` endpoints return the exact shapes above; FID endpoint's spectrum peaks match the
   selected 653 bins; control endpoints reject unauthenticated calls and enforce the single-job lock.
5. UI renders FID time-domain + spectrum with a working step slider, live progress trace, results,
   and the New-run/Stop controls (against a mock or live API).
6. New deps optional/guarded; no regressions to other `/api` routes; package imports without extras.
