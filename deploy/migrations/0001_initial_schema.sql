-- D1 initial schema for CiRA Quantum.
--
-- Mirrors the SQLite schema in backend/app/models.py::init_db() as of
-- 2026-07-02. When you add a column to the SQLite path (via idempotent
-- ALTER TABLE), add a matching migration file here:
--
--     ./migrations/000N_add_<column>.sql
--
-- Apply with:
--
--     wrangler d1 migrations apply cira-quantum-db --remote
--
-- D1 quirks worth knowing:
--   * D1 supports SQLite's `INTEGER PRIMARY KEY AUTOINCREMENT` idiom
--     but the underlying storage doesn't preserve monotonic IDs across
--     restarts — don't rely on the numeric value being sequential.
--   * FTS5 is NOT available. We don't currently use it, but if a
--     future migration reaches for it (e.g. full-text search over
--     problem_statement), the fallback is either D1's built-in
--     `LIKE` for small tables or Vectorize for real semantic search.
--   * PRAGMA is a no-op in D1 — the runtime handles WAL, foreign_keys,
--     etc. internally. If backend code shells out `PRAGMA table_info`
--     for schema introspection (models.py does), it needs a shim path
--     for D1 that queries `sqlite_schema` instead.

-- ---- users --------------------------------------------------------

CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT UNIQUE NOT NULL,
    email         TEXT UNIQUE,
    password_hash TEXT NOT NULL,
    display_name  TEXT,
    role          TEXT NOT NULL DEFAULT 'user',
    is_active     INTEGER NOT NULL DEFAULT 1,
    created_at    TEXT NOT NULL,
    last_login    TEXT
);

CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);

-- ---- api_keys (BYOK) ----------------------------------------------

CREATE TABLE IF NOT EXISTS api_keys (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       INTEGER NOT NULL,
    provider      TEXT NOT NULL,
    encrypted_key BLOB NOT NULL,
    created_at    TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    UNIQUE (user_id, provider)
);

CREATE INDEX IF NOT EXISTS idx_api_keys_user ON api_keys(user_id);

-- ---- jobs (optimization pipeline) ---------------------------------

-- All columns from the current SQLite schema — the idempotent ALTERs
-- in init_db() are inlined here so a fresh D1 deploy matches the
-- most recent shipped shape.

CREATE TABLE IF NOT EXISTS jobs (
    id                        TEXT PRIMARY KEY,
    user_id                   INTEGER NOT NULL,
    problem_statement         TEXT NOT NULL,
    provider                  TEXT NOT NULL,
    status                    TEXT NOT NULL,
    cqm_json                  TEXT,
    variable_registry         TEXT,
    validation_report         TEXT,
    solution                  TEXT,
    interpreted_solution      TEXT,
    error                     TEXT,
    num_variables             INTEGER,
    num_constraints           INTEGER,
    solve_time_ms             INTEGER,
    created_at                TEXT NOT NULL,
    completed_at              TEXT,
    -- Phase 5B — validation harness + template + oracle payloads.
    template_id               TEXT,
    expected_optimum          REAL,
    -- Phase 5D — multi-solver fan-out.
    solver_results            TEXT,
    solvers_requested         TEXT,
    -- 2026-06-30 — approval gate payload.
    preflight                 TEXT,
    -- 2026-07-01 — per-solve overrides (Origin backend selection).
    solver_params_overrides   TEXT,
    -- 2026-07-01 — LLM plain-English rewrite of the technical solution.
    plain_english_solution    TEXT,
    -- 2026-07-02 — classifier→hardcoded routing audit trail.
    formulation_route         TEXT,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_jobs_user_created ON jobs(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_template ON jobs(template_id);

-- ---- qml_jobs (QML training runs) ---------------------------------

CREATE TABLE IF NOT EXISTS qml_jobs (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id           INTEGER NOT NULL,
    dataset_id        TEXT NOT NULL,
    model             TEXT NOT NULL,
    status            TEXT NOT NULL,
    hyperparameters   TEXT,
    training_history  TEXT,
    metrics           TEXT,
    error             TEXT,
    created_at        TEXT NOT NULL,
    completed_at      TEXT,
    train_time_ms     INTEGER,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_qml_jobs_user_created
    ON qml_jobs(user_id, created_at DESC);

-- ---- qml_qpu_runs (real-QPU inference on a trained VQC) -----------

CREATE TABLE IF NOT EXISTS qml_qpu_runs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    qml_job_id          INTEGER NOT NULL,
    user_id             INTEGER NOT NULL,
    provider            TEXT NOT NULL,
    backend_name        TEXT,
    shots               INTEGER NOT NULL,
    status              TEXT NOT NULL,
    cloud_job_id        TEXT,
    queue_position      INTEGER,
    live_status         TEXT,
    metrics             TEXT,
    error               TEXT,
    created_at          TEXT NOT NULL,
    completed_at        TEXT,
    wall_time_ms        INTEGER,
    -- QML-6: per-provider submission state (Origin sample_index, etc.).
    submission_context  TEXT,
    FOREIGN KEY (qml_job_id) REFERENCES qml_jobs(id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_qml_qpu_runs_job
    ON qml_qpu_runs(qml_job_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_qml_qpu_runs_status
    ON qml_qpu_runs(status);
