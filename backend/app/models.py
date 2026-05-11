"""SQLite schema + thin model helpers.

Three tables ship in Phase 0:

* ``users``    — authentication subjects.
* ``api_keys`` — BYOK ciphertext per (user, provider). Encrypted via
                 ``app.crypto`` under ``KEY_ENCRYPTION_SECRET``.
* ``jobs``     — the optimization-pipeline run history. Phase 0 creates
                 the table; Phase 4 wires the orchestrator that writes
                 into it.

The helpers below are deliberately function-shaped (no ORM, no class
hierarchy) — the schema is small, the query surface is narrow, and a
plain ``sqlite3`` cursor is the simplest thing that works. When Phase 7
hardening or Phase 11 multi-tenancy demands more (migrations, indexes,
analytics), this is the file to grow.
"""

from __future__ import annotations

import os
import sqlite3
import uuid
from datetime import datetime
from typing import Any

from werkzeug.security import check_password_hash, generate_password_hash

from app.config import (
    DATABASE_PATH,
    DEFAULT_ADMIN_PASSWORD,
    DEFAULT_ADMIN_USERNAME,
    ROLE_ADMIN,
)

# ---- Connection ----


def get_db_connection() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    # Enforce foreign-key constraints on every connection (sqlite default is OFF).
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# ---- Schema bootstrap ----


def init_db() -> None:
    """Create tables and seed the default admin if no users exist yet."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()

        cursor.execute(
            """
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
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS api_keys (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id       INTEGER NOT NULL,
                provider      TEXT NOT NULL,
                encrypted_key BLOB NOT NULL,
                created_at    TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                UNIQUE (user_id, provider)
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id                  TEXT PRIMARY KEY,
                user_id             INTEGER NOT NULL,
                problem_statement   TEXT NOT NULL,
                provider            TEXT NOT NULL,
                status              TEXT NOT NULL,
                cqm_json            TEXT,
                variable_registry   TEXT,
                validation_report   TEXT,
                solution            TEXT,
                interpreted_solution TEXT,
                error               TEXT,
                num_variables       INTEGER,
                num_constraints     INTEGER,
                solve_time_ms       INTEGER,
                created_at          TEXT NOT NULL,
                completed_at        TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """
        )

        # Idempotent ALTERs for upgrades from earlier schemas.
        # ``ALTER TABLE ... ADD COLUMN`` is a no-op against a fresh
        # CREATE, so we only run each when the column is missing.
        existing_columns = {row[1] for row in cursor.execute("PRAGMA table_info(jobs)")}
        if "validation_report" not in existing_columns:
            cursor.execute("ALTER TABLE jobs ADD COLUMN validation_report TEXT")
        if "template_id" not in existing_columns:
            cursor.execute("ALTER TABLE jobs ADD COLUMN template_id TEXT")
        if "expected_optimum" not in existing_columns:
            cursor.execute("ALTER TABLE jobs ADD COLUMN expected_optimum REAL")

        conn.commit()

        cursor.execute("SELECT COUNT(*) FROM users")
        if cursor.fetchone()[0] == 0:
            _seed_default_admin(conn)
    finally:
        conn.close()


def _seed_default_admin(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        INSERT INTO users (username, email, password_hash, display_name, role, created_at)
        VALUES (?, NULL, ?, ?, ?, ?)
        """,
        (
            DEFAULT_ADMIN_USERNAME,
            generate_password_hash(DEFAULT_ADMIN_PASSWORD),
            "Administrator",
            ROLE_ADMIN,
            datetime.utcnow().isoformat(),
        ),
    )
    conn.commit()


# ---- User helpers ----


def create_user(
    username: str,
    password: str,
    *,
    email: str | None = None,
    display_name: str | None = None,
    role: str = "user",
) -> dict[str, Any]:
    """Insert a new user. Raises ``ValueError`` if username or email already
    exists (caller maps that to HTTP 409)."""
    conn = get_db_connection()
    try:
        try:
            conn.execute(
                """
                INSERT INTO users (username, email, password_hash, display_name, role, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    username,
                    email,
                    generate_password_hash(password),
                    display_name or username,
                    role,
                    datetime.utcnow().isoformat(),
                ),
            )
            conn.commit()
        except sqlite3.IntegrityError as e:
            raise ValueError(f"username or email already taken: {e}") from e
        # Re-query so we return whatever AUTOINCREMENT picked.
        row = conn.execute(
            "SELECT id, username, email, display_name, role FROM users WHERE username = ?",
            (username,),
        ).fetchone()
        return dict(row)
    finally:
        conn.close()


def verify_user(username: str, password: str) -> dict[str, Any] | None:
    """Return the user row (as a dict) if credentials check out and the
    account is active; ``None`` otherwise."""
    conn = get_db_connection()
    try:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
        if row is None or not row["is_active"]:
            return None
        if not check_password_hash(row["password_hash"], password):
            return None
        # Touch last_login on success.
        conn.execute(
            "UPDATE users SET last_login = ? WHERE id = ?",
            (datetime.utcnow().isoformat(), row["id"]),
        )
        conn.commit()
        return dict(row)
    finally:
        conn.close()


def get_user_by_id(user_id: int) -> dict[str, Any] | None:
    conn = get_db_connection()
    try:
        row = conn.execute(
            "SELECT id, username, email, display_name, role, is_active "
            "FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def change_password(user_id: int, current_password: str, new_password: str) -> bool:
    """Verify the current password, then store a fresh hash for the new one.
    Returns True on success, False if the current password was wrong."""
    conn = get_db_connection()
    try:
        row = conn.execute(
            "SELECT password_hash FROM users WHERE id = ?", (user_id,)
        ).fetchone()
        if row is None or not check_password_hash(row["password_hash"], current_password):
            return False
        conn.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?",
            (generate_password_hash(new_password), user_id),
        )
        conn.commit()
        return True
    finally:
        conn.close()


# ---- Job helpers (used by Phase 4; declared here so all schema-aware code
# lives in one place). ----


def create_job(
    user_id: int,
    problem_statement: str,
    provider: str,
    *,
    template_id: str | None = None,
    expected_optimum: float | None = None,
) -> str:
    """Insert a fresh job row in ``queued`` state. Returns the new job ID.

    ``template_id`` and ``expected_optimum`` are populated only when the
    job was launched from a template (Phase 5B route
    ``POST /api/solve/from-template/<id>``). The detail view's match
    badge consumes them to compare the solver's actual answer to the
    template's documented one.
    """
    job_id = str(uuid.uuid4())
    conn = get_db_connection()
    try:
        conn.execute(
            """
            INSERT INTO jobs (
                id, user_id, problem_statement, provider, status,
                template_id, expected_optimum, created_at
            )
            VALUES (?, ?, ?, ?, 'queued', ?, ?, ?)
            """,
            (
                job_id, user_id, problem_statement, provider,
                template_id, expected_optimum,
                datetime.utcnow().isoformat(),
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return job_id


def update_job(job_id: str, **fields: Any) -> None:
    if not fields:
        return
    conn = get_db_connection()
    try:
        keys = ", ".join(f"{k} = ?" for k in fields)
        conn.execute(
            f"UPDATE jobs SET {keys} WHERE id = ?",
            (*fields.values(), job_id),
        )
        conn.commit()
    finally:
        conn.close()


def get_job(
    job_id: str,
    *,
    user_id: int | None = None,
    is_admin: bool = False,
) -> dict[str, Any] | None:
    """Fetch a job. Non-admins can only see their own; admins see any."""
    conn = get_db_connection()
    try:
        if is_admin:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        else:
            row = conn.execute(
                "SELECT * FROM jobs WHERE id = ? AND user_id = ?",
                (job_id, user_id),
            ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def list_jobs(
    user_id: int | None,
    *,
    is_admin: bool = False,
    page: int = 1,
    page_size: int = 20,
) -> dict[str, Any]:
    """Paginated job list. Non-admins are filtered by ``user_id``.
    Returns ``{"jobs": [...], "page": <n>, "page_size": <n>, "total": <n>}``.
    """
    page = max(1, int(page))
    page_size = max(1, min(int(page_size), 100))
    offset = (page - 1) * page_size

    conn = get_db_connection()
    try:
        if is_admin:
            total = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
            rows = conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (page_size, offset),
            ).fetchall()
        else:
            total = conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE user_id = ?", (user_id,)
            ).fetchone()[0]
            rows = conn.execute(
                "SELECT * FROM jobs WHERE user_id = ? "
                "ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (user_id, page_size, offset),
            ).fetchall()
        return {
            "jobs": [dict(r) for r in rows],
            "page": page,
            "page_size": page_size,
            "total": int(total),
        }
    finally:
        conn.close()


def delete_job(
    job_id: str, *, user_id: int | None = None, is_admin: bool = False
) -> bool:
    """Delete a job. Returns ``True`` if a row was removed."""
    conn = get_db_connection()
    try:
        if is_admin:
            cur = conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        else:
            cur = conn.execute(
                "DELETE FROM jobs WHERE id = ? AND user_id = ?", (job_id, user_id)
            )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


# ---- BYOK API-key helpers ----


def list_api_keys(user_id: int) -> list[dict[str, Any]]:
    """List a user's stored BYOK providers. Never returns ciphertext —
    the route layer can't accidentally leak keys."""
    conn = get_db_connection()
    try:
        rows = conn.execute(
            "SELECT provider, created_at FROM api_keys WHERE user_id = ? "
            "ORDER BY provider",
            (user_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def put_api_key(user_id: int, provider: str, encrypted_key: bytes) -> None:
    """Upsert the ciphertext for ``(user_id, provider)``."""
    conn = get_db_connection()
    try:
        conn.execute(
            """
            INSERT INTO api_keys (user_id, provider, encrypted_key, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT (user_id, provider)
            DO UPDATE SET encrypted_key = excluded.encrypted_key,
                          created_at = excluded.created_at
            """,
            (user_id, provider, encrypted_key, datetime.utcnow().isoformat()),
        )
        conn.commit()
    finally:
        conn.close()


def get_api_key_ciphertext(user_id: int, provider: str) -> bytes | None:
    conn = get_db_connection()
    try:
        row = conn.execute(
            "SELECT encrypted_key FROM api_keys WHERE user_id = ? AND provider = ?",
            (user_id, provider),
        ).fetchone()
        return bytes(row["encrypted_key"]) if row else None
    finally:
        conn.close()


def delete_api_key(user_id: int, provider: str) -> bool:
    """Remove a stored key. Returns ``True`` if a row was deleted."""
    conn = get_db_connection()
    try:
        cur = conn.execute(
            "DELETE FROM api_keys WHERE user_id = ? AND provider = ?",
            (user_id, provider),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()
