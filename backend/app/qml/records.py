"""QML-7 — public TrainRecord archive.

Mirror of :mod:`app.benchmarking.records` for the QML side, with one
important shape difference:

* The optimization archive captures a single solver invocation against
  a single instance (a one-shot ``record_run``).
* The QML archive captures a single *training run* — which already
  includes per-epoch history, classical baselines on the same split,
  and any real-QPU evaluations the user submitted.

So a ``TrainRecord`` is closer to a "lab notebook entry" than a
benchmark cell. The on-disk JSON is the canonical source of truth;
``GET /api/qml/benchmarks`` reads ``qml_benchmarks/archive/*.json`` at
request time.

Reproducibility model
---------------------

Same idea as the optimization records: every ``TrainRecord`` carries
a ``repro_hash`` (SHA-256 truncated) over ``code_version + dataset_id
+ model + hyperparameters``. Two runs with the same hash should
produce identical numbers up to library nondeterminism.

Privacy
-------

The archive is **public** (the dashboard renders without auth, like
the optimization benchmark dashboard). Records include the contributor's
``display_name`` but never their internal user id, email, or any
ciphertext. Real-QPU runs' API tokens never enter the record.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import platform
import secrets
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# ---- Archive root ----------------------------------------------------------

# qml_benchmarks/archive/ at the repo root; allow override for tests.
_DEFAULT_ARCHIVE = (
    Path(__file__).resolve().parents[2].parent / "qml_benchmarks" / "archive"
)


def _archive_root() -> Path:
    override = os.environ.get("CIRA_QML_BENCH_ARCHIVE")
    if override:
        return Path(override)
    return _DEFAULT_ARCHIVE


def archive_path(record_id: str) -> Path:
    """Resolve the JSON file path for ``record_id``."""
    return _archive_root() / f"{record_id}.json"


# ---- Identifiers + provenance ---------------------------------------------


def _now() -> datetime:
    return datetime.now(tz=UTC)


def _new_record_id() -> str:
    """Sortable: ``20260521T143501.123456Z_a3c81d``-style."""
    now = _now()
    ts = now.strftime("%Y%m%dT%H%M%S") + f".{now.microsecond:06d}Z"
    return f"{ts}_{secrets.token_hex(3)}"


def _detect_code_version() -> str:
    """Best-effort git rev-parse, falling back to a versioned dev tag."""
    package_root = Path(__file__).resolve().parents[2]
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=package_root,
            capture_output=True, text=True, timeout=5, check=False,
        )
        if sha.returncode == 0 and sha.stdout.strip():
            return sha.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "0.1.0+dev"


def _detect_hardware_id() -> str:
    """The classical host's identity. Real-QPU runs append per-backend
    metadata into their own ``qpu_runs`` field — this is the box that
    trained the local VQC."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            cc = torch.cuda.get_device_capability(0)
            cuda_v = torch.version.cuda or "unknown"
            return f"cuda:{name}|cc{cc[0]}.{cc[1]}|cuda_runtime={cuda_v}"
    except Exception:  # pragma: no cover — torch may not be installed
        pass
    return f"cpu:{platform.processor() or platform.machine()}|{platform.system()}"


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def compute_repro_hash(
    code_version: str,
    dataset_id: str,
    model: str,
    hyperparameters: dict,
) -> str:
    payload = "\n".join([
        f"code_version={code_version}",
        f"dataset_id={dataset_id}",
        f"model={model}",
        f"hyperparameters={_canonical_json(hyperparameters)}",
    ])
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ---- TrainRecord dataclass -------------------------------------------------


@dataclass
class TrainRecord:
    """One archived QML training run.

    Fields cluster into:
    * ``record_id`` / ``code_version`` / ``repro_hash`` — provenance.
    * ``model`` / ``dataset_id`` / ``hyperparameters`` — *what* was trained.
    * ``metrics`` / ``baselines`` / ``qpu_runs`` — *how it did*.
    * ``contributor_display_name`` — who archived it.
    """

    record_id: str
    code_version: str
    repro_hash: str
    model: str                       # "vqc" today; future-proof for kernel/QNN
    dataset_id: str
    hyperparameters: dict            # n_qubits, n_layers, epochs, etc.
    hardware_id: str                 # classical host the local train ran on
    started_at: datetime
    completed_at: datetime
    metrics: dict                    # final_test_accuracy, confusion_matrix, etc.
    baselines: list[dict]            # the four classical baselines we run alongside
    training_history: list[dict]     # per-epoch loss/accuracy
    qpu_runs: list[dict]             # any real-QPU evaluations the user submitted
    contributor_display_name: str
    notes: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["started_at"] = self.started_at.isoformat()
        d["completed_at"] = self.completed_at.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> TrainRecord:
        known = {f for f in cls.__dataclass_fields__}
        kept = {k: v for k, v in d.items() if k in known}
        kept["started_at"] = datetime.fromisoformat(kept["started_at"])
        kept["completed_at"] = datetime.fromisoformat(kept["completed_at"])
        return cls(**kept)


# ---- Build + write ---------------------------------------------------------


def _strip_weights(metrics: dict) -> dict:
    """Drop large/unnecessary fields from the in-DB metrics blob before
    archiving. Trained weights are model state — interesting in the
    job row, useless in a citation-ready record. Scatter points are
    redundant once the dataset is identifiable from ``dataset_id``."""
    if not metrics:
        return {}
    out = dict(metrics)
    for k in ("weights", "scatter_points", "decision_grid"):
        out.pop(k, None)
    return out


def build_record_from_job(
    *,
    job_row: dict,
    contributor_display_name: str,
    qpu_run_rows: list[dict] | None = None,
    notes: str = "",
) -> TrainRecord:
    """Snapshot a completed ``qml_jobs`` row into a ``TrainRecord``.

    The row must be in status ``complete`` and have a non-null
    ``metrics`` JSON blob.
    """
    metrics_json = job_row.get("metrics") or "{}"
    metrics = json.loads(metrics_json)
    hyperparameters_json = job_row.get("hyperparameters") or "{}"
    hyperparameters = json.loads(hyperparameters_json)
    training_history_json = job_row.get("training_history") or "[]"
    training_history = json.loads(training_history_json)

    code_version = _detect_code_version()
    repro = compute_repro_hash(
        code_version=code_version,
        dataset_id=str(job_row["dataset_id"]),
        model=str(job_row["model"]),
        hyperparameters=hyperparameters,
    )

    qpu_serialized: list[dict] = []
    for run in (qpu_run_rows or []):
        m_json = run.get("metrics")
        try:
            m = json.loads(m_json) if m_json else None
        except Exception:
            m = None
        qpu_serialized.append({
            "id": run["id"],
            "provider": run["provider"],
            "backend_name": run.get("backend_name"),
            "shots": run.get("shots"),
            "status": run.get("status"),
            "cloud_job_id": run.get("cloud_job_id"),
            "wall_time_ms": run.get("wall_time_ms"),
            "metrics": m,
        })

    started_at = (
        datetime.fromisoformat(job_row["created_at"])
        if job_row.get("created_at") else _now()
    )
    completed_at = (
        datetime.fromisoformat(job_row["completed_at"])
        if job_row.get("completed_at") else _now()
    )

    return TrainRecord(
        record_id=_new_record_id(),
        code_version=code_version,
        repro_hash=repro,
        model=str(job_row["model"]),
        dataset_id=str(job_row["dataset_id"]),
        hyperparameters=hyperparameters,
        hardware_id=_detect_hardware_id(),
        started_at=started_at,
        completed_at=completed_at,
        metrics=_strip_weights(metrics),
        baselines=metrics.get("baselines", []) or [],
        training_history=training_history,
        qpu_runs=qpu_serialized,
        contributor_display_name=contributor_display_name,
        notes=notes,
    )


def write_record(record: TrainRecord) -> Path:
    """Persist ``record`` to ``archive_path(record_id)``."""
    path = archive_path(record.record_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record.to_dict(), f, indent=2)
    return path


# ---- Read + filter ---------------------------------------------------------


def iter_records():
    """Yield every record in the archive. Corrupt files are skipped
    (logged at debug level by callers if needed)."""
    pattern = str(_archive_root() / "*.json")
    for path in sorted(glob.glob(pattern)):
        try:
            with open(path, encoding="utf-8") as f:
                yield TrainRecord.from_dict(json.load(f))
        except Exception:
            continue


def load_record(record_id: str) -> TrainRecord | None:
    path = archive_path(record_id)
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return TrainRecord.from_dict(json.load(f))
    except Exception:
        return None


def delete_record(record_id: str) -> bool:
    path = archive_path(record_id)
    if not path.exists():
        return False
    path.unlink()
    return True


def summarize(record: TrainRecord) -> dict:
    """Lean projection used by the list endpoint. Drops the full
    training_history + qpu_runs blobs to keep payload small (~1 KB per
    summary vs ~10 KB per full record)."""
    return {
        "record_id": record.record_id,
        "model": record.model,
        "dataset_id": record.dataset_id,
        "repro_hash": record.repro_hash,
        "started_at": record.started_at.isoformat(),
        "completed_at": record.completed_at.isoformat(),
        "contributor_display_name": record.contributor_display_name,
        "hardware_id": record.hardware_id,
        # Headline numbers from metrics (defensive against shape drift):
        "final_test_accuracy": record.metrics.get("final_test_accuracy"),
        "final_train_accuracy": record.metrics.get("final_train_accuracy"),
        "final_loss": record.metrics.get("final_loss"),
        "train_time_ms": record.metrics.get("train_time_ms"),
        "n_qubits": record.metrics.get("n_qubits"),
        "n_baselines": len(record.baselines),
        "n_qpu_runs": len(record.qpu_runs),
        "has_real_qpu_run": any(
            (r.get("metrics") or {}).get("is_real_hardware") for r in record.qpu_runs
        ),
    }


# ---- BibTeX-style citation (mirrors the optimization side) -----------------


def bibtex_entry(record: TrainRecord) -> str:
    """Return a BibTeX ``@misc`` entry citing ``record``. Same format
    as ``app.benchmarking.cite.bibtex_entry`` so a class-wide bibliography
    can mix optimization + QML records freely."""
    key = f"cira_qml_{record.record_id.split('_')[-1]}"
    year = record.completed_at.year
    title = f"VQC on {record.dataset_id} (CiRA Quantum QML run {record.record_id})"
    author = record.contributor_display_name or "CiRA Quantum"
    return (
        "@misc{" + key + ",\n"
        f"  title = {{{title}}},\n"
        f"  author = {{{author}}},\n"
        f"  year = {{{year}}},\n"
        "  howpublished = {CiRA Quantum --- QML benchmark archive},\n"
        f"  note = {{repro\\_hash={record.repro_hash}, "
        f"test\\_accuracy={record.metrics.get('final_test_accuracy')}, "
        f"code\\_version={record.code_version}}},\n"
        "}"
    )
