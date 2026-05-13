"""Convert a completed cloud QAOA job into a ``RunRecord`` in the
benchmark archive. The materialization helper is what the
``POST /api/benchmarks/cloud-jobs/<id>/materialize`` route calls, and
also what one-off scripts like ``fetch_wukong_result.py`` call.

Inputs are a ``PendingJob`` (carrying instance + parameter context)
and an active ``QCloudService``-backed query for the job. Output is
the new ``RunRecord``'s id; the record is written to disk and the
pending entry is removed.
"""

from __future__ import annotations

import json

import dimod
import numpy as np

from app.benchmarking.instances import get_instance
from app.benchmarking.pending_jobs import PendingJob
from app.benchmarking.pending_jobs import remove as remove_pending
from app.benchmarking.records import (
    RunRecord,
    _archive_sample_set,
    _detect_code_version,
    _detect_hardware_id,
    _new_record_id,
    _now,
    _summarize,
    archive_path,
    compute_repro_hash,
)
from app.benchmarking.registry import get_solver
from app.optimization.compiler import compile_cqm_json


class JobNotReadyError(RuntimeError):
    """Raised when a job has no probabilities to materialize yet."""


class JobErroredError(RuntimeError):
    """Raised when a job hit a cloud-side error (e.g. cluster reply)."""


def materialize(pending: PendingJob, api_key: str) -> str:
    """Materialize a completed cloud job into a ``RunRecord``.

    Returns the new record_id. Raises ``JobNotReadyError`` if the job
    has no measurement probabilities yet (still queued / computing),
    or ``JobErroredError`` if the cloud reported a failure.

    Side effects on success: writes the record JSON + a gzipped
    SampleSet to ``benchmarks/archive/`` and removes ``pending`` from
    the pending-jobs list.
    """
    import pyqpanda3.qcloud as qcloud_mod

    # Construct the service so pyqpanda3 binds the credential into its
    # internal handle before we query the job — the QCloudJob ctor on
    # its own does not authenticate.
    qcloud_mod.QCloudService(api_key=api_key)
    job = qcloud_mod.QCloudJob(pending.job_id)
    q = job.query()
    err = q.error_message()
    if err:
        raise JobErroredError(
            f"Job {pending.job_id[:16]}… reported error: {err[:200]}"
        )
    probs = q.get_probs()
    if not probs:
        raise JobNotReadyError(
            f"Job {pending.job_id[:16]}… has no probabilities yet; "
            f"status is {q.job_status()}"
        )

    sample_set = _probs_to_sampleset(pending, probs, q)
    record = _wrap_in_record(pending, sample_set)

    out_path = archive_path(record.record_id)
    with open(out_path, "w") as f:
        json.dump(record.to_dict(), f, indent=2)

    # Defensive: api_key must NEVER end up in the serialized record.
    # ``302e0201`` is the standardized PKCS#8 DER prefix that every EC
    # private key (including Origin Quantum's) begins with. Checking
    # for this prefix is a leak-detector, not a credential — it's the
    # same string that would appear in any PKCS#8 EC key.
    serialized = json.dumps(record.to_dict())
    if "302e0201" in serialized or "api_key" in record.parameters:
        # If this fires, undo the disk write and refuse to remove from pending.
        out_path.unlink(missing_ok=True)
        raise RuntimeError(
            "Refused to materialize: credential fragment detected in serialized record. "
            "This is a bug — investigate before retrying."
        )

    remove_pending(pending.job_id)
    return record.record_id


def _probs_to_sampleset(
    pending: PendingJob,
    probs: dict[str, float],
    cloud_result,
) -> dimod.SampleSet:
    """Convert the cloud's bitstring → probability dict into a proper
    ``dimod.SampleSet`` with energies recomputed against the BQM."""
    inst = get_instance(pending.instance_id)
    cqm_json = inst.load_cqm_json()
    cqm, _registry, _sense = compile_cqm_json(cqm_json)
    bqm, _invert = dimod.cqm_to_bqm(
        cqm, lagrange_multiplier=pending.lagrange_multiplier
    )
    bqm_bin = bqm.change_vartype(dimod.BINARY, inplace=False)
    variables = list(bqm_bin.variables)
    n = len(variables)

    top_k = int(pending.parameters.get("top_k", 10))
    top_by_prob = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:top_k]

    rows: list[tuple[float, np.ndarray, str, float]] = []
    for bitstring, p in top_by_prob:
        padded = bitstring.zfill(n)[-n:]
        row = np.zeros(n, dtype=np.int8)
        for i, ch in enumerate(padded):
            row[i] = int(ch)
        energy = float(bqm_bin.energy(dict(zip(variables, row, strict=True))))
        rows.append((energy, row, bitstring, float(p)))
    rows.sort(key=lambda t: t[0])

    samples_array = np.stack([r[1] for r in rows]) if rows else np.zeros((0, n), dtype=np.int8)
    ss = dimod.SampleSet.from_samples(
        (samples_array, variables),
        vartype=dimod.BINARY,
        energy=[r[0] for r in rows],
    )
    ss.info["qaoa_top_bitstrings"] = [r[2] for r in rows]
    ss.info["qaoa_top_probabilities"] = [r[3] for r in rows]
    ss.info["qaoa_layer"] = pending.parameters.get("layer")
    ss.info["cloud_backend"] = pending.parameters.get("backend_name")
    ss.info["cloud_job_id"] = pending.job_id
    ss.info["cloud_shots"] = pending.parameters.get("shots")
    ss.info["cloud_is_real_hardware"] = (
        pending.parameters.get("backend_name") in {"WK_C180", "HanYuan_01"}
    )

    return ss


def _wrap_in_record(pending: PendingJob, ss: dimod.SampleSet) -> RunRecord:
    """Bundle a SampleSet into the same RunRecord shape that
    ``record_run`` produces synchronously."""
    inst = get_instance(pending.instance_id)
    cqm_json = inst.load_cqm_json()
    cqm, _registry, sense = compile_cqm_json(cqm_json)

    identity, _cls = get_solver(pending.solver_name)
    parameters = dict(pending.parameters)

    record_id = _new_record_id()
    code_version = _detect_code_version()
    hardware_id = _detect_hardware_id(identity.hardware)
    repro_hash = compute_repro_hash(code_version, pending.instance_id, identity, parameters)

    results = _summarize(
        ss,
        cqm=cqm,
        sense=sense,
        expected_optimum=inst.expected_optimum,
    )
    # We don't have wall-clock timing on the async path — the cloud
    # has its own queue + execution timing in QCloudResult.timing_info()
    # but it raises on some result formats today. Surface zero rather
    # than risk a crash.
    results.setdefault("elapsed_ms", 0)

    sample_set_path = _archive_sample_set(record_id, ss)

    return RunRecord(
        record_id=record_id,
        code_version=code_version,
        solver=identity,
        parameters=parameters,
        instance_id=pending.instance_id,
        hardware_id=hardware_id,
        started_at=_now(),
        completed_at=_now(),
        repro_hash=repro_hash,
        results=results,
        sample_set_path=sample_set_path,
    )
