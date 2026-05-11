"""Citation generator for archived ``RunRecord``s.

CLI usage::

    python -m app.benchmarking.cite <record_id>           # BibTeX entry
    python -m app.benchmarking.cite <record_id> --string  # short citation

Both paths read directly from ``benchmarks/archive/`` — no DB lookup.
"""

from __future__ import annotations

import argparse
import sys

from app.benchmarking.records import RunRecord, load_record


def bibtex_entry(record: RunRecord) -> str:
    """Render a BibTeX ``@misc`` entry for a single run record.

    The cite key is the record_id; year/month derive from ``started_at``;
    title encodes solver and instance; note includes the repro_hash so a
    reader can pin the exact preserved record."""
    started = record.started_at
    cite_key = record.record_id.replace(":", "").replace(".", "")
    title = (
        f"CiRA Quantum Benchmark: {record.solver.name} v{record.solver.version} "
        f"on {record.instance_id}"
    )
    note = (
        f"reproducibility hash {record.repro_hash}; "
        f"hardware {record.hardware_id}; code version {record.code_version}"
    )
    return (
        "@misc{" + cite_key + ",\n"
        f"  author = {{CiRA Quantum Project}},\n"
        f"  title  = {{{title}}},\n"
        f"  year   = {{{started.year}}},\n"
        f"  month  = {{{started.strftime('%B').lower()}}},\n"
        f"  howpublished = {{CiRA Quantum benchmark archive}},\n"
        f"  note   = {{{note}}}\n"
        "}\n"
    )


def short_citation(record: RunRecord) -> str:
    """Inline-citation-style string. Useful when a BibTeX entry is overkill."""
    started = record.started_at
    return (
        f"CiRA Quantum Benchmark, record {record.record_id} "
        f"({record.solver.name} v{record.solver.version} on {record.instance_id}, "
        f"{started.strftime('%Y-%m-%d')}, repro_hash {record.repro_hash})"
    )


def cite(record_id: str, *, kind: str = "bibtex") -> str:
    try:
        record = load_record(record_id)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"No archived record for {record_id!r} under benchmarks/archive/"
        ) from e
    if kind == "string":
        return short_citation(record)
    return bibtex_entry(record)


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.benchmarking.cite",
        description="Print a citation for an archived RunRecord.",
    )
    parser.add_argument("record_id", help="Record ID to cite")
    parser.add_argument(
        "--string", action="store_true", help="Emit a short inline citation instead of BibTeX"
    )
    args = parser.parse_args(argv)
    try:
        out = cite(args.record_id, kind="string" if args.string else "bibtex")
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(out, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
