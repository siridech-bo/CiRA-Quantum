# CiRA Quantum — Benchmark archive

This directory holds the platform's append-only Benchmark archive. Every
solver invocation that we want to *cite, replay, or compare* later
produces a `RunRecord` here. The Phase 5C dashboard reads from the same
data this CLI writes, so the file layout is the contract that ties
Phase 2 (foundation), Phase 5C (dashboard), and Phase 10 (contribution
pipeline) together.

## Layout

```
benchmarks/
├── archive/                       # one record per file, append-only
│   ├── 20260507T120052Z_a3c81d.json           # the run record
│   ├── 20260507T120052Z_a3c81d_samples.json.gz # the dimod.SampleSet (optional)
│   └── ...
└── README.md                      # this file
```

The `archive/` directory is treated as durable: records are not edited or
deleted. A spec change that retires a solver removes it from the live
registry but leaves its historical records in place — that is what makes
the archive *citable* and lets the time-series view in Phase 5C track
real spec drift instead of papering over it.

## Record schema

Each `<record_id>.json` follows
[`backend/app/benchmarking/schemas/record_v1.json`](../backend/app/benchmarking/schemas/record_v1.json).
Key fields:

- `record_id` — sortable timestamp + 6 random hex chars; e.g.
  `20260507T120052.446115Z_a3c81d`. Microsecond resolution keeps IDs
  monotonic even when a suite runs many records inside the same wall-clock
  second.
- `code_version` — git SHA at run time, or `<package_version>+dev` when
  the run happened outside a git tree. Two records with the same
  `code_version` and `repro_hash` should produce identical results.
- `solver` — frozen copy of the registry entry at run time. Includes the
  solver's `name`, `version`, `source` package, `hardware`, and its
  declared `parameter_schema`.
- `parameters` — the flat dict passed into `sampler.sample(...)` plus any
  `__init__` kwargs (e.g. GPU SA's `kernel` mode). Recorded verbatim;
  this is the input to the reproducibility hash.
- `instance_id` — the registered instance ID
  (e.g. `knapsack/small/knapsack_5item`).
- `hardware_id` — concrete hardware observed at run time. May be more
  specific than `solver.hardware` (e.g. exact GPU model + CUDA runtime
  version).
- `repro_hash` — SHA-256 truncated to 16 hex chars, computed over
  `(code_version, instance_id, solver, parameters)`.
- `results` — small JSON-friendly summary: `best_energy`,
  `best_user_energy` (sense-aware), `num_samples`, `num_feasible`,
  `elapsed_ms`, plus an open-ended `extras` map.
- `sample_set_path` — relative path under `archive/` to a gzipped
  `dimod.SampleSet` serialization, or `null` if archiving was disabled
  for this run.

## Reproducibility model

`replay_record(record_id, bqm=...)` re-executes the run. Three things
must match for `replay_result.agree` to be true:

1. **Code version.** A different `code_version` is reported as a
   `notes` entry, *not* a hard failure. A new commit that legitimately
   changes solver behavior is the *signal* the archive exists to detect.
2. **Reproducibility hash.** Recomputed from the live registry entry +
   the archived parameters. A drift here means either the registry
   identity changed (e.g. a solver version bump) or the inputs were
   modified.
3. **Best energy.** Compared within `1e-9` of the original. This relies
   on seeded determinism — solvers that don't accept a `seed` won't
   reproduce exactly; that's expected and surfaced.

## Citation

Every record is citable. The CLI has two modes:

```
python -m app.benchmarking.cite <record_id>           # BibTeX entry
python -m app.benchmarking.cite <record_id> --string  # short citation
```

The BibTeX entry uses the `record_id` as its cite key, encodes the
solver and instance in the title, and pins the `repro_hash` and
`hardware_id` in the `note` field so a reader following the citation
ends up at *exactly* the preserved record.

## Long-term preservation policy

The archive is meant to last the lifetime of the platform, which under
the v2 strategic vision is measured in years. Three commitments:

- **Append-only.** Once written, records are not edited or deleted by
  the platform. A correction goes in as a new record with a `notes`
  link to the old one.
- **Self-describing.** Every record includes the schemas it conforms
  to (via `code_version`). Future readers can resolve the schema
  version directly from the platform's git history.
- **Externally backed up.** Beyond Phase 7, the archive is mirrored to
  a public read-only location so the records survive any single host
  failure or repository move.

The Phase 5C dashboard, the contribution pipeline (Phase 10), and the
publication track in Appendix A all read from this archive. Treat it as
the most important on-disk asset the platform has.
