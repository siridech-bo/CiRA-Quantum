"""Package the raw reservoir traces for Zenodo (plan, build, draft upload).

Reads docs/QRC/data/TRACES_MANIFEST.json (qrc_trace_manifest.py) and splits the files into
records that each fit under the Zenodo per-record quota (50 GB; target LIMIT_GB), grouping
whole run directories so every record is self-contained. Feature matrices that are a pure
function of a saved FID trace (X = |FFT(fids_H)|[:, bins]) are left out of the default
package; their FFT-bin selections are kept as tiny `<stem>.bins.npy` sidecars and
`--regen-features` rebuilds them from the traces.

  python scripts/qrc_zenodo_package.py                       # plan -> docs/QRC/data/ZENODO_PLAN.{md,json}
  python scripts/qrc_zenodo_package.py --build OUT [--part-gb 10] [--record A] [--dry-run]
  python scripts/qrc_zenodo_package.py --upload OUT [--sandbox] [--record A] [--dry-run]
  python scripts/qrc_zenodo_package.py --regen-features ROOT [--only GLOB] [--check]

--build writes OUT/qrc_traces_<id>/ : qrc_traces_<id>.tar (uncompressed; split into
.tar.000, .tar.001, ... with --part-gb), README.md, SHA256SUMS, ARCHIVE_SHA256SUMS. Every
data file is re-hashed while it is streamed into the tar and must match the manifest.
--upload creates one Zenodo deposition per built record, PUTs its files to the deposition
bucket and sets metadata from docs/QRC/data/ZENODO_METADATA.json. It NEVER publishes: the
depositions stay drafts for review on the Zenodo web UI. Token: env ZENODO_TOKEN only.
--regen-features ROOT: ROOT is an extracted `artifacts/` directory (or backend/artifacts).
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import io
import json
import os
import re
import shutil
import sys
import tarfile
import time
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
DATA = ROOT.parent / "docs/QRC/data"
MANIFEST = DATA / "TRACES_MANIFEST.json"
PLAN_JSON, PLAN_MD, META_JSON = DATA / "ZENODO_PLAN.json", DATA / "ZENODO_PLAN.md", DATA / "ZENODO_METADATA.json"
DIRS = ("traces", "qrc_runs")
SUFFIXES = {".npz", ".pt", ".json", ".log", ".txt", ".csv"}            # = qrc_trace_manifest.py
RAW, FEAT = "raw FID trace", "features (recomputable from trace)"
ZENODO_GB, LIMIT_GB = 50, 45
CHUNK = 2048                                                            # trace rows per FFT block

# First match on the file's run unit wins; the last record catches everything else.
RECORDS = (
    {"id": "A", "short": "ERA5 weather benchmark + confirmation cities",
     "units": ("qrc_runs/realworld/era5_*", "qrc_runs/confirm*"),
     "about": "Real-world benchmark on Open-Meteo ERA5 daily temperature + humidity for the five "
              "development cities (Delhi, Mumbai, Bangkok, Beijing, Berlin; design grids C1 gain, C5 "
              "tau, learned encodings) and the pre-registered confirmation runs on untouched cities "
              "(qrc_runs/confirm*)."},
    {"id": "C", "short": "pre-registered steps 5-6: Confirmations 3-4, drift test, encoding/readout-channel controls",
     "units": ("qrc_runs/step5", "qrc_runs/step6", "qrc_runs/confirm4"),
     "about": "Step 5 / Confirmation 3 (untouched ERA5 Lagos, Mexico City, Stockholm) and the drift test "
              "(untouched exchange-rate and electricity series), qrc_runs/step5; Step 6, the paper-encoding "
              "control (theta = arcsin s) and the readout-channel passes (proton + 13C FID, single-spin <sz>, "
              "diagonal populations) on the development datasets, qrc_runs/step6; Confirmation 4 if run, "
              "qrc_runs/confirm4. Each directory has the tuned ESN grid, tuned LSTM predictions, test "
              "predictions and logs next to the raw traces."},
    {"id": "B", "short": "other real-world benchmarks + synthetic / development runs",
     "units": ("*",),
     "about": "Non-weather real-world benchmarks (UCI Beijing PM2.5, ETTh1, Santa Fe laser A, Kaggle "
              "Daily Delhi Climate), the C6 bounded-feature check on ETTh2 (qrc_runs/c6), and the "
              "synthetic / development runs: crotonic-9 and DCCB-7 learnable-encoding campaigns "
              "(NARMA, Mackey-Glass), learnable / memory-capacity / phase runs, weather development "
              "runs and the trace cache (traces/, incl. weather_full.npz)."},
)

gb = lambda b: f"{b / 1e9:.2f}"  # noqa: E731


def unit(path):
    s = path.split("/")
    if s[:2] == ["qrc_runs", "realworld"] and len(s) > 3:
        return "/".join(s[:3])
    return "/".join(s[:2]) if len(s) > 2 else s[0]


def record_of(u):
    return next(r["id"] for r in RECORDS if any(fnmatch.fnmatchcase(u, p) for p in r["units"]))


def npz_info(p):
    """{member: (dtype, shape)} from the npy headers only (no array data read)."""
    out = {}
    with zipfile.ZipFile(p) as z:
        for n in z.namelist():
            with z.open(n) as f:
                v = np.lib.format.read_magic(f)
                shape, _, dt = (np.lib.format.read_array_header_1_0 if v == (1, 0)
                                else np.lib.format.read_array_header_2_0)(f)
            out[n[:-4]] = (str(dt), list(shape))
    return out


def bins_source(path, paths):
    """Path of the npz whose `bins` a feature file was read out at (None if not stored)."""
    d, name = path.rsplit("/", 1)
    if name != "X_learned.npz":
        return path
    parent, leaf = d.rsplit("/", 1)
    if leaf.startswith("learned_seed"):                                 # gain 1: the arcsin pass bins
        cand = f"{parent}/X_arcsin.npz"
    elif (m := re.fullmatch(r"g([\d.]+)_seed\d+", leaf)) and parent.endswith("/design_learn"):
        cand = f"{parent.rsplit('/', 1)[0]}/design_c1/g{m.group(1)}/X.npz"
    else:
        return None
    return cand if cand in paths else None


def feature_decision(r, paths):
    """('exclude', entry) or ('keep', reason) for a file of kind FEAT."""
    d, name = r["path"].rsplit("/", 1)
    if "esn" in name:
        return "keep", "classical ESN states (not derived from an FID trace)"
    tr = f"{d}/trace{name[1:]}"                                         # X.npz->trace.npz, X_learned->trace_learned
    if tr not in paths:
        return "keep", "no FID trace for it in the same directory"
    src = bins_source(r["path"], paths)
    if src is None:
        return "keep", "its FFT-bin selection is not stored"
    info, tinfo = npz_info(ART / r["path"]), npz_info(ART / tr)
    if src != r["path"] and "bins" not in npz_info(ART / src):
        return "keep", f"bins source {src} has no `bins`"
    if src == r["path"] and "bins" not in info:
        return "keep", "its FFT-bin selection is not stored"
    if "fids_H" not in tinfo or tinfo["fids_H"][1][0] != info["X"][1][0]:
        return "keep", "trace has no matching fids_H"
    return "exclude", {"path": r["path"], "bytes": r["bytes"], "sha256": r["sha256"], "trace": tr,
                       "fid_key": "fids_H", "bins_from": src, "bins_file": src[:-4] + ".bins.npy",
                       "keys": sorted(info), "dtype": info["X"][0], "shape": info["X"][1]}


def anomalies(man, rows):
    """Things in the manifest worth knowing before archiving."""
    out = []
    paths = {r["path"]: r for r in rows}
    disk = {f.relative_to(ART).as_posix(): f.stat().st_size for d in DIRS for f in (ART / d).rglob("*")
            if f.is_file() and f.suffix in SUFFIXES and ".ckpt" not in f.parts}
    new = sorted(set(disk) - set(paths))
    if new:
        units = sorted({unit(p) for p in new})
        out.append(f"Manifest is stale: {len(new)} file(s), {gb(sum(disk[p] for p in new))} GB on disk are not in it "
                   f"(units: {', '.join(units)}). Re-run `python scripts/qrc_trace_manifest.py` before --build; "
                   "they are NOT in this plan.")
    gone = [p for p in paths if p not in disk]
    changed = [p for p in paths if p in disk and disk[p] != paths[p]["bytes"]]
    if gone:
        out.append(f"{len(gone)} manifest file(s) missing on disk, e.g. {gone[:3]}.")
    if changed:
        out.append(f"{len(changed)} file(s) changed size since the manifest, e.g. {changed[:3]}.")
    if man.get("provenance", {}).get("git_dirty"):
        out.append(f"Manifest was generated on a dirty tree (commit {man['provenance'].get('git_commit', '?')[:7]}).")
    zero = [r["path"] for r in rows if r["bytes"] == 0]
    if zero:
        out.append(f"Zero-byte file(s): {', '.join(zero)}.")
    dup = defaultdict(list)
    for r in rows:
        dup[r["sha256"]].append(r)
    dups = [v for v in dup.values() if len(v) > 1]
    if dups:
        out.append(f"{len(dups)} sets of byte-identical files ({gb(sum(v[0]['bytes'] * (len(v) - 1) for v in dups))} GB "
                   "redundant), e.g. " + "; ".join(" = ".join(x["path"] for x in v) for v in dups[:3]) + ".")
    mis = [r["path"] for r in rows if r["kind"] != RAW and r["path"].endswith(".npz") and (ART / r["path"]).exists()
           and {"fids", "fids_H", "fids_C"} & set(npz_info(ART / r["path"]))]
    if mis:
        out.append(f"{len(mis)} file(s) are raw FID traces (have fids/fids_H/fids_C) but the manifest calls them "
                   f"\"other array\" (manifest `kind()` keys on file name), "
                   f"{gb(sum(paths[p]['bytes'] for p in mis))} GB, e.g. {', '.join(mis[:4])}. They ARE archived; "
                   "only the kind label is off, so the manifest's 'raw FID trace' total is an undercount.")
    feat_other = [r["path"] for r in rows if r["kind"] == "other array" and re.search(r"(_X|set_[^/]*)\.npz$", r["path"])]
    if feat_other:
        out.append(f"{len(feat_other)} feature matrices under traces/weather_block are labelled \"other array\" and are "
                   "archived as-is (small).")
    other = sorted({f.suffix or "(none)" for d in DIRS for f in (ART / d).rglob("*")
                    if f.is_file() and f.suffix not in SUFFIXES and ".ckpt" not in f.parts})
    if other:
        out.append(f"Files with suffixes {other} (launcher events.jsonl / progress.html, .sh, .png, DONE markers) "
                   "are outside the manifest and not archived.")
    return out


def make_plan():
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rows = man["files"]
    paths = {r["path"] for r in rows}
    recs = {r["id"]: {**{k: r[k] for k in ("id", "short", "about")}, "units": defaultdict(lambda: [0, 0]),
                      "files": [], "excluded": [], "kept_features": []} for r in RECORDS}
    for r in sorted(rows, key=lambda r: r["path"]):
        u = unit(r["path"])
        rec = recs[record_of(u)]
        if r["kind"] == FEAT:
            what, x = feature_decision(r, paths)
            if what == "exclude":
                rec["excluded"].append(x)
                continue
            rec["kept_features"].append({"path": r["path"], "bytes": r["bytes"], "reason": x})
        rec["files"].append({k: r[k] for k in ("path", "bytes", "sha256", "kind")})
        rec["units"][u][0] += 1
        rec["units"][u][1] += r["bytes"]
    for rec in recs.values():
        rec["bins_sidecars"] = sorted({e["bins_file"] for e in rec["excluded"]})
        rec["units"] = {u: {"files": n, "bytes": b} for u, (n, b) in sorted(rec["units"].items())}
        rec["n_files"], rec["bytes"] = len(rec["files"]), sum(f["bytes"] for f in rec["files"])
        rec["raw_trace_bytes"] = sum(f["bytes"] for f in rec["files"] if f["kind"] == RAW)
        rec["excluded_bytes"] = sum(e["bytes"] for e in rec["excluded"])
        rec["bytes_with_features"] = rec["bytes"] + rec["excluded_bytes"]
    from app.qrc.provenance import provenance  # noqa: E402
    plan = {"manifest": "docs/QRC/data/TRACES_MANIFEST.json", "manifest_n_files": man["n_files"],
            "manifest_total_bytes": man["total_bytes"], "manifest_provenance": man.get("provenance"),
            "zenodo_record_limit_gb": ZENODO_GB, "target_limit_gb": LIMIT_GB,
            "records": list(recs.values()), "anomalies": anomalies(man, rows),
            "provenance": provenance(step="zenodo package plan")}
    over = [r["id"] for r in plan["records"] if r["bytes"] > LIMIT_GB * 1e9]
    if over:
        raise SystemExit(f"records over {LIMIT_GB} GB: {over} -- adjust RECORDS")
    return plan


def write_plan(plan):
    PLAN_JSON.write_text(json.dumps(plan, indent=1), encoding="utf-8")
    R = plan["records"]
    tot = lambda k: sum(r[k] for r in R)  # noqa: E731
    md = ["# Zenodo archive plan for the raw QRC traces", "",
          f"Generated by `backend/scripts/qrc_zenodo_package.py` from `TRACES_MANIFEST.json` "
          f"({plan['manifest_n_files']} files, {gb(plan['manifest_total_bytes'])} GB). Zenodo allows "
          f"{ZENODO_GB} GB per record; each record here is kept under {LIMIT_GB} GB and holds whole run "
          "directories, so it can be used on its own. Per-file lists + SHA-256 in `ZENODO_PLAN.json`. "
          "Nothing has been built or uploaded.", "",
          "## Records", "",
          "| record | contents | files | GB packaged | of which raw FID traces | features left out (GB) | GB if features included |",
          "|---|---|---|---|---|---|---|"]
    md += [f"| {r['id']} | {r['short']} | {r['n_files']} (+{len(r['bins_sidecars'])} bins sidecars) | {gb(r['bytes'])} | "
           f"{gb(r['raw_trace_bytes'])} | {len(r['excluded'])} files, {gb(r['excluded_bytes'])} | "
           f"{gb(r['bytes_with_features'])} |" for r in R]
    md += [f"| **total** | | {tot('n_files')} | {gb(tot('bytes'))} | {gb(tot('raw_trace_bytes'))} | "
           f"{sum(len(r['excluded']) for r in R)} files, {gb(tot('excluded_bytes'))} | {gb(tot('bytes_with_features'))} |", ""]
    for r in R:
        md += [f"### Record {r['id']}: {r['short']}", "", r["about"], "", "| run unit | files | GB |", "|---|---|---|"]
        md += [f"| `{u}` | {v['files']} | {gb(v['bytes'])} |" for u, v in r["units"].items()]
        md += [""]
    md += ["## Left out: recomputable feature matrices", "",
           "Feature files of kind \"features (recomputable from trace)\" are excluded when they are a pure "
           "function of a raw FID trace in the same directory: `X = |FFT(fids_H, axis=1)|[:, bins]` "
           "(the standard 653-bin FID-spectrum readout, SOP). The bin selection (`bins`, top-653 of the "
           "first step's spectrum of the base arcsin / gain pass) is archived as a sidecar "
           "`<stem>.bins.npy` next to the file it came from; learned-encoding features "
           "(`X_learned.npz`) reuse the bins of their base pass (`X_arcsin` for `learned_seed*`, "
           "`design_c1/g<g>/X` for `design_learn/g<g>_seed*`). Rebuild after extracting:", "",
           "```", "python tools/qrc_zenodo_package.py --regen-features artifacts --plan tools/ZENODO_PLAN.json",
           "python tools/qrc_zenodo_package.py --regen-features artifacts --plan tools/ZENODO_PLAN.json --only 'qrc_runs/realworld/santafe/*'",
           "```", "",
           "NumPy only (no GPU). The traces are stored complex64 while `X.npz` / `X_arcsin.npz` were computed "
           "from the complex128 pass, so rebuilt values agree to float32 precision (relative ~1e-7 measured), not "
           "bit-for-bit: their SHA-256 will differ from the manifest. `--check` compares against originals "
           "where they exist.", "",
           "Feature files kept in the package (not safely recomputable from an archived trace):", ""]
    kept = [k for r in R for k in r["kept_features"]]
    md += [f"- `{k['path']}` ({gb(k['bytes'])} GB): {k['reason']}" for k in kept] or ["- none"]
    md += ["", "## Manifest anomalies", ""] + [f"- {a}" for a in plan["anomalies"]]
    md += ["", "## Build / upload", "",
           "```",
           "python scripts/qrc_zenodo_package.py --build D:/zenodo_out --part-gb 10   # tar per record + README + SHA256SUMS",
           "python scripts/qrc_zenodo_package.py --upload D:/zenodo_out --sandbox      # test on sandbox.zenodo.org first",
           "python scripts/qrc_zenodo_package.py --upload D:/zenodo_out                # drafts on zenodo.org (never published)",
           "```", "",
           "`--build` needs about the packaged size free on the output drive. `--upload` reads the token "
           "from the environment variable `ZENODO_TOKEN` (scope deposit:write), refuses while "
           "`ZENODO_METADATA.json` still contains `TODO`, and leaves every deposition as an unpublished "
           "draft; review and publish on the Zenodo web UI.", "",
           "## Quota-increase request (if one record is preferred)", "",
           "Draft for Zenodo support (https://zenodo.org/support, category \"File upload / quota\"):", "",
           "> Subject: Quota increase request for a single dataset record (~"
           f"{round(tot('bytes_with_features') / 1e9) + 1} GB)", ">",
           "> Hello Zenodo team,", ">",
           "> We would like to deposit the raw simulation outputs behind a manuscript on quantum reservoir "
           "computing with NMR spin systems (code on GitHub, linked from the record). The dataset is "
           f"{gb(tot('bytes_with_features'))} GB in {sum(r['n_files'] + len(r['excluded']) for r in R)} files: "
           f"{gb(tot('raw_trace_bytes'))} GB of raw free-induction-decay waveforms (NumPy .npz) that took GPU "
           "days to produce, plus the derived feature matrices, results and logs, every file listed with a "
           "SHA-256 checksum. We would prefer one record (one DOI, one citation) over splitting by experiment. "
           "Could the per-record limit for this upload be raised to "
           f"{int(-(-tot('bytes_with_features') * 1.1 // 1e10) * 10)} GB? We will upload it as a few uncompressed "
           "tar parts of at most 10 GB each. If that is not possible we will split it into "
           f"{len(R)} linked records of under {LIMIT_GB} GB each.", ">",
           "> Thank you,", "> TODO author list", ""]
    PLAN_MD.write_text("\n".join(md) + "\n", encoding="utf-8")


def load_plan(p=None):
    return json.loads(Path(p or PLAN_JSON).read_text(encoding="utf-8"))


def pick(plan, rid):
    R = [r for r in plan["records"] if rid in (None, r["id"])]
    if not R:
        raise SystemExit(f"no record {rid}")
    return R


# ---------------------------------------------------------------- build

class SplitWriter:
    """Write-only stream into name.000, name.001, ... (or one file if part_bytes == 0), SHA-256 per part."""

    def __init__(self, base, part_bytes):
        self.base, self.part, self.parts, self.f = base, part_bytes, [], None

    def _open(self):
        name = self.base.name if not self.part else f"{self.base.name}.{len(self.parts):03d}"
        self.f, self.n, self.h = open(self.base.parent / name, "wb"), 0, hashlib.sha256()
        self.parts.append([name, None, 0])

    def _close(self):
        if self.f:
            self.f.close()
            self.parts[-1][1:] = [self.h.hexdigest(), self.n]
            self.f = None

    def write(self, b):
        b = memoryview(b).cast("B")
        n0 = len(b)
        while len(b):
            if self.f is None or (self.part and self.n >= self.part):
                self._close()
                self._open()
            k = len(b) if not self.part else min(len(b), self.part - self.n)
            self.f.write(b[:k])
            self.h.update(b[:k])
            self.n += k
            b = b[k:]
        return n0

    def close(self):
        self._close()


class HashReader:
    def __init__(self, f):
        self.f, self.h = f, hashlib.sha256()

    def read(self, n=-1):
        b = self.f.read(n)
        self.h.update(b)
        return b


def readme(plan, r):
    p = plan.get("manifest_provenance") or {}
    return "\n".join([
        f"# QRC raw reservoir traces, record {r['id']}: {r['short']}", "", r["about"], "",
        f"Part of a {len(plan['records'])}-record set: " + "; ".join(f"{x['id']} = {x['short']}" for x in plan["records"]) + ".",
        f"Produced on commit {p.get('git_commit', '?')[:7]} of the CiRA Quantum repository (branch qrc-simulation); "
        "the scripts named below live in backend/scripts/.", "",
        "## Unpacking", "",
        "Check the archive parts with `sha256sum -c ARCHIVE_SHA256SUMS`, then (if split) "
        f"`cat qrc_traces_{r['id']}.tar.* > qrc_traces_{r['id']}.tar` (Windows: `copy /b qrc_traces_{r['id']}.tar.000+"
        f"qrc_traces_{r['id']}.tar.001+... qrc_traces_{r['id']}.tar`) and `tar -xf qrc_traces_{r['id']}.tar`.", "",
        "## Layout", "",
        "- `artifacts/` mirrors `backend/artifacts/` of the repository (run directories under `qrc_runs/`, trace cache under `traces/`).",
        "- `SHA256SUMS`: every file, `sha256sum -c SHA256SUMS` from this directory.",
        "- `tools/TRACES_MANIFEST.json` (all files of all records), `tools/ZENODO_PLAN.json`, `tools/qrc_zenodo_package.py`.", "",
        "## Formats", "",
        "- `trace*.npz`: `fids_H` complex64 (steps, 2048) proton FID per input step (2048 x 0.3 ms), plus inputs (`w` / `theta`) and `meta` (JSON).",
        "- `learned_encoding_fid.npz`, `traces/*.npz`: trace-cache schema (`fids` or `fids_H`/`fids_C`, `split`, `seed`, `fid_dwell`, task input, `meta`).",
        "- `encoder.pt`: learned encoder state_dict (torch). `*.json` / `*.log`: results and run logs. `*predictions*.npz`, `esn_grid.npz`, `lstm.npz`: test predictions / baselines.", "",
        "## Feature matrices not included", "",
        f"{len(r['excluded'])} feature files ({gb(r['excluded_bytes'])} GB) are left out; each is `|FFT(fids_H)|[:, bins]` of a trace in this record "
        "(the standard 653-bin FID-spectrum readout). Their `bins` are kept as `<stem>.bins.npy`. Rebuild with NumPy:", "",
        "    python tools/qrc_zenodo_package.py --regen-features artifacts --plan tools/ZENODO_PLAN.json", "",
        "Rebuilt values match the originals to float32 precision (traces are stored complex64), so their checksums differ from the manifest.", ""])


def build(plan, out, part_gb, rid, dry):
    out = Path(out)
    for r in pick(plan, rid):
        name = f"qrc_traces_{r['id']}"
        missing = [f["path"] for f in r["files"] if not (ART / f["path"]).is_file()
                   or (ART / f["path"]).stat().st_size != f["bytes"]]
        need = r["bytes"] + 512 * (len(r["files"]) * 3 + 64)
        print(f"record {r['id']}: {r['n_files']} files + {len(r['bins_sidecars'])} sidecars, {gb(r['bytes'])} GB -> "
              f"{out / name / (name + '.tar')}{'.NNN' if part_gb else ''}", flush=True)
        if missing:
            raise SystemExit(f"  {len(missing)} files missing / changed size, e.g. {missing[:3]} -- re-run the manifest")
        if dry:
            free = shutil.disk_usage(out if out.exists() else out.anchor or ".").free
            print(f"  [dry-run] would write ~{gb(need)} GB ({'parts of ' + str(part_gb) + ' GB' if part_gb else 'one tar'}); "
                  f"free on target drive {gb(free)} GB{'  <-- NOT ENOUGH' if free < need else ''}")
            continue
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        side = {}
        for s in r["bins_sidecars"]:
            src = next(e["bins_from"] for e in r["excluded"] if e["bins_file"] == s)
            buf = io.BytesIO()
            np.save(buf, np.load(ART / src)["bins"])
            side[s] = buf.getvalue()
        sums = [f"{f['sha256']}  artifacts/{f['path']}" for f in r["files"]]
        sums += [f"{hashlib.sha256(b).hexdigest()}  artifacts/{s}" for s, b in side.items()]
        tools = {"tools/qrc_zenodo_package.py": Path(__file__).read_bytes(),
                 "tools/ZENODO_PLAN.json": PLAN_JSON.read_bytes(), "tools/TRACES_MANIFEST.json": MANIFEST.read_bytes()}
        sums += [f"{hashlib.sha256(b).hexdigest()}  {k}" for k, b in tools.items()]
        text = {"README.md": readme(plan, r).encode("utf-8"), "SHA256SUMS": ("\n".join(sums) + "\n").encode("utf-8")}
        for k, b in text.items():
            (d / k).write_bytes(b)
        sw = SplitWriter(d / f"{name}.tar", int(part_gb * 1e9))
        t0 = time.time()
        with tarfile.open(fileobj=sw, mode="w|", format=tarfile.PAX_FORMAT) as tar:
            def add_bytes(arc, b):
                ti = tarfile.TarInfo(f"{name}/{arc}")
                ti.size, ti.mtime, ti.mode = len(b), int(time.time()), 0o644
                tar.addfile(ti, io.BytesIO(b))
            for k, b in {**text, **tools}.items():
                add_bytes(k, b)
            for s, b in side.items():
                add_bytes(f"artifacts/{s}", b)
            done = 0
            for i, f in enumerate(r["files"], 1):
                p = ART / f["path"]
                ti = tar.gettarinfo(str(p), arcname=f"{name}/artifacts/{f['path']}")
                ti.uid = ti.gid = 0
                ti.uname = ti.gname = ""
                with open(p, "rb") as fh:
                    hr = HashReader(fh)
                    tar.addfile(ti, hr)
                if hr.h.hexdigest() != f["sha256"]:
                    raise SystemExit(f"checksum mismatch: {f['path']} -- archive aborted")
                done += f["bytes"]
                if i % 25 == 0 or i == len(r["files"]):
                    print(f"  {i}/{len(r['files'])}  {gb(done)} GB  {time.time() - t0:.0f}s", flush=True)
        sw.close()
        (d / "ARCHIVE_SHA256SUMS").write_text("".join(f"{h}  {n}\n" for n, h, _ in sw.parts), encoding="utf-8")
        print(f"  wrote {len(sw.parts)} archive file(s), {gb(sum(n for *_, n in sw.parts))} GB", flush=True)


# ---------------------------------------------------------------- upload (drafts only)

def api(method, url, token, body=None, data=None, size=None):
    headers = {"Authorization": f"Bearer {token}"}
    if body is not None:
        data, headers["Content-Type"] = json.dumps(body).encode("utf-8"), "application/json"
    else:
        headers["Content-Type"] = "application/octet-stream"
    if size is not None:
        headers["Content-Length"] = str(size)
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=3600) as resp:
        return json.loads(resp.read() or b"{}")


def record_metadata(plan, r):
    meta = json.loads(META_JSON.read_text(encoding="utf-8"))["metadata"]
    m = dict(meta)
    m["title"] = f"{meta['title']} (record {r['id']} of {len(plan['records'])}: {r['short']})"
    m["description"] = (meta["description"] + f"<p><b>This record ({r['id']}):</b> {r['about']} "
                        f"{r['n_files']} files, {gb(r['bytes'])} GB, of which {gb(r['raw_trace_bytes'])} GB raw FID traces. "
                        f"{len(r['excluded'])} recomputable feature matrices ({gb(r['excluded_bytes'])} GB) are omitted; "
                        "see README.md for how to rebuild them.</p>")
    return m


def upload(plan, out, sandbox, rid, dry):
    base = "https://sandbox.zenodo.org" if sandbox else "https://zenodo.org"
    token = os.environ.get("ZENODO_TOKEN", "")
    for r in pick(plan, rid):
        d = Path(out) / f"qrc_traces_{r['id']}"
        meta = record_metadata(plan, r)
        files = sorted(p for p in d.glob("*") if p.is_file() and p.name != "zenodo_deposition.json") if d.is_dir() else []
        print(f"record {r['id']} -> {base}: {len(files)} file(s) in {d}")
        todo = "TODO" in json.dumps(meta)
        if dry:
            print(f"  [dry-run] POST {base}/api/deposit/depositions  (new draft, or reuse {d / 'zenodo_deposition.json'})")
            for p in files:
                print(f"  [dry-run] PUT <bucket>/{p.name}  {gb(p.stat().st_size)} GB")
            for n in [] if files else [f"qrc_traces_{r['id']}.tar[.NNN]", "README.md", "SHA256SUMS", "ARCHIVE_SHA256SUMS"]:
                print(f"  [dry-run] PUT <bucket>/{n}  (not built yet)")
            print(f"  [dry-run] PUT {base}/api/deposit/depositions/<id>  metadata: {meta['title'][:90]}")
            print("  [dry-run] NOT published (drafts only)" + ("; metadata still has TODO -> a real upload would refuse"
                                                           if todo else ""))
            print(f"  [dry-run] token from env ZENODO_TOKEN: {'set' if token else 'NOT set'}")
            continue
        if todo:
            raise SystemExit(f"fill the TODO fields in {META_JSON} first")
        if not token:
            raise SystemExit("set the environment variable ZENODO_TOKEN (deposit:write scope)")
        if not files:
            raise SystemExit(f"nothing built in {d}; run --build first")
        dep_file = d / "zenodo_deposition.json"
        if dep_file.exists():
            dep = json.loads(dep_file.read_text(encoding="utf-8"))
        else:
            dep = api("POST", f"{base}/api/deposit/depositions", token, body={})
            dep = {"id": dep["id"], "bucket": dep["links"]["bucket"], "html": dep["links"].get("html"), "base": base}
            dep_file.write_text(json.dumps(dep, indent=1), encoding="utf-8")
        print(f"  draft deposition {dep['id']}  {dep.get('html')}", flush=True)
        for p in files:
            t0 = time.time()
            with open(p, "rb") as fh:
                api("PUT", f"{dep['bucket']}/{p.name}", token, data=fh, size=p.stat().st_size)
            print(f"  PUT {p.name}  {gb(p.stat().st_size)} GB  {time.time() - t0:.0f}s", flush=True)
        api("PUT", f"{base}/api/deposit/depositions/{dep['id']}", token, body={"metadata": meta})
        print("  metadata set; left as DRAFT (review + publish on the Zenodo web UI)")


# ---------------------------------------------------------------- feature regeneration

def regen(plan, root, only, check):
    root = Path(root)
    E = [e for r in plan["records"] for e in r["excluded"] if not only or fnmatch.fnmatchcase(e["path"], only)]
    print(f"{len(E)} feature file(s) to {'check' if check else 'rebuild'} under {root}")
    worst = 0.0
    for e in E:
        tr = root / e["trace"]
        if not tr.exists():
            print(f"  skip {e['path']}: trace {e['trace']} not here")
            continue
        sc, src = root / e["bins_file"], root / e["bins_from"]
        bins = np.load(sc) if sc.exists() else np.load(src)["bins"]
        F = np.load(tr)[e["fid_key"]]
        cdt = np.complex128 if e["dtype"].endswith("8") else np.complex64
        X = np.empty(e["shape"], dtype=e["dtype"])
        for i in range(0, len(F), CHUNK):
            X[i:i + CHUNK] = np.abs(np.fft.fft(F[i:i + CHUNK].astype(cdt), axis=1))[:, bins]
        dst = root / e["path"]
        if check:
            if not dst.exists():
                print(f"  {e['path']}: no original to compare")
                continue
            X0 = np.load(dst)["X"]
            err = float(np.max(np.abs(X - X0)) / (np.max(np.abs(X0)) + 1e-30))
            worst = max(worst, err)
            print(f"  {e['path']}: max |dX| / max|X| = {err:.2e}", flush=True)
            continue
        if dst.exists():
            print(f"  keep existing {e['path']}")
            continue
        np.savez(dst, **({"X": X, "bins": bins} if "bins" in e["keys"] else {"X": X}))
        print(f"  wrote {e['path']}", flush=True)
    if check:
        print(f"worst relative deviation {worst:.2e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--build", metavar="OUTDIR")
    ap.add_argument("--upload", metavar="OUTDIR")
    ap.add_argument("--regen-features", metavar="ROOT")
    ap.add_argument("--plan", help="ZENODO_PLAN.json to use (default docs/QRC/data/ZENODO_PLAN.json)")
    ap.add_argument("--record", help="only this record id")
    ap.add_argument("--part-gb", type=float, default=0, help="split each tar into parts of this size (0 = one file)")
    ap.add_argument("--sandbox", action="store_true", help="upload to sandbox.zenodo.org")
    ap.add_argument("--only", help="glob on feature paths for --regen-features")
    ap.add_argument("--check", action="store_true", help="--regen-features: compare with existing originals, write nothing")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.regen_features:
        return regen(load_plan(a.plan), a.regen_features, a.only, a.check)
    if a.build or a.upload:
        plan = load_plan(a.plan)
        if a.build:
            build(plan, a.build, a.part_gb, a.record, a.dry_run)
        if a.upload:
            upload(plan, a.upload, a.sandbox, a.record, a.dry_run)
        return
    plan = make_plan()
    write_plan(plan)
    for r in plan["records"]:
        print(f"record {r['id']}: {r['n_files']} files, {gb(r['bytes'])} GB (raw traces {gb(r['raw_trace_bytes'])} GB); "
              f"excluded {len(r['excluded'])} features {gb(r['excluded_bytes'])} GB; with features {gb(r['bytes_with_features'])} GB")
    for x in plan["anomalies"]:
        print("  ! " + x.encode("ascii", "replace").decode())
    print(f"wrote {PLAN_MD.name}, {PLAN_JSON.name}")


if __name__ == "__main__":
    main()
