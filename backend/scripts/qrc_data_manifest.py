"""Build docs/QRC/data/MANIFEST.md + manifest.json (publication data package).

For every file in docs/QRC/data: SHA-256, size, the commit that first added it,
the producing script, a status (current / superseded / interim) and a one-line
description. Headline numbers are EXTRACTED from the files here (never typed),
so every number in the paper traces to a file + key. Re-run after adding data:

  python backend/scripts/qrc_data_manifest.py
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "docs" / "QRC" / "data"

# file -> (status, producing script, description)
FILES = {
    "qrc_6spin_campaign.json": ("current", "qrc_learnable_campaign.py (plan 6spin)",
                                "6-spin x2 learned vs arcsin, NARMA-2/10 + Mackey-Glass h=1, 3 seeds + tau->0 ablations (run learnable-campaign-01ce463b, 41 h)"),
    "qrc_6spin_mackey_h10_seed1.json": ("current", "qrc_learnable_9spin.py --task mackey_glass --horizon 10",
                                        "6-spin Mackey-Glass h=10, seed 1 (run learnable-0a45c7b0)"),
    "qrc_6spin_mackey_h10_confirm.json": ("current", "qrc_mackey_h10_confirm.py",
                                          "6-spin Mackey-Glass h=10, seeds 2-3 (run mackey-h10-confirm-2d21b56f); its seed-1 entry is a rounded copy (0.0013) — use qrc_6spin_mackey_h10_seed1.json"),
    "qrc_classical_baseline_v1_superseded.json": ("superseded", "qrc_classical_baseline.py",
                                                  "first classical baseline: size-only ESN, data seed 1 only — superseded by qrc_classical_fair_*"),
    "qrc_classical_fair_esn.json": ("current", "qrc_classical_fair.py --models esn",
                                    "fully tuned ESN + size-matched ESN, NARMA-2/10, MG h=10, data seeds 1-3"),
    "qrc_classical_fair_lstm.json": ("current", "qrc_classical_fair.py --models lstm",
                                     "tuned LSTM, NARMA-2/10, MG h=10, data seeds 1-3"),
    "qrc_crotonic9_campaign.json": ("current", "qrc_learnable_campaign.py (plan crotonic9)",
                                    "crotonic-9 learned vs arcsin, NARMA-2/10 + MG h=10, 3 seeds + ablations"),
    "qrc_dccb7_tau100.json": ("current", "qrc_learnable_campaign.py (plan dccb7, tau 0.1)", "dccb7 tau=100 ms"),
    "qrc_dccb7_tau30.json": ("current", "qrc_learnable_campaign.py (plan dccb7, tau 0.03)", "dccb7 tau=30 ms"),
    "qrc_dccb7_tau100_j0.json": ("current", "qrc_learnable_campaign.py (plan dccb7_j0)",
                                 "dccb7 tau=100 ms, J=0 ablation (readout basis of the coupled molecule)"),
    "qrc_block_bench_cira3.jsonl": ("current", "qrc_block_bench.py", "block-propagator scaling benchmark 9-12 spins (A100)"),
    "qrc_paper4_narma.json": ("current", "qrc_reproduce_paper4.py", "Paper-4 reproduction, NARMA"),
    "qrc_paper4_weather.json": ("superseded", "qrc_reproduce_paper4.py", "Paper-4 weather reproduction, fid_points 1024"),
    "qrc_paper4_weather_v2.json": ("current", "qrc_reproduce_paper4.py", "Paper-4 weather reproduction, full 653-FID readout"),
    "qrc_weather_fair_esn.json": ("superseded", "qrc_weather_fair.py", "weather tuned ESN, single 150-day validation (fragile selection)"),
    "qrc_weather_fair_bootstrap_noleak.json": ("superseded", "qrc_weather_fair.py + bootstrap", "weather QRC vs no-leak ESN"),
    "qrc_weather_bench_step1_v300.json": ("current", "qrc_weather_bench.py", "weather readout variants, symmetric protocol"),
    "qrc_weather_bench_step2_v300.json": ("current", "qrc_weather_bench.py + qrc_weather_block.py", "weather reservoir variants"),
    "qrc_weather_learned_leakfree_seed0.json": ("current", "qrc_weather_learn.py + qrc_weather_bench.py", "weather learned encoding, seed 0"),
    "qrc_weather_learned_leakfree_seed1.json": ("current", "qrc_weather_learn.py + qrc_weather_bench.py", "weather learned encoding, seed 1"),
    "qrc_weather_learned_leakfree_seed2.json": ("current", "qrc_weather_learn.py + qrc_weather_bench.py", "weather learned encoding, seed 2"),
    "qrc_weather_pool_delhi_kaggle.json": ("current", "qrc_weather_pool.py", "Kaggle Delhi pooled seeds: learned QRC vs ESN, ESN with trained input stage"),
    "qrc_era5_delhi_eval_interim_seed0.json": ("superseded", "qrc_era5_weather.py eval", "ERA5 Delhi, encoder seed 0 only (interim)"),
    "qrc_era5_delhi_eval_final.json": ("current", "qrc_era5_weather.py eval",
                                       "ERA5 Delhi 1970-2025, test 2005-2025: arcsin / learned QRC (3 seeds) / tuned + matched ESN"),
}


def sha256(p):
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def first_commit(p):
    out = subprocess.run(["git", "log", "--diff-filter=A", "--format=%h %ad", "--date=short", "--", str(p)],
                         cwd=REPO, capture_output=True, text=True).stdout.strip().splitlines()
    return out[-1] if out else "uncommitted"


def J(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def headline():
    """(claim, value, source file, key) — computed from the data."""
    rows = []
    six = J("qrc_6spin_campaign.json")["aggregate"]
    for t in ("narma2", "narma10", "mackey_glass"):
        a = six[t]
        rows.append((f"6-spin learned {t} (h=1) test NMSE", f"{a['perspin_mean']:.3g} ± {a['perspin_std']:.2g} (arcsin {a['arcsin_mean']:.3g})",
                     "qrc_6spin_campaign.json", f"aggregate.{t}"))
    s1 = J("qrc_6spin_mackey_h10_seed1.json")["conditions"]["perspin"]["test_nmse"]
    conf = J("qrc_6spin_mackey_h10_confirm.json")["seeds"]
    vals = [s1, conf["2"]["perspin"], conf["3"]["perspin"]]
    rows.append(("6-spin learned Mackey-Glass h=10 (3 seeds, exact)", f"{np.mean(vals):.5f} ± {np.std(vals):.5f}",
                 "qrc_6spin_mackey_h10_seed1.json + _confirm.json", "conditions.perspin / seeds.{2,3}.perspin"))
    rows.append(("  (old paper value, used rounded seed-1 0.0013)", f"{J('qrc_6spin_mackey_h10_confirm.json')['aggregate']['perspin_mean']:.5f}",
                 "qrc_6spin_mackey_h10_confirm.json", "aggregate.perspin_mean"))
    for f, lab in (("qrc_crotonic9_campaign.json", "crotonic-9"), ("qrc_dccb7_tau100.json", "dccb7 tau=100 ms"),
                   ("qrc_dccb7_tau30.json", "dccb7 tau=30 ms"), ("qrc_dccb7_tau100_j0.json", "dccb7 J=0")):
        for t, a in J(f)["aggregate"].items():
            rows.append((f"{lab} learned {t}", f"{a['perspin_mean']:.4g} ± {a['perspin_std']:.2g} (arcsin {a['arcsin_mean']:.3g})",
                         f, f"aggregate.{t}"))
    esn, lstm = J("qrc_classical_fair_esn.json")["results"], J("qrc_classical_fair_lstm.json")["results"]
    for t in esn:
        rows.append((f"tuned ESN {t}", f"{esn[t]['esn_best']['mean']:.3g} (size-matched {esn[t]['esn_matched']['mean']:.3g})",
                     "qrc_classical_fair_esn.json", f"results.{t}"))
        rows.append((f"tuned LSTM {t}", f"{lstm[t]['lstm_best']['mean']:.3g} ± {lstm[t]['lstm_best']['std']:.2g}",
                     "qrc_classical_fair_lstm.json", f"results.{t}.lstm_best"))
    pool = J("qrc_weather_pool_delhi_kaggle.json")
    for g, v in pool["groups"].items():
        rows.append((f"Kaggle Delhi weather {g} pooled test R²", f"{v['mean']:.3f} ± {v['sd']:.3f} (ensemble {v['ensemble']:.3f})",
                     "qrc_weather_pool_delhi_kaggle.json", f"groups.{g}"))
    for k, v in pool["comparisons"].items():
        rows.append((f"Kaggle Delhi {k}", f"95% CI [{v['pooled_ci95'][0]:+.3f}, {v['pooled_ci95'][1]:+.3f}]",
                     "qrc_weather_pool_delhi_kaggle.json", f"comparisons.{k}"))
    for f in sorted(DATA.glob("qrc_realworld_*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        if "groups" not in d:                         # factor / memory analyses, not benchmark evals
            continue
        for g, v in d["groups"].items():
            rows.append((f"{d['dataset']} {g} pooled test R²", f"{v['mean']:.3f} ± {v['sd']:.3f}", f.name, f"groups.{g}"))
        for k, v in d["comparisons"].items():
            rows.append((f"{d['dataset']} {k}", f"95% CI [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]", f.name, f"comparisons.{k}"))
    return rows


def main():
    files = sorted(p for p in DATA.iterdir() if p.is_file() and p.name not in ("MANIFEST.md", "manifest.json"))
    entries = []
    for p in files:
        if p.name.startswith("qrc_design_c5c7_"):
            FILES.setdefault(p.name, ("current", "qrc_design_c5c7.py",
                                      "Step 3 C5 tau + C7 compact readout (pre-registered ed96b17), validation only — not adopted"))
        if p.name.startswith("qrc_design_c6_"):
            FILES.setdefault(p.name, ("current", "qrc_design_c6.py",
                                      "Step 2 C6 bounded features (pre-registered 53d8e6a): freeze / post-hoc dev test / ETTh2 confirmation — NOT supported"))
        if p.name.startswith("qrc_confirm1_"):
            FILES.setdefault(p.name, ("current", "qrc_design_confirm.py",
                                      "Confirmation 1 (pre-registered 29152bf): frozen gain g=0.35 on an untouched city, test 2005-2025"))
        if p.name.startswith("qrc_design_c1_"):
            FILES.setdefault(p.name, ("current", "qrc_design_c1.py",
                                      "design concept C1 (encoding gain) sweep, validation only — tests H1/H2"))
        if p.name in ("qrc_realworld_factors.json", "qrc_realworld_memory_function.json"):
            FILES.setdefault(p.name, ("current", "qrc_realworld_factors.py / memory-function analysis",
                                      "factor analysis of the real-world benchmark (why QRC wins/comparable/loses)"))
        st, script, desc = FILES.get(p.name, ("current" if p.name.startswith("qrc_realworld_") else "unlisted",
                                              "qrc_realworld_bench.py eval" if p.name.startswith("qrc_realworld_") else "?",
                                              "real-world benchmark eval" if p.name.startswith("qrc_realworld_") else ""))
        entries.append({"file": p.name, "sha256": sha256(p), "bytes": p.stat().st_size, "added": first_commit(p),
                        "status": st, "script": script, "description": desc})
    rows = headline()
    (DATA / "manifest.json").write_text(json.dumps({"files": entries, "headline": [dict(zip(("claim", "value", "file", "key"), r))
                                                                                   for r in rows]}, indent=2), encoding="utf-8")
    md = ["# QRC data manifest", "",
          "Generated by `backend/scripts/qrc_data_manifest.py` — do not edit by hand. Every paper number must",
          "come from a row of the *Headline numbers* table (value extracted from the file/key shown).", "",
          "Raw reservoir traces (FID `.npz`, ~1 GB) live in `backend/artifacts/` (gitignored) and are to be",
          "archived (Zenodo) with checksums; result files below are version-controlled.", "",
          "## Files", "", "| file | status | added | script | description | sha256 (12) |", "|---|---|---|---|---|---|"]
    md += [f"| `{e['file']}` | {e['status']} | {e['added']} | `{e['script']}` | {e['description']} | `{e['sha256'][:12]}` |"
           for e in entries]
    md += ["", "## Headline numbers (extracted)", "", "| claim | value | file | key |", "|---|---|---|---|"]
    md += [f"| {c} | {v} | `{f}` | `{k}` |" for c, v, f, k in rows]
    (DATA / "MANIFEST.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"{len(entries)} files, {len(rows)} headline numbers -> {DATA / 'MANIFEST.md'}")


if __name__ == "__main__":
    main()
