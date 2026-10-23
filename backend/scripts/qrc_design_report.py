"""Design evidence log: docs/QRC/QRC_Design_Evidence.md, generated from
docs/QRC/data/qrc_design_*.json (one section per concept step). Never hand-edited.

Each section states the pre-registered hypothesis and criterion
(QRC_Design_Concepts.md), the validation-only results, and the verdict.

  python backend/scripts/qrc_design_report.py
"""
from __future__ import annotations

import json
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "docs" / "QRC" / "data"
OUT = DATA.parent / "QRC_Design_Evidence.md"
ORDER = ["era5_delhi", "era5_mumbai", "era5_bangkok", "era5_beijing", "era5_berlin",
         "beijing_pm25", "etth1", "santafe", "kaggle_delhi"]
SMALL = {"kaggle_delhi"}                     # 300 training days: inner split too small (flagged)


def c1_section():
    files = {n: DATA / f"qrc_design_c1_{n}.json" for n in ORDER}
    files = {n: f for n, f in files.items() if f.exists()}
    if not files:
        return ""
    gains = None
    rows_val, rows_mem, verdict_rows = [], [], []
    h1_support = h2_support = n_used = 0
    for n, f in files.items():
        d = json.loads(f.read_text(encoding="utf-8"))
        G = d["gains"]
        gains = sorted(G, key=float)
        ref = G["1"]
        best_g = max(gains, key=lambda g: G[g]["val"]["pooled"])
        mem_lag = sorted(ref["memory"], key=int)
        mid = [L for L in mem_lag if int(L) >= 20][0] if any(int(L) >= 20 for L in mem_lag) else mem_lag[-1]
        rows_val.append(f"| {n}{' †' if n in SMALL else ''} | " + " | ".join(
            f"{G[g]['val']['pooled']:.3f}" for g in gains) + f" | {best_g} |")
        rows_mem.append(f"| {n}{' †' if n in SMALL else ''} (lag {mid}) | " + " | ".join(
            f"{G[g]['memory'][mid]:.2f}" for g in gains) + " |")
        long1, longb = ref["val"]["long"], G[best_g]["val"]["long"]
        short1, shortb = ref["val"]["short"], G[best_g]["val"]["short"]
        verdict_rows.append(f"| {n}{' †' if n in SMALL else ''} | {best_g} | {ref['val']['pooled']:.3f} → "
                            f"{G[best_g]['val']['pooled']:.3f} | {long1:.3f} → {longb:.3f} | {short1:.3f} → {shortb:.3f} |")
        if n in SMALL:
            continue
        n_used += 1
        mem = [G[g]["memory"][mid] for g in gains]
        # H1: memory at the reference lag is higher at some g < 1 than at g = 1
        h1_support += int(max(mem[:-1]) > mem[-1] + 0.01)
        h2_support += int(longb - long1 >= 0.5 * 0.041 and shortb >= short1 - 0.005)
    hdr = "| dataset | " + " | ".join(f"g={g}" for g in gains) + " | best g |"
    sep = "|---" * (len(gains) + 2) + "|"
    memhdr = "| dataset | " + " | ".join(f"g={g}" for g in gains) + " |"
    memsep = "|---" * (len(gains) + 1) + "|"
    return f"""## Step 1 — C1 encoding gain (tests H1; first part of H2)

**Pre-registered:** H1 — input overwrite limits memory (memory at a mid lag rises as the encoding gain
g falls, until injection fails). H2 (partial, validation) — long-horizon R² improves by ≥ half the
benchmark gap (0.0205) without losing short horizons (> 0.005).
**Setup:** θ = g·arcsin(√s), arcsin readout, no training; ridge α per target on an inner split of the
training period, scored on the validation period. **No test data used.** Source files:
`qrc_design_c1_<dataset>.json` (script `backend/scripts/qrc_design_c1.py`; traces saved per gain).
† Kaggle Delhi: 300 training days, inner split of 60 days — too small for stable selection; reported,
not counted.

### Pooled validation R² vs gain
{hdr}
{sep}
{chr(10).join(rows_val)}

### Memory function (validation R² of reconstructing the input at the given lag) vs gain
{memhdr}
{memsep}
{chr(10).join(rows_mem)}

### Best gain vs g = 1 (validation)
| dataset | best g | pooled R² | long-horizon R² | short-horizon R² |
|---|---|---|---|---|
{chr(10).join(verdict_rows)}

**Verdict (validation, {n_used} datasets counted):** H1 supported on {h1_support}/{n_used};
H2 criterion met on {h2_support}/{n_used}.

**Caveats recorded with the result.** (1) "Best g" is chosen and scored on the same validation period,
so those numbers are optimistic; the robust finding is the trend (memory and long-horizon skill rise as
g falls). The gain rule must be frozen and confirmed on untouched data. (2) Small-data regime: on Kaggle
Delhi (300 training days) low gains are unstable (validation R² down to −6.3, erratic memory) — weak
injection makes the features small and near-collinear, so the readout is ill-conditioned with little
data. The best gain is therefore data-regime dependent, which bears on H5 (protecting the QRC's
small-data advantage).
"""


def _ci(x):
    return f"{x['point']:+.4f} [{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}]"


def confirm1_section():
    cities = [c for c in ("chennai", "cairo", "moscow") if (DATA / f"qrc_confirm1_{c}.json").exists()]
    if not cities:
        return ""
    D = {c: json.loads((DATA / f"qrc_confirm1_{c}.json").read_text(encoding="utf-8")) for c in cities}
    s1 = [D[c]["tests"]["S1_S2 g0.35 - g1"] for c in cities]
    n_pass = sum(x["all"]["ci95"][0] > 0 for x in s1)
    long_m = sum(x["long"]["point"] for x in s1) / len(s1)
    short_m = sum(x["short"]["point"] for x in s1) / len(s1)
    s2 = long_m >= 0.0205 and short_m >= -0.005
    rows, sec, s3 = [], [], []
    for c in cities:
        d, t = D[c], D[c]["tests"]
        x = t["S1_S2 g0.35 - g1"]
        rows.append(f"| {c} | {d['groups']['qrc_g1']['pooled']:.4f} | {d['groups']['qrc_g035']['pooled']:.4f} | "
                    f"{_ci(x['all'])} | {_ci(x['long'])} | {_ci(x['short'])} |")
        gs = d["rule_secondary_g"]
        sec.append(f"| {c} | g = {gs:g} | {_ci(t[f'secondary g{gs:g} - g1']['all'])} |")
        s3.append(f"| {c} | " + " | ".join(f"{_ci(t[f'S3 g1 - {b}']['all'])} → {_ci(t[f'S3 g0.35 - {b}']['all'])}"
                                          for b in ("esn_tuned", "lstm_tuned", "linear_ar")) + " |")
    nl = "\n"
    return f"""## Confirmation 1 — frozen gain g = 0.35 on untouched cities (pre-registered, commit 29152bf)

**Pre-registered before the data were downloaded** (`QRC_Design_Concepts.md` §6). Untouched ERA5 cities;
benchmark protocol (train 1971–1999, validation 2000–2004, **test 2005–2025, read once**); source files
`qrc_confirm1_<city>.json` (script `backend/scripts/qrc_design_confirm.py`; predictions and traces saved).

### S1 / S2 — QRC g = 0.35 vs g = 1 (pooled test R², 95% block-bootstrap CI)
| city | g = 1 | g = 0.35 | Δ all targets | Δ long (h ≥ 30) | Δ short (h ≤ 10) |
|---|---|---|---|---|---|
{nl.join(rows)}

**S1 (≥ 2/3 cities with CI > 0): {"PASS" if n_pass >= 2 else "FAIL"} ({n_pass}/{len(cities)}).**
**S2 (mean long-horizon gain ≥ +0.0205, short not worse than −0.005): {"PASS" if s2 else "FAIL"}**
(long {long_m:+.4f}, short {short_m:+.4f}).

### Secondary rule — per-city validation-selected gain vs g = 1
| city | selected | Δ all targets |
|---|---|---|
{nl.join(sec)}

The fixed rule beat the per-city selected rule where they differed (Chennai, Cairo): one frozen gain
generalised better than re-tuning the gain on each city's validation period.

### S3 (descriptive) — QRC − baseline, g = 1 → g = 0.35
| city | − tuned ESN | − tuned LSTM | − tuned linear AR |
|---|---|---|---|
{nl.join(s3)}

The gain rule closes most of the gap to the tuned ESN but does not overtake it (comparable in Cairo,
ESN still significantly better in Chennai and Moscow).
"""


def main():
    sections = [c1_section(), confirm1_section()]
    md = ("# QRC design evidence log\n\nGenerated by `backend/scripts/qrc_design_report.py` from "
          "`docs/QRC/data/qrc_design_*.json` — do not edit by hand. Concepts, hypotheses and the protocol "
          "are pre-registered in `QRC_Design_Concepts.md`; all development steps use validation data only; "
          "the frozen design is confirmed later on untouched datasets.\n\n" + "\n".join(s for s in sections if s))
    OUT.write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
