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


def c6_section():
    fz_f, ph_f, et_f = (DATA / "qrc_design_c6_frozen.json", DATA / "qrc_design_c6_posthoc.json",
                        DATA / "qrc_design_c6_etth2.json")
    if not fz_f.exists():
        return ""
    fz = json.loads(fz_f.read_text(encoding="utf-8"))
    tr = list(fz["mean_val"])
    dev_rows = "\n".join(f"| {n} | " + " | ".join(f"{v[t]:.4f}" for t in tr) + " |" for n, v in fz["datasets"].items())
    mean_row = "| **mean** | " + " | ".join(f"**{fz['mean_val'][t]:.4f}**" for t in tr) + " |"
    out = f"""## Step 2 — C6 bounded readout features (pre-registered, commit 53d8e6a)

**Pre-registered** (`QRC_Design_Concepts.md` §7): transforms clipz / tanhz / rank (fitted on training data)
vs raw; freeze the best bounded transform on development *validation* data at the frozen gain g = 0.35;
confirm on untouched **ETTh2** (real distribution shift). Source files: `qrc_design_c6_frozen.json`,
`qrc_design_c6_posthoc.json`, `qrc_design_c6_etth2.json` (script `backend/scripts/qrc_design_c6.py`).

### Freeze (validation only, pooled R²)
| dataset | {" | ".join(tr)} |
|---|{"---|" * len(tr)}
{dev_rows}
{mean_row}

**Frozen transform: `{fz['frozen']}`** (committed 34156a7, before ETTh2 was downloaded).
"""
    if et_f.exists():
        e = json.loads(et_f.read_text(encoding="utf-8"))
        ref = f"qrc_g035_{e['frozen_transform']}"
        g_rows = "\n".join(f"| {g} | {v['pooled']:+.4f} |" for g, v in e["groups"].items())
        t_rows = "\n".join(f"| {k} | {_ci(v)} |" for k, v in e["tests"].items())
        out += f"""
### R1 (primary) — untouched ETTh2, pooled test R²
| model | test R² |
|---|---|
{g_rows}

| difference | point [95% CI] |
|---|---|
{t_rows}

**R1: {"PASS" if e["R1_pass"] else "FAIL"}** — the frozen transform must have R² ≥ 0 and not be worse than raw
(CI lower bound > −0.005). ESN transform chosen on validation: `{e['esn_transform_choice']}`.
"""
    if ph_f.exists():
        p = json.loads(ph_f.read_text(encoding="utf-8"))
        f = p["frozen"]
        p_rows = "\n".join(f"| {n} | {v['raw']:+.4f} | {v[f]:+.4f} | {_ci(v['diff'])} |" for n, v in p["datasets"].items())
        out += f"""
### R2 (post-hoc, labelled — development test sets already seen)
| dataset | raw | {f} | {f} − raw |
|---|---|---|---|
{p_rows}

**R2: {"PASS" if p["R2_pass"] else "FAIL"}** — negative R²: {p['R2_detail']['negative_r2'] or 'none'};
loss > 0.005: {p['R2_detail']['loss_gt_0.005'] or 'none'}.
"""
    out += """
**Verdict: C6 not supported.** Bounding the feature magnitudes does not fix the drift failure (ETTh1 stays
negative; on untouched ETTh2 the transform is worse than raw). The ETTh1 failure **replicates out of
sample** on ETTh2 for the standard QRC (g = 1: R² −0.15), while the tuned ESN / LSTM / linear AR reach
0.75–0.81. The cause is not feature size but extrapolation: under drift the inputs leave the training
range, where the QRC's strongly nonlinear feature map gives the linear readout no reliable relation;
the linear AR and ESN extrapolate far more gracefully. Supplementary (not pre-registered for ETTh2): the
frozen gain g = 0.35 raises the QRC from −0.15 to +0.36 there, consistent with Confirmation 1, but it
remains ~0.4 R² behind the classical models — a documented limitation under input drift.
"""
    return out


def c5c7_section():
    fz_f = DATA / "qrc_design_c5c7_frozen.json"
    if not fz_f.exists():
        return ""
    fz = json.loads(fz_f.read_text(encoding="utf-8"))
    taus = sorted({k.split("|")[0] for k in fz["mean_val"]}, key=float)
    ros = list(dict.fromkeys(k.split("|")[1] for k in fz["mean_val"]))
    hdr = "| τ (s) | " + " | ".join(ros) + " |"
    rows = "\n".join(f"| {t} | " + " | ".join(
        (f"**{fz['mean_val'][f'{t}|{r}']:.4f}**" if f"{t}|{r}" == fz["best"] else f"{fz['mean_val'][f'{t}|{r}']:.4f}")
        for r in ros) + " |" for t in taus)
    out = f"""## Step 3 — C5 step time τ + C7 compact readout (pre-registered, commit ed96b17)

**Pre-registered** (`QRC_Design_Concepts.md` §8): at the frozen gain g = 0.35, τ ∈ {{15, 30, 60, 100}} ms ×
readout ∈ {{bins653, bins200, bins100, pca128, pca64, lines}}; adopt the best mean development-validation
combination only if it beats the current design (τ = 30 ms, bins653) by ≥ 0.005. Source files:
`qrc_design_c5c7_dev_<dataset>.json`, `qrc_design_c5c7_frozen.json` (script `backend/scripts/qrc_design_c5c7.py`;
FID traces saved for every τ).

### Mean pooled validation R² over the 8 development datasets (bold = best)
{hdr}
|---|{"---|" * len(ros)}
{rows}

**Decision:** best `{fz['best']}` = {fz['mean_val'][fz['best']]:.4f} vs current {fz['current']:.4f}
(Δ {fz['mean_val'][fz['best']] - fz['current']:+.4f}; threshold +0.005) → **{"ADOPTED " + fz['adopted'] if fz['changed']
else "NOT ADOPTED — current design kept"}**.
"""
    kg = DATA / "qrc_design_c5c7_dev_kaggle_delhi.json"
    if kg.exists():
        S = json.loads(kg.read_text(encoding="utf-8"))["scores"]
        out += ("\n**T4 small-data check (Kaggle Delhi, 300 training days, validation R²):** "
                + "; ".join(f"τ={t}: bins653 {S[t]['bins653']:.3f}, bins100 {S[t]['bins100']:.3f}, lines {S[t]['lines']:.3f}"
                            for t in taus) + ".\n")
    confs = [c for c in ("sydney", "sao_paulo", "toronto") if (DATA / f"qrc_design_c5c7_confirm_{c}.json").exists()]
    if confs:
        out += "\n### Confirmation on untouched cities (pooled test R², 95% CI)\n| city | current | adopted | adopted − current |\n|---|---|---|---|\n"
        for c in confs:
            d = json.loads((DATA / f"qrc_design_c5c7_confirm_{c}.json").read_text(encoding="utf-8"))
            out += (f"| {c} | {d['groups']['qrc_current']['pooled']:.4f} | {d['groups']['qrc_adopted']['pooled']:.4f} | "
                    f"{_ci(d['tests']['T1 adopted - current'])} |\n")
    return out


def main():
    sections = [c1_section(), confirm1_section(), c6_section(), c5c7_section()]
    md = ("# QRC design evidence log\n\nGenerated by `backend/scripts/qrc_design_report.py` from "
          "`docs/QRC/data/qrc_design_*.json` — do not edit by hand. Concepts, hypotheses and the protocol "
          "are pre-registered in `QRC_Design_Concepts.md`; all development steps use validation data only; "
          "the frozen design is confirmed later on untouched datasets.\n\n" + "\n".join(s for s in sections if s))
    OUT.write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
