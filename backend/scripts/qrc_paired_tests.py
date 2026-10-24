"""Paired statistical tests for the synthetic-benchmark headline claims.

Reads the published campaign files in docs/QRC/data and, per (system, task),
compares the learned per-spin encoding (regime B) against:

  * the fixed arcsin encoding (regime A)          -- paired by seed, same run
  * the tau->0 ablation of the learned encoding   -- paired by seed (where run)
  * tuned LSTM / tuned ESN / size-matched ESN     -- qrc_classical_fair_*.json
  * the in-run LSTM(64) (untuned, same job)       -- paired by seed

Pairing with the classical baselines is legitimate: qrc_classical_fair.py and the
QRC runners (qrc_learnable_campaign.py / qrc_learnable_9spin.train) both build the
data with the SAME ``qrc_learnable_9spin.make_task(task, T, seed, horizon)`` and the
SAME post-washout ``_three_way`` 50/25/25 split; T=1500 and washout=30 are checked
from the file metadata below. So classical data seed s == QRC seed s (the QRC seed
additionally seeds the encoder init; the classical models average over 3 inits).
Seeds with no classical counterpart (e.g. seeds 4-5 from the 6spin_pub campaign)
are left out of the paired test; an unpaired Welch t on log NMSE over all available
seeds is reported alongside in that case.

Statistic per comparison: d_i = ln NMSE_learned,i - ln NMSE_other,i (d < 0 favours
the learned encoding). Reported: n, wins, mean d, sample s.d. of d (ddof=1), the
geometric-mean ratio exp(mean d), exact sign test, exact sign-flip permutation test
(all 2^n flips of the mean of d) and paired t on log NMSE, each one-sided (learned
better) and two-sided. With n = 3 the smallest attainable p is 0.125 one-sided /
0.25 two-sided for the sign and permutation tests -- these tests CANNOT reach 0.05
at n = 3; the t-test can, but rests on normality of d with 2 d.o.f.

Extra campaign files (same schema as qrc_6spin_campaign.json, e.g. the 6spin_pub
shards) are merged with ``--extra path.json[:system]``; a (system, task, seed,
ablation) key already present keeps its first value and the duplicate is listed
under ``duplicates`` (useful for the 6spin_retrace reproducibility check).

  python backend/scripts/qrc_paired_tests.py
  python backend/scripts/qrc_paired_tests.py --extra runs/6spin-pub/6spin_pub_shard0/campaign.json \
      --extra runs/6spin-pub/6spin_pub_shard1/campaign.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "docs" / "QRC" / "data"

CAMPAIGNS = [  # (file, system label)
    ("qrc_6spin_campaign.json", "6spin"),
    ("qrc_crotonic9_campaign.json", "crotonic9"),
    ("qrc_dccb7_tau30.json", "dccb7_tau30"),
    ("qrc_dccb7_tau100.json", "dccb7_tau100"),
    ("qrc_dccb7_tau100_j0.json", "dccb7_tau100_J0"),
]
# learned(A) vs learned(B) on the same seeds: does the coupling J matter?
CROSS_SYSTEM = [("dccb7_tau100", "dccb7_tau100_J0")]


def J(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def system_label(d, path):
    plan, sysn = d.get("plan"), str(d.get("system"))
    if plan in ("6spin", "6spin_pub", "6spin_retrace") or sysn in ("6", "6-spin"):
        return "6spin"
    if "crotonic" in sysn:
        return "crotonic9"
    if sysn == "dccb7":
        lab = f"dccb7_tau{round(1000 * float(d.get('tau', 0.03)))}"
        return lab + ("_J0" if float(d.get("coupling_scale", 1.0)) == 0.0 else "")
    raise SystemExit(f"cannot infer system for {path}; pass --extra {path}:<system>")


# ---- loading ---------------------------------------------------------------


def load_records(extras):
    recs, dups, meta = {}, [], {}

    def add(system, j, source):
        if "error" in j or j.get("perspin") is None:
            return
        key = (system, j["task"], int(j["seed"]), bool(j.get("ablation", False)))
        r = {"arcsin": j.get("arcsin"), "perspin": j["perspin"], "lstm64": j.get("lstm"), "source": source}
        if key in recs:
            dups.append({"key": list(key), "kept": recs[key], "duplicate": r})
        else:
            recs[key] = r

    for f, lab in CAMPAIGNS:
        p = DATA / f
        if not p.exists():
            continue
        d = J(p)
        meta[f] = {"system": lab, "T": d.get("T"), "washout": d.get("washout", 30)}
        for j in d["jobs"]:
            add(lab, j, f)

    # 6-spin Mackey-Glass h=10: seed 1 exact from its own file, seeds 2-3 from the
    # confirm run (its seed-1 entry is a rounded copy and is ignored)
    s1 = J(DATA / "qrc_6spin_mackey_h10_seed1.json")
    meta["qrc_6spin_mackey_h10_seed1.json"] = {"system": "6spin", "T": s1.get("T"), "washout": 30}
    add("6spin", {"task": "mackey_glass_h10", "seed": 1,
                  "arcsin": s1["conditions"]["arcsin"]["test_nmse"],
                  "perspin": s1["conditions"]["perspin"]["test_nmse"]}, "qrc_6spin_mackey_h10_seed1.json")
    conf = J(DATA / "qrc_6spin_mackey_h10_confirm.json")
    for s, v in conf["seeds"].items():
        if "arcsin" in v:
            add("6spin", {"task": "mackey_glass_h10", "seed": int(s), "arcsin": v["arcsin"],
                          "perspin": v["perspin"]}, "qrc_6spin_mackey_h10_confirm.json")

    for e in extras:
        path, _, lab = e.rpartition(":") if not Path(e).exists() else (e, "", "")
        d = J(path)
        lab = lab or system_label(d, path)
        meta[str(path)] = {"system": lab, "T": d.get("T"), "washout": d.get("washout", 30), "extra": True}
        for j in d["jobs"]:
            add(lab, j, str(path))
    return recs, dups, meta


def load_classical():
    esn = J(DATA / "qrc_classical_fair_esn.json")
    lstm = J(DATA / "qrc_classical_fair_lstm.json")
    out = {}
    for t, r in lstm["results"].items():
        out.setdefault(t, {})["tuned_lstm"] = {int(s): v["test_mean"] for s, v in r["per_seed"].items()}
    for t, r in esn["results"].items():
        out.setdefault(t, {})["tuned_esn"] = {int(s): v["esn_best"]["test_mean"] for s, v in r["per_seed"].items()}
        out[t]["matched_esn"] = {int(s): v["esn_matched"]["test_mean"] for s, v in r["per_seed"].items()}
    cmeta = {"T": [esn["T"], lstm["T"]], "washout": [esn["washout"], lstm["washout"]],
             "data_seeds": [esn["data_seeds"], lstm["data_seeds"]], "split": esn["split"]}
    return out, cmeta


# ---- statistics ------------------------------------------------------------


def paired(a, b, seeds):
    """a = learned, b = comparator (NMSE, per seed). d = ln a - ln b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = np.log(a) - np.log(b)
    n = len(d)
    res = {"n": n, "seeds": list(seeds), "learned": a.tolist(), "other": b.tolist(),
           "log_ratio": d.tolist(), "wins": int(np.sum(d < 0)),
           "mean_log_ratio": float(d.mean()), "geo_mean_ratio": float(np.exp(d.mean()))}
    res["sd_log_ratio"] = float(d.std(ddof=1)) if n > 1 else None
    # exact sign test (ties dropped)
    nz = d[d != 0]
    k, m = int(np.sum(nz < 0)), len(nz)
    if m:
        res["sign_p_one_sided"] = float(stats.binomtest(k, m, 0.5, alternative="greater").pvalue)
        res["sign_p_two_sided"] = float(stats.binomtest(k, m, 0.5, alternative="two-sided").pvalue)
    # exact sign-flip permutation on the mean of d
    if n:
        flips = np.array(list(itertools.product((1.0, -1.0), repeat=n)))
        means = (flips * d).mean(axis=1)
        eps = 1e-12 * max(1.0, abs(d.mean()))
        res["perm_p_one_sided"] = float(np.mean(means <= d.mean() + eps))
        res["perm_p_two_sided"] = float(np.mean(np.abs(means) >= abs(d.mean()) - eps))
        res["min_attainable_p"] = {"one_sided": 2.0 ** -n, "two_sided": min(1.0, 2.0 ** (1 - n))}
    if n > 1 and res["sd_log_ratio"] > 0:
        t = d.mean() / (res["sd_log_ratio"] / math.sqrt(n))
        res["t"] = float(t)
        res["df"] = n - 1
        res["t_p_one_sided"] = float(stats.t.cdf(t, n - 1))
        res["t_p_two_sided"] = float(2 * stats.t.sf(abs(t), n - 1))
    return res


def welch(a, b):
    la, lb = np.log(np.asarray(a, float)), np.log(np.asarray(b, float))
    if len(la) < 2 or len(lb) < 2:
        return None
    r = stats.ttest_ind(la, lb, equal_var=False)
    return {"n_learned": len(la), "n_other": len(lb), "mean_log_learned": float(la.mean()),
            "mean_log_other": float(lb.mean()), "geo_mean_ratio": float(np.exp(la.mean() - lb.mean())),
            "t": float(r.statistic), "p_two_sided": float(r.pvalue),
            "p_one_sided": float(r.pvalue / 2 if r.statistic < 0 else 1 - r.pvalue / 2)}


# ---- main ------------------------------------------------------------------


def run(extras):
    recs, dups, meta = load_records(extras)
    classical, cmeta = load_classical()
    for f, m in meta.items():
        if m["T"] not in (None, 1500) or m["washout"] != 30:
            raise SystemExit(f"{f}: T={m['T']} washout={m['washout']} != classical T=1500 washout=30")
    if set(cmeta["T"]) != {1500} or set(cmeta["washout"]) != {30}:
        raise SystemExit(f"classical metadata mismatch: {cmeta}")

    groups = sorted({(s, t) for (s, t, _, ab) in recs if not ab})
    comps = []
    for system, task in groups:
        main_ = {sd: r for (s, t, sd, ab), r in recs.items() if s == system and t == task and not ab}
        abl = {sd: r for (s, t, sd, ab), r in recs.items() if s == system and t == task and ab}
        seeds = sorted(main_)
        base = {"system": system, "task": task, "all_seeds": seeds}

        sd = [s for s in seeds if main_[s]["arcsin"] is not None]
        comps.append({**base, "comparison": "learned vs arcsin", "pairing": "same run, same seed",
                      **paired([main_[s]["perspin"] for s in sd], [main_[s]["arcsin"] for s in sd], sd)})
        sd = sorted(s for s in abl if s in main_)
        if sd:
            comps.append({**base, "comparison": "learned vs tau->0 ablation", "pairing": "same seed",
                          **paired([main_[s]["perspin"] for s in sd], [abl[s]["perspin"] for s in sd], sd)})
        for name, lab in (("tuned_lstm", "tuned LSTM"), ("tuned_esn", "tuned ESN"),
                          ("matched_esn", "size-matched ESN")):
            ref = classical.get(task, {}).get(name)
            if not ref:
                continue
            sd = [s for s in seeds if s in ref]
            c = {**base, "comparison": f"learned vs {lab}", "pairing": "same data seed (make_task + _three_way)",
                 **paired([main_[s]["perspin"] for s in sd], [ref[s] for s in sd], sd)}
            if len(sd) < len(seeds):
                c["welch_all_seeds"] = welch([main_[s]["perspin"] for s in seeds], list(ref.values()))
                c["note"] = f"QRC seeds {sorted(set(seeds) - set(sd))} have no classical counterpart"
            comps.append(c)
        sd = [s for s in seeds if main_[s].get("lstm64") is not None]
        if sd:
            comps.append({**base, "comparison": "learned vs in-run LSTM(64) (untuned)", "pairing": "same run, same seed",
                          **paired([main_[s]["perspin"] for s in sd], [main_[s]["lstm64"] for s in sd], sd)})

    for sa, sb in CROSS_SYSTEM:
        for task in sorted({t for (s, t) in groups if s == sa} & {t for (s, t) in groups if s == sb}):
            A = {sd: r for (s, t, sd, ab), r in recs.items() if s == sa and t == task and not ab}
            B = {sd: r for (s, t, sd, ab), r in recs.items() if s == sb and t == task and not ab}
            sd = sorted(set(A) & set(B))
            comps.append({"system": f"{sa} vs {sb}", "task": task, "all_seeds": sd,
                          "comparison": f"learned {sa} vs learned {sb}", "pairing": "same data seed",
                          **paired([A[s]["perspin"] for s in sd], [B[s]["perspin"] for s in sd], sd)})

    # pooled sign test over every (system, task, seed) learned-vs-arcsin pair
    lva = [c for c in comps if c["comparison"] == "learned vs arcsin"]
    k = sum(c["wins"] for c in lva)
    m = sum(c["n"] for c in lva)
    pooled = {"comparison": "learned vs arcsin, all systems x tasks x seeds", "wins": k, "n": m,
              "sign_p_one_sided": float(stats.binomtest(k, m, 0.5, alternative="greater").pvalue),
              "caveat": "pairs share data seeds across systems, so they are not fully independent; "
                        "treat as descriptive"}
    return {"experiment": "paired_tests", "regime": "B (trained encoding) vs A / classical",
            "readout": "fid_reduced (D = 2*D_eff per campaign; labeled in the source files) -- NOT the 653-FID standard",
            "metric": "test NMSE; d = ln(learned) - ln(other), d < 0 favours learned",
            "seed_pairing_check": {"qrc_files": meta, "classical": cmeta,
                                   "conclusion": "same make_task(task, T=1500, seed, horizon) + washout 30 + _three_way "
                                                 "split in both code paths -> classical data seed s == QRC seed s; paired"},
            "small_n_note": "n = 3 seeds: the exact sign / permutation tests cannot go below p = 0.125 (one-sided) "
                            "or 0.25 (two-sided); the paired t (df = 2) assumes normal log-ratios",
            "extras": list(extras), "comparisons": comps, "pooled_learned_vs_arcsin": pooled,
            "duplicates": dups}


def fmt(x, nd=3):
    if x is None:
        return "-"
    return f"{x:.{nd}g}"


def to_md(out):
    L = ["# Paired tests -- synthetic headline claims", "",
         f"Generated by `backend/scripts/qrc_paired_tests.py` from `docs/QRC/data` "
         f"(extras: {', '.join(out['extras']) or 'none'}). Readout: {out['readout']}.", "",
         f"Metric: {out['metric']}. Ratio = geometric mean of learned/other NMSE (< 1 favours learned). "
         "p-values: one-sided (learned better) / two-sided.", "",
         f"**Small n:** {out['small_n_note']}.", "",
         f"**Seed pairing:** {out['seed_pairing_check']['conclusion']}.", "",
         "| system | task | comparison | n | wins | ratio | mean d ± s.d. | sign p | perm p | paired-t p |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for c in out["comparisons"]:
        p = lambda a, b: f"{fmt(c.get(a), 2)} / {fmt(c.get(b), 2)}"  # noqa: E731
        sd = f"{c['mean_log_ratio']:+.2f} ± {c['sd_log_ratio']:.2f}" if c.get("sd_log_ratio") is not None \
            else f"{c['mean_log_ratio']:+.2f}"
        row = (f"| {c['system']} | {c['task']} | {c['comparison']} | {c['n']} | {c['wins']}/{c['n']} | "
               f"{fmt(c['geo_mean_ratio'])} | {sd} | {p('sign_p_one_sided', 'sign_p_two_sided')} | "
               f"{p('perm_p_one_sided', 'perm_p_two_sided')} | {p('t_p_one_sided', 't_p_two_sided')} |")
        L.append(row)
        if c.get("welch_all_seeds"):
            w = c["welch_all_seeds"]
            L.append(f"| | | ↳ Welch on log NMSE, all seeds ({w['n_learned']} vs {w['n_other']}) | | | "
                     f"{fmt(w['geo_mean_ratio'])} | | | | {fmt(w['p_one_sided'], 2)} / {fmt(w['p_two_sided'], 2)} |")
    pl = out["pooled_learned_vs_arcsin"]
    L += ["", f"**Pooled** {pl['comparison']}: {pl['wins']}/{pl['n']} wins, exact sign p (one-sided) = "
              f"{pl['sign_p_one_sided']:.2g} ({pl['caveat']}).", ""]
    if out["duplicates"]:
        L += ["Duplicates (first kept):", ""]
        for d in out["duplicates"]:
            L.append(f"- {tuple(d['key'])}: kept {fmt(d['kept']['perspin'], 4)} ({d['kept']['source']}), "
                     f"duplicate {fmt(d['duplicate']['perspin'], 4)} ({d['duplicate']['source']})")
        L.append("")
    L += ["Comparisons with the tuned classical baselines are run on the SAME data seeds/split; "
          "the in-run LSTM(64) is the campaign's untuned reference (kept for continuity only).", ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", action="append", default=[],
                    help="extra campaign json (qrc_6spin_campaign schema), optionally path:system")
    ap.add_argument("--out", default=str(DATA / "qrc_paired_tests.json"))
    args = ap.parse_args()
    out = run(args.extra)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    Path(args.out).with_suffix(".md").write_text(to_md(out), encoding="utf-8")
    print(f"wrote {args.out} (+ .md): {len(out['comparisons'])} comparisons")


if __name__ == "__main__":
    main()
