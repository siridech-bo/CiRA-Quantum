"""QRC Phase-1 v2 — a *rigorous* feature-representation study (offline).

The v1 sweep (``qrc_phase1.py``) appended each feature block to the 653-magnitude
baseline and ranked on a single horizon. That is underpowered/confounded and
cannot answer "does phase / multimodal information carry signal": the 18
multimodal descriptors are ~0.9% of a 1977-column matrix (swamped), blocks are
appended not isolated (marginal value unattributable), and with 1977 features vs
600 train samples (p >> n) a ±0.01 change on one split is noise.

This script removes those confounds, all offline on the cached FID trace:

  1. **Standalone** — evaluate each representation *on its own* (magnitude /
     phase-only / complex / multimodal-only / full), so each is judged by its own
     predictive content, not as an addition.
  2. **Random-feature null** — compare ``baseline + block`` against
     ``baseline + K random features`` (matched K, many draws). If the real block
     doesn't beat the null distribution, it adds no marginal signal — a *clean*
     negative. If it does, it does.
  3. **Dimensionality-matched** — reduce every representation to the same width
     (PCA-k) and compare, killing the 18-vs-1959 imbalance.
  4. **Proper readout** — per-fit z-scoring + ``RidgeCV`` alpha selection.
  5. **Error bars** — blocked (time-respecting) CV → mean ± std, all horizons.

Reservoir-*seed* variance is NOT covered here (it needs fresh GPU traces); this
isolates the *feature-representation* question given the one cached trace.

Run from ``backend``::

    python scripts/qrc_phase1_v2.py --trace artifacts/traces/weather_full.npz \
        --out-dir artifacts/qrc_phase1_v2 --n-null 25
"""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from app.qrc.feature_methods import build_features

ALPHAS = np.logspace(-3, 5, 25)


def ridge_r2(Xtr, ytr, Xte, yte) -> float:
    """Z-score on train, RidgeCV (alpha via internal CV), R² on test — the
    same readout family as the reproduction, but with per-representation alpha
    selection so no single representation is unfairly regularised."""
    sc = StandardScaler().fit(Xtr)
    model = RidgeCV(alphas=ALPHAS).fit(sc.transform(Xtr), ytr)
    return float(r2_score(yte, model.predict(sc.transform(Xte))))


def load_trace(path: str) -> dict:
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def column_masks(names: list[str]) -> dict[str, np.ndarray]:
    names = np.asarray(names)
    mag = np.char.startswith(names, "fid_mag_")
    re = np.char.startswith(names, "fid_re_")
    im = np.char.startswith(names, "fid_im_")
    mm = (np.char.startswith(names, "td_") | np.char.startswith(names, "wav_")
          | np.char.startswith(names, "nl_"))
    return {
        "magnitude653": mag,
        "phase_only(re+im)": re | im,
        "complex(mag+re+im)": mag | re | im,
        "multimodal_only": mm,
        "full": np.ones(len(names), dtype=bool),
    }


def blocked_folds(lo: int, hi: int, k: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """k contiguous (time-respecting) folds over rows [lo, hi): each fold is the
    test block, the rest is train. Preserves temporal structure (no shuffling)."""
    idx = np.arange(lo, hi)
    splits = np.array_split(idx, k)
    folds = []
    for f in splits:
        test = f
        train = np.setdiff1d(idx, test)
        folds.append((train, test))
    return folds


def main() -> None:
    ap = argparse.ArgumentParser(description="QRC Phase-1 v2 (rigorous)")
    ap.add_argument("--trace", required=True)
    ap.add_argument("--out-dir", default="artifacts/qrc_phase1_v2")
    ap.add_argument("--n-peaks", type=int, default=653)
    ap.add_argument("--n-null", type=int, default=25, help="random-feature null draws")
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--match-k", type=int, default=18, help="dimensionality-matched width")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    npz = load_trace(args.trace)
    fids = np.asarray(npz["fids"])
    n = fids.shape[0]
    washout, n_train, n_test = (int(x) for x in np.asarray(npz["split"]).ravel()[:3])
    temp = np.asarray(npz["weather_norm"], dtype=float)[:, 0]
    horizons = [int(h) for h in np.asarray(npz["horizons"]).ravel()]
    print(f"trace: n={n} split=({washout},{n_train},{n_test}) horizons={horizons}", flush=True)

    X, names = build_features(fids, "multimodal", n_peaks=args.n_peaks, select="first")
    masks = column_masks(names)
    for k, m in masks.items():
        print(f"  block {k:<22} width={int(m.sum())}")

    tr = np.arange(washout, washout + n_train)
    te = np.arange(washout + n_train, washout + n_train + n_test)

    def y_of(h: int) -> np.ndarray:
        return temp[np.arange(n) + h]

    # ---- 1. Standalone: fixed split + blocked-CV mean±std, all horizons ------
    folds = blocked_folds(washout, washout + n_train + n_test, args.cv_folds)
    standalone: dict[str, dict] = {}
    for rep, mask in masks.items():
        Xr = X[:, mask]
        per_h = {}
        for h in horizons:
            y = y_of(h)
            fixed = ridge_r2(Xr[tr], y[tr], Xr[te], y[te])
            cv = [ridge_r2(Xr[a], y[a], Xr[b], y[b]) for a, b in folds]
            per_h[f"h{h}"] = {"fixed": fixed, "cv_mean": float(np.mean(cv)),
                              "cv_std": float(np.std(cv))}
        standalone[rep] = {"width": int(mask.sum()), "by_horizon": per_h}
        row = "  ".join(f"h{h}={per_h[f'h{h}']['cv_mean']:.3f}±{per_h[f'h{h}']['cv_std']:.3f}"
                        for h in horizons)
        print(f"[standalone] {rep:<22} w={int(mask.sum()):>4}  {row}", flush=True)

    # ---- 2. Random-feature null: marginal value of a block over baseline -----
    base = masks["magnitude653"]
    null_tests: dict[str, dict] = {}
    for block_name in ("phase_only(re+im)", "multimodal_only"):
        bmask = masks[block_name]
        w = int(bmask.sum())
        Xbase, Xblock = X[:, base], X[:, bmask]
        per_h = {}
        for h in horizons:
            y = y_of(h)
            r2_base = ridge_r2(Xbase[tr], y[tr], Xbase[te], y[te])
            r2_real = ridge_r2(np.hstack([Xbase, Xblock])[tr], y[tr],
                               np.hstack([Xbase, Xblock])[te], y[te])
            d_real = r2_real - r2_base
            d_null = []
            for _ in range(args.n_null):
                R = rng.standard_normal((n, w))
                r2_r = ridge_r2(np.hstack([Xbase, R])[tr], y[tr],
                                np.hstack([Xbase, R])[te], y[te])
                d_null.append(r2_r - r2_base)
            d_null = np.array(d_null)
            p95 = float(np.percentile(d_null, 95))
            per_h[f"h{h}"] = {
                "r2_base": r2_base, "r2_real": r2_real, "delta_real": d_real,
                "delta_null_mean": float(d_null.mean()), "delta_null_std": float(d_null.std()),
                "delta_null_p95": p95, "significant": bool(d_real > p95),
            }
        null_tests[block_name] = per_h
        sig = "  ".join(f"h{h}={'YES' if per_h[f'h{h}']['significant'] else 'no'}"
                        for h in horizons)
        print(f"[null] {block_name:<22} marginal>random?  {sig}", flush=True)

    # ---- 3. Dimensionality-matched (PCA-k) -----------------------------------
    k = args.match_k
    dim_matched: dict[str, dict] = {}
    for rep, mask in masks.items():
        Xr = X[:, mask]
        per_h = {}
        for h in horizons:
            y = y_of(h)
            if Xr.shape[1] <= k:
                Xtr, Xte = Xr[tr], Xr[te]           # already ≤ k wide
            else:
                pca = PCA(n_components=k, random_state=args.seed).fit(Xr[tr])
                Xtr, Xte = pca.transform(Xr[tr]), pca.transform(Xr[te])
            per_h[f"h{h}"] = ridge_r2(Xtr, y[tr], Xte, y[te])
        dim_matched[rep] = {"width": min(k, int(mask.sum())), "by_horizon": per_h}
        row = "  ".join(f"h{h}={per_h[f'h{h}']:.3f}" for h in horizons)
        print(f"[dim-matched k={k}] {rep:<22} {row}", flush=True)

    result = {
        "generated_utc": datetime.now(UTC).isoformat(),
        "trace": args.trace, "n": n, "split": [washout, n_train, n_test],
        "horizons": horizons, "n_null": args.n_null, "cv_folds": args.cv_folds,
        "match_k": k, "seed": args.seed,
        "standalone": standalone, "null_tests": null_tests, "dim_matched": dim_matched,
    }
    (out / "phase1_v2.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nDone. Wrote {out / 'phase1_v2.json'}")


if __name__ == "__main__":
    main()
