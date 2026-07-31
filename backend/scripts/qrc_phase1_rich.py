"""Does extracting *more* multimodal features help? — rigorous 18-vs-rich test.

The multimodal block in the pipeline is only 18 numbers: each modality collapses
the whole 2048-point FID into a few global scalars, discarding all time- and
scale-localization. This script builds a **rich** multimodal extractor (~130
features: windowed time-domain + per-band wavelet energy/entropy + an expanded
complexity panel) and runs magnitude653 / multimodal-18 / multimodal-rich through
the SAME rigorous harness as ``qrc_phase1_v2`` (standalone blocked-CV, marginal
random-feature null over the magnitude baseline, dimensionality-matched PCA).

Answers, rigorously: (a) does rich multimodal predict better standalone than the
18? (b) does it add marginal signal over magnitude that the 18 didn't? (c) at
matched dimensionality, is richer better?

Run from ``backend``::

    python scripts/qrc_phase1_rich.py --trace artifacts/traces/weather_full.npz \
        --out-dir artifacts/qrc_phase1_rich --n-null 25
"""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from app.qrc.feature_methods import build_features
from qrc_phase1_v2 import ALPHAS, blocked_folds, column_masks, load_trace, ridge_r2  # noqa: F401


def rich_multimodal(fids: np.ndarray, n_windows: int = 16, wav_levels: int = 5):
    """~130 time/scale-localized descriptors per FID (vs the pipeline's 18).

    Windowed time-domain on both the |FID| envelope and the real part, richer
    per-band wavelet stats, and an expanded nonlinear/complexity panel. Every
    block has a fixed, stable length; non-finite values → 0 (StandardScaler
    handles scale downstream)."""
    import pywt
    try:
        import antropy as ant
    except ImportError:
        ant = None

    n = fids.shape[0]
    wav = pywt.Wavelet("db4")
    rows: list[list[float]] = []
    names: list[str] | None = None

    for r in range(n):
        fid = fids[r]
        x = np.real(fid).astype(float)
        env = np.abs(fid).astype(float)
        vals: list[float] = []
        nm: list[str] = []

        # 1) windowed time-domain (localizes the decay envelope + oscillation)
        for arr, tag in ((env, "env"), (x, "re")):
            for i, w in enumerate(np.array_split(arr, n_windows)):
                vals += [float(np.mean(np.abs(w))), float(np.std(w)), float(np.sum(w ** 2))]
                nm += [f"{tag}w{i}_meanabs", f"{tag}w{i}_std", f"{tag}w{i}_energy"]

        # 2) richer wavelet: energy / log-energy / std / mean|c| / coeff-entropy per band
        lev = max(1, min(wav_levels, pywt.dwt_max_level(len(x), wav.dec_len)))
        for i, c in enumerate(pywt.wavedec(x, wav, level=lev)):
            c = np.asarray(c, float)
            e = float(np.sum(c ** 2))
            p = c ** 2 / (e + 1e-12)
            ent = float(-np.sum(p * np.log(p + 1e-12)))
            vals += [e, float(np.log(e + 1e-12)), float(np.std(c)), float(np.mean(np.abs(c))), ent]
            nm += [f"wav{i}_energy", f"wav{i}_logE", f"wav{i}_std", f"wav{i}_meanabs", f"wav{i}_entropy"]

        # 3) expanded complexity panel
        if ant is not None:
            panel = [
                ("sampen", lambda: ant.sample_entropy(x)),
                ("permen", lambda: ant.perm_entropy(x, normalize=True)),
                ("specen", lambda: ant.spectral_entropy(x, sf=1.0, method="fft", normalize=True)),
                ("svden", lambda: ant.svd_entropy(x, normalize=True)),
                ("higuchi", lambda: ant.higuchi_fd(x)),
                ("petrosian", lambda: ant.petrosian_fd(x)),
                ("katz", lambda: ant.katz_fd(x)),
                ("dfa", lambda: ant.detrended_fluctuation(x)),
            ]
            for label, fn in panel:
                try:
                    v = float(fn())
                except Exception:  # noqa: BLE001 - a degenerate FID must not abort
                    v = 0.0
                vals.append(v if np.isfinite(v) else 0.0)
                nm.append(f"nl_{label}")
            try:
                mob, cmp = ant.hjorth_params(x)
            except Exception:  # noqa: BLE001
                mob, cmp = 0.0, 0.0
            vals += [float(mob), float(cmp)]
            nm += ["nl_hjorth_mob", "nl_hjorth_cmp"]

        rows.append(vals)
        if names is None:
            names = nm

    X = np.nan_to_num(np.asarray(rows, float), nan=0.0, posinf=0.0, neginf=0.0)
    return X, (names or [])


def standalone(reps, y_of, tr, te, folds, horizons):
    out = {}
    for name, Xr in reps.items():
        per = {}
        for h in horizons:
            y = y_of(h)
            fixed = ridge_r2(Xr[tr], y[tr], Xr[te], y[te])
            cv = [ridge_r2(Xr[a], y[a], Xr[b], y[b]) for a, b in folds]
            per[f"h{h}"] = {"fixed": fixed, "cv_mean": float(np.mean(cv)), "cv_std": float(np.std(cv))}
        out[name] = {"width": int(Xr.shape[1]), "by_horizon": per}
        row = "  ".join(f"h{h}={per[f'h{h}']['cv_mean']:.3f}±{per[f'h{h}']['cv_std']:.3f}" for h in horizons)
        print(f"[standalone] {name:<18} w={int(Xr.shape[1]):>4}  {row}", flush=True)
    return out


def null_test(base, block, y_of, tr, te, horizons, rng, n_null):
    w = block.shape[1]
    per = {}
    for h in horizons:
        y = y_of(h)
        r2_base = ridge_r2(base[tr], y[tr], base[te], y[te])
        Xr = np.hstack([base, block])
        r2_real = ridge_r2(Xr[tr], y[tr], Xr[te], y[te])
        d_real = r2_real - r2_base
        d_null = []
        for _ in range(n_null):
            R = np.hstack([base, rng.standard_normal((base.shape[0], w))])
            d_null.append(ridge_r2(R[tr], y[tr], R[te], y[te]) - r2_base)
        d_null = np.array(d_null)
        per[f"h{h}"] = {"delta_real": d_real, "delta_null_p95": float(np.percentile(d_null, 95)),
                        "delta_null_mean": float(d_null.mean()), "significant": bool(d_real > np.percentile(d_null, 95))}
    return per


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--out-dir", default="artifacts/qrc_phase1_rich")
    ap.add_argument("--n-peaks", type=int, default=653)
    ap.add_argument("--n-null", type=int, default=25)
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    npz = load_trace(args.trace)
    fids = np.asarray(npz["fids"])
    n = fids.shape[0]
    washout, n_train, n_test = (int(x) for x in np.asarray(npz["split"]).ravel()[:3])
    temp = np.asarray(npz["weather_norm"], dtype=float)[:, 0]
    horizons = [int(h) for h in np.asarray(npz["horizons"]).ravel()]

    X, names = build_features(fids, "multimodal", n_peaks=args.n_peaks, select="first")
    masks = column_masks(names)
    print("building rich multimodal features…", flush=True)
    X_rich, rich_names = rich_multimodal(fids)
    print(f"rich multimodal width = {X_rich.shape[1]} (vs pipeline's 18)", flush=True)

    reps = {
        "magnitude653": X[:, masks["magnitude653"]],
        "multimodal_18": X[:, masks["multimodal_only"]],
        "multimodal_rich": X_rich,
    }
    tr = np.arange(washout, washout + n_train)
    te = np.arange(washout + n_train, washout + n_train + n_test)
    folds = blocked_folds(washout, washout + n_train + n_test, args.cv_folds)

    def y_of(h):
        return temp[np.arange(n) + h]

    sa = standalone(reps, y_of, tr, te, folds, horizons)

    print("\n[null] marginal value over magnitude653 (real vs random-feature null):", flush=True)
    nulls = {}
    for blk in ("multimodal_18", "multimodal_rich"):
        nulls[blk] = null_test(reps["magnitude653"], reps[blk], y_of, tr, te, horizons, rng, args.n_null)
        sig = "  ".join(f"h{h}={'YES' if nulls[blk][f'h{h}']['significant'] else 'no'}(Δ={nulls[blk][f'h{h}']['delta_real']:+.4f})" for h in horizons)
        print(f"  {blk:<16} {sig}", flush=True)

    print("\n[dim-matched] PCA to matched widths:", flush=True)
    dim = {}
    for k in (18, 50):
        dim[k] = {}
        for name, Xr in reps.items():
            per = {}
            for h in horizons:
                y = y_of(h)
                if Xr.shape[1] <= k:
                    A, B = Xr[tr], Xr[te]
                else:
                    pca = PCA(n_components=k, random_state=args.seed).fit(Xr[tr])
                    A, B = pca.transform(Xr[tr]), pca.transform(Xr[te])
                per[f"h{h}"] = ridge_r2(A, y[tr], B, y[te])
            dim[k][name] = per
            row = "  ".join(f"h{h}={per[f'h{h}']:.3f}" for h in horizons)
            print(f"  k={k:>2} {name:<18} {row}", flush=True)

    result = {"generated_utc": datetime.now(UTC).isoformat(), "trace": args.trace,
              "rich_width": int(X_rich.shape[1]), "horizons": horizons,
              "standalone": sa, "null_tests": nulls, "dim_matched": dim}
    (out / "phase1_rich.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nDone. Wrote {out / 'phase1_rich.json'}")


if __name__ == "__main__":
    main()
