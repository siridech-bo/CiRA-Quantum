"""QRC Phase-1 feature experiments (offline, from a cached FID trace).

Runs Phase-1 of ``QRC_Next_Stage_Experiments.md`` end to end against a cached
raw-FID trace (Coder A's ``.npz``, schema §1) — **no reservoir re-evolution**:

* **Exp 1.1 (phase)**   — baseline ``magnitude653`` vs the ``phase`` set
  (magnitude + real + imag at the same 653 bins).
* **Exp 1.2 (multimodal)** — the ``multimodal`` set (phase + time-domain +
  wavelet + entropy descriptors).
* **Exp 1.3 (selection)** — take the multimodal set and sweep every reducer
  (``none/pca/kpca/umap/lasso/mi/random``) over a range of feature counts.

For each experiment it calls Coder A's :func:`app.qrc.feature_lab.evaluate_features`
(the Evaluation-Protocol harness, §2), writes one JSON per experiment in the
Output-Format shape, emits a summary table (JSON + console), and generates the
key Phase-1 figure: **weather R² @ h=30 vs #features, one curve per selection
method** (Exp 1.3's publication figure). For a NARMA trace the figure plots
NARMA-10 NMSE (log scale) instead.

Run from ``d:\\CiRA Quantum\\backend`` (needs the ``[qrc]`` + ``[featurelab]``
extras for the multimodal/umap paths)::

    PYTHONIOENCODING=utf-8 python scripts/qrc_phase1.py \\
        --trace artifacts/traces/weather.npz --out-dir artifacts/qrc_phase1

The trace cache (Coder A) and ``evaluate_features`` (Coder A,
``app/qrc/feature_lab.py``) are coded to the §1/§2 contracts; this runner works
once those land.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from app.qrc.feature_lab import evaluate_features
from app.qrc.feature_methods import build_features, reduce_features
from app.qrc.tasks import narma_sequence_sine

# Feature methods (Exp 1.1 / 1.2) and their experiment ids.
FEATURE_EXPERIMENTS = [
    ("magnitude653", "1.0_baseline_magnitude653"),
    ("phase", "1.1_fid_phase"),
    ("multimodal", "1.2_multimodal"),
]

# Exp 1.3 reducers. Projection/filter methods sweep over ``k``; ``none`` and
# ``lasso`` are single points (full set / CV-selected sparsity).
SWEEP_REDUCERS = ("pca", "kpca", "umap", "mi", "random")
POINT_REDUCERS = ("none", "lasso")
DEFAULT_K_SWEEP = (50, 100, 200, 400)
KEY_HORIZON = 30


# --------------------------------------------------------------------------
# Trace loading / split
# --------------------------------------------------------------------------


def load_trace(path: str) -> dict:
    """Load Coder A's trace ``.npz`` (§1) into a plain dict of arrays/scalars."""
    with np.load(path, allow_pickle=True) as z:
        npz = {k: z[k] for k in z.files}
    return npz


def _scalar(v) -> int:
    return int(np.asarray(v).reshape(-1)[0])


def split_slices(npz: dict) -> tuple[slice, slice]:
    """Train / test row slices from the ``split=[washout, n_train, n_test]`` key
    (standard washout→train→test layout, matching the reservoir loop order)."""
    washout, n_train, n_test = (int(x) for x in np.asarray(npz["split"]).ravel()[:3])
    tr = slice(washout, washout + n_train)
    te = slice(washout + n_train, washout + n_train + n_test)
    return tr, te


def reducer_target(npz: dict, tr: slice, key_horizon: int) -> np.ndarray:
    """A train-split supervised target to *guide* the task-aware reducers
    (LASSO/MI). This only ranks features; the authoritative scoring is done by
    ``evaluate_features``. Weather → temperature ``key_horizon`` days ahead;
    NARMA → the NARMA-10 target recomputed from the trace seed."""
    task = str(np.asarray(npz["task"]).reshape(-1)[0])
    if task == "weather":
        temp = np.asarray(npz["weather_norm"], dtype=float)[:, 0]
        idx = np.arange(tr.start, tr.stop) + key_horizon
        idx = np.clip(idx, 0, temp.size - 1)
        return temp[idx]
    # NARMA: reconstruct the order-10 target for the trace's step count/seed.
    n_steps = _scalar(npz["split"][:3].sum()) if hasattr(npz["split"], "sum") \
        else sum(int(x) for x in np.asarray(npz["split"]).ravel()[:3])
    seed = _scalar(npz.get("seed", 0))
    _, y = narma_sequence_sine(n_steps, order=10, seed=seed)
    return y[tr]


# --------------------------------------------------------------------------
# Experiments
# --------------------------------------------------------------------------


def _write_json(out_dir: Path, result: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{result['experiment']}.json"
    path.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    return path


def run_feature_experiments(
    npz: dict, out_dir: Path, *, n_peaks: int, select: str
) -> dict[str, dict]:
    """Exp 1.1 / 1.2: build each feature set and score it. Returns
    ``{feature_method: eval_result}`` and keeps the built matrices' widths."""
    fids = np.asarray(npz["fids"])
    results: dict[str, dict] = {}
    for method, exp_id in FEATURE_EXPERIMENTS:
        X, names = build_features(fids, method, n_peaks=n_peaks, select=select)
        result = evaluate_features(X, npz, exp_id)
        result.setdefault("feature_count", X.shape[1])
        result["experiment"] = exp_id
        result["timestamp"] = datetime.now(UTC).isoformat()
        result["feature_method"] = method
        _write_json(out_dir, result)
        print(f"[feat] {exp_id:<28} n_feat={X.shape[1]:>5} "
              f"({len(names)} names)")
        results[method] = result
    return results


def run_selection_sweep(
    npz: dict, out_dir: Path, *, n_peaks: int, select: str, k_sweep: tuple[int, ...],
    key_horizon: int,
) -> list[dict]:
    """Exp 1.3: expand to the multimodal set, then apply every reducer over the
    k-sweep. Returns one record per (method, k) with the eval result."""
    fids = np.asarray(npz["fids"])
    Xfull, _ = build_features(fids, "multimodal", n_peaks=n_peaks, select=select)
    tr, te = split_slices(npz)
    Xtr, Xte = Xfull[tr], Xfull[te]
    ytr = reducer_target(npz, tr, key_horizon)
    n_steps = Xfull.shape[0]

    records: list[dict] = []
    plans = [(m, k) for m in SWEEP_REDUCERS for k in k_sweep]
    plans += [(m, None) for m in POINT_REDUCERS]

    for method, k in plans:
        try:
            Xtr2, Xte2 = reduce_features(Xtr, ytr, Xte, method, k)
        except ImportError as exc:            # e.g. umap not installed
            print(f"[sel]  {method} k={k}: skipped ({exc})")
            continue
        except Exception as exc:              # noqa: BLE001 - one degenerate
            # reducer (e.g. UMAP's spectral init needs k < n_samples, which
            # fails on tiny traces) must not abort the whole sweep — skip it
            # and keep going so the summary + figure are still produced.
            print(f"[sel]  {method} k={k}: skipped ({type(exc).__name__}: {exc})")
            continue
        full = _assemble_full(Xtr2, Xte2, tr, te, n_steps)
        tag = f"{method}_k{k}" if k is not None else method
        exp_id = f"1.3_{tag}"
        result = evaluate_features(full, npz, exp_id)
        result.setdefault("feature_count", Xtr2.shape[1])
        result["experiment"] = exp_id
        result["timestamp"] = datetime.now(UTC).isoformat()
        result["reducer"] = method
        result["reducer_k"] = k
        _write_json(out_dir, result)
        print(f"[sel]  {exp_id:<22} n_feat={Xtr2.shape[1]:>5}")
        records.append(result)
    return records


def _assemble_full(
    Xtr2: np.ndarray, Xte2: np.ndarray, tr: slice, te: slice, n_steps: int
) -> np.ndarray:
    """Re-embed reduced train/test rows into a full ``[n_steps, width]`` matrix
    (washout rows zero-filled; ``evaluate_features`` re-splits via the npz)."""
    width = Xtr2.shape[1]
    full = np.zeros((n_steps, width), dtype=float)
    full[tr] = Xtr2
    full[te] = Xte2
    return full


# --------------------------------------------------------------------------
# Summary + figure
# --------------------------------------------------------------------------


def _metric(result: dict, task: str, key_horizon: int) -> float | None:
    """The scalar plotted / tabulated: weather R²@h=key (temp) or NARMA-10 NMSE."""
    if task == "weather":
        wr = result.get("weather_r2") or {}
        return wr.get(f"h{key_horizon}")
    return result.get("narma10_nmse")


def write_summary(
    out_dir: Path, feat_results: dict[str, dict], sel_records: list[dict],
    task: str, key_horizon: int,
) -> Path:
    rows = []
    for method, _exp in FEATURE_EXPERIMENTS:
        r = feat_results.get(method)
        if r:
            rows.append({
                "experiment": r["experiment"], "kind": "feature",
                "feature_count": r.get("feature_count"),
                "metric": _metric(r, task, key_horizon),
            })
    for r in sel_records:
        rows.append({
            "experiment": r["experiment"], "kind": "selection",
            "reducer": r.get("reducer"), "reducer_k": r.get("reducer_k"),
            "feature_count": r.get("feature_count"),
            "metric": _metric(r, task, key_horizon),
        })
    summary = {
        "generated_utc": datetime.now(UTC).isoformat(),
        "task": task,
        "metric_name": (f"weather_r2_h{key_horizon}" if task == "weather"
                        else "narma10_nmse"),
        "rows": rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")

    label = summary["metric_name"]
    print(f"\n{'experiment':<26} {'n_feat':>7}  {label}")
    print("-" * 52)
    for row in rows:
        val = row["metric"]
        val_s = f"{val:.4f}" if isinstance(val, float) else "—"
        print(f"{row['experiment']:<26} {str(row['feature_count']):>7}  {val_s}")
    return path


def make_figure(
    out_dir: Path, sel_records: list[dict], feat_results: dict[str, dict],
    task: str, key_horizon: int,
) -> Path | None:
    """Exp 1.3 key figure: metric vs #features, one curve per reducer."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[fig]  matplotlib unavailable; skipping figure")
        return None

    by_method: dict[str, list[tuple[int, float]]] = {}
    for r in sel_records:
        m = _metric(r, task, key_horizon)
        fc = r.get("feature_count")
        if m is None or fc is None:
            continue
        by_method.setdefault(r["reducer"], []).append((int(fc), float(m)))

    fig, ax = plt.subplots(figsize=(7, 4.6))
    for method, pts in sorted(by_method.items()):
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        style = "o-" if len(pts) > 1 else "o"
        ax.plot(xs, ys, style, lw=2, ms=6, label=method)

    # Full multimodal (no reduction) as a horizontal reference line.
    mm = feat_results.get("multimodal")
    if mm is not None:
        ref = _metric(mm, task, key_horizon)
        if ref is not None:
            ax.axhline(ref, ls="--", color="gray", alpha=0.7,
                       label=f"multimodal (full, {mm.get('feature_count')})")

    ax.set_xlabel("number of features")
    if task == "weather":
        ax.set_ylabel(f"weather R²  (temperature, h={key_horizon})")
        ax.set_title(f"Phase-1 feature selection: R²@h={key_horizon} vs #features")
    else:
        ax.set_ylabel("NARMA-10 NMSE (lower is better)")
        ax.set_yscale("log")
        ax.set_title("Phase-1 feature selection: NARMA-10 NMSE vs #features")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "phase1_selection.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig]  wrote {path}")
    return path


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="QRC Phase-1 feature experiments")
    ap.add_argument("--trace", required=True, help="cached FID trace .npz (§1)")
    ap.add_argument("--out-dir", default="artifacts/qrc_phase1",
                    help="output dir for per-experiment JSON + summary + figure")
    ap.add_argument("--n-peaks", type=int, default=653)
    ap.add_argument("--select", choices=["mean", "first"], default="first",
                    help="peak-bin selection: 'first' (default) caches bins on "
                         "the first step, reproducing the online FID readout "
                         "bit-exactly so the magnitude653 baseline matches v2; "
                         "'mean' ranks bins by the mean spectrum across steps "
                         "(an approximation that can diverge — verified up to "
                         "~0.76 in weather R2 on the QA config)")
    ap.add_argument("--k-sweep", type=int, nargs="+", default=list(DEFAULT_K_SWEEP),
                    help="feature counts for the Exp-1.3 reducer sweep")
    ap.add_argument("--key-horizon", type=int, default=KEY_HORIZON,
                    help="weather horizon for the key figure/metric")
    ap.add_argument("--skip-selection", action="store_true",
                    help="run only the feature-set experiments (1.1/1.2)")
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    npz = load_trace(args.trace)
    task = str(np.asarray(npz["task"]).reshape(-1)[0])
    print(f"trace: {args.trace}  task={task}  fids={np.asarray(npz['fids']).shape}")

    feat_results = run_feature_experiments(
        npz, out_dir, n_peaks=args.n_peaks, select=args.select
    )

    sel_records: list[dict] = []
    if not args.skip_selection:
        sel_records = run_selection_sweep(
            npz, out_dir, n_peaks=args.n_peaks, select=args.select,
            k_sweep=tuple(args.k_sweep), key_horizon=args.key_horizon,
        )

    write_summary(out_dir, feat_results, sel_records, task, args.key_horizon)
    if not args.no_figure and sel_records:
        make_figure(out_dir, sel_records, feat_results, task, args.key_horizon)

    print(f"\nDone. Outputs in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
