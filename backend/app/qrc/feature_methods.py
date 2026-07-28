"""Offline QRC feature builders + reducers (Stage A, Phase 1).

These are the *offline* counterparts to the online reservoir readout in
:mod:`app.qrc.features`: instead of building features step-by-step from live
QuTiP states, they operate on a **cached raw-FID trace** — the
``fids: complex[n_steps, fid_points]`` array written by Coder A's trace cache
(``QRC_StageA_Contracts.md`` §1). This is what makes Phase-1 experiments cheap:
re-extract features from the saved FID without re-evolving the reservoir.

Two public entry points, per the §2 builder/reducer contract:

* :func:`build_features` — turn the FID trace into a design matrix ``X`` with
  stable feature names, for one of three methods:

  - ``"magnitude653"`` — the Paper-4 baseline. FFT each FID row, take the
    magnitude spectrum, and read the magnitude at a fixed set of ``n_peaks``
    (default 653) bins. **Reproduces** the :meth:`FeatureExtractor.from_fid`
    magnitude path so the baseline matches prior (v2) results.
  - ``"phase"`` — baseline magnitude **plus** the real and imaginary FFT
    coefficients at the *same* fixed bins (Experiment 1.1, ``fid_complex``).
  - ``"multimodal"`` — the phase set **plus** time-domain, wavelet and
    nonlinear-entropy descriptors of each FID (Experiment 1.2). Reuses the
    :class:`FeatureExtractor` modality helpers verbatim; optional libraries
    (PyWavelets / antropy) degrade gracefully.

* :func:`reduce_features` — apply a dimensionality reduction / feature
  selection method (``none``/``pca``/``kpca``/``umap``/``lasso``/``mi``/
  ``random``) fit on the train split and applied to both splits
  (Experiment 1.3 / Phase 4).

**Bin-selection note (see report / §5 gate).** Online, ``from_fid`` fixes the
peak bins on the *first* reservoir step and caches them. To reproduce the v2
baseline, :func:`build_features` defaults to the same rule (``select="first"``),
which is **bit-exact** to the online FID readout on the same config (QA gate:
``max|Δ|=0`` vs ``Reservoir.run``). The alternative ``select="mean"`` ranks bins
by the mean magnitude spectrum across steps; although the NMR Hamiltonian's
dominant resonances are broadly step-stable, this is only an approximation and
was observed to diverge (up to ~0.76 in weather R² on the QA config), so it must
not be used for the reproduction baseline.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from app.qrc.features import FeatureConfig, FeatureExtractor

# --------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------


def _select_bins(mag: np.ndarray, n_peaks: int, select: str) -> np.ndarray:
    """Return the fixed peak-bin indices (sorted ascending, deterministic).

    ``mag`` is the per-step magnitude spectrum ``[n_steps, fid_points]``.
    ``select`` picks the ranking statistic:

    * ``"mean"`` — top ``n_peaks`` bins by mean magnitude across steps
      (the offline default; robust when all steps are available).
    * ``"first"`` — top ``n_peaks`` bins of the *first* step's spectrum,
      exactly mirroring :meth:`FeatureExtractor.from_fid`'s cached selection
      for a bit-exact v2 reproduction.

    Sorting the chosen indices ascending gives a stable, frequency-ordered
    feature layout identical to :meth:`FeatureExtractor._select_peak_bins`.
    """
    if select == "mean":
        rank = mag.mean(axis=0)
    elif select == "first":
        rank = mag[0]
    else:
        raise ValueError(f"unknown select={select!r}; use 'mean' or 'first'")
    k = min(n_peaks, rank.size)
    idx = np.argsort(rank)[::-1][:k]
    return np.sort(idx)


def _gather(source: np.ndarray, bins: np.ndarray, n_peaks: int,
            prefix: str) -> tuple[np.ndarray, list[str]]:
    """Sample ``source[n_steps, fid_points]`` at ``bins``, zero-padded to
    ``n_peaks`` columns — the vectorized analog of
    :meth:`FeatureExtractor._gather_bins` (same layout and ``fid_<prefix>_p<i>``
    names, so the offline matrix lines up with the online readout)."""
    n_steps = source.shape[0]
    out = np.zeros((n_steps, n_peaks), dtype=float)
    out[:, : bins.size] = source[:, bins]
    names = [f"fid_{prefix}_p{i}" for i in range(n_peaks)]
    return out, names


# A stub-backed extractor lets us reuse the FeatureExtractor modality helpers
# (_time_domain/_wavelet/_nonlinear) offline. Those helpers are pure functions
# of a single FID (plus wavelet_scales); they never touch the quantum system,
# so a namespace stub for the ``.qt``/``.n``/``.sim`` attributes is enough.
def _modality_extractor(wavelet_scales: int) -> FeatureExtractor:
    stub = SimpleNamespace(qt=None, n=0, sim=SimpleNamespace(n_virtual=0))
    cfg = FeatureConfig(wavelet_scales=wavelet_scales)
    return FeatureExtractor(stub, cfg)


def _multimodal_blocks(
    fids: np.ndarray,
    *,
    time_domain: bool,
    wavelet: bool,
    nonlinear: bool,
    wavelet_scales: int,
) -> tuple[list[np.ndarray], list[str]]:
    """Per-step time-domain / wavelet / nonlinear descriptors of each FID.

    Reuses :class:`FeatureExtractor`'s modality helpers per row and stacks the
    results. Block widths depend only on ``fid_points`` (constant across steps)
    and ``wavelet_scales``, so the output has a stable, deterministic length and
    stable names. Optional libs (PyWavelets/antropy) missing → that block is
    silently empty (helper returns width 0), consistent within the run."""
    ext = _modality_extractor(wavelet_scales)
    parts: list[tuple[bool, str]] = [
        (time_domain, "_time_domain"),
        (wavelet, "_wavelet"),
        (nonlinear, "_nonlinear"),
    ]
    per_step: list[list[np.ndarray]] = []
    names: list[str] | None = None
    for row in fids:
        fid = np.asarray(row)
        vecs: list[np.ndarray] = []
        nms: list[str] = []
        for enabled, meth in parts:
            if not enabled:
                continue
            vals, nm = getattr(ext, meth)(fid)
            if vals.size:
                vecs.append(np.asarray(vals, dtype=float))
                nms.extend(nm)
        per_step.append(vecs)
        if names is None:
            names = nms
    if not names:
        return [], []
    # Concatenate each step's blocks, then stack into [n_steps, n_multimodal].
    rows = [np.concatenate(vecs) if vecs else np.empty(0) for vecs in per_step]
    mat = np.vstack(rows)
    return [mat], names


def build_features(
    fids: np.ndarray,
    method: str,
    *,
    n_peaks: int = 653,
    select: str = "first",
    fid_complex: bool | None = None,
    time_domain: bool = True,
    wavelet: bool = True,
    nonlinear: bool = True,
    wavelet_scales: int = 4,
    **_ignored,
) -> tuple[np.ndarray, list[str]]:
    """Build a design matrix from a cached raw-FID trace (§2 builder contract).

    Parameters
    ----------
    fids
        Complex FID trace, shape ``[n_steps, fid_points]`` (index 0 = t=0),
        as stored under the ``fids`` key of Coder A's trace ``.npz`` (§1).
    method
        ``"magnitude653"``, ``"phase"``, or ``"multimodal"``.
    n_peaks
        Number of fixed spectral bins (Paper-4 default 653).
    select
        Bin-selection statistic, ``"first"`` (default; bit-exact v2/online
        reproduction) or ``"mean"`` (approximation — see module docstring).
    fid_complex
        Whether to append real+imag coefficients. ``None`` (default) resolves
        per method: ``False`` for magnitude653, ``True`` for phase/multimodal.
    time_domain, wavelet, nonlinear, wavelet_scales
        Multimodal toggles (only used by ``method="multimodal"``).

    Returns
    -------
    (X, names)
        ``X`` has shape ``[n_steps, n_feat]``; ``names`` is the matching
        feature-name list. Deterministic and stable-length across calls.
    """
    fids = np.asarray(fids)
    if fids.ndim != 2:
        raise ValueError(f"fids must be 2-D [n_steps, fid_points], got {fids.shape}")

    spec = np.fft.fft(fids, axis=1)   # full complex FFT, matches from_fid
    mag = np.abs(spec)
    bins = _select_bins(mag, n_peaks, select)

    blocks: list[np.ndarray] = []
    names: list[str] = []

    mblock, mnames = _gather(mag, bins, n_peaks, "mag")
    blocks.append(mblock)
    names.extend(mnames)

    if method == "magnitude653":
        want_complex = False if fid_complex is None else fid_complex
        want_multimodal = False
    elif method == "phase":
        want_complex = True if fid_complex is None else fid_complex
        want_multimodal = False
    elif method == "multimodal":
        want_complex = True if fid_complex is None else fid_complex
        want_multimodal = True
    else:
        raise ValueError(
            f"unknown method={method!r}; use 'magnitude653', 'phase', or 'multimodal'"
        )

    if want_complex:
        reblock, renames = _gather(spec.real, bins, n_peaks, "re")
        imblock, imnames = _gather(spec.imag, bins, n_peaks, "im")
        blocks += [reblock, imblock]
        names += renames + imnames

    if want_multimodal:
        mm_blocks, mm_names = _multimodal_blocks(
            fids,
            time_domain=time_domain,
            wavelet=wavelet,
            nonlinear=nonlinear,
            wavelet_scales=wavelet_scales,
        )
        blocks += mm_blocks
        names += mm_names

    X = np.concatenate(blocks, axis=1)
    return X, names


# --------------------------------------------------------------------------
# Reducers
# --------------------------------------------------------------------------


def _as_1d_target(y: np.ndarray) -> np.ndarray:
    """Collapse a target array to 1-D for the supervised reducers.

    LASSO / mutual-information selection want a single response. Weather traces
    carry a 2-column target (temp, humidity); we select on the **primary**
    target (column 0 = temperature, which the key R²@h=30 figure reports)."""
    y = np.asarray(y)
    if y.ndim == 1:
        return y
    if y.shape[1] == 1:
        return y.ravel()
    return y[:, 0]


def reduce_features(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    method: str,
    k: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce/select features, fit on train, apply to train+test (§2 reducer).

    Methods (``k`` = target #components / #selected features):

    * ``"none"``   — identity (``k`` ignored).
    * ``"pca"``    — PCA linear projection to ``k`` components.
    * ``"kpca"``   — RBF Kernel-PCA projection to ``k`` components.
    * ``"umap"``   — UMAP nonlinear projection to ``k`` (needs ``umap-learn``;
      import guarded — raises a clear error if the ``[featurelab]`` extra is
      not installed).
    * ``"lasso"``  — LassoCV embedded selection (nonzero-coef columns; ``k``
      ignored — the CV picks the sparsity).
    * ``"mi"``     — mutual-information filter, top-``k`` columns.
    * ``"random"`` — Gaussian random projection to ``k`` components (baseline).

    Returns the transformed ``(Xtr2, Xte2)``.
    """
    Xtr = np.asarray(Xtr, dtype=float)
    Xte = np.asarray(Xte, dtype=float)

    if method == "none":
        return Xtr, Xte

    n_feat = Xtr.shape[1]
    n_samp = Xtr.shape[0]

    def _need_k() -> int:
        if k is None:
            raise ValueError(f"method={method!r} requires an integer k")
        return int(k)

    if method == "pca":
        from sklearn.decomposition import PCA

        n = min(_need_k(), n_feat, n_samp)
        model = PCA(n_components=n, random_state=0)
        return model.fit_transform(Xtr), model.transform(Xte)

    if method == "kpca":
        from sklearn.decomposition import KernelPCA

        n = min(_need_k(), n_feat, n_samp)
        model = KernelPCA(n_components=n, kernel="rbf", random_state=0)
        return model.fit_transform(Xtr), model.transform(Xte)

    if method == "random":
        from sklearn.random_projection import GaussianRandomProjection

        n = min(_need_k(), n_feat)
        model = GaussianRandomProjection(n_components=n, random_state=0)
        return model.fit_transform(Xtr), model.transform(Xte)

    if method == "umap":
        try:
            import umap
        except ImportError as exc:   # pragma: no cover - env-dependent
            raise ImportError(
                "reduce_features(method='umap') needs umap-learn; "
                "install the extra:  pip install '.[featurelab]'"
            ) from exc
        n = min(_need_k(), n_feat)
        model = umap.UMAP(n_components=n, random_state=42)
        Xtr2 = model.fit_transform(Xtr)
        return np.asarray(Xtr2), np.asarray(model.transform(Xte))

    if method == "lasso":
        from sklearn.linear_model import LassoCV

        y1 = _as_1d_target(ytr)
        lasso = LassoCV(cv=min(10, max(2, n_samp // 3)), max_iter=10000,
                        random_state=0).fit(Xtr, y1)
        sel = np.where(np.abs(lasso.coef_) > 0)[0]
        if sel.size == 0:   # degenerate: keep the strongest single feature
            sel = np.array([int(np.argmax(np.abs(lasso.coef_)))])
        return Xtr[:, sel], Xte[:, sel]

    if method == "mi":
        from sklearn.feature_selection import mutual_info_regression

        y1 = _as_1d_target(ytr)
        n = min(_need_k(), n_feat)
        mi = mutual_info_regression(Xtr, y1, random_state=42)
        top = np.sort(np.argsort(mi)[::-1][:n])
        return Xtr[:, top], Xte[:, top]

    raise ValueError(
        f"unknown method={method!r}; use one of "
        "none/pca/kpca/umap/lasso/mi/random"
    )
