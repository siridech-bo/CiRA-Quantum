"""QRC multi-modal feature extraction (plan §9).

Paper 4 reads out 653 features from the FID spectrum. The plan's novel
contribution is *multi-modal* features — combining observables, spectral,
time-domain, time-frequency, and nonlinear-dynamics descriptors — to
enrich the linear readout without growing the quantum system.

Inputs are the ``V`` density matrices sampled within one input step
(the temporal-multiplexing nodes). From them we build:

* **observables**   ⟨σ_a⟩ per axis per qubit per node (the standard QRC
  readout; always available),
* a simulated **FID** — the complex transverse magnetization
  ``M⁺(t) = Σ_i ⟨σx_i⟩(t) + i⟨σy_i⟩(t)`` sampled over the V nodes — from
  which the richer modalities are derived,
* **spectral** features (FFT magnitude peaks of the FID),
* **time-domain** statistics (moments, energy, zero-crossings),
* **wavelet** features (needs ``PyWavelets``; skipped if absent),
* **nonlinear** features (sample/permutation entropy; needs ``antropy``;
  skipped if absent).

Optional modalities degrade gracefully: if a library isn't installed the
feature block is silently omitted and its names don't appear, so the
design matrix stays consistent within a run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass
class FeatureConfig:
    """Which feature modalities to extract (plan §9.7: start small)."""

    # Readout axes. Default is all three, and that default matters: the
    # NMR Hamiltonian (chemical-shift σz + J-coupling σzσz) *commutes with
    # every σz*, so ⟨σz⟩ is frozen under coherent evolution and only moves
    # via T1 (~5 s ≫ τ). The reservoir's computation lives in the
    # transverse components σx/σy — which is exactly what NMR physically
    # reads out as the FID. σz-only readout gives a broken memory curve
    # (corr²(0)≈0.45); adding σx,σy restores it (corr²(0)≈0.92).
    observables: tuple[str, ...] = ("x", "y", "z")   # subset of ("x","y","z")
    spectral: bool = False
    spectral_peaks: int = 8
    time_domain: bool = False
    wavelet: bool = False
    wavelet_scales: int = 4
    nonlinear: bool = False
    # -- Paper-4 FID readout (plan §0 / §3) ------------------------------
    # ``"observables"`` keeps the standard ⟨σ_a⟩ multiplexed readout;
    # ``"fid"`` switches the reservoir loop (Coder A, evolution.py) to
    # feed each step's FID signal through :meth:`FeatureExtractor.from_fid`.
    readout: Literal["observables", "fid"] = "observables"
    # Number of fixed spectral-peak bins read from the FID magnitude
    # spectrum (Paper 4 uses 653). The peak *positions* are set once (on
    # the first call) and cached so the design matrix stays consistent.
    n_peaks: int = 653
    # When True, append the real and imaginary parts at the cached bins to
    # the magnitude features (richer, but larger). Default magnitude-only
    # for the faithful reproduction.
    fid_complex: bool = False


class FeatureExtractor:
    def __init__(self, qrc_system, cfg: FeatureConfig) -> None:
        self.sys = qrc_system
        self.qt = qrc_system.qt
        self.cfg = cfg
        self.n = qrc_system.n
        # Cached FID peak-bin positions (set on the first ``from_fid`` call).
        self._peak_bins: np.ndarray | None = None

    @property
    def is_observables_only(self) -> bool:
        """True when the standard ⟨σ_a⟩ readout with no FID-derived
        modality is requested, so the fast vectorized-observable path
        (:meth:`QRCSystem.multiplex_observables`) can be used instead of
        building Qobj states. FID readout (``readout=="fid"``) or any
        multimodal toggle takes the slow state-building path."""
        c = self.cfg
        return c.readout == "observables" and not (
            c.spectral or c.time_domain or c.wavelet or c.nonlinear
        )

    def observable_names(self) -> list[str]:
        """Feature names for the observables-only fast path (matches the
        order produced by :meth:`QRCSystem.multiplex_observables`)."""
        return [
            f"obs_{a}_q{q}_v{v}"
            for v in range(self.sys.sim.n_virtual)
            for a in self.cfg.observables
            for q in range(self.n)
        ]

    # -- FID construction ------------------------------------------------

    def _fid(self, states) -> np.ndarray:
        """Complex transverse magnetization over the V sampled nodes."""
        mx = np.array(
            [[float(self.qt.expect(op, r)) for op in self.sys.sx] for r in states]
        )
        my = np.array(
            [[float(self.qt.expect(op, r)) for op in self.sys.sy] for r in states]
        )
        return mx.sum(axis=1) + 1j * my.sum(axis=1)   # length V, complex

    # -- modality blocks -------------------------------------------------

    def _observables(self, states):
        vals = self.sys.expectations(states, which=self.cfg.observables)
        names = [
            f"obs_{a}_q{q}_v{v}"
            for v in range(len(states))
            for a in self.cfg.observables
            for q in range(self.n)
        ]
        return vals, names

    def _spectral(self, fid):
        # The FID is a *complex* quadrature signal, so use the full ``fft``
        # (``rfft`` is real-input only and raises a TypeError on complex);
        # this matches :meth:`from_fid`. Peak selection over the magnitude
        # is unchanged.
        spec = np.abs(np.fft.fft(fid))
        k = min(self.cfg.spectral_peaks, spec.size)
        idx = np.argsort(spec)[::-1][:k]
        idx_sorted = np.sort(idx)
        vals = spec[idx_sorted]
        names = [f"spec_peak{i}" for i in range(k)]
        # pad if the FID is shorter than requested peak count
        if vals.size < self.cfg.spectral_peaks:
            pad = self.cfg.spectral_peaks - vals.size
            vals = np.concatenate([vals, np.zeros(pad)])
            names += [f"spec_peak{k + i}" for i in range(pad)]
        return vals, names

    def _time_domain(self, fid):
        x = np.real(fid)
        mean = np.mean(x)
        var = np.var(x)
        std = np.sqrt(var) if var > 0 else 1.0
        skew = np.mean(((x - mean) / std) ** 3)
        kurt = np.mean(((x - mean) / std) ** 4)
        energy = float(np.sum(np.abs(fid) ** 2))
        zcr = float(np.mean(np.abs(np.diff(np.sign(x))) > 0))
        vals = np.array([mean, var, skew, kurt, energy, zcr])
        names = ["td_mean", "td_var", "td_skew", "td_kurt", "td_energy", "td_zcr"]
        return vals, names

    def _wavelet(self, fid):
        try:
            import pywt
        except ImportError:
            return np.empty(0), []
        x = np.real(fid)
        wavelet = "db1"
        max_level = pywt.dwt_max_level(len(x), pywt.Wavelet(wavelet).dec_len)
        level = max(1, min(self.cfg.wavelet_scales, max_level))
        coeffs = pywt.wavedec(x, wavelet, level=level)
        vals, names = [], []
        for i, c in enumerate(coeffs):
            vals.append(float(np.mean(np.abs(c))))
            vals.append(float(np.std(c)))
            names += [f"wav_L{i}_meanabs", f"wav_L{i}_std"]
        return np.asarray(vals), names

    def _nonlinear(self, fid):
        try:
            import antropy as ant
        except ImportError:
            return np.empty(0), []
        x = np.real(fid).astype(float)
        vals, names = [], []
        try:
            vals.append(float(ant.sample_entropy(x)))
            names.append("nl_sampen")
        except Exception:
            pass
        try:
            vals.append(float(ant.perm_entropy(x, normalize=True)))
            names.append("nl_permen")
        except Exception:
            pass
        arr = np.asarray(vals, dtype=float)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr, names

    # -- FID readout (plan §3, Paper-4-style 653-peak spectrum) ----------

    def _select_peak_bins(self, mag: np.ndarray) -> None:
        """Choose and cache the ``n_peaks`` largest-magnitude bin indices.

        Called once (on the first :meth:`from_fid`). The peak *positions*
        are fixed by the (time-invariant) Hamiltonian, so caching them
        keeps the design matrix consistent across every reservoir step.
        Indices are stored sorted (ascending frequency) for a stable,
        deterministic feature order.
        """
        k = min(self.cfg.n_peaks, mag.size)
        idx = np.argsort(mag)[::-1][:k]
        self._peak_bins = np.sort(idx)

    def _gather_bins(self, arr: np.ndarray, prefix: str) -> tuple[np.ndarray, list[str]]:
        """Sample ``arr`` at the cached bins, padding to ``n_peaks`` with
        zeros. Out-of-range bins (shorter spectrum than at cache time)
        also read as zero, so the vector length is always ``n_peaks``."""
        bins = self._peak_bins
        n_peaks = self.cfg.n_peaks
        vals = np.zeros(n_peaks, dtype=float)
        if bins is not None and bins.size:
            mask = bins < arr.size
            vals[: bins.size][mask] = arr[bins[mask]]
        names = [f"fid_{prefix}_p{i}" for i in range(n_peaks)]
        return vals, names

    def from_fid(self, fid: np.ndarray) -> tuple[np.ndarray, list[str]]:
        """Feature vector + names from one step's complex FID (plan §3).

        rfft → magnitude spectrum; feature = magnitude at the cached peak
        bins (selected/cached on the first call). With ``fid_complex`` the
        real and imaginary parts at those bins are appended. When the
        ``time_domain``/``wavelet``/``nonlinear`` toggles are on, those
        multimodal descriptors are computed on the FID time series and
        appended (Tier B). Deterministic, stable-length, stable-named.
        """
        fid = np.asarray(fid)
        # Deviation from plan §3's literal ``rfft``: the FID is a *complex*
        # quadrature NMR signal (§0 contract, complex128), whose two-sided
        # spectrum is not conjugate-symmetric, so ``rfft`` (real-input only,
        # and a TypeError on complex in current NumPy) would be wrong. The
        # full ``fft`` is the correct transform; peak selection over its
        # magnitude is unchanged. For a real input the two agree up to the
        # mirrored half.
        spec = np.fft.fft(fid)
        mag = np.abs(spec)
        if self._peak_bins is None:
            self._select_peak_bins(mag)

        vals, names = self._gather_bins(mag, "mag")
        vecs, nms = [vals], list(names)

        if self.cfg.fid_complex:
            rv, rn = self._gather_bins(np.real(spec), "re")
            iv, inm = self._gather_bins(np.imag(spec), "im")
            vecs += [rv, iv]
            nms += rn + inm

        def add(vec, nm):
            if vec.size:
                vecs.append(vec)
                nms.extend(nm)

        if self.cfg.time_domain:
            add(*self._time_domain(fid))
        if self.cfg.wavelet:
            add(*self._wavelet(fid))
        if self.cfg.nonlinear:
            add(*self._nonlinear(fid))

        return np.concatenate(vecs), nms

    # -- public ----------------------------------------------------------

    def extract(self, states):
        """Return ``(feature_vector, feature_names)`` for one input step."""
        vals, names = self._observables(states)
        vecs, nms = [vals], list(names)

        need_fid = (
            self.cfg.spectral or self.cfg.time_domain
            or self.cfg.wavelet or self.cfg.nonlinear
        )
        fid = self._fid(states) if need_fid else None

        def add(vec, nm):
            if vec.size:
                vecs.append(vec)
                nms.extend(nm)

        if self.cfg.spectral:
            add(*self._spectral(fid))
        if self.cfg.time_domain:
            add(*self._time_domain(fid))
        if self.cfg.wavelet:
            add(*self._wavelet(fid))
        if self.cfg.nonlinear:
            add(*self._nonlinear(fid))

        return np.concatenate(vecs), nms
