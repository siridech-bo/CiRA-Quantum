"""Independent validation of our QuTiP NMR simulator against SLEEPY.

SLEEPY (pip ``sleepy-nmr``, import name ``SLEEPY``) is an independent,
NMR-specialised Liouville-space engine (Nature Comms 2025, CPU-only). This
module reproduces a small homonuclear two-proton subsystem in *both* SLEEPY
and our QuTiP Lindblad simulator, then cross-checks that the free-induction
decay (FID) — peak positions and relaxation lineshape — agree. A matching
result is independent evidence that our QuTiP FID/relaxation physics is
correct (plan ``docs/QRC/QRC_Reproduction_Plan.md`` §5).

Verified SLEEPY API (checked against ``sleepy-nmr`` 1.1.2 — both the source
under ``SLEEPY/`` and a live two-proton run — not guessed):

* ``ex = sl.ExpSys(v0H=<MHz>, Nucs=['1H', '1H'], pwdavg=0)`` — proton Larmor
  in MHz, homonuclear pair. ``pwdavg=0`` picks a single orientation; for the
  purely isotropic interactions used here (CS + J) the powder average is
  redundant, and skipping it keeps the run fast.
* ``ex.set_inter('CS', i=<int>, Hz=<float>)`` — chemical shift as a
  rotating-frame offset in **Hz** (``HamTypes.CS``: ``H = sign*Hz*S.z``).
  With ``Defaults['Hz_gyro_sign_depend']=True`` (the default) and ¹H's
  positive gyromagnetic ratio the sign is ``+``, so a ``+Hz`` offset yields a
  spectral peak at ``+Hz`` — the same convention as our ``π ν σz`` term.
* ``ex.set_inter('J', i0=<int>, i1=<int>, J=<Hz>)`` — J-coupling. **Deviation
  to note:** ``HamTypes.J`` uses the *full isotropic* coupling
  ``J*(Sx·Ix + Sy·Iy + Sz·Iz)`` for a **homonuclear** pair, whereas our
  QuTiP Hamiltonian uses the weak-coupling Ising form ``(π/2) J σz σz``.
  These agree in the AX limit ``|Δν| >> J`` (here ``|985.9-520.3| = 465.6 Hz
  >> 15.8 Hz``): the strong-coupling correction to the line positions is
  ``O(J²/2Δν) ≈ 0.3 Hz``, comfortably inside the < 1 Hz tolerance, and the
  plan (§5) chooses well-separated protons precisely so the Ising H is exact
  there.
* ``L = sl.Liouvillian(ex)`` then per spin
  ``L.add_relax('T1', i=, T1=)`` and ``L.add_relax('T2', i=, T2=)``, finally
  ``L.add_relax('recovery')``. In ``RelaxMat`` the T1 matrix has its
  transverse part explicitly removed (only population transfer survives), and
  the T2 matrix damps coherences at exactly ``1/T2``. So SLEEPY's transverse
  FID envelope is ``exp(-t/T2)``. Our QuTiP ``dephasing_model='physical'``
  (``γ_φ = 1/T2 − 1/(2 T1)`` on top of the ``1/(2 T1)`` from the amplitude-
  damping σ₋) gives the same *total* transverse rate ``1/T2`` — matched.
* ``seq = L.Sequence(Dt=fid_dwell)``; ``rho = sl.Rho('1Hx', '1Hp')`` starts
  the protons transverse (σx, i.e. immediately after a 90° pulse) and detects
  σ₊; ``rho.DetProp(seq, n=fid_points)`` acquires the FID. ``DetProp``
  detects *then* propagates, so sample 0 is at ``t=0`` — matching our
  ``fid_signal`` contract (index 0 = t=0).
* Complex signal accessor: ``np.asarray(rho.I[0])`` → shape ``(fid_points,)``,
  dtype ``complex128`` (``Rho.I`` returns an ``Nd × Nt`` powder-weighted
  detection matrix; row 0 is our single σ₊ detector).

Our QuTiP side is built to the interface contract in the plan (§0/§2):
``QRCSystem.fid_signal(rho) -> np.ndarray`` and the ``SimConfig.fid_points /
fid_dwell`` fields (owned by Coder A). SLEEPY starts from a transverse state,
while ``fid_signal`` applies its own π/2 readout pulse, so we hand it a
σz-polarised initial state (the reservoir's thermal-like ``rho0``); the pulse
then rotates it into the transverse plane. The residual x-vs-y phase between
the two conventions is a global phase, removed by the normalisation in
:func:`compare`.

Install the optional dependency with ``pip install ".[validation]"``.
"""

from __future__ import annotations

import numpy as np

# Default two-proton subsystem: H1/H2 of the paper-4 crotonic-acid molecule
# (plan §1), a well-separated homonuclear pair with a single J-coupling.
DEFAULT_NU = (985.9, 520.3)     # Hz, rotating-frame offsets
DEFAULT_J = 15.8                # Hz, H1-H2 scalar coupling
DEFAULT_T1 = (3.2, 3.4)         # s
DEFAULT_T2 = (0.203, 0.332)     # s
DEFAULT_V0H = 400.2118          # MHz, ¹H Larmor
DEFAULT_FID_POINTS = 2048
DEFAULT_FID_DWELL = 3e-4        # s (Nyquist 1667 Hz > proton offsets)


def _require_sleepy():
    """Import SLEEPY lazily with a helpful error if the extra is missing."""
    try:
        import SLEEPY  # noqa: F401
    except ImportError as exc:  # pragma: no cover - env dependent
        raise ImportError(
            "SLEEPY validation needs the 'sleepy-nmr' package. Install it "
            "with the optional extra:\n"
            '    pip install ".[validation]"\n'
            "(import name is 'SLEEPY'; the extra also pulls IPython, which "
            "sleepy-nmr imports at load time but does not declare)."
        ) from exc
    return SLEEPY


def build_sleepy_two_proton(
    nu: tuple[float, float] = DEFAULT_NU,
    J: float = DEFAULT_J,
    T1: tuple[float, float] = DEFAULT_T1,
    T2: tuple[float, float] = DEFAULT_T2,
    v0H: float = DEFAULT_V0H,
    fid_points: int = DEFAULT_FID_POINTS,
    fid_dwell: float = DEFAULT_FID_DWELL,
) -> np.ndarray:
    """Acquire the two-proton FID from SLEEPY (independent reference).

    Returns a complex ``(fid_points,)`` array, index 0 = t=0.
    """
    sl = _require_sleepy()

    ex = sl.ExpSys(v0H=v0H, Nucs=["1H", "1H"], pwdavg=0)
    ex.set_inter("CS", i=0, Hz=nu[0])
    ex.set_inter("CS", i=1, Hz=nu[1])
    ex.set_inter("J", i0=0, i1=1, J=J)

    liouv = sl.Liouvillian(ex)
    liouv.add_relax("T1", i=0, T1=T1[0])
    liouv.add_relax("T2", i=0, T2=T2[0])
    liouv.add_relax("T1", i=1, T1=T1[1])
    liouv.add_relax("T2", i=1, T2=T2[1])
    liouv.add_relax("recovery")

    seq = liouv.Sequence(Dt=fid_dwell)
    rho = sl.Rho("1Hx", "1Hp")     # transverse start (post-90°), detect σ₊
    rho.DetProp(seq, n=fid_points)

    fid = np.asarray(rho.I[0], dtype=complex)
    return fid


def build_qutip_two_proton(
    nu: tuple[float, float] = DEFAULT_NU,
    J: float = DEFAULT_J,
    T1: tuple[float, float] = DEFAULT_T1,
    T2: tuple[float, float] = DEFAULT_T2,
    v0H: float = DEFAULT_V0H,
    fid_points: int = DEFAULT_FID_POINTS,
    fid_dwell: float = DEFAULT_FID_DWELL,
) -> np.ndarray:
    """Acquire the same two-proton FID from our QuTiP simulator.

    Builds a 2-qubit :class:`~app.qrc.config.SystemConfig` (the two proton
    chemical shifts, the single J-coupling, and the per-spin T1/T2), a
    :class:`~app.qrc.system.QRCSystem` with matching ``fid_points/fid_dwell``
    and the exact sparse ``action`` evolution mode, and returns
    ``fid_signal(rho0)``. ``rho0`` is the thermal-like σz-polarised initial
    state; ``fid_signal`` applies the π/2 readout pulse, rotating it into the
    transverse plane to match SLEEPY's ``'1Hx'`` start.

    ``v0H`` is accepted for a symmetric signature; the rotating-frame QuTiP
    dynamics do not depend on the absolute Larmor frequency.
    """
    del v0H  # rotating-frame dynamics are Larmor-independent
    from app.qrc.config import SimConfig, SystemConfig
    from app.qrc.system import QRCSystem

    j_matrix = [[0.0, float(J)], [float(J), 0.0]]
    system = SystemConfig(
        n_qubits=2,
        chemical_shifts=[float(nu[0]), float(nu[1])],
        j_coupling=j_matrix,
        t1=[float(T1[0]), float(T1[1])],
        t2=[float(T2[0]), float(T2[1])],
        labels=["1H", "1H"],
    )
    sim = SimConfig(
        fid_points=fid_points,
        fid_dwell=fid_dwell,
        evolution_mode="action",
        dephasing_model="physical",
    )
    qsys = QRCSystem(system, sim)
    fid = np.asarray(qsys.fid_signal(qsys.rho0), dtype=complex)
    return fid


# ---------------------------------------------------------------------------
# Spectral / trajectory comparison helpers
# ---------------------------------------------------------------------------


def _spectrum(fid: np.ndarray, dwell: float) -> tuple[np.ndarray, np.ndarray]:
    """Full complex spectrum of a complex FID.

    The FID is complex (quadrature σ₊ detection), so a full FFT — not
    ``rfft`` — is required to place peaks at their true signed frequencies;
    we fftshift so the frequency axis runs low→high symmetrically about 0.
    """
    spec = np.fft.fftshift(np.fft.fft(fid))
    freq = np.fft.fftshift(np.fft.fftfreq(len(fid), d=dwell))
    return freq, np.abs(spec)


def _find_peaks(
    freq: np.ndarray, mag: np.ndarray, rel_height: float = 0.05
) -> np.ndarray:
    """Sub-bin peak frequencies above ``rel_height`` × max magnitude.

    Each local maximum is refined by parabolic (3-point) interpolation, so a
    reported position is the true line centre rather than the nearest FFT bin.
    Without this a genuinely sub-Hz peak difference can straddle a bin
    boundary and read as a full ``1/(N·dwell)`` Hz apart.
    """
    if mag.max() <= 0:
        return np.asarray([], dtype=float)
    thr = mag.max() * rel_height
    binw = float(freq[1] - freq[0])
    peaks = []
    for i in range(1, len(mag) - 1):
        if mag[i] >= thr and mag[i] >= mag[i - 1] and mag[i] > mag[i + 1]:
            y0, y1, y2 = mag[i - 1], mag[i], mag[i + 1]
            denom = y0 - 2.0 * y1 + y2
            delta = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
            peaks.append(freq[i] + delta * binw)
    return np.sort(np.asarray(peaks, dtype=float))


def _max_peak_diff(peaks_a: np.ndarray, peaks_b: np.ndarray) -> float:
    """Max over each peak in ``peaks_a`` of its nearest distance in ``peaks_b``."""
    if len(peaks_a) == 0 or len(peaks_b) == 0:
        return float("inf")
    return float(max(np.min(np.abs(pa - peaks_b)) for pa in peaks_a))


def _normalise(fid: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(fid)
    return fid / norm if norm > 0 else fid


def _phase_align_rms(a: np.ndarray, b: np.ndarray) -> float:
    """RMS of ``|a - b'|`` where ``b'`` is ``b`` rotated to cancel the global
    phase against ``a`` (both already L2-normalised)."""
    overlap = np.vdot(a, b)
    b_aligned = b * np.exp(-1j * np.angle(overlap))
    return float(np.sqrt(np.mean(np.abs(a - b_aligned) ** 2)))


def compare_fids(
    fid_sleepy: np.ndarray,
    fid_qutip: np.ndarray,
    fid_dwell: float = DEFAULT_FID_DWELL,
) -> dict:
    """Compare two complex FIDs sampled on the same ``fid_dwell`` grid.

    Returns a metrics dict:

    * ``sleepy_peaks_hz`` / ``qutip_peaks_hz`` — detected peak frequencies (Hz);
    * ``max_peak_diff_hz`` — worst nearest-peak mismatch (Hz);
    * ``fid_rms`` — RMS difference of the L2-normalised, phase-aligned FIDs.

    Overall phase and scale are normalised away. A global frequency-sign /
    conjugation convention (which mirrors the spectrum about 0 and conjugates
    the FID) is also tolerated: both orientations are scored and the better is
    reported, so an unrelated sign convention never masquerades as a physics
    disagreement.
    """
    fs, ms = _spectrum(fid_sleepy, fid_dwell)
    peaks_s = _find_peaks(fs, ms)
    a = _normalise(fid_sleepy)

    best: dict | None = None
    # Orientation 0: as-is. Orientation 1: conjugate the QuTiP FID (mirror
    # its spectrum) — a legitimate global-convention ambiguity.
    for conj in (False, True):
        fq = np.conjugate(fid_qutip) if conj else fid_qutip
        fqf, mqf = _spectrum(fq, fid_dwell)
        peaks_q = _find_peaks(fqf, mqf)
        peak_diff = max(
            _max_peak_diff(peaks_s, peaks_q),
            _max_peak_diff(peaks_q, peaks_s),
        )
        rms = _phase_align_rms(a, _normalise(fq))
        score = peak_diff + rms
        if best is None or score < best["_score"]:
            best = {
                "_score": score,
                "qutip_peaks_hz": [float(x) for x in peaks_q],
                "max_peak_diff_hz": peak_diff,
                "fid_rms": rms,
                "qutip_conjugated": conj,
            }
    assert best is not None
    best.pop("_score")
    return {
        "sleepy_peaks_hz": [float(x) for x in peaks_s],
        "n_points": int(len(fid_sleepy)),
        "dwell_s": float(fid_dwell),
        **best,
    }


def compare(
    nu: tuple[float, float] = DEFAULT_NU,
    J: float = DEFAULT_J,
    T1: tuple[float, float] = DEFAULT_T1,
    T2: tuple[float, float] = DEFAULT_T2,
    v0H: float = DEFAULT_V0H,
    fid_points: int = DEFAULT_FID_POINTS,
    fid_dwell: float = DEFAULT_FID_DWELL,
) -> dict:
    """Build the two-proton FID in both engines and compare them.

    Returns the metrics dict from :func:`compare_fids`.
    """
    fid_sleepy = build_sleepy_two_proton(
        nu=nu, J=J, T1=T1, T2=T2, v0H=v0H,
        fid_points=fid_points, fid_dwell=fid_dwell,
    )
    fid_qutip = build_qutip_two_proton(
        nu=nu, J=J, T1=T1, T2=T2, v0H=v0H,
        fid_points=fid_points, fid_dwell=fid_dwell,
    )
    return compare_fids(fid_sleepy, fid_qutip, fid_dwell=fid_dwell)
