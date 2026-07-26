"""SLEEPY vs QuTiP cross-validation of the QRC NMR physics (plan §5).

Independent confirmation that our QuTiP Lindblad FID matches SLEEPY (an
NMR-specialised Liouville engine) on a small homonuclear two-proton system:

* peak positions must agree to **< 1 Hz**;
* the normalised FID trajectories must agree to **< 5 % RMS**.

Both engines are optional, so the whole module skips cleanly when either is
absent. The QuTiP FID readout depends on Coder A's ``QRCSystem.fid_signal`` /
``SimConfig.fid_points`` contract; if that has not landed yet the test skips
with a clear reason rather than failing.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("SLEEPY", reason="sleepy-nmr not installed (pip install '.[validation]')")
pytest.importorskip("qutip", reason="qutip not installed (pip install '.[qrc]')")

from app.qrc import validate_sleepy  # noqa: E402
from app.qrc.config import SimConfig  # noqa: E402
from app.qrc.system import QRCSystem  # noqa: E402

# Coder A's FID-readout contract (plan §0/§2) this validation builds on.
_HAS_FID_CONTRACT = hasattr(QRCSystem, "fid_signal") and all(
    hasattr(SimConfig(), attr) for attr in ("fid_points", "fid_dwell")
)
_needs_fid = pytest.mark.skipif(
    not _HAS_FID_CONTRACT,
    reason="QRCSystem.fid_signal / SimConfig.fid_points not implemented yet (Coder A)",
)

# Smaller acquisition than the 2048-point default keeps the test fast while
# still resolving the ~465 Hz proton separation and the 15.8 Hz J-splitting.
_FID_POINTS = 1024
_FID_DWELL = 3e-4


def test_sleepy_fid_is_physical():
    """The SLEEPY reference alone must produce the expected proton spectrum:
    H1/H2 doublets centred on their offsets, split by J. Guards our SLEEPY
    setup independently of the QuTiP side."""
    fid = validate_sleepy.build_sleepy_two_proton(
        fid_points=_FID_POINTS, fid_dwell=_FID_DWELL
    )
    assert fid.shape == (_FID_POINTS,)
    assert np.iscomplexobj(fid)

    freq, mag = validate_sleepy._spectrum(fid, _FID_DWELL)
    peaks = validate_sleepy._find_peaks(freq, mag)
    # Expect four lines: 985.9 ± 7.9 and 520.3 ± 7.9 Hz.
    expected = [985.9 - 7.9, 985.9 + 7.9, 520.3 - 7.9, 520.3 + 7.9]
    for line in expected:
        assert np.min(np.abs(peaks - line)) < 2.0, (
            f"no SLEEPY peak within 2 Hz of {line}; peaks={peaks}"
        )


@_needs_fid
def test_qutip_matches_sleepy_fid():
    """Peak positions < 1 Hz apart and normalised-FID RMS < 5 % (plan §5)."""
    metrics = validate_sleepy.compare(
        fid_points=_FID_POINTS, fid_dwell=_FID_DWELL
    )
    assert metrics["max_peak_diff_hz"] < 1.0, metrics
    assert metrics["fid_rms"] < 0.05, metrics
