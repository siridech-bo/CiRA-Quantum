"""Fast weather reservoir pass on the exact block propagator (+ FID readout).

Same physics and loop as ``benchmarks.run_weather_reservoir`` in FID mode:
per day k, pulse the weather inputs (``arcsin(sqrt(s))`` R_x per target spin),
evolve tau, then acquire the FID of a copy of the state (π/2 readout pulse on a
channel's spins, ``fid_points`` samples at ``dwell``). The block propagator makes
the step and the FID exact linear maps, so a full 1474-day pass takes seconds
instead of ~15 GPU-hours. Validated against the saved trace (``--check``).

Channels: ``H`` (protons) and/or ``C`` (carbons), each with its own dwell (the
¹³C band of crotonic spans ±7.7 kHz → dwell ≤ 65 µs; ¹H ±1.2 kHz → 0.3 ms).

Writes (persist-waveform rule): a trace ``.npz`` with every channel's raw FIDs
plus ``weather_norm``/``split``/``meta``, and a features ``.npz`` (key ``X``:
|FFT| at the 653 strongest bins of each channel, concatenated) for
``qrc_weather_bench.py report --extra``.

  python scripts/qrc_weather_block.py --system crotonic9_paper4 --tau 0.03 \
      --channels H --out-trace t.npz --out-features f.npz [--check]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.qrc.block_propagator import BlockPropagator  # noqa: E402
from app.qrc.config import SimConfig  # noqa: E402
from app.qrc.encoding import angle_for  # noqa: E402
from app.qrc.system import QRCSystem  # noqa: E402
from qrc_gen_traces import _resolve_system  # noqa: E402

C128 = torch.complex128


def _rx(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return torch.tensor([[c, -1j * s], [-1j * s, c]], dtype=C128)


def channel_spins(sysc, ch):
    lab = [x.upper() for x in sysc.labels]
    return [i for i, x in enumerate(lab) if x.startswith(ch)]


def weather_drive(w, n, temp_idx, hum_idx):
    """(n_days, n) per-spin input: temp on ``temp_idx``, humidity on ``hum_idx``."""
    seq = np.zeros((len(w), n))
    seq[:, temp_idx] = w[:, 0:1]
    seq[:, hum_idx] = w[:, 1:2]
    return seq


def run(sysc, tau, w, n_steps, temp_idx, hum_idx, channels, fid_points, dwells, device,
        enc_fn="arcsin_sqrt", log_every=200):
    sim = SimConfig(tau=tau, n_virtual=1, evolution_mode="action")
    sysm = QRCSystem(sysc, sim)
    bp = BlockPropagator(sysc, sim, device=device, cdtype=C128, which=("z",))
    ros = {ch: bp.fid_readout(channel_spins(sysc, ch), fid_points, dwells[ch]) for ch in channels}
    vec = torch.tensor(sysm.qt.operator_to_vector(sysm.rho0).full().ravel(), dtype=C128, device=device)
    seq = weather_drive(w, sysm.n, temp_idx, hum_idx)
    fids = {ch: np.empty((n_steps, fid_points), np.complex64) for ch in channels}
    t0 = time.time()
    for k in range(n_steps):
        pulse = [_rx(angle_for(enc_fn, float(seq[k, q]))).to(device) for q in range(sysm.n)]
        _, vec = bp.step(vec, pulse)
        for ch in channels:
            fids[ch][k] = bp.fid(vec, ros[ch]).cpu().numpy()
        if log_every and (k + 1) % log_every == 0:
            print(f"  day {k + 1}/{n_steps} ({time.time() - t0:.0f}s)", flush=True)
    return fids


def features(fids_by_ch, n_peaks=653):
    """|FFT| at the n_peaks strongest bins of each channel's first FID, concatenated."""
    out = []
    for ch, F in fids_by_ch.items():
        mag = np.abs(np.fft.fft(F, axis=1))
        bins = np.sort(np.argsort(mag[0])[::-1][:n_peaks])
        out.append(mag[:, bins])
    return np.hstack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="crotonic9_paper4")
    ap.add_argument("--tau", type=float, default=0.03)
    ap.add_argument("--channels", default="H", help="comma list of H,C")
    ap.add_argument("--fid-points", type=int, default=2048)
    ap.add_argument("--dwell-H", type=float, default=3e-4)
    ap.add_argument("--dwell-C", type=float, default=5e-5)
    ap.add_argument("--temp-spins", default=None, help="comma list (default: protons)")
    ap.add_argument("--hum-spins", default=None, help="comma list (default: carbons)")
    ap.add_argument("--trace", default="artifacts/traces/weather_full.npz",
                    help="reference trace: weather_norm + split (and FIDs for --check)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--check", action="store_true", help="compare H-channel FIDs to the reference trace")
    ap.add_argument("--out-trace", required=True)
    ap.add_argument("--out-features", required=True)
    a = ap.parse_args()

    ref = np.load(a.trace, allow_pickle=True)
    w = np.asarray(ref["weather_norm"], float)
    n_steps = int(np.sum(ref["split"]))
    sysc = _resolve_system(a.system)
    temp_idx = ([int(x) for x in a.temp_spins.split(",")] if a.temp_spins else channel_spins(sysc, "H"))
    hum_idx = ([int(x) for x in a.hum_spins.split(",")] if a.hum_spins else channel_spins(sysc, "C"))
    channels = a.channels.split(",")
    dwells = {"H": a.dwell_H, "C": a.dwell_C}
    print(f"system={a.system} tau={a.tau} channels={channels} temp->{temp_idx} humidity->{hum_idx} "
          f"days={n_steps} device={a.device}", flush=True)
    t0 = time.time()
    fids = run(sysc, a.tau, w, n_steps, temp_idx, hum_idx, channels, a.fid_points, dwells, a.device)
    print(f"pass done in {time.time() - t0:.1f}s", flush=True)
    if a.check and "H" in fids:
        r = np.asarray(ref["fids"])
        err = np.abs(fids["H"] - r).max() / np.abs(r).max()
        print(f"CHECK vs reference trace (H channel, all {n_steps} days): max rel err {err:.2e}", flush=True)

    meta = {"experiment": "weather_block", "system": a.system, "tau": a.tau, "channels": channels,
            "dwell": {c: dwells[c] for c in channels}, "fid_points": a.fid_points,
            "temp_spins": temp_idx, "hum_spins": hum_idx, "encoding": "arcsin_sqrt R_x",
            "readout": "FID |FFT| top-653 bins per channel", "split": ref["split"].tolist()}
    np.savez(a.out_trace, **{f"fids_{c}": fids[c] for c in channels},
             weather_norm=w, split=ref["split"], seed=ref["seed"], fid_dwell=dwells[channels[0]],
             task="weather", meta=json.dumps(meta))
    X = features(fids)
    np.savez(a.out_features, X=X)
    print(f"wrote {a.out_trace} and {a.out_features} (X {X.shape})")


if __name__ == "__main__":
    main()
