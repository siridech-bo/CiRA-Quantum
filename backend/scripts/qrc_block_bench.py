"""Scaling benchmark for the block-structured differentiable reservoir.

For each system size: propagator build time + resident GB, forward-only
ms/step, and forward+backward ms/step + backprop GB/step (slope between two
sequence lengths, so the static propagator is excluded). Uses the
physics-informed reduced-FID readout (the paper's readout) with V = D_eff
(the 6-spin campaign used V=137 ~ D_eff). Prints one JSON line per size and a
projection for the paper's protocol (T=1500, 100 Adam steps).

Run: python backend/scripts/qrc_block_bench.py --systems 9,10,11,12 --device cuda
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

from app.qrc.spectral_lines import physics_informed_lines  # noqa: E402
from qrc_gen_traces import _resolve_system  # noqa: E402
from qrc_learnable_9spin import (  # noqa: E402
    CDT_MAP, EncoderPerSpin, build_system, forward_features, make_fid_reduced_post,
)


def _sync(dev):
    if dev == "cuda":
        torch.cuda.synchronize()


def bench_one(name, dev, cdt, coupling, seq_lens, ckpt_every, T_proj, steps_proj):
    # V = D_eff of the coupling-scaled system (stronger J widens the multiplets)
    d_eff = physics_informed_lines(build_system(name, 1, coupling_scale=coupling).system,
                                   None).d_eff
    V = d_eff
    sysm = build_system(name, V, coupling_scale=coupling)
    if dev == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    g = sysm.ensure_diff(device=dev, cdtype=cdt, block=True)
    _sync(dev)
    build_s = time.time() - t0
    post, nfeat, ls = make_fid_reduced_post(sysm, g, dev)
    static = torch.cuda.memory_allocated() / 1e9 if dev == "cuda" else 0.0
    enc = EncoderPerSpin(sysm.n).to(dev)
    u = np.random.default_rng(0).random(max(seq_lens))

    with torch.no_grad():                      # forward-only (eval / baseline cost)
        forward_features([enc(float(s)) for s in u[:4]], sysm, g, dev, cdt, post=post)
        _sync(dev); t = time.time()
        forward_features([enc(float(s)) for s in u[:seq_lens[-1]]], sysm, g, dev, cdt, post=post)
        _sync(dev); fwd_ms = 1000 * (time.time() - t) / seq_lens[-1]

    peaks, ms = [], []
    for L in seq_lens:
        if dev == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        enc.zero_grad()
        _sync(dev); t = time.time()
        F = forward_features([enc(float(s)) for s in u[:L]], sysm, g, dev, cdt, post=post,
                             ckpt_every=ckpt_every)
        (F ** 2).mean().backward()
        _sync(dev)
        ms.append(1000 * (time.time() - t) / L)
        peaks.append(torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else 0.0)
    gb_per_step = (peaks[-1] - peaks[0]) / (seq_lens[-1] - seq_lens[0])
    fb_ms = ms[-1]
    # projection: each Adam step = 1 fwd+bwd over T (+ the arcsin baseline and
    # final eval forwards, negligible); per-condition wall-clock
    proj_h = fb_ms * T_proj * steps_proj / 1000 / 3600
    proj_gb = static + gb_per_step * T_proj
    rec = {
        "system": name, "n": sysm.n, "V": V, "D_eff": d_eff, "feats": nfeat,
        "dtype": str(cdt).split(".")[-1], "ckpt_every": ckpt_every,
        "build_s": round(build_s, 1), "propagator_GB": round(g["block"].nbytes() / 1e9, 3),
        "static_GB": round(static, 2), "fwd_ms_per_step": round(fwd_ms, 1),
        "fwdbwd_ms_per_step": round(fb_ms, 1), "backprop_GB_per_step": round(gb_per_step, 4),
        "peak_GB": [round(p, 2) for p in peaks], "seq_lens": seq_lens,
        f"proj_T{T_proj}_GB": round(proj_gb, 1),
        f"proj_T{T_proj}_{steps_proj}steps_h_per_condition": round(proj_h, 2),
    }
    print(json.dumps(rec), flush=True)
    del g, sysm
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default="9,10,11,12")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="complex64", choices=list(CDT_MAP))
    ap.add_argument("--coupling-scale", type=float, default=2.0)
    ap.add_argument("--seq-lens", default="8,24")
    ap.add_argument("--ckpt-every", type=int, default=0)
    ap.add_argument("--T-proj", type=int, default=1500)
    ap.add_argument("--steps-proj", type=int, default=100)
    ap.add_argument("--out", default=None, help="append JSON lines here")
    a = ap.parse_args()
    recs = []
    for name in a.systems.split(","):
        try:
            recs.append(bench_one(name, a.device, CDT_MAP[a.dtype], a.coupling_scale,
                                  [int(x) for x in a.seq_lens.split(",")], a.ckpt_every,
                                  a.T_proj, a.steps_proj))
        except torch.OutOfMemoryError as e:
            rec = {"system": name, "error": "OOM", "msg": str(e).splitlines()[0]}
            print(json.dumps(rec), flush=True)
            recs.append(rec)
            torch.cuda.empty_cache()
    if a.out:
        with open(a.out, "a", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
