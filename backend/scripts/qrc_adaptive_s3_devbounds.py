"""Stage-3 pilot (disclosed, QRC_AdaptiveEst.md sec. 4.7 item 4): bounds at the dev points vs dev RMSE.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
import qrc_adaptive_s3 as s3
from app.qrc import estimation as est
from app.qrc.adaptive_batch import BatchedEpisodes2D

system, eta, N = "R3", 0.1, 49
z = np.load(s3.s3_trace(system))
sig = est.m0_sigma(eta, float(z["A"]))
be = BatchedEpisodes2D(z["table"], z["th_grid"], z["nu_grid"], sig,
                       ((s3.TH_LO, s3.TH_HI), (s3.NU_LO, s3.NU_HI)), device="cpu")
rd = np.random.default_rng(990)                       # same dev points as eval --dev 3
th = rd.uniform(s3.TH_LO + 0.1, s3.TH_HI - 0.1, 3)
nu = rd.uniform(s3.NU_LO + 0.1, s3.NU_HI - 0.1, 3)
pts = torch.tensor(np.c_[th, nu])[None]
_, d1, d2 = be._model(pts)
d1, d2 = d1[0].numpy(), d2[0].numpy()                 # (3, P, L)
F = np.stack([np.stack([(d1.conj() * d1).real.sum(-1), (d1.conj() * d2).real.sum(-1)], -1),
              np.stack([(d2.conj() * d1).real.sum(-1), (d2.conj() * d2).real.sum(-1)], -1)], -2) / sig ** 2
tr = {t: np.load(f"artifacts/traces/qrc_adaptive_stage3_eval_{system}_eta{eta:g}_{t}_dev.npz") for t in sys.argv[1:]}
for k in range(3):
    Fm = F[k].mean(0)
    b = {"known_opt": 1 / F[k][:, 0, 0].max(), "nuis_opt": s3.c_optimal(F[k])[0],
         "nuis_bank": np.linalg.inv(Fm)[0, 0]}
    print(f"point {k} ({th[k]:.3f},{nu[k]:.3f}) bound RMSE@49: " + "  ".join(f"{a}={np.sqrt(v / N):.2e}" for a, v in b.items()))
    for t, e in tr.items():
        P = list(e["policies"])
        for ip, p in enumerate(P):
            mn = e["means"][k, ip]
            if np.isnan(mn).all():
                continue
            mse = np.mean((mn[:, -1] - th[k]) ** 2)
            print(f"   [{t}] {p:18s} rmse={np.sqrt(mse):.2e}  nuis_opt/MSE={b['nuis_opt'] / N / mse:.2f}"
                  f"  nuis_bank/MSE={b['nuis_bank'] / N / mse:.2f}  known_opt/MSE={b['known_opt'] / N / mse:.2f}")
