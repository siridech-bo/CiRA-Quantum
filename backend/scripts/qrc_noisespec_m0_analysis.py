"""Milestone 0 analysis (pre-registered 09e3bfd, QRC_NoiseSpec_Milestone0.md sec. 3.5-3.7).

Loads the raw noiseless ensemble FIDs (traces), adds measurement noise from fixed seeds,
and scores M1 (R3 ridge), M2 (S1 ridge), M3 (S1 filter-function inversion), M4 (S1 MLP),
M5 (R3 MLP, descriptive). All selection on dev; confirmation used once.

  python scripts/qrc_noisespec_m0_analysis.py --task a --jobs 64
  python scripts/qrc_noisespec_m0_analysis.py --task b --jobs 64
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import qrc_noisespec_m0 as m  # noqa: E402
from app.qrc.spectral_lines import physics_informed_lines  # noqa: E402
from app.qrc.training import _fit_ridge  # noqa: E402

ETAS = [0.0, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
ETA_PRIMARY = 1e-2
ALPHAS = np.logspace(-4, 4, 17)
FAMILIES = ("lines-lin", "lines-log", "time-lin", "time-log")
T_IDX = np.arange(0, m.N_ACQ, 16)                      # 32 time points per probe
FF_IDX = np.array([0, 50, 100, 200, 400])              # 0, 2.5, 5, 10, 20 ms into acquisition
OMEGA = np.logspace(-1, 6, 3000)
EDGES = [(1e-1, m.BINS[0]), (m.BINS[-1], 1e6)]


# --- data --------------------------------------------------------------------
def amplitudes():
    """A = noiseless |S(0)| right after the pi/2 pulse, per system."""
    out = {}
    for name, sysc in m.make_systems().items():
        sp = m.Spins(sysc)
        out[name] = float(abs((sp.rho0 @ sp.P90.T) @ sp.o))
    return out


def load(task, split):
    z = np.load(m.TRACES / f"qrc_noisespec_m0{task}_{split}.npz", allow_pickle=True)
    return {"R3": z["fids_R3"].astype(np.complex128), "S1": z["fids_S1"].astype(np.complex128),
            "y": z["targets"], "P": z["noise_params"]}


def add_noise(fid, eta, A, seed):
    if eta == 0:
        return fid
    rng = np.random.default_rng(seed)
    s = eta * A / np.sqrt(2)
    return fid + s * (rng.standard_normal(fid.shape) + 1j * rng.standard_normal(fid.shape))


def noise_seed(task, split, system, ie):
    return {"a": 1, "b": 2}[task] * 100000 + {"dev": 1, "confirm": 2}[split] * 10000 \
        + {"R3": 1, "S1": 2, "R3eq": 3}[system] * 1000 + ie


# --- features ----------------------------------------------------------------
_PHI = {}


def phi(system):
    if system not in _PHI:
        sysc = m.make_systems()[system]
        ls = physics_informed_lines(sysc)
        t = np.arange(m.N_ACQ) * m.DWELL
        _PHI[system] = {s: np.exp(s * 1j * 2 * np.pi * np.outer(t, ls.freqs_hz)) for s in (1, -1)}
    return _PHI[system]


def line_sign(fid, system):
    """Pick the projection sign that captures the lines (engine FID rotation sense)."""
    p = phi(system)
    pw = {s: float(np.abs(fid[:50, 0] @ p[s]).sum()) for s in (1, -1)}
    return max(pw, key=pw.get), pw


def features(fid, family, A, sign, system, probes=None):
    f = fid if probes is None else fid[:, probes]
    if family.startswith("lines"):
        X = (f @ phi(system)[sign]) / (A * m.N_ACQ)
    else:
        X = f[:, :, T_IDX] / A
    X = X.reshape(X.shape[0], -1)
    if family.endswith("lin"):
        return np.hstack([X.real, X.imag])
    return np.log(np.abs(X) + 1e-6)


# --- metrics -----------------------------------------------------------------
def r2_cols(y, p):
    return 1 - ((y - p) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)


# --- ridge (M1, M2) ------------------------------------------------------------
def _std(Xtr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
    return (Xtr - mu) / sd, (Xte - mu) / sd


def ridge_cv(X, y, folds=5, seed=0):
    """Mean CV R^2 for every alpha (dual form per fold; identical to _fit_ridge with an
    unpenalised bias on train-standardised features)."""
    n = len(y)
    perm = np.random.default_rng(seed).permutation(n)
    parts = np.array_split(perm, folds)
    scores = np.zeros(len(ALPHAS))
    for i in range(folds):
        te = parts[i]
        tr = np.concatenate([parts[j] for j in range(folds) if j != i])
        Xtr, Xte = _std(X[tr], X[te])
        ym = y[tr].mean(0)
        K = Xtr @ Xtr.T
        ev, U = np.linalg.eigh(K)
        Uy = U.T @ (y[tr] - ym)
        KteU = (Xte @ Xtr.T) @ U
        for ia, al in enumerate(ALPHAS):
            pred = KteU @ (Uy / (ev + al)[:, None]) + ym
            scores[ia] += r2_cols(y[te], pred).mean() / folds
    return scores


def ridge_fit_predict(Xtr, ytr, Xte, alpha):
    Xs, Xt = _std(Xtr, Xte)
    out = np.zeros((len(Xt), ytr.shape[1]))
    for k in range(ytr.shape[1]):
        w = _fit_ridge(Xs, ytr[:, k], alpha, "cpu")
        out[:, k] = np.hstack([Xt, np.ones((len(Xt), 1))]) @ w
    return out


def ridge_method(F_dev, F_con, ydev):
    best = None
    for fam in FAMILIES:
        sc = ridge_cv(F_dev[fam], ydev)
        ia = int(np.argmax(sc))
        if best is None or sc[ia] > best[2]:
            best = (fam, float(ALPHAS[ia]), float(sc[ia]))
    fam, al, cv = best
    pred = ridge_fit_predict(F_dev[fam], ydev, F_con[fam], al)
    return pred, {"family": fam, "alpha": al, "cv_r2": cv, "D": int(F_dev[fam].shape[1])}


# --- MLP (M4, M5) --------------------------------------------------------------
MLP_GRID = [(h, l2) for h in ((128, 128), (256, 256)) for l2 in (1e-4, 1e-2)]


def _mlp(Xtr, ytr, Xte, h, l2, seed=0):
    from sklearn.neural_network import MLPRegressor
    Xs, Xt = _std(Xtr, Xte)
    ym, ys = ytr.mean(0), ytr.std(0) + 1e-12
    net = MLPRegressor(hidden_layer_sizes=h, alpha=l2, early_stopping=True, max_iter=2000,
                       random_state=seed, learning_rate_init=1e-3, n_iter_no_change=30)
    net.fit(Xs, (ytr - ym) / ys)
    return net.predict(Xt).reshape(len(Xt), -1) * ys + ym


def _mlp_cv_job(X, y, h, l2, folds=3):
    perm = np.random.default_rng(0).permutation(len(y))
    parts = np.array_split(perm, folds)
    s = 0.0
    for i in range(folds):
        te = parts[i]
        tr = np.concatenate([parts[j] for j in range(folds) if j != i])
        s += r2_cols(y[te], _mlp(X[tr], y[tr], X[te], h, l2)).mean() / folds
    return s


def mlp_method(F_dev, F_con, ydev, jobs):
    from joblib import Parallel, delayed
    cfgs = [(fam, h, l2) for fam in FAMILIES for h, l2 in MLP_GRID]
    sc = Parallel(n_jobs=jobs)(delayed(_mlp_cv_job)(F_dev[f], ydev, h, l2) for f, h, l2 in cfgs)
    i = int(np.argmax(sc))
    fam, h, l2 = cfgs[i]
    pred = _mlp(F_dev[fam], ydev, F_con[fam], h, l2)
    return pred, {"family": fam, "hidden": list(h), "l2": l2, "cv_r2": float(sc[i]),
                  "D": int(F_dev[fam].shape[1])}


# --- filter-function inversion (M3) -------------------------------------------
def filter_sq(n_pi, T, ta):
    """|F(omega)|^2 for toggling f over [0, T + ta] (CPMG over T, then free with the last sign)."""
    if n_pi == 0:
        bounds, signs = [0.0, T + ta], [1.0]
    else:
        tau = T / n_pi
        pulses = [tau / 2 + i * tau for i in range(n_pi)]
        bounds = [0.0] + pulses + [T + ta]
        signs = [(-1.0) ** i for i in range(n_pi + 1)]
    w = OMEGA
    F = np.zeros_like(w, dtype=complex)
    for s, a, b in zip(signs, bounds[:-1], bounds[1:]):
        F += s * (np.exp(-1j * w * b) - np.exp(-1j * w * a)) / (-1j * w)
    return np.abs(F) ** 2


def ff_design(probes=None):
    """Rows: (probe, acquisition offset). Returns |F|^2 matrix (rows, omega) and elapsed times."""
    pr = range(len(m.PROBES)) if probes is None else probes
    rows, el, idx = [], [], []
    for j in pr:
        n_pi, t_ms = m.PROBES[j]
        for k in FF_IDX:
            ta = k * m.DWELL
            rows.append(filter_sq(n_pi, t_ms * 1e-3, ta))
            el.append(t_ms * 1e-3 + ta)
            idx.append((j, k))
    return np.array(rows), np.array(el), idx


def _int(F2, S):
    """chi = (1/2pi) int_0^inf S |F|^2 d omega, trapezoid on the log grid."""
    g = F2 * S[None, :] * OMEGA[None, :]
    return np.trapezoid(g, np.log(OMEGA), axis=1) / (2 * np.pi)


def ff_chi(fid, A, eta, T2, idx, el):
    mag = np.array([np.abs(fid[:, j, k]) for j, k in idx]).T            # (n, rows)
    floor = max(eta, 1e-4) * A
    magc = np.maximum(mag, floor)
    chi = -np.log(magc / A) - el[None, :] / T2
    w = (magc / floor) ** 2
    return chi, w


class FFa:
    """M0a: chi = sigma^2 K(tau); NLLS in (log sigma, log tau) from a grid start; prior-range clip."""

    def __init__(self, F2):
        self.ltau = np.linspace(np.log10(1e-5), np.log10(1.0), 300)
        self.K = np.array([_int(F2, 2 * 10 ** lt / (1 + (OMEGA * 10 ** lt) ** 2)) for lt in self.ltau])

    def fit(self, chi, w):
        from scipy.optimize import least_squares
        cfg = m.TASKS["a"]
        out = []
        for c, ww in zip(chi, w):
            sw = np.sqrt(ww)
            num = (self.K * ww).dot(c)
            den = (self.K ** 2 * ww).sum(1)
            s2 = np.maximum(num / den, 1e-6)
            loss = ((s2[:, None] * self.K - c) ** 2 * ww).sum(1)
            i = int(np.argmin(loss))
            x0 = [0.5 * np.log10(s2[i]), self.ltau[i]]

            def res(x):
                u = (np.clip(x[1], self.ltau[0], self.ltau[-1]) - self.ltau[0]) / (self.ltau[1] - self.ltau[0])
                i0 = min(int(u), len(self.ltau) - 2)
                f = u - i0
                Kx = (1 - f) * self.K[i0] + f * self.K[i0 + 1]
                return sw * (10 ** (2 * x[0]) * Kx - c)
            try:
                x = least_squares(res, x0, bounds=([-1, -5], [5, 0])).x
            except Exception:
                x = np.array(x0)
            out.append(x)
        out = np.array(out)
        out[:, 0] = np.clip(out[:, 0], *np.log10(cfg["sig"]))
        out[:, 1] = np.clip(out[:, 1], *np.log10(cfg["tau"]))
        return out


class FFb:
    """M0b: chi = sum_b S_b K_b + edges; weighted NNLS + ridge lambda; log floor. lambda/floor on dev."""

    def __init__(self, F2):
        cols = []
        for a, b in [EDGES[0]] + list(zip(m.BINS[:-1], m.BINS[1:])) + [EDGES[1]]:
            S = ((OMEGA >= a) & (OMEGA < b)).astype(float)
            cols.append(_int(F2, S))
        self.K = np.array(cols).T                                         # (rows, 8)

    def fit(self, chi, w, lam, floor):
        from scipy.optimize import nnls
        out = []
        for c, ww in zip(chi, w):
            sw = np.sqrt(ww)
            A = self.K * sw[:, None]
            nrm = np.linalg.norm(A, axis=0) + 1e-30
            An = A / nrm
            if lam > 0:
                An = np.vstack([An, np.sqrt(lam) * np.eye(An.shape[1])])
                bb = np.concatenate([c * sw, np.zeros(An.shape[1])])
            else:
                bb = c * sw
            s, _ = nnls(An, bb, maxiter=2000)
            out.append(np.log10(np.maximum(s / nrm, 10.0 ** floor))[1:-1])
        return np.array(out)


FF_LAMS = [0.0, 1e-6, 1e-4, 1e-2]
FF_FLOORS = [-4.0, -3.0, -2.0, -1.0]


# --- bootstrap ---------------------------------------------------------------
def boot_diff(y, pa, pb, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    N = len(y)
    d = []
    for _ in range(n):
        i = rng.integers(0, N, N)
        d.append(r2_cols(y[i], pa[i]).mean() - r2_cols(y[i], pb[i]).mean())
    d = np.array(d)
    return [float(r2_cols(y, pa).mean() - r2_cols(y, pb).mean()),
            float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


# --- main --------------------------------------------------------------------
def run(task, jobs, etas, budget, smoke=False):
    t0 = time.time()
    A = amplitudes()
    if smoke:                       # dev-only dry run: dev[:800] vs dev[800:], confirm untouched
        d = load(task, "dev")
        dev = {k: v[:800] for k, v in d.items()}
        con = {k: v[800:] for k, v in d.items()}
    else:
        dev, con = load(task, "dev"), load(task, "confirm")
    ydev, ycon = dev["y"], con["y"]
    sign, pw = line_sign(dev["R3"], "R3")
    T2 = m.make_systems()["S1"].t2[0]
    F2, el, idx = ff_design()
    ff = FFa(F2) if task == "a" else FFb(F2)
    res = {"task": task, "prereg": "09e3bfd", "A": A, "line_sign": sign, "line_power": pw,
           "D_lines_R3": int(physics_informed_lines(m.make_systems()["R3"]).d_eff),
           "levels": {}}
    preds_primary = {}

    # sanity: FF forward model vs SLE at eta = 0 (true params), S1
    chi0, _ = ff_chi(dev["S1"][:200], A["S1"], 0.0, T2, idx, el)
    Ptrue = dev["P"][:200]
    chim = np.array([_int(F2, m.S_two_sided(OMEGA, [tuple(c) for c in p])) for p in Ptrue])
    ok = chim > 0.05
    res["ff_model_check"] = {"median_abs_rel_err": float(np.median(np.abs(chi0[ok] - chim[ok]) / chim[ok])),
                             "n_points": int(ok.sum())}
    print("FF forward model vs SLE (eta=0):", res["ff_model_check"], flush=True)

    for ie, eta in enumerate(ETAS):
        if eta not in etas:
            continue
        L = {}
        fd = {s: add_noise(dev[s], eta, A[s], noise_seed(task, "dev", s, ie)) for s in ("R3", "S1")}
        fc = {s: add_noise(con[s], eta, A[s], noise_seed(task, "confirm", s, ie)) for s in ("R3", "S1")}
        Fd = {s: {f: features(fd[s], f, A[s], sign if s == "R3" else 1, s) for f in FAMILIES} for s in fd}
        Fc = {s: {f: features(fc[s], f, A[s], sign if s == "R3" else 1, s) for f in FAMILIES} for s in fc}
        P = {}
        P["M1"], L["M1"] = ridge_method(Fd["R3"], Fc["R3"], ydev)
        P["M2"], L["M2"] = ridge_method(Fd["S1"], Fc["S1"], ydev)
        # M3
        chid, wd = ff_chi(fd["S1"], A["S1"], eta, T2, idx, el)
        chic, wc = ff_chi(fc["S1"], A["S1"], eta, T2, idx, el)
        if task == "a":
            P["M3"] = ff.fit(chic, wc)
            L["M3"] = {"method": "NLLS (log sigma, log tau), grid start, prior-range clip",
                       "D": int(len(idx))}
        else:
            sub = np.random.default_rng(0).permutation(len(ydev))[:300]
            best = None
            for lam in FF_LAMS:
                pdv = ff.fit(chid[sub], wd[sub], lam, FF_FLOORS[0])
                for fl in FF_FLOORS:
                    r = r2_cols(ydev[sub], np.maximum(pdv, fl)).mean()
                    if best is None or r > best[2]:
                        best = (lam, fl, r)
            P["M3"] = ff.fit(chic, wc, best[0], best[1])
            L["M3"] = {"method": "weighted NNLS over 6 bins + 2 edges", "lambda": best[0],
                       "log_floor": best[1], "dev_r2": float(best[2]), "D": int(len(idx))}
        P["M4"], L["M4"] = mlp_method(Fd["S1"], Fc["S1"], ydev, jobs)
        P["M5"], L["M5"] = mlp_method(Fd["R3"], Fc["R3"], ydev, jobs)
        # descriptive: R3 at equal per-spin noise (its 3-spin signal advantage kept)
        fd_eq = add_noise(dev["R3"], eta, A["S1"], noise_seed(task, "dev", "R3eq", ie))
        fc_eq = add_noise(con["R3"], eta, A["S1"], noise_seed(task, "confirm", "R3eq", ie))
        Fd_eq = {f: features(fd_eq, f, A["R3"], sign, "R3") for f in FAMILIES}
        Fc_eq = {f: features(fc_eq, f, A["R3"], sign, "R3") for f in FAMILIES}
        P["M1eq"], L["M1eq"] = ridge_method(Fd_eq, Fc_eq, ydev)
        lv = {"eta": eta, "methods": {}}
        for k in P:
            r = r2_cols(ycon, P[k])
            lv["methods"][k] = {"r2_per_target": r.tolist(), "r2_mean": float(r.mean()), **L[k]}
        for k in ("M2", "M3", "M4"):
            lv[f"M1_minus_{k}"] = boot_diff(ycon, P["M1"], P[k])
        res["levels"][str(eta)] = lv
        print(f"[{task}] eta={eta:g}  " + "  ".join(f"{k}={lv['methods'][k]['r2_mean']:.3f}" for k in P)
              + f"   ({time.time() - t0:.0f}s)", flush=True)
        if eta == ETA_PRIMARY:
            preds_primary = P

    # verdict
    lv = res["levels"].get(str(ETA_PRIMARY))
    if lv:
        r1 = lv["methods"]["M1"]["r2_per_target"]
        if task == "a":
            res["verdict"] = {"rule": "M1 R2 >= 0.90 on both log sigma and log tau_c at eta=1e-2",
                              "M1_r2": r1, "pass": bool(min(r1) >= 0.90)}
        else:
            conds = {k: (lv[f"M1_minus_{k}"][0] >= 0.05 and lv[f"M1_minus_{k}"][1] > 0)
                     for k in ("M2", "M3", "M4")}
            res["verdict"] = {"rule": "M1 mean-bin R2 exceeds M2, M3, M4 each by >= 0.05 with CI lower bound > 0",
                              "diffs": {k: lv[f"M1_minus_{k}"] for k in ("M2", "M3", "M4")},
                              "conditions": conds, "pass": bool(all(conds.values()))}
        print("VERDICT", json.dumps(res["verdict"]), flush=True)

    # descriptive: probe budget at eta = 1e-2
    if budget and lv:
        ie = ETAS.index(ETA_PRIMARY)
        perm = np.random.default_rng(7).permutation(len(m.PROBES))
        fd = {s: add_noise(dev[s], ETA_PRIMARY, A[s], noise_seed(task, "dev", s, ie)) for s in ("R3", "S1")}
        fc = {s: add_noise(con[s], ETA_PRIMARY, A[s], noise_seed(task, "confirm", s, ie)) for s in ("R3", "S1")}
        bud = {}
        for nb in (7, 14, 28, 49):
            pr = np.sort(perm[:nb])
            Fd = {s: {f: features(fd[s], f, A[s], sign if s == "R3" else 1, s, pr) for f in FAMILIES} for s in fd}
            Fc = {s: {f: features(fc[s], f, A[s], sign if s == "R3" else 1, s, pr) for f in FAMILIES} for s in fc}
            row = {}
            row["M1"] = r2_cols(ycon, ridge_method(Fd["R3"], Fc["R3"], ydev)[0]).mean()
            row["M2"] = r2_cols(ycon, ridge_method(Fd["S1"], Fc["S1"], ydev)[0]).mean()
            F2b, elb, idxb = ff_design(pr)
            chic, wc = ff_chi(fc["S1"], A["S1"], ETA_PRIMARY, T2, idxb, elb)
            if task == "a":
                row["M3"] = r2_cols(ycon, FFa(F2b).fit(chic, wc)).mean()
            else:
                c3 = lv["methods"]["M3"]
                row["M3"] = r2_cols(ycon, FFb(F2b).fit(chic, wc, c3["lambda"], c3["log_floor"])).mean()
            c4 = lv["methods"]["M4"]
            row["M4"] = r2_cols(ycon, _mlp(Fd["S1"][c4["family"]], ydev, Fc["S1"][c4["family"]],
                                           tuple(c4["hidden"]), c4["l2"])).mean()
            bud[nb] = {k: float(v) for k, v in row.items()}
            print(f"[{task}] budget {nb}: " + "  ".join(f"{k}={v:.3f}" for k, v in row.items()), flush=True)
        res["budget"] = bud

    res["wall_s"] = time.time() - t0
    if smoke:
        print("smoke run: nothing written")
        return
    m.DATA.mkdir(parents=True, exist_ok=True)
    (m.DATA / f"qrc_noisespec_m0{task}_results.json").write_text(json.dumps(res, indent=1))
    if preds_primary:
        np.savez(m.DATA / f"qrc_noisespec_m0{task}_preds.npz", y=ycon, P=con["P"],
                 **{k: v for k, v in preds_primary.items()})
    print("done", task, f"{time.time() - t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["a", "b"], required=True)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--etas", type=float, nargs="*", default=ETAS)
    ap.add_argument("--no-budget", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    run(a.task, a.jobs, a.etas, not a.no_budget, a.smoke)


if __name__ == "__main__":
    main()
