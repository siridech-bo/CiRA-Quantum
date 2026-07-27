"""QRC benchmark orchestration — runs, ESN baseline, and the scaling study.

This is where the pieces meet: build a system, drive a task through the
reservoir, fit the readout, score it. Three entry points:

* :func:`run_memory_capacity` — the fading-memory gate (plan §5.5). Must
  pass (smooth corr²-vs-delay decay) before any downstream result means
  anything.
* :func:`run_narma` — NARMA-n NMSE against Paper 4's numbers (§10.1).
* :func:`scaling_study` — **the headline deliverable**: sweep qubit count
  N ∈ {3,5,7,9,…} and show memory capacity / NARMA accuracy *improve* with
  N. This is the quantitative argument that acquiring a larger machine
  than the 3-qubit SPINQ buys real computational power (§10.4 — "no
  experimental group has done this systematically").

A classical **Echo State Network** baseline (:func:`esn_baseline`) gives
the "is the quantum reservoir actually pulling its weight?" comparison
Paper 4 reports against.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.qrc.config import (
    EncodingConfig,
    QRCConfig,
    SimConfig,
    TrainingConfig,
    make_system_config,
)
from app.qrc.encoding import Encoder
from app.qrc.evolution import Reservoir
from app.qrc.features import FeatureConfig
from app.qrc.system import QRCSystem
from app.qrc.tasks import (
    delayed_targets,
    memory_task,
    narma_sequence,
    narma_sequence_sine,
    nmse_paper,
)
from app.qrc.training import evaluate, train_readout
from app.qrc.utils import nmse, normalize, r2_score, squared_correlation

# ---------------------------------------------------------------------------
# Reservoir assembly
# ---------------------------------------------------------------------------


def build_reservoir(cfg: QRCConfig, feature_cfg: FeatureConfig | None = None) -> Reservoir:
    system = QRCSystem(cfg.system, cfg.sim)
    encoder = Encoder(system, cfg.encoding)
    return Reservoir(system, encoder, feature_cfg)


def _split(X, y, washout, n_train, n_test):
    """Discard washout, then take contiguous train/test blocks.

    Raises if the sequence is too short to fill both blocks — otherwise
    an empty test slice silently yields NaN correlations downstream.
    """
    need = washout + n_train + n_test
    if X.shape[0] < need:
        raise ValueError(
            f"sequence has {X.shape[0]} steps but the split needs "
            f"{need} (washout {washout} + train {n_train} + test {n_test}). "
            f"Increase --steps/n_steps or lower the split sizes."
        )
    lo = washout
    tr = slice(lo, lo + n_train)
    te = slice(lo + n_train, lo + n_train + n_test)
    return X[tr], y[tr], X[te], y[te]


# ---------------------------------------------------------------------------
# Memory capacity  (the fading-memory gate)
# ---------------------------------------------------------------------------


@dataclass
class MemoryCapacityResult:
    total_mc: float
    per_delay: dict[int, float]
    n_qubits: int
    passed_gate: bool
    backend: str


def run_memory_capacity(
    cfg: QRCConfig,
    max_delay: int = 30,
    n_steps: int = 1200,
    feature_cfg: FeatureConfig | None = None,
) -> MemoryCapacityResult:
    """Drive a random sequence and measure corr² for each delay.

    Gate criterion (plan §5.5.3): corr² high at d=0 and *decaying* — not
    stuck at 1 (missing dissipation) nor collapsed to 0 (over-damped /
    wrong Hamiltonian). We accept when corr²(d=0) ≥ 0.9 and the curve is
    non-increasing on average (monotone decay allowing small noise).
    """
    res = build_reservoir(cfg, feature_cfg)
    u = memory_task(n_steps, seed=cfg.sim.seed)
    out = res.run(u)
    X = out.X
    targets = delayed_targets(u, max_delay)

    tr_cfg = cfg.training
    per_delay: dict[int, float] = {}
    for d, yd in targets.items():
        Xtr, ytr, Xte, yte = _split(
            X, yd, tr_cfg.washout, tr_cfg.n_train, tr_cfg.n_test
        )
        model = train_readout(Xtr, ytr, tr_cfg)
        per_delay[d] = squared_correlation(yte, model.predict(Xte))

    total = float(sum(per_delay.values()))
    c0 = per_delay.get(0, 0.0)
    # smoothness: average forward difference should be ≤ small positive
    ds = [per_delay[d] for d in sorted(per_delay)]
    diffs = np.diff(ds)
    decays = c0 >= 0.9 and np.mean(diffs) <= 0.02
    return MemoryCapacityResult(
        total_mc=total,
        per_delay=per_delay,
        n_qubits=cfg.system.n_qubits,
        passed_gate=bool(decays),
        backend=out.backend,
    )


# ---------------------------------------------------------------------------
# NARMA
# ---------------------------------------------------------------------------


@dataclass
class NarmaResult:
    order: int
    metrics: dict
    n_qubits: int


def run_narma(
    cfg: QRCConfig,
    order: int = 2,
    n_steps: int | None = None,
    feature_cfg: FeatureConfig | None = None,
    input_kind: str = "uniform",
    progress_cb=None,
) -> NarmaResult:
    """Drive a NARMA-``order`` task through the reservoir and score it.

    ``input_kind`` selects the driving signal: ``"uniform"`` (classic
    i.i.d. NARMA, unchanged default) or ``"sine"`` (Paper-4 multi-tone
    input via :func:`narma_sequence_sine`). The returned metrics bundle
    adds ``nmse_paper`` (Σ(y-ŷ)²/Σy²) alongside the standard metrics.
    ``progress_cb(done,total)`` is forwarded to the reservoir loop for
    step-level progress on long runs.
    """
    tr_cfg = cfg.training
    if n_steps is None:
        n_steps = tr_cfg.washout + tr_cfg.n_train + tr_cfg.n_test
    res = build_reservoir(cfg, feature_cfg)
    if input_kind == "sine":
        u, y = narma_sequence_sine(n_steps, order, seed=cfg.sim.seed)
    else:
        u, y = narma_sequence(n_steps, order, seed=cfg.sim.seed)
    out = res.run(u, progress_cb=progress_cb)
    Xtr, ytr, Xte, yte = _split(
        out.X, y, tr_cfg.washout, tr_cfg.n_train, tr_cfg.n_test
    )
    model = train_readout(Xtr, ytr, tr_cfg)
    metrics = evaluate(model, Xte, yte)
    metrics["nmse_paper"] = nmse_paper(yte, model.predict(Xte))
    return NarmaResult(order=order, metrics=metrics, n_qubits=out.n_qubits)


def run_narma_multitask(
    cfg: QRCConfig,
    orders,
    n_steps: int | None = None,
    feature_cfg: FeatureConfig | None = None,
    input_kind: str = "sine",
    progress_cb=None,
) -> tuple[dict[int, dict], int]:
    """Emulate several NARMA orders from a *single* reservoir pass.

    The NARMA driving input is the same for every order (it depends only on
    ``n_steps``/``seed``, not the order) — so the reservoir readout matrix X
    is identical across orders and only the target changes. Paper 4 calls
    this "multitasking": run the (expensive) reservoir once, then fit a
    separate ridge readout per order. This is ~len(orders)× cheaper than
    calling :func:`run_narma` per order, which re-runs the whole reservoir.

    Returns ``({order: metrics}, n_qubits)``.
    """
    orders = list(orders)
    tr_cfg = cfg.training
    if n_steps is None:
        n_steps = tr_cfg.washout + tr_cfg.n_train + tr_cfg.n_test
    seq = narma_sequence_sine if input_kind == "sine" else narma_sequence
    # Input is order-independent; take it from the first order.
    u, _ = seq(n_steps, orders[0], seed=cfg.sim.seed)
    res = build_reservoir(cfg, feature_cfg)
    out = res.run(u, progress_cb=progress_cb)      # the ONE expensive pass
    results: dict[int, dict] = {}
    for order in orders:
        _, y = seq(n_steps, order, seed=cfg.sim.seed)
        Xtr, ytr, Xte, yte = _split(
            out.X, y, tr_cfg.washout, tr_cfg.n_train, tr_cfg.n_test
        )
        model = train_readout(Xtr, ytr, tr_cfg)
        m = evaluate(model, Xte, yte)
        m["nmse_paper"] = nmse_paper(yte, model.predict(Xte))
        results[order] = m
    return results, out.n_qubits


# ---------------------------------------------------------------------------
# Classical ESN baseline (plan §10.2)
# ---------------------------------------------------------------------------


def _spectral_radius(W: np.ndarray, method: str, seed: int, iters: int = 200) -> float:
    """Largest-magnitude eigenvalue of ``W``.

    ``"exact"`` uses a full eigendecomposition (accurate, but O(n³) and
    memory-heavy — impractical past a few thousand units). ``"power"``
    uses power iteration, which returns the dominant magnitude in O(iters·n²)
    with no extra memory — good enough to scale a random reservoir and the
    only feasible option for the ESN(5000/10000) sweep sizes.
    """
    if method == "power":
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(W.shape[0])
        nrm = np.linalg.norm(v)
        if nrm == 0:
            return 0.0
        v /= nrm
        lam = 0.0
        for _ in range(iters):
            w = W @ v
            lam = float(np.linalg.norm(w))
            if lam == 0.0:
                return 0.0
            v = w / lam
        return lam
    return float(np.max(np.abs(np.linalg.eigvals(W))))


def esn_baseline(
    u: np.ndarray,
    y: np.ndarray,
    n_reservoir: int = 100,
    spectral_radius: float = 0.9,
    leak: float = 1.0,
    tr_cfg: TrainingConfig | None = None,
    seed: int = 0,
    radius_method: str = "exact",
) -> dict:
    """A standard leaky-integrator Echo State Network for comparison.

    Not quantum — a plain classical reservoir with a random recurrent
    matrix scaled to ``spectral_radius``. Gives the yardstick Paper 4
    compares its QRC against ("QRC beats an ESN of size N"). The returned
    metrics bundle adds ``nmse_paper`` (Σ(y-ŷ)²/Σy²). Set
    ``radius_method="power"`` for large reservoirs where a full eig is
    infeasible (see :func:`esn_sweep`).
    """
    tr_cfg = tr_cfg or TrainingConfig()
    rng = np.random.default_rng(seed)
    n = len(u)
    Win = rng.uniform(-0.5, 0.5, size=(n_reservoir, 1))
    W = rng.uniform(-0.5, 0.5, size=(n_reservoir, n_reservoir))
    radius = _spectral_radius(W, radius_method, seed)
    if radius > 0:
        W *= spectral_radius / radius

    x = np.zeros(n_reservoir)
    X = np.zeros((n, n_reservoir))
    for k in range(n):
        pre = np.tanh(Win[:, 0] * u[k] + W @ x)
        x = (1 - leak) * x + leak * pre
        X[k] = x

    Xtr, ytr, Xte, yte = _split(X, y, tr_cfg.washout, tr_cfg.n_train, tr_cfg.n_test)
    model = train_readout(Xtr, ytr, tr_cfg)
    metrics = evaluate(model, Xte, yte)
    metrics["nmse_paper"] = nmse_paper(yte, model.predict(Xte))
    metrics["n_reservoir"] = n_reservoir
    return metrics


def esn_sweep(
    u: np.ndarray,
    y: np.ndarray,
    sizes: tuple[int, ...] = (500, 1000, 5000, 10000),
    tr_cfg: TrainingConfig | None = None,
    seed: int = 0,
) -> dict[int, dict]:
    """Run :func:`esn_baseline` at several reservoir sizes (plan §4).

    Returns ``{size: metrics}``. Large sizes automatically switch to the
    power-iteration spectral-radius estimate (a full eig at N=10000 is
    infeasible); the ≤2000 sizes keep the exact eig.
    """
    tr_cfg = tr_cfg or TrainingConfig()
    out: dict[int, dict] = {}
    for n_res in sizes:
        method = "exact" if n_res <= 2000 else "power"
        out[int(n_res)] = esn_baseline(
            u, y, n_reservoir=int(n_res), tr_cfg=tr_cfg, seed=seed,
            radius_method=method,
        )
    return out


def svr_readout(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    C: float = 10.0,
    gamma: str | float = "scale",
    epsilon: float = 1e-4,
) -> dict | None:
    """Optional RBF-SVR nonlinear readout ("QRC+RBF", plan §4).

    Z-score standardises the design matrix on the train split (matching
    the paper's preprocessing) and fits an RBF-kernel SVR. Returns the
    standard metric bundle plus ``nmse_paper``, or ``None`` if scikit-learn
    isn't installed (guarded, per the optional-dependency pattern).
    """
    try:
        from sklearn.svm import SVR
    except ImportError:
        return None

    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    sigma[sigma == 0] = 1.0
    Xtr = (X_train - mu) / sigma
    Xte = (X_test - mu) / sigma

    model = SVR(kernel="rbf", C=C, gamma=gamma, epsilon=epsilon)
    model.fit(Xtr, np.asarray(y_train, dtype=float).ravel())
    pred = model.predict(Xte)
    return {
        "r2": r2_score(y_test, pred),
        "corr2": squared_correlation(y_test, pred),
        "nmse": nmse(y_test, pred),
        "nmse_paper": nmse_paper(y_test, pred),
        "readout": "rbf_svr",
    }


# ---------------------------------------------------------------------------
# Scaling study — THE proof-of-benefit deliverable
# ---------------------------------------------------------------------------


@dataclass
class ScalingResult:
    qubit_counts: list[int]
    memory_capacity: list[float]
    narma_nmse: list[float]
    mc_gate_passed: list[bool]
    # Classical ESN NARMA NMSE at a *matched* reservoir size (empty when
    # the ESN comparison is disabled). Lower is better; QRC beating ESN at
    # equal readout dimension is the strongest form of "quantum benefit".
    esn_narma_nmse: list[float] = field(default_factory=list)
    details: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        has_esn = len(self.esn_narma_nmse) == len(self.qubit_counts)
        lines = ["N-qubit scaling study (QRC benefit vs system size)", ""]
        if has_esn:
            lines.append(
                f"{'N':>3} | {'MemCap':>8} | {'QRC NMSE':>11} | "
                f"{'ESN NMSE':>11} | {'winner':>7} | gate"
            )
            lines.append("-" * 62)
        else:
            lines.append(f"{'N':>3} | {'MemCap':>8} | {'NARMA NMSE':>12} | gate")
            lines.append("-" * 40)
        for i, (n, mc, nm, g) in enumerate(zip(
            self.qubit_counts, self.memory_capacity, self.narma_nmse,
            self.mc_gate_passed, strict=True,
        )):
            if has_esn:
                e = self.esn_narma_nmse[i]
                winner = "QRC" if nm < e else "ESN"
                lines.append(
                    f"{n:>3} | {mc:8.3f} | {nm:11.3e} | {e:11.3e} | "
                    f"{winner:>7} | {'ok' if g else 'FAIL'}"
                )
            else:
                lines.append(
                    f"{n:>3} | {mc:8.3f} | {nm:12.3e} | {'ok' if g else 'FAIL'}"
                )
        lines.append("")
        if len(self.memory_capacity) >= 2:
            grew = self.memory_capacity[-1] > self.memory_capacity[0]
            lines.append(
                "Memory capacity "
                + ("INCREASES" if grew else "does NOT increase")
                + f" with N: {self.memory_capacity[0]:.2f} (N={self.qubit_counts[0]})"
                + f" -> {self.memory_capacity[-1]:.2f} (N={self.qubit_counts[-1]})"
            )
        if has_esn:
            wins = sum(q < e for q, e in zip(self.narma_nmse, self.esn_narma_nmse,
                                             strict=True))
            lines.append(
                f"QRC beats a size-matched classical ESN on NARMA at "
                f"{wins}/{len(self.qubit_counts)} system sizes."
            )
        return "\n".join(lines)


def scaling_study(
    qubit_counts=(3, 5, 7, 9),
    *,
    sim: SimConfig | None = None,
    encoding: EncodingConfig | None = None,
    training: TrainingConfig | None = None,
    feature_cfg: FeatureConfig | None = None,
    max_delay: int = 30,
    narma_order: int = 2,
    mc_steps: int = 1200,
    compare_esn: bool = True,
    verbose: bool = True,
) -> ScalingResult:
    """Sweep qubit count and record memory capacity + NARMA accuracy.

    For each N we build a generic N-spin molecule (:func:`make_system_config`)
    with identical relaxation/encoding/training settings, so the *only*
    thing changing is the reservoir's Hilbert-space dimension. Rising
    memory capacity and falling NARMA NMSE across N is the quantitative
    evidence that more qubits deliver more computational power.

    When ``compare_esn`` is set, each N is also run against a classical
    Echo State Network whose reservoir size is *matched* to the quantum
    readout dimension (``len(observables) · N · V``), so any QRC advantage
    isn't just "more features". This is the QRC-vs-classical comparison
    Paper 4 reports.

    Note: theory says memory capacity is bounded by the number of *linearly
    independent observables* the readout sees, which grows with N (and with
    the number of virtual nodes V). This sweep measures how close the real
    dissipative dynamics get to that bound.
    """
    mcs, nmses, gates, esn_nmses, details = [], [], [], [], []
    ns = list(qubit_counts)
    for n in ns:
        cfg = QRCConfig(
            system=make_system_config(n),
            sim=sim or SimConfig(),
            encoding=encoding or EncodingConfig(),
            training=training or TrainingConfig(),
        )
        if verbose:
            print(f"[scaling] N={n} (dim={2**n}) — memory capacity …", flush=True)
        mc = run_memory_capacity(cfg, max_delay=max_delay, n_steps=mc_steps,
                                 feature_cfg=feature_cfg)
        if verbose:
            print(f"[scaling] N={n} — NARMA{narma_order} …", flush=True)
        na = run_narma(cfg, order=narma_order, feature_cfg=feature_cfg)

        esn_metrics = None
        if compare_esn:
            # Match the ESN reservoir size to the QRC observable-readout
            # dimension: n_axes · n_qubits · V.
            n_axes = len((feature_cfg or FeatureConfig()).observables)
            n_res = n_axes * n * cfg.sim.n_virtual
            n_steps = (cfg.training.washout + cfg.training.n_train
                       + cfg.training.n_test)
            u, y = narma_sequence(n_steps, narma_order, seed=cfg.sim.seed)
            esn_metrics = esn_baseline(
                u, y, n_reservoir=n_res, tr_cfg=cfg.training, seed=cfg.sim.seed
            )
            esn_nmses.append(esn_metrics["nmse"])

        mcs.append(mc.total_mc)
        nmses.append(na.metrics["nmse"])
        gates.append(mc.passed_gate)
        details.append(
            {
                "n_qubits": n,
                "hilbert_dim": 2**n,
                "memory_capacity": mc.total_mc,
                "mc_gate_passed": mc.passed_gate,
                "mc_per_delay": mc.per_delay,
                "narma_metrics": na.metrics,
                "esn_metrics": esn_metrics,
                "backend": mc.backend,
            }
        )
        if verbose:
            extra = (f" ESN NMSE={esn_metrics['nmse']:.3e}"
                     if esn_metrics else "")
            print(
                f"[scaling] N={n}: MC={mc.total_mc:.3f} "
                f"NARMA{narma_order} NMSE={na.metrics['nmse']:.3e}{extra} "
                f"gate={'ok' if mc.passed_gate else 'FAIL'}",
                flush=True,
            )
    return ScalingResult(ns, mcs, nmses, gates, esn_nmses, details)


# ---------------------------------------------------------------------------
# Weather forecasting (plan §10.2 / Paper 4) — the quantum-advantage task
# ---------------------------------------------------------------------------


def load_weather_series(
    train_csv: str,
    test_csv: str | None = None,
    columns=("meantemp", "humidity"),
):
    """Load (+ optionally concatenate) the Delhi climate CSV(s) and min-max
    normalize each column to [0, 1].

    Returns ``(norm, scalers)`` where ``norm`` is ``(n_days, n_vars)`` in
    [0,1] and ``scalers`` is a list of ``(lo, hi)`` per column for
    denormalization. Concatenating train+test gives the ~1576-day series
    the paper's 374/600/600 split needs.
    """
    import pandas as pd

    frames = [pd.read_csv(train_csv)]
    if test_csv:
        frames.append(pd.read_csv(test_csv))
    df = pd.concat(frames, ignore_index=True)
    cols = []
    scalers = []
    for c in columns:
        s, lo, hi = normalize(df[c].to_numpy(dtype=float))
        cols.append(s)
        scalers.append((lo, hi))
    return np.column_stack(cols), scalers


def _weather_input(weather_norm, n_steps, n_qubits, proton_idx, carbon_idx):
    """Build the (n_steps, n_qubits) reservoir drive: temperature (col 0) as a
    global rotation on the proton spins, humidity (col 1) on the carbon spins
    (Paper 4's multivariate encoding)."""
    seq = np.zeros((n_steps, n_qubits))
    seq[:, proton_idx] = weather_norm[:n_steps, 0:1]      # temp -> protons
    seq[:, carbon_idx] = weather_norm[:n_steps, 1:2]      # humidity -> carbons
    return seq


def run_weather_reservoir(
    cfg: QRCConfig,
    weather_norm: np.ndarray,
    n_steps: int,
    feature_cfg: FeatureConfig | None = None,
    proton_idx=(4, 5, 6, 7, 8),
    carbon_idx=(0, 1, 2, 3),
    progress_cb=None,
) -> np.ndarray:
    """One reservoir pass over the weather series → readout matrix X.

    The (expensive) FID reservoir is run once; the caller fits cheap
    per-horizon, per-variable readouts on the returned X (multitasking)."""
    system = QRCSystem(cfg.system, cfg.sim)
    encoder = Encoder(system, cfg.encoding)
    res = Reservoir(system, encoder, feature_cfg)
    seq = _weather_input(weather_norm, n_steps, system.n,
                         list(proton_idx), list(carbon_idx))
    out = res.run(seq, progress_cb=progress_cb)
    return out.X


def forecast_from_X(
    X: np.ndarray,
    weather_norm: np.ndarray,
    horizons,
    tr_cfg: TrainingConfig,
    var_names=("temp", "humidity"),
    use_rbf: bool = False,
) -> dict:
    """Fit ridge (and optional RBF-SVR) readouts predicting ``weather_norm``
    at each forecast horizon from the reservoir features ``X`` (row k → day
    k+h). Returns ``{h: {var: {"r2":.., "rbf_r2":..}}}`` on the test block."""
    n_steps = X.shape[0]
    out: dict = {}
    for h in horizons:
        per_var: dict = {}
        for j, name in enumerate(var_names):
            y = weather_norm[np.arange(n_steps) + h, j]
            Xtr, ytr, Xte, yte = _split(
                X, y, tr_cfg.washout, tr_cfg.n_train, tr_cfg.n_test
            )
            model = train_readout(Xtr, ytr, tr_cfg)
            rec = {"r2": r2_score(yte, model.predict(Xte))}
            if use_rbf:
                rbf = svr_readout(Xtr, ytr, Xte, yte)
                rec["rbf_r2"] = rbf["r2"] if rbf else None
            per_var[name] = rec
        out[h] = per_var
    return out


def esn_weather_sweep(
    weather_norm: np.ndarray,
    horizons,
    sizes=(500, 1000, 5000, 10000),
    tr_cfg: TrainingConfig | None = None,
    var_names=("temp", "humidity"),
    seed: int = 0,
) -> dict:
    """Classical multivariate-ESN baseline for weather forecasting.

    A leaky ESN driven by the 2-D weather series; per size, fits a ridge
    readout per variable per horizon. Returns
    ``{size: {h: {var: r2}}}``. This is the Paper-4 comparison the QRC's
    advantage claim rests on."""
    tr_cfg = tr_cfg or TrainingConfig()
    rng = np.random.default_rng(seed)
    n_in = weather_norm.shape[1]
    n_steps = weather_norm.shape[0] - max(horizons)
    u = weather_norm[:n_steps]
    result: dict = {}
    for m in sizes:
        Win = rng.uniform(-0.5, 0.5, size=(m, n_in))
        W = rng.uniform(-0.5, 0.5, size=(m, m))
        radius = _spectral_radius(W, "power" if m > 2000 else "exact", seed)
        if radius > 0:
            W *= 0.9 / radius
        x = np.zeros(m)
        X = np.zeros((n_steps, m))
        for k in range(n_steps):
            x = np.tanh(Win @ u[k] + W @ x)
            X[k] = x
        per_h: dict = {}
        for h in horizons:
            per_var = {}
            for j, name in enumerate(var_names):
                y = weather_norm[np.arange(n_steps) + h, j]
                Xtr, ytr, Xte, yte = _split(
                    X, y, tr_cfg.washout, tr_cfg.n_train, tr_cfg.n_test
                )
                model = train_readout(Xtr, ytr, tr_cfg)
                per_var[name] = r2_score(yte, model.predict(Xte))
            per_h[h] = per_var
        result[m] = per_h
    return result
