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
)
from app.qrc.training import evaluate, train_readout
from app.qrc.utils import squared_correlation

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
) -> NarmaResult:
    tr_cfg = cfg.training
    if n_steps is None:
        n_steps = tr_cfg.washout + tr_cfg.n_train + tr_cfg.n_test
    res = build_reservoir(cfg, feature_cfg)
    u, y = narma_sequence(n_steps, order, seed=cfg.sim.seed)
    out = res.run(u)
    Xtr, ytr, Xte, yte = _split(
        out.X, y, tr_cfg.washout, tr_cfg.n_train, tr_cfg.n_test
    )
    model = train_readout(Xtr, ytr, tr_cfg)
    return NarmaResult(order=order, metrics=evaluate(model, Xte, yte), n_qubits=out.n_qubits)


# ---------------------------------------------------------------------------
# Classical ESN baseline (plan §10.2)
# ---------------------------------------------------------------------------


def esn_baseline(
    u: np.ndarray,
    y: np.ndarray,
    n_reservoir: int = 100,
    spectral_radius: float = 0.9,
    leak: float = 1.0,
    tr_cfg: TrainingConfig | None = None,
    seed: int = 0,
) -> dict:
    """A standard leaky-integrator Echo State Network for comparison.

    Not quantum — a plain classical reservoir with a random recurrent
    matrix scaled to ``spectral_radius``. Gives the yardstick Paper 4
    compares its QRC against ("QRC beats an ESN of size N").
    """
    tr_cfg = tr_cfg or TrainingConfig()
    rng = np.random.default_rng(seed)
    n = len(u)
    Win = rng.uniform(-0.5, 0.5, size=(n_reservoir, 1))
    W = rng.uniform(-0.5, 0.5, size=(n_reservoir, n_reservoir))
    radius = np.max(np.abs(np.linalg.eigvals(W)))
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
    return evaluate(model, Xte, yte)


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
