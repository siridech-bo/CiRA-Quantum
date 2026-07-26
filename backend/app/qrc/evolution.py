"""QRC reservoir evolution — driving an input sequence through the spins.

This realizes the plan's six-step pipeline (§7) end to end for a whole
input sequence:

    for each input s_k:
        ρ ← U(s_k) ρ U(s_k)†          # step 1: encode (RF pulse)
        states, ρ ← multiplex(ρ)      # steps 2–3: evolve τ, sample V nodes
        x_k ← features(states)        # steps 4–5: observables / multi-modal

The reservoir state ρ **persists across steps** — that persistence, made
to *fade* by the Lindblad dissipation in :mod:`app.qrc.system`, is the
memory that makes reservoir computing work (plan §5). The output is a
design matrix ``X`` (n_steps × n_features) consumed by
:mod:`app.qrc.training`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.qrc.encoding import Encoder
from app.qrc.features import FeatureConfig, FeatureExtractor
from app.qrc.system import QRCSystem


@dataclass
class ReservoirOutput:
    """Result of driving one input sequence through the reservoir."""

    X: np.ndarray                 # (n_steps, n_features)
    feature_names: list[str]
    n_qubits: int
    n_virtual: int
    backend: str


class Reservoir:
    """A configured QRC reservoir ready to process input sequences."""

    def __init__(
        self,
        qrc_system: QRCSystem,
        encoder: Encoder,
        feature_cfg: FeatureConfig | None = None,
    ) -> None:
        self.sys = qrc_system
        self.encoder = encoder
        self.features = FeatureExtractor(qrc_system, feature_cfg or FeatureConfig())

    def run(
        self,
        inputs,
        rho0=None,
        progress: bool = False,
        progress_cb=None,
    ) -> ReservoirOutput:
        """Drive ``inputs`` through the reservoir.

        ``inputs`` is a 1-D sequence (single-channel) or a 2-D array of
        shape ``(n_steps, n_channels)`` for multi-nucleus parallel
        encoding, where ``n_channels`` must match the encoder's target
        qubit count. Returns a :class:`ReservoirOutput`.

        ``progress_cb``, if given, is called as ``progress_cb(done, total)``
        after each input step — used by the long-running reproduction runner
        to emit step-level progress + ETA to its event log.
        """
        arr = np.asarray(inputs, dtype=float)
        if arr.ndim == 1:
            seq = arr[:, None]
        else:
            seq = arr
        n_steps = seq.shape[0]

        rho = rho0 if rho0 is not None else self.sys.rho0
        rows: list[np.ndarray] = []
        names: list[str] | None = None

        # Fast path: observables-only features skip Qobj-state construction
        # and per-observable traces (critical at N≥8). The action backend
        # gets an extra turbo: the whole loop stays in vec space (no Qobj
        # density matrices at all), which is what makes N=9 tractable.
        fid_mode = self.features.cfg.readout == "fid"
        fast = (not fid_mode) and self.features.is_observables_only
        which = self.features.cfg.observables
        vec_loop = fast and self.sys.evolution_mode == "action"
        gpu_loop = fast and self.sys.evolution_mode == "gpu"
        # FID readout on the GPU backend: carry the state forward on the GPU
        # (reusing the exact Taylor stepper) then acquire the FID per step.
        fid_gpu = fid_mode and self.sys.evolution_mode == "gpu"
        if fast:
            names = self.features.observable_names()

        rho_mat = self.sys.init_mat(rho) if vec_loop else None
        vec_gpu = (
            self.sys.gpu_init(rho, which=which) if (gpu_loop or fid_gpu) else None
        )

        for k in range(n_steps):
            values = seq[k] if seq.shape[1] > 1 else seq[k, 0]
            if fid_gpu:
                # Reservoir update stays on the GPU; then read out the FID.
                u_mat = self.encoder.pulse_unitary(values).full()
                _, vec_gpu = self.sys.step_observables_gpu(
                    vec_gpu, u_mat, which=which
                )
                d = self.sys.dim
                rho_mat = vec_gpu.reshape(d, d).T.cpu().numpy()
                fid = self.sys.fid_signal(rho_mat)
                feats, fnames = self.features.from_fid(fid)
                if names is None:
                    names = fnames
            elif fid_mode:
                # Reservoir update (encode + free evolution τ), then FID readout.
                rho = self.encoder.apply(rho, values)
                _, rho = self.sys.multiplex(rho)
                fid = self.sys.fid_signal(rho)
                feats, fnames = self.features.from_fid(fid)
                if names is None:
                    names = fnames
            elif gpu_loop:
                u_mat = self.encoder.pulse_unitary(values).full()
                feats, vec_gpu = self.sys.step_observables_gpu(
                    vec_gpu, u_mat, which=which
                )
            elif vec_loop:
                u_mat = self.encoder.pulse_unitary(values).full()
                feats, rho_mat = self.sys.step_observables_mat(
                    rho_mat, u_mat, which=which
                )
            elif fast:
                rho = self.encoder.apply(rho, values)
                feats, rho = self.sys.multiplex_observables(rho, which=which)
            else:
                rho = self.encoder.apply(rho, values)
                states, rho = self.sys.multiplex(rho)
                feats, fnames = self.features.extract(states)
                if names is None:
                    names = fnames
            rows.append(feats)
            if progress and (k % max(1, n_steps // 10) == 0):
                print(f"  reservoir step {k}/{n_steps}", flush=True)
            if progress_cb is not None:
                progress_cb(k + 1, n_steps)

        X = np.vstack(rows)
        return ReservoirOutput(
            X=X,
            feature_names=names or [],
            n_qubits=self.sys.n,
            n_virtual=self.sys.sim.n_virtual,
            backend=self.sys.active_backend,
        )
