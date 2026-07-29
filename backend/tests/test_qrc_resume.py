"""Crash-resume correctness for the trace-cache generator.

The full trace-gen streams each step's FID to an on-disk memmap and checkpoints
the GPU state every N steps, so a crash resumes from the last checkpoint instead
of losing hours of compute. The contract that matters: a run resumed from a
mid-way checkpoint must produce a trace **bit-identical** to an uninterrupted
one. (GPU-only: the resumable state lives on the ``fid_gpu`` path.)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("resume path is GPU-only (fid_gpu)", allow_module_level=True)

from app.qrc.config import QRCConfig, SimConfig, TrainingConfig, make_system_config
from app.qrc.encoding import Encoder
from app.qrc.evolution import Reservoir
from app.qrc.features import FeatureConfig
from app.qrc.system import QRCSystem
from qrc_gen_traces import StreamingTrace


def _reservoir():
    cfg = QRCConfig(
        system=make_system_config(6),
        sim=SimConfig(evolution_mode="gpu", fid_points=64, n_virtual=6, tau=0.01, seed=1),
        training=TrainingConfig(washout=5, n_train=15, n_test=10, seed=1),
    )
    system = QRCSystem(cfg.system, cfg.sim)
    return Reservoir(system, Encoder(system, cfg.encoding),
                     FeatureConfig(readout="fid", n_peaks=32))


def test_resume_is_bit_identical(tmp_path: Path):
    n_steps, fidp = 30, 64
    inputs = np.sin(np.arange(n_steps) * 0.7) * 0.5 + 0.5   # in [0, 1]

    # Reference: an uninterrupted streamed run.
    ref = StreamingTrace(tmp_path / "ref", n_steps, fidp, "H", total=n_steps, every=5)
    assert ref.try_resume() == (0, None)
    _reservoir().run(inputs, fid_cb=ref, checkpoint_cb=ref.checkpoint)
    fids_ref = ref.finalize()
    ref.cleanup()

    # Crash at step 17 (a checkpoint was written at step 14 since every=5).
    run = tmp_path / "run"
    chk1 = StreamingTrace(run, n_steps, fidp, "H", total=n_steps, every=5)
    s0, st0 = chk1.try_resume()

    def crashing(k, fid):
        chk1(k, fid)
        if k == 17:
            raise RuntimeError("simulated crash")

    with pytest.raises(RuntimeError):
        _reservoir().run(inputs, fid_cb=crashing, checkpoint_cb=chk1.checkpoint,
                         start_step=s0, state0=st0)
    # deliberately no cleanup — the checkpoint must survive for the resume

    # Resume: a fresh process re-launching the same config picks up mid-way.
    chk2 = StreamingTrace(run, n_steps, fidp, "H", total=n_steps, every=5)
    start_step, state0 = chk2.try_resume()
    assert start_step == 15, "should resume from the step-14 checkpoint"
    assert state0 is not None
    _reservoir().run(inputs, fid_cb=chk2, checkpoint_cb=chk2.checkpoint,
                     start_step=start_step, state0=state0)
    fids_res = chk2.finalize()
    chk2.cleanup()

    assert fids_res.shape == fids_ref.shape == (n_steps, fidp)
    assert np.array_equal(fids_ref, fids_res), "resumed trace must be bit-identical"


def test_wrong_config_hash_does_not_resume(tmp_path: Path):
    """A checkpoint from a different config must be ignored (no cross-config resume)."""
    n_steps, fidp = 12, 64   # fidp must match the reservoir's fid_points
    inputs = np.linspace(0, 1, n_steps)
    a = StreamingTrace(tmp_path / "d", n_steps, fidp, "HASH_A", total=n_steps, every=3)
    _reservoir().run(inputs, fid_cb=a, checkpoint_cb=a.checkpoint)
    # A different hash pointing at the same dir must not resume.
    b = StreamingTrace(tmp_path / "d", n_steps, fidp, "HASH_B", total=n_steps, every=3)
    assert b.try_resume() == (0, None)
    a.cleanup()
