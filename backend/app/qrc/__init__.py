"""QRC (Quantum Reservoir Computing) sister app — package root.

Implements ``docs/QRC_Simulation_Plan.md``: a GPU-capable simulator for
Quantum Reservoir Computing on NMR spin networks, used to prototype
experiments before running on the SPINQ Gemini Lab and to make the
quantitative case for larger hardware.

Deliberately mirrors the QML/qLDPC layout. The quantum core depends on
QuTiP (optional extra ``pip install ".[qrc]"``); it's imported lazily so
this package — and the NumPy-only ``config``/``utils`` — load without it.

Layout (plan §12 module map):

* ``config``      — system presets (SPINQ-3, generic-N, crotonic-9) + knobs
* ``system``      — N-qubit Hamiltonian + Lindblad dynamics + propagators
* ``encoding``    — input → RF pulse (all encoding functions, multi-nucleus)
* ``evolution``   — reservoir stepping + temporal multiplexing
* ``features``    — multi-modal feature extraction
* ``training``    — ridge readout on GPU (torch)
* ``tasks``       — NARMA, memory capacity, weather
* ``benchmarks``  — orchestration, ESN baseline, N-qubit scaling study
* ``main``        — CLI entry point
"""

from app.qrc.config import (
    QRCConfig,
    SimConfig,
    SystemConfig,
    make_system_config,
)

__all__ = ["QRCConfig", "SimConfig", "SystemConfig", "make_system_config"]
