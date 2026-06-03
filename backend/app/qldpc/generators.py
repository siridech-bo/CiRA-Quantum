"""qLDPC code-family factory functions (Sprint 1).

Each factory wraps one constructor from the ``qldpc`` PyPI package
(``qLDPCOrg/qLDPC``) with a parameter set chosen to land at the
⟦n, k, d⟧ values the gallery advertises. The parameters were picked
to be recognizable to working QEC researchers:

* Surface code  — Kitaev's rotated d=13 patch                    → ⟦169, 1, 13⟧
* Toric code    — unrotated 10×10                                → ⟦200, 2, 10⟧
* Hypergraph product — HGP of the classical Hamming(7,4) code    → ⟦58, 16, 3⟧
* Bicycle (BB)  — IBM "gross code" parameters (Bravyi et al.)    → ⟦144, 12, 12⟧

``qldpc`` is imported lazily inside each factory so that Sprint 0's
metadata-only endpoints keep working when the lib isn't installed.
The capability check at ``/api/qldpc/health`` is the gate the routes
use to decide whether to call into here.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _hamming_7_4_parity_check() -> np.ndarray:
    """Standard (7,4) Hamming code parity-check matrix.

    Three checks over seven bits — the textbook classical LDPC building
    block. We use this as the base classical code for the HGP family.
    Returned as a plain numpy int array so callers can construct an
    ``HGPCode`` without taking a dependency on the qldpc ``ClassicalCode``
    type at import time.
    """
    return np.array(
        [
            [1, 0, 1, 0, 1, 0, 1],
            [0, 1, 1, 0, 0, 1, 1],
            [0, 0, 0, 1, 1, 1, 1],
        ],
        dtype=int,
    )


def build_surface() -> Any:
    """Return a rotated d=13 Surface code (⟦169, 1, 13⟧).

    Rotated boundary saves a factor of 2× in physical qubits versus the
    unrotated lattice while preserving the same logical-qubit count and
    distance. This is the canonical reference patch used in nearly every
    superconducting-qubit roadmap.
    """
    from qldpc.codes import SurfaceCode
    return SurfaceCode(13)


def build_toric() -> Any:
    """Return an unrotated 10×10 Toric code (⟦200, 2, 10⟧).

    Unrotated so that ``n = 2·L²`` and the two non-contractible loops of
    the torus give two independent logical qubits (a key distinguishing
    feature versus the surface code's k=1). 10×10 hits the ``n = 200``
    target the gallery advertises.
    """
    from qldpc.codes import ToricCode
    return ToricCode(10, 10, rotated=False)


def build_hypergraph_product() -> Any:
    """Return a hypergraph-product code from Hamming(7,4) (⟦58, 16, 3⟧).

    The HGP construction takes a classical LDPC code with parity-check
    matrix ``H`` (m×n) and produces a quantum code with ``n² + m²``
    physical qubits and ``(n−m)²`` logical qubits when ``H`` has full
    row rank. For Hamming(7,4): 49 + 9 = 58 physical, 16 logical.
    Distance is modest (3) — the gallery card calls out HGP as a
    teaching example of finite-rate codes, not a fault-tolerance
    candidate at these small sizes.
    """
    from qldpc.codes import HGPCode
    return HGPCode(_hamming_7_4_parity_check())


def build_bicycle() -> Any:
    """Return the IBM "gross code" bivariate-bicycle BB code (⟦144, 12, 12⟧).

    Parameters from Bravyi, Cross, Gambetta, Maslov, Rall & Yoder
    (Nature 2024, *High-threshold and low-overhead fault-tolerant
    quantum memory*). ``orders={x:12, y:6}`` defines the abelian group
    ``Z_12 × Z_6``; the two polynomials A(x,y) and B(x,y) define the
    circulant blocks. The result is the most famous modern finite-rate
    qLDPC code — 12 logical qubits, distance 12, on 144 physical qubits.
    Roughly 10× the encoding rate of the equivalent surface code.

    Note: ``BBCode`` is the modern *bivariate* bicycle — distinct from
    MacKay's original 2004 "bicycle" construction, which the upstream
    lib does not ship. We label this entry "Bicycle (BB)" in the
    gallery to be honest about which construction users get.
    """
    import sympy
    from qldpc.codes import BBCode
    x, y = sympy.symbols("x y")
    return BBCode(
        orders={x: 12, y: 6},
        poly_a=x**3 + y + y**2,
        poly_b=y**3 + x + x**2,
    )


# Map family id (matching code_families.py) to factory. Routes look up
# the factory here when serving the matrix/tanner/distance endpoints.
FACTORIES = {
    "surface": build_surface,
    "toric": build_toric,
    "hypergraph_product": build_hypergraph_product,
    "bicycle": build_bicycle,
}


def get_factory(family_id: str):
    """Return the factory function for a family, or ``None`` if unknown."""
    return FACTORIES.get(family_id)
