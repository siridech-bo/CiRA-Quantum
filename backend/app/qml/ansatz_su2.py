"""Pure-numpy Riemannian SU(2) primitives for an alternate VQC ansatz.

Library-only. No PennyLane / autograd / Flask imports — these are the
manifold-bookkeeping building blocks. The trainer wiring lives in a
follow-up file once these primitives are validated end-to-end.

The math is the standard setup for optimization on the unitary group
U(n) (Edelman, Arias & Smith 1998), recently revisited in the
quantum-circuit-compilation context by Guo & Yang 2025
(arXiv:2501.07387v2):

    Manifold:     U(n) = { U ∈ ℂ^{n×n} : U†U = I }
    Tangent at U: T_U U(n) = { U · A : A skew-Hermitian }
    Projection:   π_U(G) = U · skew(U†G),  skew(M) = ½(M − M†)
    Retraction:   r_U(ξ) = polar(U − ξ),   ξ a tangent vector

For SU(2) (n = 2) the per-gate cost is dominated by a 2×2 SVD, so the
overall optimizer is bottlenecked by loss-and-gradient evaluation, not
by the manifold bookkeeping.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "riemannian_grad",
    "retract_polar",
    "is_unitary",
    "random_u2",
    "init_near_identity",
    "riemannian_adam_step",
]


def riemannian_grad(U: np.ndarray, grad_eucl: np.ndarray) -> np.ndarray:
    """Project ``grad_eucl`` onto the tangent space of U(n) at ``U``.

    Returns ``U · skew(U†G)`` where ``skew(M) = ½(M − M†)``. The output
    is a tangent vector at ``U``, which is equivalent to saying that
    ``output @ U.conj().T`` is skew-Hermitian.
    """
    if U.shape != grad_eucl.shape:
        raise ValueError(
            f"U and grad shape mismatch: {U.shape} vs {grad_eucl.shape}"
        )
    A = U.conj().T @ grad_eucl
    skew = 0.5 * (A - A.conj().T)
    return U @ skew


def retract_polar(U: np.ndarray, tangent_step: np.ndarray) -> np.ndarray:
    """Move from ``U`` along ``-tangent_step``, then project back onto U(n).

    Polar retraction: take the unitary factor of the polar decomposition
    of ``U − tangent_step``. We compute it via SVD because that's the
    cheapest numerically-stable route for small (2×2) matrices, and the
    SVD's ``Q · V†`` *is* the polar unitary factor.

    Sign convention matches plain gradient descent: pass
    ``tangent_step = lr * riemannian_grad(U, grad)`` and the result will
    have moved in the descent direction.
    """
    Q, _, Vh = np.linalg.svd(U - tangent_step, full_matrices=False)
    return Q @ Vh


def is_unitary(U: np.ndarray, atol: float = 1e-10) -> bool:
    """Cheap check for ``U†U ≈ I``. Used in tests, not on hot paths."""
    n = U.shape[0]
    return np.allclose(U.conj().T @ U, np.eye(n), atol=atol)


def riemannian_adam_step(
    U: np.ndarray,
    grad_eucl: np.ndarray,
    *,
    m: np.ndarray,
    v: float,
    t: int,
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, float]:
    """One Riemannian Adam step on the unitary manifold.

    Follows Becigneul & Ganea 2018, "Riemannian Adaptive Optimization
    Methods" with two cheap choices:

    * **Vector transport via re-projection.** The incoming momentum
      buffer ``m`` lives in the tangent space at the *previous*
      iterate. We move it to the tangent space at ``U`` (the current
      iterate) by re-projecting: ``π_U(m) = U·skew(U†·m)``. This is
      not a true Levi-Civita parallel transport but it's a valid
      retraction-compatible vector transport, and on SU(2) it costs
      one 2×2 matmul.
    * **Scalar second moment.** ``v`` tracks ``β₂ · v_prev + (1−β₂) ·
      ‖g‖_F²``. Being a real number, it doesn't need transport. The
      B&G paper proves this scalar variant retains Adam's adaptive
      step-size property on retraction-compatible metrics.

    Caller is responsible for any conjugation on ``grad_eucl`` (e.g.,
    PennyLane autograd needs ``.conj()`` first — see the gotcha in
    docs/research_notes/2501.07387_*).

    Returns ``(U_new, m_new, v_new)``. ``m_new`` lives in ``T_U U(n)`` —
    on the next call, pass it as ``m`` along with the new ``U`` and the
    transport step will move it forward.
    """
    # Step 1 — transport the incoming momentum into the current tangent
    # space. At t=1 this is a no-op because m starts as zeros.
    m_carry = riemannian_grad(U, m)
    # Step 2 — Riemannian gradient at U.
    g = riemannian_grad(U, grad_eucl)
    # Step 3 — blend in T_U.
    m_new = beta1 * m_carry + (1 - beta1) * g
    # Step 4 — scalar second moment.
    g_norm_sq = float(np.real(np.sum(g.conj() * g)))
    v_new = beta2 * v + (1 - beta2) * g_norm_sq
    # Step 5 — Adam bias correction and step.
    m_hat = m_new / (1 - beta1 ** t)
    v_hat = v_new / (1 - beta2 ** t)
    step = lr * m_hat / (np.sqrt(v_hat) + eps)
    # Step 6 — retract back to U(n).
    U_new = retract_polar(U, step)
    return U_new, m_new, v_new


def init_near_identity(
    rng: np.random.Generator, scale: float = 0.1
) -> np.ndarray:
    """Sample a 2×2 unitary close to the identity.

    Construction: U = exp(i·scale·H) where H is a random Hermitian with
    Frobenius norm ~ 1. As ``scale → 0`` the result tends to I; as it
    grows, the distribution widens but is no longer Haar-uniform.

    Used as the default initialization for the per-qubit SU(2) bricks
    in the VQC trainer. Haar-random init (``random_u2``) gives a much
    more scrambled starting state and tends to look like the barren-
    plateau regime even at 2 qubits, which makes gradient descent
    finicky. A near-identity init effectively warm-starts at "do
    nothing" and lets the optimizer discover useful rotations from
    there. ``scale ≈ 0.1`` works well at the n ≤ 6-qubit scales we
    train at; larger systems may want smaller scales.
    """
    A = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
    H = scale * 0.5 * (A + A.conj().T)  # Hermitian
    eigvals, eigvecs = np.linalg.eigh(H)
    return eigvecs @ np.diag(np.exp(1j * eigvals)) @ eigvecs.conj().T


def random_u2(rng: np.random.Generator) -> np.ndarray:
    """Sample a Haar-random 2×2 unitary.

    Standard recipe: QR-decompose a complex-Gaussian matrix, then
    rescale Q's columns by the sign of the diagonal of R so the
    distribution is uniform on U(2) (Mezzadri 2007). The phase is
    *not* fixed, so the determinant has unit modulus but arbitrary
    argument — i.e. this is U(2), not SU(2). For optimization purposes
    a free global phase is harmless and we don't bother projecting.
    """
    Z = (rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))) / np.sqrt(2)
    Q, R = np.linalg.qr(Z)
    d = np.diag(R)
    Q = Q * (d / np.abs(d))
    return Q
