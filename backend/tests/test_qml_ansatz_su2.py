"""Tests for the SU(2) Riemannian manifold primitives.

These verify the *library-level* invariants of ``ansatz_su2``:

1. ``random_u2`` actually returns unitaries.
2. ``riemannian_grad(U, G)`` lands in the tangent space at ``U``.
3. ``retract_polar`` always returns something unitary.
4. The projected gradient agrees with a numerical-difference baseline
   on a known loss.
5. Riemannian gradient descent converges on a convex toy problem
   (Frobenius distance to a fixed target unitary).

The trainer-level wiring (per-qubit SU(2) bricks inside a VQC, with a
real classification dataset) is the next step and lives in a separate
test file once this layer is green.
"""
from __future__ import annotations

import numpy as np

from app.qml.ansatz_su2 import (
    is_unitary,
    random_u2,
    retract_polar,
    riemannian_adam_step,
    riemannian_grad,
)


def test_random_u2_is_unitary():
    rng = np.random.default_rng(0)
    for _ in range(16):
        U = random_u2(rng)
        assert U.shape == (2, 2)
        assert is_unitary(U), "random_u2 must produce a unitary matrix"


def test_riemannian_grad_is_in_tangent_space():
    """Tangent space of U(n) at U is { U·A : A skew-Hermitian }, so
    ``T @ U.conj().T`` must be skew-Hermitian for any T in the tangent
    space at U.
    """
    rng = np.random.default_rng(1)
    for _ in range(8):
        U = random_u2(rng)
        G = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
        T = riemannian_grad(U, G)
        M = T @ U.conj().T
        assert np.allclose(M + M.conj().T, 0, atol=1e-10), (
            "T·U† must be skew-Hermitian for T in T_U U(n)"
        )


def test_retract_polar_returns_unitary():
    """Polar retraction must land on U(n) regardless of step size,
    including absurdly large steps where the linearized update would
    leave the manifold by a wide margin.
    """
    rng = np.random.default_rng(2)
    U = random_u2(rng)
    G = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
    T = riemannian_grad(U, G)
    for lr in (1e-3, 1e-1, 1.0, 10.0):
        U_next = retract_polar(U, lr * T)
        assert is_unitary(U_next), f"retract_polar broke unitarity at lr={lr}"


def test_riemannian_grad_matches_numerical_on_frobenius_loss():
    """For f(U) = ½‖U − V‖_F², the Euclidean gradient is U − V. The
    Riemannian gradient should equal U · skew(U†(U − V)). We verify
    against a numerical-difference reading along a known tangent
    direction.
    """
    rng = np.random.default_rng(3)
    U = random_u2(rng)
    V = random_u2(rng)
    G_eucl = U - V
    T_grad = riemannian_grad(U, G_eucl)

    eps = 1e-6
    A = U.conj().T @ T_grad  # skew-Hermitian
    U_plus = retract_polar(U, -eps * T_grad)   # move along +T_grad
    U_minus = retract_polar(U, +eps * T_grad)  # move along -T_grad
    f_plus = 0.5 * np.linalg.norm(U_plus - V, "fro") ** 2
    f_minus = 0.5 * np.linalg.norm(U_minus - V, "fro") ** 2

    # Directional derivative along T_grad direction:
    # d/dε f(retract(U, -ε·T))|_{ε=0} = ⟨G_eucl, T⟩_Re = Re tr(G† T)
    expected = float(np.real(np.trace(G_eucl.conj().T @ T_grad)))
    numerical = (f_plus - f_minus) / (2 * eps)
    assert np.isclose(numerical, expected, rtol=1e-3, atol=1e-6), (
        f"directional derivative mismatch: numerical={numerical}, "
        f"expected={expected}"
    )
    # Bonus sanity: the skew-Hermitian assertion underlying the formula.
    assert np.allclose(A + A.conj().T, 0, atol=1e-10)


def test_riemannian_descent_converges_to_target():
    """Minimise ½‖U − V‖_F² over U ∈ U(2) starting from a random U.

    The global minimiser is V itself (loss = 0). With a reasonable step
    size, Riemannian gradient descent should drive the loss to ≪ initial
    within a few hundred steps. We assert a generous bound so this test
    doesn't go flaky if numpy's BLAS changes phase conventions.
    """
    rng = np.random.default_rng(4)
    V = random_u2(rng)
    U = random_u2(rng)
    initial_loss = 0.5 * np.linalg.norm(U - V, "fro") ** 2
    assert initial_loss > 1e-3, "test setup: U and V should not coincide"

    lr = 0.2
    for _ in range(400):
        G_eucl = U - V
        T = riemannian_grad(U, G_eucl)
        U = retract_polar(U, lr * T)
        assert is_unitary(U, atol=1e-9), "iterate left the manifold"

    final_loss = 0.5 * np.linalg.norm(U - V, "fro") ** 2
    assert final_loss < 1e-4, (
        f"descent did not converge: initial={initial_loss:.4f}, "
        f"final={final_loss:.6f}"
    )


def test_riemannian_adam_step_preserves_unitarity():
    """Adam with vector transport must still land on U(n) at every step,
    even with the bias-corrected step sizes that are larger than plain
    SGD's at the same nominal lr.
    """
    rng = np.random.default_rng(5)
    U = random_u2(rng)
    V = random_u2(rng)
    m = np.zeros_like(U)
    v = 0.0
    for t in range(1, 50):
        G_eucl = U - V
        U, m, v = riemannian_adam_step(U, G_eucl, m=m, v=v, t=t, lr=0.05)
        assert is_unitary(U, atol=1e-9), f"Adam left manifold at step {t}"


def test_riemannian_adam_descent_converges_on_frobenius_loss():
    """Adam should also drive the Frobenius distance to a target unitary
    to near-zero — same toy problem as vanilla SGD, just adaptive steps.
    """
    rng = np.random.default_rng(6)
    V = random_u2(rng)
    U = random_u2(rng)
    initial = 0.5 * np.linalg.norm(U - V, "fro") ** 2
    m = np.zeros_like(U)
    v = 0.0
    for t in range(1, 400):
        G_eucl = U - V
        U, m, v = riemannian_adam_step(U, G_eucl, m=m, v=v, t=t, lr=0.05)
    final = 0.5 * np.linalg.norm(U - V, "fro") ** 2
    assert final < 1e-3, (
        f"Adam did not converge: initial={initial:.4f}, final={final:.6f}"
    )


def test_riemannian_adam_momentum_carries_across_steps():
    """The first-moment buffer should be non-zero after the first step
    and stay in the tangent space at the current iterate (modulo the
    re-projection that happens on the next call).
    """
    rng = np.random.default_rng(7)
    U = random_u2(rng)
    V = random_u2(rng)
    m = np.zeros_like(U)
    v = 0.0
    U1, m1, v1 = riemannian_adam_step(U, U - V, m=m, v=v, t=1, lr=0.05)
    assert not np.allclose(m1, 0), "Adam momentum should be non-zero after step 1"
    assert v1 > 0, "Adam second moment should be positive after step 1"
    # m1 lives in T_U U(n), i.e. m1 @ U.conj().T should be skew-Hermitian.
    M = m1 @ U.conj().T
    assert np.allclose(M + M.conj().T, 0, atol=1e-10), (
        "Adam momentum drifted out of tangent space"
    )


def test_riemannian_grad_shape_mismatch_raises():
    U = np.eye(2, dtype=complex)
    bad = np.zeros((3, 3), dtype=complex)
    try:
        riemannian_grad(U, bad)
    except ValueError as e:
        assert "shape mismatch" in str(e)
    else:
        raise AssertionError("expected ValueError on shape mismatch")
