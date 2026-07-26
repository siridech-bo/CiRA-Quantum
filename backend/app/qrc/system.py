"""QRC quantum system — N-qubit NMR Hamiltonian + Lindblad dynamics.

This is the physics foundation. It builds, for an arbitrary ``n``-spin
weak-coupling NMR network:

* the rotating-frame Hamiltonian
  ``H = Σ_i π ν_i σz_i  +  Σ_{i<j} (π/2) J_ij σz_i σz_j``  (rad/s),
* the Lindblad collapse operators (T1 amplitude damping + T2 pure
  dephasing) per spin — **the source of fading memory** (plan §5.2.2),
* a thermal-like product initial state,
* and — the performance-critical bit — a *precomputed set of Liouvillian
  propagators* over the temporal-multiplexing grid, so a reservoir run
  over thousands of input steps reuses one matrix-exponential instead of
  re-integrating the master equation every step.

Backends
--------
QuTiP 5's pluggable data layer lets the same code run on CPU (NumPy) or
GPU (JAX via ``qutip-jax``). At ≤ ~9 qubits the Hilbert space is tiny
(512×512) and CPU wins — GPU only pays off for larger systems or big
batched sweeps (plan §2.3). The backend is selected by
``SimConfig.backend`` and degrades gracefully to NumPy if ``qutip-jax``
isn't importable.

QuTiP is an optional dependency (``pip install ".[qrc]"``). Importing
this module never imports QuTiP; it's imported lazily inside the class so
the package (and the lightweight utils/metrics) load without it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from app.qrc.config import SimConfig, SystemConfig


def _require_qutip():
    """Import QuTiP lazily with a helpful error if the extra is missing."""
    try:
        import qutip  # noqa: F401
    except ImportError as exc:  # pragma: no cover - env dependent
        raise ImportError(
            "QRC simulation needs QuTiP. Install the extra:\n"
            '    pip install ".[qrc]"\n'
            "(QuTiP 5.x + qutip-jax for the GPU backend)."
        ) from exc
    return qutip


def _maybe_enable_jax(backend: str) -> str:
    """Switch QuTiP's default data layer to JAX for GPU execution.

    Returns the backend actually in effect ("jax" or "numpy") so callers
    can log the truth rather than the request.
    """
    if backend != "jax":
        return "numpy"
    try:  # pragma: no cover - GPU/env dependent
        import qutip
        import qutip_jax  # noqa: F401  (registers the "jax"/"jaxdia" dtypes)

        qutip.settings.core["default_dtype"] = "jaxdia"
        return "jax"
    except Exception:  # pragma: no cover
        return "numpy"


class QRCSystem:
    """A configured N-spin reservoir: operators, dynamics, propagators."""

    def __init__(self, system: SystemConfig, sim: SimConfig) -> None:
        self.qt = _require_qutip()
        self.system = system
        self.sim = sim
        self.n = system.n_qubits
        self.dim = 2**self.n
        self.active_backend = _maybe_enable_jax(sim.backend)

        self._build_operators()
        self.H = self._build_hamiltonian()
        self.c_ops = self._build_collapse_ops()
        self.rho0 = self._build_initial_state()
        self.evolution_mode = self._resolve_evolution_mode(sim.evolution_mode)
        # Filled on first evolve() via _ensure_propagators() (propagator mode).
        self._props: list[Any] | None = None
        self._sub_times: np.ndarray | None = None
        # Sub-step grid: [0, t_1, ..., t_V=tau]; states at index 1: are V nodes.
        self._sub_tlist = np.linspace(0.0, sim.tau, sim.n_virtual + 1)
        # action-mode: sparse Liouvillian + observable row-vectors, lazy.
        self._L_sparse = None
        self._obs_rows_cache: dict[tuple, Any] = {}
        # gpu-mode: torch CUDA tensors, lazy.
        self._gpu = None
        # FID readout: pulse/observable/L_read caches, lazy.
        self._fid_cache = None
        self._fid_gpu = None

    def _resolve_evolution_mode(self, mode: str) -> str:
        """Pick the evolution strategy. ``auto`` uses the exact sparse
        ``action`` path from N ≥ 6, where it is both memory-lean and
        *faster* than the dense propagator (measured: N=6 propagator ≈
        13.5 min to precompute vs ``action`` < 1 min; the dense 4^N
        superoperator also balloons — ~0.27 GB at N=6, ~4.3 GB at N=7,
        ~550 GB at N=9). ``action`` applies exp(t·L) via Krylov (scipy
        ``expm_multiply``) on the *sparse* Liouvillian (~0.001% dense at
        N=9) — exact, so unlike an adaptive ODE (``mesolve``) it doesn't
        choke on the stiff mix of fast precession (~kHz) and slow
        relaxation (~Hz). Propagator stays the default only for N ≤ 5,
        where its one-time cost is negligible and per-step reuse is
        fastest."""
        if mode != "auto":
            return mode
        return "action" if self.n >= 6 else "propagator"

    # -- operator construction ------------------------------------------

    def _op_on(self, op, i: int):
        """Embed single-qubit ``op`` on spin ``i`` into the n-qubit space."""
        qt = self.qt
        ops = [qt.qeye(2) for _ in range(self.n)]
        ops[i] = op
        return qt.tensor(ops)

    def _build_operators(self) -> None:
        qt = self.qt
        self.sx = [self._op_on(qt.sigmax(), i) for i in range(self.n)]
        self.sy = [self._op_on(qt.sigmay(), i) for i in range(self.n)]
        self.sz = [self._op_on(qt.sigmaz(), i) for i in range(self.n)]
        self.sm = [self._op_on(qt.sigmam(), i) for i in range(self.n)]

    def _build_hamiltonian(self):
        """Rotating-frame NMR Hamiltonian in rad/s.

        Chemical-shift term ``Σ_i π ν_i σz_i`` (ν in Hz; the ½·2π folds
        into π since σz has eigenvalues ±1). Weak-coupling J term
        ``Σ_{i<j} (π/2) J_ij σz_i σz_j`` — matches the plan §6.1.2 form
        ``Σ π J σz σz / 2``.
        """
        H = 0
        nu = self.system.chemical_shifts
        for i in range(self.n):
            H += np.pi * nu[i] * self.sz[i]
        j = np.asarray(self.system.j_coupling, dtype=float)
        for i in range(self.n):
            for k in range(i + 1, self.n):
                if j[i, k] != 0.0:
                    H += (np.pi / 2.0) * j[i, k] * self.sz[i] * self.sz[k]
        return H

    def _build_collapse_ops(self) -> list:
        """T1 amplitude damping + T2 pure dephasing per spin (plan §6.2).

        * T1:  L = √(1/T1) · σ₋
        * T2 pure dephasing:
            - "physical":  γ_φ = 1/T2 − 1/(2 T1),  L = √(γ_φ/2) · σz
            - "simple":    L = √(1/(2 T2)) · σz     (plan §12 shortcut)

        The σz collapse operator contributes a decay rate 2·(coeff)² to
        off-diagonal coherences, hence the ½ inside the physical-model
        coefficient so that the total transverse rate reproduces 1/T2.
        """
        c_ops = []
        t1 = self.system.t1
        t2 = self.system.t2
        model = self.sim.dephasing_model
        for i in range(self.n):
            if t1[i] > 0:
                c_ops.append(np.sqrt(1.0 / t1[i]) * self.sm[i])
            if t2[i] > 0:
                if model == "physical":
                    gamma_phi = 1.0 / t2[i] - 1.0 / (2.0 * t1[i])
                    gamma_phi = max(gamma_phi, 0.0)
                    coeff = np.sqrt(gamma_phi / 2.0)
                else:  # "simple"
                    coeff = np.sqrt(1.0 / (2.0 * t2[i]))
                if coeff > 0:
                    c_ops.append(coeff * self.sz[i])
        return c_ops

    def _build_initial_state(self):
        """Thermal-like product state ρ0 = ⊗_i (I + ε σz)/2 (plan §6.4.1).

        High-temperature NMR limit: nearly maximally mixed with a small
        longitudinal polarization ε. Dynamics are linear in ρ, so ε only
        sets the overall signal scale.
        """
        qt = self.qt
        eps = self.sim.init_polarization
        singles = []
        for _ in range(self.n):
            rho_i = 0.5 * (qt.qeye(2) + eps * qt.sigmaz())
            singles.append(rho_i)
        return qt.tensor(singles)

    # -- dynamics -------------------------------------------------------

    def steady_state(self):
        """Lindblad steady state ρ_ss (validation: should be ~maximally
        mixed for these dissipators)."""
        return self.qt.steadystate(self.H, self.c_ops)

    def _ensure_propagators(self) -> None:
        """Precompute superoperator propagators U(t_j) from 0 → t_j over
        the temporal-multiplexing grid, once. Reused for every input step
        because H and c_ops are time-independent within a step.

        ``n_virtual`` sample points span (0, tau]; index -1 is the full
        step propagator that carries the reservoir state to the next step.
        """
        if self._props is not None:
            return
        qt = self.qt
        V = self.sim.n_virtual
        # Sample points within the step: exclude t=0 (pre-evolution),
        # include t=tau. propagator() needs t=0 as the first entry to
        # anchor the identity, so build [0, t_1, ..., t_V=tau] and drop 0.
        tlist = np.linspace(0.0, self.sim.tau, V + 1)
        props = qt.propagator(self.H, tlist, self.c_ops)
        # qutip.propagator returns a list (len == len(tlist)); [0] is I.
        self._props = list(props)[1:]        # V superoperators
        self._sub_times = tlist[1:]

    @property
    def sub_times(self) -> np.ndarray:
        if self.evolution_mode in ("mesolve", "action"):
            return self._sub_tlist[1:]
        self._ensure_propagators()
        assert self._sub_times is not None
        return self._sub_times

    def _apply_super(self, U, rho):
        """Apply a superoperator ``U`` to a density matrix ``rho``."""
        qt = self.qt
        return qt.vector_to_operator(U * qt.operator_to_vector(rho))

    def _ensure_liouvillian(self):
        """Build (once) the sparse Liouvillian for the ``action`` backend."""
        if self._L_sparse is None:
            L = self.qt.liouvillian(self.H, self.c_ops)
            self._L_sparse = L.data.as_scipy()      # scipy sparse, ~0.001% dense
        return self._L_sparse

    def _action_vecs(self, rho):
        """Exact Krylov action exp(t·L)·vec over the sub-step grid.

        Returns ``(out, dims)`` where ``out`` is the ``(V+1, dim²)`` array
        of vectorized states (column-stacking) and ``dims`` is the Qobj
        dims to rebuild density matrices if needed. Index 0 is t=0."""
        from scipy.sparse.linalg import expm_multiply

        L = self._ensure_liouvillian()
        vec = self.qt.operator_to_vector(rho).full().ravel()
        out = expm_multiply(
            L, vec, start=0.0, stop=self.sim.tau,
            num=self.sim.n_virtual + 1, endpoint=True,
        )
        return out, rho.dims

    def _vec_to_rho(self, vec, dims):
        return self.qt.Qobj(vec.reshape((self.dim, self.dim), order="F"), dims=dims)

    def _multiplex_action(self, rho):
        """Temporal multiplexing via the Krylov action, returning Qobj
        states (general interface; rebuilds a Qobj per node)."""
        out, dims = self._action_vecs(rho)
        states = [self._vec_to_rho(out[j], dims) for j in range(1, out.shape[0])]
        return states, states[-1]

    def _ensure_obs_rows(self, which: tuple):
        """Sparse ``(n_obs, dim²)`` matrix M whose rows are ``vec(O^T)`` so
        that ``M · vec(ρ)`` yields ``⟨O⟩`` for every observable at once.

        Identity ``Tr(Oρ) = vec(O^T)·vec(ρ)`` (column-stacking). Row order
        is state-then-axis-then-qubit-compatible: axis-major, qubit-inner,
        matching :meth:`expectations`. Cached per ``which``."""
        import scipy.sparse as sp

        key = tuple(which)
        if key in self._obs_rows_cache:
            return self._obs_rows_cache[key]
        axis_ops = {"x": self.sx, "y": self.sy, "z": self.sz}
        rows = []
        for a in which:
            for op in axis_ops[a]:
                rows.append(self.qt.operator_to_vector(op.trans()).full().ravel())
        M = sp.csr_matrix(np.asarray(rows))
        self._obs_rows_cache[key] = M
        return M

    def multiplex_observables(self, rho, which=("x", "y", "z")):
        """Fast path: evolve one step (action backend) and return
        ``(feature_vector, rho_end)`` computing observables directly from
        the vectorized states — no per-node Qobj, no per-observable trace.

        Feature order matches :meth:`expectations` (state-major, then axis,
        then qubit). Falls back to the Qobj path for non-action backends."""
        if self.evolution_mode != "action":
            states, rho_end = self.multiplex(rho)
            return self.expectations(states, which=which), rho_end
        out, dims = self._action_vecs(rho)
        M = self._ensure_obs_rows(which)
        vecs = out[1:].T                       # (dim², V)
        feats = np.asarray((M @ vecs).real)    # (n_obs, V)
        rho_end = self._vec_to_rho(out[-1], dims)
        # flatten state-major: for each node, all (axis,qubit) observables
        return feats.T.ravel(), rho_end

    # -- raw-numpy action hot loop (no Qobj density matrices) ------------

    def init_mat(self, rho) -> np.ndarray:
        """Density matrix as a plain ndarray, for the raw hot loop."""
        return rho.full()

    def step_observables_mat(self, rho_mat, U_mat, which=("x", "y", "z")):
        """One reservoir step in raw NumPy (action backend only).

        Pulse ``ρ → U ρ U†`` as a dense matmul, free evolution via the
        sparse Krylov action, observables via the sparse row-vector product
        — no QuTiP Qobj wrappers in the loop, which is what makes N=9
        tractable (the Qobj round-trips, not the math, were the overhead).
        Returns ``(feature_vector, rho_mat_end)``.

        Column-stacking (``order='F'``) matches the Liouvillian and the
        observable rows built by :meth:`_ensure_obs_rows`."""
        from scipy.sparse.linalg import expm_multiply

        L = self._ensure_liouvillian()
        rho_mat = U_mat @ rho_mat @ U_mat.conj().T
        vec = rho_mat.reshape(-1, order="F")
        out = expm_multiply(
            L, vec, start=0.0, stop=self.sim.tau,
            num=self.sim.n_virtual + 1, endpoint=True,
        )
        M = self._ensure_obs_rows(which)
        feats = np.asarray((M @ out[1:].T).real)      # (n_obs, V)
        rho_end = out[-1].reshape((self.dim, self.dim), order="F")
        return feats.T.ravel(), rho_end

    # -- GPU action hot loop (torch CUDA) --------------------------------

    def _ensure_gpu(self, which):
        """Build (once) the CUDA tensors for the ``gpu`` backend: the
        sparse Liouvillian, the observable row-matrix, and the sub-stepping
        schedule for the Taylor exp(t·L) action.

        The sub-step count is chosen so ``‖h·L‖ ≲ 1`` (from a one-norm
        estimate of ``tau·L``), which makes a fixed-order Taylor series
        converge to well under fp32 precision; it's rounded up to a
        multiple of V so the temporal-multiplexing nodes fall on sub-step
        boundaries."""
        import torch
        from scipy.sparse.linalg import onenormest

        if not torch.cuda.is_available():
            raise RuntimeError(
                "evolution_mode='gpu' needs a CUDA-enabled torch build. "
                "Use 'action' for the exact CPU path."
            )
        key = tuple(which)
        if self._gpu is not None and self._gpu["which"] == key:
            return self._gpu
        dev = "cuda"
        L = self._ensure_liouvillian().tocsr().astype(np.complex64)
        Lt = torch.sparse_csr_tensor(
            torch.tensor(L.indptr, dtype=torch.int64),
            torch.tensor(L.indices, dtype=torch.int64),
            torch.tensor(L.data), size=L.shape, device=dev,
        )
        import scipy.sparse as sp
        Mrows = sp.csr_matrix(self._ensure_obs_rows(which)).astype(np.complex64)
        Mt = torch.sparse_csr_tensor(
            torch.tensor(Mrows.indptr, dtype=torch.int64),
            torch.tensor(Mrows.indices, dtype=torch.int64),
            torch.tensor(Mrows.data), size=Mrows.shape, device=dev,
        )
        V = self.sim.n_virtual
        nrm = float(onenormest(self.sim.tau * self._ensure_liouvillian()))
        base = max(V, int(np.ceil(nrm)))
        substeps = V * int(np.ceil(base / V))     # multiple of V, ‖hL‖≲1
        self._gpu = {
            "which": key, "dev": dev, "L": Lt, "M": Mt,
            "substeps": substeps, "per": substeps // V,
            "h": self.sim.tau / substeps, "K": 18, "torch": torch,
        }
        return self._gpu

    def gpu_init(self, rho, which=("x", "y", "z")):
        """Initial column-stacked state vector on the GPU (complex64)."""
        g = self._ensure_gpu(which)
        vec = self.qt.operator_to_vector(rho).full().ravel().astype(np.complex64)
        return g["torch"].tensor(vec, device=g["dev"])

    def step_observables_gpu(self, vec, U_mat, which=("x", "y", "z")):
        """One reservoir step fully on the GPU.

        Pulse ρ→UρU† (dense complex64 matmul), free evolution via a
        Taylor + sub-stepping ``exp(t·L)`` built from sparse CUDA matvecs,
        observables via a sparse row-matrix product — the state never
        leaves the GPU between steps. Returns ``(feature_vector_np,
        vec_end_gpu)``. ~15-25x faster than the CPU ``action`` path at N=9.
        """
        g = self._ensure_gpu(which)
        torch = g["torch"]
        d = self.dim
        U = torch.as_tensor(U_mat, dtype=torch.complex64, device=g["dev"])
        # pulse in matrix form (col-stack <-> row-major transpose bookkeeping)
        rho = vec.reshape(d, d).T
        rho = U @ rho @ U.conj().T
        x = rho.T.reshape(-1).contiguous()
        L, h, K, per = g["L"], g["h"], g["K"], g["per"]
        node_vecs = []
        for step in range(1, g["substeps"] + 1):
            term = x
            acc = x
            for k in range(1, K + 1):
                term = (h / k) * torch.mv(L, term)
                acc = acc + term
            x = acc
            if step % per == 0:
                node_vecs.append(x)
        Vmat = torch.stack(node_vecs, dim=1)          # (dim², V)
        feats = torch.sparse.mm(g["M"], Vmat).real    # (n_obs, V)
        return feats.T.reshape(-1).cpu().numpy(), x

    # -- FID spectral readout (Paper 4, Hou et al. 2026) -----------------

    def _readout_indices(self) -> list[int]:
        """Spins that are pulsed and detected by the FID readout (plan §2).

        Priority: explicit ``SimConfig.readout_qubits`` → the protons for a
        multi-nucleus system (labels starting with ``H`` while some spins are
        not protons, i.e. the crotonic ¹³C/¹H case where carbons are the
        bath) → otherwise all spins.
        """
        ro = self.sim.readout_qubits
        if ro:
            return list(ro)
        protons = [
            i for i, lbl in enumerate(self.system.labels)
            if lbl.upper().startswith("H")
        ]
        if protons and len(protons) < self.n:
            return protons
        return list(range(self.n))

    def _build_readout_hamiltonian(self, readout_set):
        """Reduced readout Hamiltonian ``H_read`` (plan §2 stiffness trick).

        Identical to :meth:`_build_hamiltonian` except it **omits the
        chemical-shift σz terms of the non-readout (bath) spins**. Those
        terms commute with every σz and with the readout transverse
        operators σ_±, so they leave the readout FID ⟨O_FID⟩(t) exactly
        unchanged while dominating ‖L‖ (carbon shifts ~7.7 kHz vs proton
        ~1.2 kHz) — dropping them cuts the exponential cost ~6–8×.
        """
        qt = self.qt
        H = qt.qzero([2] * self.n)
        nu = self.system.chemical_shifts
        for i in range(self.n):
            if i in readout_set:
                H = H + np.pi * nu[i] * self.sz[i]
        j = np.asarray(self.system.j_coupling, dtype=float)
        for i in range(self.n):
            for k in range(i + 1, self.n):
                if j[i, k] != 0.0:
                    H = H + (np.pi / 2.0) * j[i, k] * self.sz[i] * self.sz[k]
        return H

    def _ensure_fid_cache(self):
        """Build (once) the FID readout operators: the π/2 readout pulse, the
        ``O_FID`` row-vector, and the reduced sparse Liouvillian ``L_read``.

        Column-stacking (``order='F'``) is kept throughout to match the
        ``action`` backend and :meth:`_ensure_obs_rows`:
        ``⟨O⟩ = vec(O^T)·vec(ρ)``.
        """
        if getattr(self, "_fid_cache", None) is not None:
            return self._fid_cache
        qt = self.qt
        readout = self._readout_indices()
        ro_set = set(readout)
        # π/2 x-pulse on each readout spin: R_x(θ) = cos(θ/2) I − i sin(θ/2) σx.
        theta = np.pi / 2.0
        c, s = np.cos(theta / 2.0), np.sin(theta / 2.0)
        rx = c * qt.qeye(2) - 1j * s * qt.sigmax()
        pulse_ops = [rx if i in ro_set else qt.qeye(2) for i in range(self.n)]
        U = qt.tensor(pulse_ops).full().astype(np.complex128)
        # O_FID = Σ_{p∈readout} (σ_y^p + i σ_x^p); detect via o = vec(O_FID^T).
        o_op = qt.qzero([2] * self.n)
        for p in readout:
            o_op = o_op + self.sy[p] + 1j * self.sx[p]
        o = qt.operator_to_vector(o_op.trans()).full().ravel().astype(np.complex128)
        H_read = self._build_readout_hamiltonian(ro_set)
        L_read = qt.liouvillian(H_read, self.c_ops).data.as_scipy()
        L_read = L_read.tocsr().astype(np.complex128)
        self._fid_cache = {"readout": readout, "U": U, "o": o, "L": L_read}
        # gpu-mode FID tensors, lazy.
        self._fid_gpu = None
        return self._fid_cache

    def fid_signal(self, rho) -> np.ndarray:
        """Simulated FID S(t) for the readout spins (plan §2, Paper Eq. 4).

        ``S(t) = Tr[ e^{tL_read}(UρU†) · O_FID ]`` sampled at ``fid_points``
        times spaced by ``fid_dwell``. Returns a length-``fid_points``
        complex128 array with index 0 = t=0. Uses the reduced ``L_read``
        (exact — see :meth:`_build_readout_hamiltonian`).
        """
        cache = self._ensure_fid_cache()
        rho_mat = rho.full() if hasattr(rho, "full") else np.asarray(rho)
        rho_mat = rho_mat.astype(np.complex128, copy=False)
        U = cache["U"]
        rho_read = U @ rho_mat @ U.conj().T
        if self.evolution_mode == "gpu":
            return self._fid_signal_gpu(rho_read, cache)
        return self._fid_signal_cpu(rho_read, cache)

    def _fid_signal_cpu(self, rho_read, cache):
        """CPU FID: exact Krylov action exp(t·L_read)·vec on the sample grid."""
        from scipy.sparse.linalg import expm_multiply

        n = self.sim.fid_points
        dwell = self.sim.fid_dwell
        o = cache["o"]
        r = rho_read.reshape(-1, order="F")           # column-stacked vec(ρ)
        if n < 2:
            return np.asarray([complex(o @ r)], dtype=np.complex128)
        out = expm_multiply(
            cache["L"], r, start=0.0, stop=(n - 1) * dwell, num=n, endpoint=True,
        )
        sig = out @ o                                 # (n,) complex
        return np.asarray(sig, dtype=np.complex128)

    def _ensure_fid_gpu(self, cache):
        """Build (once) the CUDA tensors for the GPU FID path: sparse
        ``L_read`` (complex64), the ``O_FID`` row-vector, and a Taylor
        sub-stepping schedule with ‖h·L_read‖ ≲ 1 per dwell interval."""
        import torch
        from scipy.sparse.linalg import onenormest

        if getattr(self, "_fid_gpu", None) is not None:
            return self._fid_gpu
        if not torch.cuda.is_available():
            raise RuntimeError(
                "evolution_mode='gpu' needs a CUDA-enabled torch build. "
                "Use 'action' for the exact CPU FID path."
            )
        dev = "cuda"
        L = cache["L"].tocsr().astype(np.complex64)
        Lt = torch.sparse_csr_tensor(
            torch.tensor(L.indptr, dtype=torch.int64),
            torch.tensor(L.indices, dtype=torch.int64),
            torch.tensor(L.data), size=L.shape, device=dev,
        )
        dwell = self.sim.fid_dwell
        nrm = float(onenormest(dwell * cache["L"]))
        per = max(1, int(np.ceil(nrm)))               # substeps per dwell, ‖hL‖≲1
        o = torch.tensor(cache["o"].astype(np.complex64), device=dev)
        self._fid_gpu = {
            "torch": torch, "dev": dev, "L": Lt,
            "per": per, "h": dwell / per, "K": 18, "o": o,
        }
        return self._fid_gpu

    def _fid_signal_gpu(self, rho_read, cache):
        """GPU FID: Taylor + sub-stepping exp(t·L_read) matvecs on CUDA,
        reusing the same fixed-order series as :meth:`step_observables_gpu`,
        sampled on the ``fid_points`` × ``fid_dwell`` grid."""
        g = self._ensure_fid_gpu(cache)
        torch = g["torch"]
        n = self.sim.fid_points
        L, h, K, per, o = g["L"], g["h"], g["K"], g["per"], g["o"]
        x = torch.tensor(
            rho_read.reshape(-1, order="F").astype(np.complex64), device=g["dev"]
        )
        sig = torch.empty(n, dtype=torch.complex64, device=g["dev"])
        sig[0] = (o * x).sum()
        for t in range(1, n):
            for _ in range(per):
                term = x
                acc = x
                for k in range(1, K + 1):
                    term = (h / k) * torch.mv(L, term)
                    acc = acc + term
                x = acc
            sig[t] = (o * x).sum()
        return sig.cpu().numpy().astype(np.complex128)

    def multiplex(self, rho):
        """Evolve ``rho`` one step and return
        ``(states_at_V_subtimes, rho_end)``.

        ``states_at_V_subtimes`` is the list of density matrices sampled
        at each virtual node; ``rho_end`` (== last sample) is carried to
        the next input step to realize the reservoir's memory.

        Backends (see :class:`SimConfig`.``evolution_mode``): superoperator
        ``propagator`` (fast, small N), exact sparse ``action`` (scales to
        9+), or adaptive ``mesolve`` integration.
        """
        if self.evolution_mode == "action":
            return self._multiplex_action(rho)
        if self.evolution_mode == "mesolve":
            res = self.qt.mesolve(
                self.H, rho, self._sub_tlist, c_ops=self.c_ops
            )
            states = list(res.states)[1:]     # drop t=0 (pre-evolution)
            return states, states[-1]
        self._ensure_propagators()
        assert self._props is not None
        states = [self._apply_super(U, rho) for U in self._props]
        return states, states[-1]

    def expectations(self, states, which=("z",)) -> np.ndarray:
        """⟨σ_a⟩ for each requested axis, each qubit, each state.

        Returns a flat feature vector of length
        ``len(states) · n_qubits · len(which)`` — the standard QRC
        observable readout (plan §7 step 4). Multi-modal features live in
        :mod:`app.qrc.features`.
        """
        axis_ops = {"x": self.sx, "y": self.sy, "z": self.sz}
        feats: list[float] = []
        for rho in states:
            for a in which:
                for op in axis_ops[a]:
                    feats.append(float(self.qt.expect(op, rho)))
        return np.asarray(feats, dtype=float)
