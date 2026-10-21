"""Exact block-structured propagator for the all-σz NMR reservoir.

Why this exists: the differentiable reservoir step (``ensure_diff``) used a
dense node propagator ``P = exp((τ/V)·L)`` of size ``4^n × 4^n`` — 134 MB at
6 spins, 2 GB at 7, 34 GB at 8 — so learnable-encoding training walled at
6 spins. But the Liouvillian of this reservoir is *not* a generic
``4^n × 4^n`` matrix.

Structure (exact, no approximation). For a density-matrix element ``ρ_ab``:

* ``H`` is diagonal (chemical shifts + Ising ``σz σz`` J-couplings), so the
  coherent part only multiplies ``ρ_ab`` by ``-i(E_a - E_b)``.
* σz dephasing only multiplies ``ρ_ab`` (by ``-γ_φ`` when spin s differs
  between a and b).
* σ₋ amplitude damping (T1) maps ``ρ_ab → ρ_a'b'`` only when spin s is in the
  *same* state in a and b, flipping it in both.

Hence the coherence pattern ``p = a XOR b`` is conserved, and so are the
bits of ``a`` on the pattern positions (``q = a AND p``). ``L`` splits into
independent sub-blocks labelled ``(p, q)`` of size ``m = 2^{n-|p|}``; with
``|p| = k`` there are ``C(n,k)·2^k`` of them, so the propagator holds
``Σ_k C(n,k) 2^k 4^{n-k} = 6^n`` entries instead of ``16^n``:

    n      dense P (c64)     block P (c64)
    6      134 MB            0.4 MB
    9      550 GB            81 MB
    10     8.8 TB            0.48 GB
    12     2.3 PB            17.4 GB

Within a sub-block the only off-diagonal terms are the T1 jumps, which set a
spectator bit (QuTiP's σ₋ maps basis 0 → 1), so each sub-block generator is
lower-triangular in the compressed index used here.

The FID / ⟨σ⟩ readout only touches the populations (``k = 0``, σz) and the
single-quantum coherences (``k = 1``, σx/σy). Each node feature at
``t_v = v·τ/V`` is therefore a fixed linear map of the post-pulse state on
those blocks; it is precomputed once as ``R_v = w · exp(t_v L)`` so a step
costs one batched matvec per group plus one small readout contraction —
independent of V.

Conventions match :class:`app.qrc.system.QRCSystem`: spin 0 is the most
significant bit (QuTiP tensor order), basis 0 is σz = +1, states are
column-stacked (``vec[b·d + a] = ρ_ab``). Validated against the QuTiP
Liouvillian and the dense-propagator path in ``scripts/qrc_block_check.py``.
"""

from __future__ import annotations

from math import comb

import numpy as np


def _deposit(vals: np.ndarray, positions: list[int]) -> np.ndarray:
    """Scatter the low bits of ``vals`` onto the bit ``positions`` (pdep)."""
    out = np.zeros_like(vals)
    for i, b in enumerate(positions):
        out |= ((vals >> i) & 1) << b
    return out


def _rates(system, sim):
    """Per-spin T1 rate γ₁ and pure-dephasing coherence decay rate, mirroring
    ``QRCSystem._build_collapse_ops`` (σz collapse with coefficient c decays a
    coherence at ``2c²``)."""
    n = system.n_qubits
    g1 = np.zeros(n)
    gphi = np.zeros(n)
    for s in range(n):
        t1, t2 = system.t1[s], system.t2[s]
        if t1 > 0:
            g1[s] = 1.0 / t1
        if t2 > 0:
            if sim.dephasing_model == "physical":
                c2 = max(1.0 / t2 - 1.0 / (2.0 * t1), 0.0) / 2.0
            else:
                c2 = 1.0 / (2.0 * t2)
            gphi[s] = 2.0 * c2
    return g1, gphi


def _energies(system) -> np.ndarray:
    """Diagonal of the rotating-frame H (rad/s) over the computational basis."""
    n = system.n_qubits
    d = 1 << n
    a = np.arange(d)
    Z = np.stack([1 - 2 * ((a >> (n - 1 - s)) & 1) for s in range(n)], axis=1).astype(float)
    nu = np.asarray(system.chemical_shifts, dtype=float)
    J = np.triu(np.asarray(system.j_coupling, dtype=float), 1)
    return np.pi * (Z @ nu) + (np.pi / 2.0) * np.einsum("as,st,at->a", Z, J, Z)


class BlockPropagator:
    """Exact sub-block propagators + precomputed observable readout.

    ``step(vec, U)`` is the differentiable reservoir step (pulse, free
    evolution over τ, V-node readout) returning ``(feats, vec_end)`` in the
    same layout as :meth:`QRCSystem.step_diff`: feats are v-major,
    axis-major (``which`` order), qubit-inner.
    """

    def __init__(self, system, sim, device="cpu", cdtype=None,
                 which=("x", "y", "z"), chunk_bytes=1 << 29):
        import torch

        self.torch = torch
        self.n = n = system.n_qubits
        self.d = d = 1 << n
        self.V = V = sim.n_virtual
        self.tau = sim.tau
        self.which = tuple(which)
        if not set(self.which) <= {"x", "y", "z"}:
            raise ValueError(f"block readout supports x/y/z observables, got {which}")
        self.device = device
        self.cdtype = cdtype if cdtype is not None else torch.complex64
        self.chunk_bytes = chunk_bytes

        E = _energies(system)
        g1, gphi = _rates(system, sim)
        self._E = torch.tensor(E, dtype=torch.float64, device=device)
        self._g1, self._gphi = g1, gphi

        # -- enumerate sub-blocks, grouped by coherence order k = |p| -------
        self.groups = []
        perm_parts = []
        for k in range(n + 1):
            m = 1 << (n - k)
            A_rows, P_rows, C_rows = [], [], []
            for p in range(d):
                if bin(p).count("1") != k:
                    continue
                pb = [b for b in range(n) if (p >> b) & 1]
                cb = [b for b in range(n) if not (p >> b) & 1]
                dep_c = _deposit(np.arange(m, dtype=np.int64), cb)
                dep_p = _deposit(np.arange(1 << k, dtype=np.int64), pb)
                A_rows.append(dep_p[:, None] | dep_c[None, :])   # q-major, then r
                P_rows.append(np.full(1 << k, p, dtype=np.int64))
                C_rows.append(np.full(1 << k, (d - 1) ^ p, dtype=np.int64))
            A = np.concatenate(A_rows)                      # (count, m) a-values
            P = np.concatenate(P_rows)                      # (count,) pattern
            comp = np.concatenate(C_rows)                   # (count,) spectator mask
            assert A.shape[0] == comb(n, k) << k
            idx = ((A ^ P[:, None]) * d + A)                # col-stack index of ρ_ab
            perm_parts.append(idx.ravel())
            self.groups.append({"k": k, "m": m, "A": A, "P": P, "comp": comp})

        perm = np.concatenate(perm_parts)
        assert perm.size == d * d
        self.perm = torch.tensor(perm, dtype=torch.int64, device=device)
        inv = np.empty_like(perm)
        inv[perm] = np.arange(perm.size, dtype=np.int64)
        self.invperm = torch.tensor(inv, dtype=torch.int64, device=device)
        self.sizes = [g["A"].size for g in self.groups]

        # -- step propagators exp(τ L_k), all groups ---------------------------
        for g in self.groups:
            g["B"] = self._expm_group(g, self.tau).to(self.cdtype)
        self._build_readout()

    # -- generator / exponential ---------------------------------------------

    def generator(self, g, rows=slice(None)):
        """Dense sub-block generators ``(c, m, m)`` (complex128) for ``rows``
        of group ``g``. Column i → row i' means ``d ρ_{i'}/dt ∋ L[i', i] ρ_i``."""
        torch = self.torch
        n, dev = self.n, self.device
        A = torch.tensor(g["A"][rows], device=dev)
        P = torch.tensor(g["P"][rows], device=dev)
        comp = g["comp"][rows]
        B = A ^ P[:, None]
        diag = (-1j) * (self._E[A] - self._E[B]).to(torch.complex128)
        for s in range(n):
            bp = n - 1 - s
            # float64 masks: an int/bool tensor times a float scalar would
            # silently promote to float32 and lose the rates' precision
            ab = ((A >> bp) & 1).to(torch.float64)
            bb = ((B >> bp) & 1).to(torch.float64)
            if self._gphi[s] > 0:
                diag = diag - float(self._gphi[s]) * (ab - bb).abs()
            if self._g1[s] > 0:
                diag = diag - 0.5 * float(self._g1[s]) * ((1 - ab) + (1 - bb))
        L = torch.diag_embed(diag)
        for s in range(n):
            if self._g1[s] <= 0:
                continue
            bp = n - 1 - s
            # rank of bit bp among each row's spectator bits = its bit in r
            rank = np.array([bin(int(c) & ((1 << bp) - 1)).count("1") for c in comp])
            spect = torch.tensor(((comp >> bp) & 1) == 1, device=dev)
            mask = spect[:, None] & (((A >> bp) & 1) == 0)
            rr, ii = mask.nonzero(as_tuple=True)
            step = torch.tensor(1 << rank, device=dev)[rr]
            L[rr, ii + step, ii] += float(self._g1[s])
        return L

    def _expm_group(self, g, t, dtype=None):
        """``exp(t·L)`` for every sub-block of group ``g``, computed in
        complex128 and cast to ``dtype`` (default: the propagator dtype),
        chunked so the matrix_exp workspace stays bounded."""
        torch = self.torch
        count, m = g["A"].shape
        per = max(1, int(self.chunk_bytes // (16 * m * m * 8)))
        out = []
        for lo in range(0, count, per):
            L = self.generator(g, slice(lo, min(count, lo + per)))
            out.append(torch.linalg.matrix_exp(t * L).to(dtype or self.cdtype))
            del L
        return torch.cat(out)

    # -- readout ------------------------------------------------------------

    def _build_readout(self):
        """Precompute ``R_v = w · exp(v·τ/V · L)`` for the σz (k=0) and σx/σy
        (k=1) observables, v = 1..V."""
        torch = self.torch
        n, d, V, dev = self.n, self.d, self.V, self.device
        dt = self.tau / V
        c128 = torch.complex128
        g0, g1 = self.groups[0], self.groups[1]

        # k=0: a single block over all a (r = a), σz_s weight z_s(a)
        a = torch.arange(d, device=dev)
        wz = torch.stack([(1 - 2 * ((a >> (n - 1 - s)) & 1)) for s in range(n)]).to(torch.float64).to(c128)  # (n, d)
        N0 = self._expm_group(g0, dt, c128)[0]                                         # (d, d)
        # k=1: pattern j (ascending p = 1<<j) owns rows 2j (q=0) and 2j+1 (q=p)
        N1 = self._expm_group(g1, dt, c128)                                          # (2n, m, m)
        m = d // 2
        # weights per (axis, row): σx → 1;  σy → +i on q=0 (b_s=1, a_s=0), -i on q=p
        wx = torch.ones(2 * n, m, dtype=c128, device=dev)
        wy = torch.empty(2 * n, m, dtype=c128, device=dev)
        wy[0::2] = 1j
        wy[1::2] = -1j
        Rz = torch.empty(V, n, d, dtype=c128, device=dev)
        Rxy = torch.empty(V, 2, 2 * n, m, dtype=c128, device=dev)
        Wz, Wx, Wy = wz, wx, wy
        for v in range(V):
            Wz = Wz @ N0
            Wx = torch.bmm(Wx.unsqueeze(1), N1).squeeze(1)
            Wy = torch.bmm(Wy.unsqueeze(1), N1).squeeze(1)
            Rz[v], Rxy[v, 0], Rxy[v, 1] = Wz, Wx, Wy
        # rows (2n, m) in pattern order j → (n, 2m) per pattern, then spin order
        # (pattern j ↔ bit j ↔ spin n-1-j)
        Rxy = Rxy.reshape(V, 2, n, 2 * m).flip(2)
        self.Rz = Rz.to(self.cdtype)
        self.Rxy = Rxy.to(self.cdtype)

    def features(self, x0, x1):
        """Node features ``(V·n_axes·n,)`` from the post-pulse k=0 block
        ``x0 (d,)`` and k=1 blocks ``x1 (2n·m,)``."""
        torch = self.torch
        n = self.n
        X1 = x1.reshape(n, -1).flip(0)                       # spin order, (n, 2m)
        out = {}
        if "x" in self.which or "y" in self.which:
            fxy = torch.einsum("vasd,sd->vas", self.Rxy, X1)
            out["x"], out["y"] = fxy[:, 0], fxy[:, 1]
        if "z" in self.which:
            out["z"] = torch.einsum("vsd,d->vs", self.Rz, x0)
        F = torch.stack([out[a] for a in self.which], dim=1)  # (V, n_axes, n)
        return F.real.reshape(-1)

    # -- the step -----------------------------------------------------------

    def step(self, vec, U):
        """One differentiable reservoir step. ``U`` is either the dense
        ``d×d`` pulse or a list of ``n`` per-spin ``2×2`` factors (applied
        as tensor contractions — O(n·d²) instead of O(d³))."""
        torch = self.torch
        d = self.d
        rho = vec.reshape(d, d).T
        if isinstance(U, (list, tuple)):
            if torch.is_grad_enabled() and self.n >= 8:
                # the 2n contractions would each keep a d² intermediate for
                # backward (~3 GB/step at 12 spins); keep only the input and recompute
                from torch.utils.checkpoint import checkpoint

                rho = checkpoint(lambda r, *f: apply_local_pulse(r, f, self.n),
                                 rho, *U, use_reentrant=False)
            else:
                rho = apply_local_pulse(rho, U, self.n)
        else:
            rho = U @ rho @ U.conj().T
        x = rho.T.reshape(-1)
        parts = torch.split(x[self.perm], self.sizes)
        feats = self.features(parts[0], parts[1])
        outs = []
        for g, part in zip(self.groups, parts):
            X = part.reshape(g["A"].shape)
            outs.append(torch.bmm(g["B"], X.unsqueeze(-1)).reshape(-1))
        x_new = torch.cat(outs)[self.invperm]
        return feats, x_new

    # -- FID acquisition readout (Paper-4 / trace-cache readout) -------------

    def fid_readout(self, readout_spins, fid_points, dwell):
        """Precompute the exact FID acquisition as a linear map of the state.

        Mirrors :meth:`QRCSystem.fid_signal`: a π/2 x-pulse on ``readout_spins``,
        then ``S(t_m) = Tr[e^{t_m L}(UρU†) O_FID]`` with
        ``O_FID = Σ_p (σy_p + i σx_p)``, ``t_m = m·dwell``. ``O_FID`` only
        touches the single-quantum (k=1) blocks of the readout spins, so
        ``S = R · x_k1`` with ``R[m] = w · exp(t_m L_block)`` built by repeated
        multiplication. Returns a dict for :meth:`fid`; exact (no stiffness
        sub-stepping), O(fid_points · |readout| · d) memory."""
        torch = self.torch
        n, d, dev = self.n, self.d, self.device
        c128 = torch.complex128
        g1 = self.groups[1]
        m = d // 2
        N1 = self._expm_group(g1, dwell, c128)                              # (2n, m, m)
        # rows 2j (q=0) / 2j+1 (q=p) belong to pattern bit j <-> spin n-1-j;
        # weight of O_FID = σy + iσx: q=0 -> (+i) + i·1 = 2i ; q=p -> (-i) + i = 0
        w = torch.zeros(2 * n, m, dtype=c128, device=dev)
        for s in readout_spins:
            j = n - 1 - s
            w[2 * j] = 2j
        rows = [r for s in readout_spins for r in (2 * (n - 1 - s), 2 * (n - 1 - s) + 1)]
        rows_t = torch.tensor(rows, device=dev)
        W = w[rows_t]                                                       # (r, m)
        Nr = N1[rows_t]
        R = torch.empty(fid_points, len(rows), m, dtype=c128, device=dev)
        R[0] = W
        for t in range(1, fid_points):
            W = torch.bmm(W.unsqueeze(1), Nr).squeeze(1)
            R[t] = W
        # π/2 x-pulse factors on the readout spins (identity elsewhere)
        c, s_ = np.cos(np.pi / 4), np.sin(np.pi / 4)
        rx = torch.tensor([[c, -1j * s_], [-1j * s_, c]], dtype=c128, device=dev)
        eye = torch.eye(2, dtype=c128, device=dev)
        pulse = [rx if i in set(readout_spins) else eye for i in range(n)]
        # offsets of the k=1 group inside the permuted (grouped) state vector
        off = self.sizes[0]
        idx = torch.cat([torch.arange(off + r * m, off + (r + 1) * m, device=dev) for r in rows])
        return {"R": R.reshape(fid_points, -1).to(self.cdtype), "idx": idx,
                "pulse": [p.to(self.cdtype) for p in pulse]}

    def fid(self, vec, ro):
        """FID samples (complex, length fid_points) of the column-stacked state
        ``vec`` for a readout built by :meth:`fid_readout`. Non-destructive."""
        d = self.d
        rho = apply_local_pulse(vec.reshape(d, d).T, ro["pulse"], self.n)
        x = rho.T.reshape(-1)[self.perm][ro["idx"]]
        return ro["R"] @ x

    def nbytes(self) -> int:
        tot = sum(g["B"].numel() * g["B"].element_size() for g in self.groups)
        return tot + self.Rz.numel() * self.Rz.element_size() + self.Rxy.numel() * self.Rxy.element_size()


def apply_local_pulse(rho, mats, n):
    """``U ρ U†`` for ``U = ⊗_s mats[s]`` (spin 0 = most significant), via
    per-spin tensor contractions on the ``(2,)*2n`` reshaped density matrix."""
    import torch

    d = rho.shape[0]
    T = rho.reshape([2] * (2 * n))
    for s, R in enumerate(mats):
        T = torch.movedim(torch.tensordot(R, T, dims=([1], [s])), 0, s)
        T = torch.movedim(torch.tensordot(R.conj(), T, dims=([1], [n + s])), 0, n + s)
    return T.reshape(d, d)
