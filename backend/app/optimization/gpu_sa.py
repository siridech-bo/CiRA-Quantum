"""GPU-accelerated simulated annealing sampler.

Implements ``dimod.Sampler`` and runs a population of independent Metropolis
chains in parallel on a CUDA device. Within each chain spins are updated
sequentially (Gibbs-style); the per-spin update is vectorized across chains.

This is the V1 sampler. It is intentionally simple — checkerboard
parallelization, parallel tempering, and CUDA graph capture are deferred to
later phases.

CLI usage::

    python -m app.optimization.gpu_sa --bqm tests/instances/tiny_5var.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import dimod
import numpy as np
import torch


def _sweep_eager(
    spins: torch.Tensor,
    h: torch.Tensor,
    J: torch.Tensor,
    beta: torch.Tensor,
    rand_buf: torch.Tensor,
) -> torch.Tensor:
    """In-place Gibbs sweep — used by the JIT and eager paths.

    ``spins`` is mutated in place and also returned. JIT-script trees this
    into native bytecode without dispatch overhead, which is the cheapest
    way to run the sweep when ``torch.compile`` is not available.

    ``J`` must be symmetric with zero diagonal; we read row ``i`` rather
    than column ``i`` because ``J[i]`` is a contiguous view in row-major
    storage and ``J[:, i]`` would force a strided gather.

    ``spins``   : (R, n) float32, values in {-1, +1}
    ``h``       : (n,) float32
    ``J``       : (n, n) float32, symmetric, zero diagonal
    ``beta``    : 0-d float32 tensor, current inverse temperature
    ``rand_buf``: (n, R) float32 in [0, 1), one row per spin index per chain
    """
    n = spins.shape[1]
    for i in range(n):
        row = J[i]
        local_field = h[i] + torch.mv(spins, row)
        s_i = spins[:, i].clone()
        delta_e = -2.0 * s_i * local_field
        accept = torch.logical_or(delta_e <= 0, torch.exp(-beta * delta_e) > rand_buf[i])
        spins[:, i] = torch.where(accept, -s_i, s_i)
    return spins


def _sweep_functional(
    spins_in: torch.Tensor,
    h: torch.Tensor,
    J: torch.Tensor,
    beta: torch.Tensor,
    rand_buf: torch.Tensor,
) -> torch.Tensor:
    """Out-of-place Gibbs sweep — used by the compile path.

    Inductor needs to see no in-place writes on the input tensor in order to
    enable CUDA-graph capture (``mode="reduce-overhead"``). We allocate a
    fresh ``(R, n)`` working tensor at the top and write into it; the input
    is never modified.
    """
    spins = spins_in.clone()
    n = spins.shape[1]
    for i in range(n):
        row = J[i]
        local_field = h[i] + torch.mv(spins, row)
        s_i = spins[:, i].clone()
        delta_e = -2.0 * s_i * local_field
        accept = torch.logical_or(delta_e <= 0, torch.exp(-beta * delta_e) > rand_buf[i])
        spins[:, i] = torch.where(accept, -s_i, s_i)
    return spins


def _try_jit_script(fn):
    """JIT-script ``fn`` to remove Python interpreter overhead from the inner
    loop. Falls back to eager if scripting fails for any reason — correctness
    is unaffected, only speed."""
    try:
        return torch.jit.script(fn)
    except Exception:
        return fn


def _try_compile(fn):
    """``torch.compile`` ``fn`` for maximum throughput. Triton (or
    ``triton-windows``) is required; if it isn't installed we fall back to
    JIT-script, which is faster than eager but slower than the inductor path.

    The body has a Python ``for i in range(n)`` loop which Dynamo unrolls at
    trace time, so ``dynamic=False`` is correct here — each new ``(R, n)``
    shape pays a one-time compile cost in exchange for fused Triton kernels
    + CUDA-graph capture on every later call with the same shape. The
    caller must use the *functional* sweep (``_sweep_functional``) here
    because in-place input mutation disables cudagraphs in Inductor.
    """
    try:
        return torch.compile(fn, mode="reduce-overhead", fullgraph=False, dynamic=False)
    except Exception:
        return _try_jit_script(fn)


_sweep_jit = _try_jit_script(_sweep_eager)
_sweep_compiled: Any | None = None  # lazily initialized — compile cost is paid on first use


class GPUSimulatedAnnealingSampler(dimod.Sampler):
    """Parallel-Metropolis simulated annealing sampler running on a CUDA GPU.

    The sampler stores chain state as a ``(num_reads, n)`` float32 tensor of
    ±1 values and updates one spin index per inner step, vectorized across
    chains. The inverse-temperature schedule is geometric, sweeping from
    ``beta_range[0]`` to ``beta_range[1]`` over ``num_sweeps`` steps.
    """

    def __init__(self, device: str = "cuda:0", kernel: str = "jit"):
        """Construct a GPU SA sampler.

        ``kernel`` selects the inner-sweep implementation:
            - ``"jit"`` (default) — ``torch.jit.script``. Fast, no warmup.
            - ``"compile"`` — ``torch.compile`` (Inductor + Triton). Highest
              throughput, but pays a one-time compile cost (~10–120 s) and
              recompiles when input shapes change.
            - ``"eager"`` — pure-Python fallback for diagnostics.
        """
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available; GPU SA requires a CUDA device")

        torch_device = torch.device(device)
        if torch_device.type != "cuda":
            raise ValueError(f"device must be a cuda device, got {device!r}")

        idx = torch_device.index if torch_device.index is not None else 0
        if idx >= torch.cuda.device_count():
            raise ValueError(
                f"cuda device index {idx} >= {torch.cuda.device_count()} available"
            )

        if kernel not in ("jit", "compile", "eager"):
            raise ValueError(f"kernel must be one of 'jit', 'compile', 'eager', got {kernel!r}")

        self._device = torch_device
        self._device_str = device
        self._kernel = kernel

    # ----- dimod.Sampler interface -----

    @property
    def parameters(self) -> dict:
        return {"num_reads": [], "num_sweeps": [], "beta_range": [], "seed": []}

    @property
    def properties(self) -> dict:
        idx = self._device.index if self._device.index is not None else 0
        cc = torch.cuda.get_device_capability(idx)
        gpu_props = torch.cuda.get_device_properties(idx)
        return {
            "device": self._device_str,
            "compute_capability": tuple(cc),
            "vram_total_gb": float(gpu_props.total_memory) / (1024**3),
        }

    def sample(
        self,
        bqm: dimod.BinaryQuadraticModel,
        num_reads: int = 1000,
        num_sweeps: int = 1000,
        beta_range: tuple[float, float] = (0.1, 5.0),
        seed: int | None = None,
    ) -> dimod.SampleSet:
        if num_reads <= 0:
            raise ValueError("num_reads must be positive")
        if num_sweeps <= 0:
            raise ValueError("num_sweeps must be positive")

        # Seed *both* CPU and CUDA generators so spin init and Metropolis rolls
        # are reproducible under a fixed seed.
        if seed is None:
            torch.seed()
            torch.cuda.seed_all()
        else:
            seed_int = int(seed)
            torch.manual_seed(seed_int)
            torch.cuda.manual_seed_all(seed_int)

        # ---- Empty-BQM short-circuit ----
        if bqm.num_variables == 0:
            offset = float(bqm.offset)
            samples_array = np.empty((num_reads, 0), dtype=np.int8)
            energies = np.full(num_reads, offset, dtype=np.float64)
            return dimod.SampleSet.from_samples(
                (samples_array, []),
                vartype=bqm.vartype,
                energy=energies,
            )

        # Convert to Ising for the kernel; remember original vartype for output.
        ising = bqm.change_vartype("SPIN", inplace=False)
        var_order: list[Any] = list(ising.variables)
        n = len(var_order)
        idx_of = {v: i for i, v in enumerate(var_order)}

        # Materialize linear/quadratic as numpy arrays first, then push to
        # the GPU in a single contiguous transfer. Element-by-element
        # assignment via Python is unworkable for dense BQMs (50M+ edges
        # at N=10000 takes tens of minutes).
        h_np = np.zeros(n, dtype=np.float32)
        for v, b in ising.linear.items():
            h_np[idx_of[v]] = float(b)
        h = torch.from_numpy(h_np).to(self._device)

        J = torch.zeros((n, n), dtype=torch.float32, device=self._device)
        if ising.quadratic:
            ij_np = np.empty((len(ising.quadratic), 2), dtype=np.int64)
            v_np = np.empty(len(ising.quadratic), dtype=np.float32)
            for k, ((u, v), b) in enumerate(ising.quadratic.items()):
                ij_np[k, 0] = idx_of[u]
                ij_np[k, 1] = idx_of[v]
                v_np[k] = float(b)
            i_idx = torch.from_numpy(ij_np[:, 0]).to(self._device)
            j_idx = torch.from_numpy(ij_np[:, 1]).to(self._device)
            v_buf = torch.from_numpy(v_np).to(self._device)
            J.index_put_((i_idx, j_idx), v_buf)
            J.index_put_((j_idx, i_idx), v_buf)

        offset = float(ising.offset)

        # Random initial spins ±1, shape (R, n)
        spins = (
            torch.randint(0, 2, (num_reads, n), device=self._device, dtype=torch.int32) * 2 - 1
        ).to(torch.float32)

        # Geometric beta schedule
        beta0, beta1 = float(beta_range[0]), float(beta_range[1])
        if num_sweeps == 1:
            betas = torch.tensor([beta1], dtype=torch.float32, device=self._device)
        else:
            betas = torch.logspace(
                start=math.log10(beta0),
                end=math.log10(beta1),
                steps=num_sweeps,
                base=10.0,
                dtype=torch.float32,
                device=self._device,
            )

        # Pre-roll random uniforms for each sweep × spin × chain to keep the
        # inner Python loop tight. Memory cost is num_sweeps * n * num_reads
        # float32, which is 4 bytes per cell — fine for the sizes we target.
        # For very large problems we generate per-sweep instead.
        per_sweep_bytes = n * num_reads * 4
        # Cap the pre-roll buffer at 256 MB; otherwise generate per sweep.
        if per_sweep_bytes * num_sweeps > 256 * 1024 * 1024:
            preroll = False
        else:
            preroll = True

        if preroll:
            rand_all = torch.rand(
                (num_sweeps, n, num_reads), dtype=torch.float32, device=self._device
            )

        sweep_fn = self._select_sweep_fn()
        # When the inner sweep is captured in a CUDA graph (kernel="compile",
        # mode="reduce-overhead"), the graph's output tensor lives in a memory
        # pool that the next invocation may reuse for its own intermediates.
        # Cloning between calls breaks that aliasing chain.
        clone_between_calls = self._kernel == "compile"
        for s_idx in range(num_sweeps):
            beta = betas[s_idx]
            if preroll:
                rand_buf = rand_all[s_idx]
            else:
                rand_buf = torch.rand((n, num_reads), dtype=torch.float32, device=self._device)
            spins = sweep_fn(spins, h, J, beta, rand_buf)
            if clone_between_calls:
                spins = spins.clone()

        # Final energies in Ising form: E = h·s + 0.5 s^T J s + offset
        h_term = spins @ h  # (R,)
        sJ = spins @ J  # (R, n)
        q_term = 0.5 * (sJ * spins).sum(dim=1)  # (R,)
        energies = (h_term + q_term + offset).to(torch.float64).cpu().numpy()

        spin_array = spins.to(torch.int8).cpu().numpy()

        if bqm.vartype is dimod.BINARY:
            sample_array = ((spin_array + 1) // 2).astype(np.int8)
        else:
            sample_array = spin_array

        # Sort lowest-energy first so .first is the best sample.
        order = np.argsort(energies, kind="stable")
        sample_array = sample_array[order]
        energies = energies[order]

        return dimod.SampleSet.from_samples(
            (sample_array, var_order),
            vartype=bqm.vartype,
            energy=energies,
        )

    def _select_sweep_fn(self):
        """Pick the inner-sweep implementation according to ``self._kernel``.

        ``compile`` mode lazily initializes the global compiled function so
        the warmup cost is paid once per process, not once per ``sample()``.
        """
        global _sweep_compiled
        if self._kernel == "eager":
            return _sweep_eager
        if self._kernel == "jit":
            return _sweep_jit
        # "compile" — uses the functional (out-of-place) variant so Inductor
        # can capture the body in a CUDA graph.
        if _sweep_compiled is None:
            _sweep_compiled = _try_compile(_sweep_functional)
        return _sweep_compiled


# ----- CLI harness -----


def _load_bqm_from_json(path: Path) -> dimod.BinaryQuadraticModel:
    with open(path) as f:
        data = json.load(f)

    vartype = dimod.SPIN if data["vartype"] == "SPIN" else dimod.BINARY

    def _key(s: str):
        try:
            return int(s)
        except ValueError:
            return s

    linear = {_key(k): float(v) for k, v in data["linear"].items()}
    quadratic: dict[tuple, float] = {}
    for k, v in data["quadratic"].items():
        u, w = k.split(",")
        quadratic[(_key(u), _key(w))] = float(v)
    offset = float(data.get("offset", 0.0))
    return dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype)


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.optimization.gpu_sa",
        description="Run GPU SA on a JSON-encoded BQM.",
    )
    parser.add_argument("--bqm", type=Path, required=True, help="Path to BQM JSON file")
    parser.add_argument("--num-reads", type=int, default=1000)
    parser.add_argument("--num-sweeps", type=int, default=1000)
    parser.add_argument("--beta-min", type=float, default=0.1)
    parser.add_argument("--beta-max", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--top", type=int, default=5, help="How many of the lowest-energy samples to print"
    )
    args = parser.parse_args(argv)

    bqm = _load_bqm_from_json(args.bqm)
    print(
        f"Loaded BQM: {bqm.num_variables} vars, "
        f"{bqm.num_interactions} interactions, vartype={bqm.vartype.name}",
        file=sys.stderr,
    )

    sampler = GPUSimulatedAnnealingSampler(device=args.device)
    print(f"Sampler properties: {sampler.properties}", file=sys.stderr)

    t0 = time.perf_counter()
    sampleset = sampler.sample(
        bqm,
        num_reads=args.num_reads,
        num_sweeps=args.num_sweeps,
        beta_range=(args.beta_min, args.beta_max),
        seed=args.seed,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"Sampled {len(sampleset)} reads in {elapsed_ms:.1f} ms", file=sys.stderr)

    print(sampleset.truncate(args.top))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
