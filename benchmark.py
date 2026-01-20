"""Benchmark script for klujax dot performance testing.

Run this before and after loop reordering to measure the impact.
Target runtime: ~20s total.
"""

import time

import jax.numpy as jnp
import numpy as np

import klujax
from klujax import coalesce


def generate_sparse_diagonally_dominant(
    n_col, density, n_lhs, dtype=np.float64, seed=42
):
    """Generate a sparse diagonally dominant matrix (guaranteed invertible)."""
    rng = np.random.RandomState(seed)

    n_offdiag = int(n_col * n_col * density)
    Ai_off = rng.randint(0, n_col, size=n_offdiag).astype(np.int32)
    Aj_off = rng.randint(0, n_col, size=n_offdiag).astype(np.int32)
    Ax_off = rng.randn(n_lhs, n_offdiag).astype(dtype)

    Ai_diag = np.arange(n_col, dtype=np.int32)
    Aj_diag = np.arange(n_col, dtype=np.int32)
    Ax_diag = np.ones((n_lhs, n_col), dtype=dtype) * (n_col * 2)

    Ai = jnp.concatenate([jnp.array(Ai_off), jnp.array(Ai_diag)])
    Aj = jnp.concatenate([jnp.array(Aj_off), jnp.array(Aj_diag)])
    Ax = jnp.concatenate([jnp.array(Ax_off), jnp.array(Ax_diag)], axis=1)

    Ai, Aj, Ax = coalesce(Ai, Aj, Ax)
    return Ai, Aj, Ax


def benchmark_dot(Ai, Aj, Ax, x, warmup=2, iterations=10):
    """Benchmark klujax.dot performance."""
    for _ in range(warmup):
        b = klujax.dot(Ai, Aj, Ax, x)
        b.block_until_ready()

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        b = klujax.dot(Ai, Aj, Ax, x)
        b.block_until_ready()
        end = time.perf_counter()
        times.append(end - start)

    return times


def main():
    # Fixed config that gives ~20s runtime
    n_col = 404
    n_lhs = 14580
    density = 0.1
    iterations = 10

    print("KLUJAX dot Benchmark")
    print("=" * 50)
    print(f"n_col={n_col}, n_lhs={n_lhs}, density={density}")

    Ai, Aj, Ax = generate_sparse_diagonally_dominant(n_col, density, n_lhs)
    x = jnp.array(np.random.RandomState(123).randn(n_lhs, n_col, 1))

    print(f"n_nz={len(Ai)}, Ax.shape={Ax.shape}, x.shape={x.shape}")
    print(f"Running {iterations} iterations...")
    print()

    times = benchmark_dot(Ai, Aj, Ax, x, warmup=2, iterations=iterations)
    times = np.array(times)

    print(f"Mean:  {times.mean() * 1000:8.2f} ms")
    print(f"Std:   {times.std() * 1000:8.2f} ms")
    print(f"Min:   {times.min() * 1000:8.2f} ms")
    print(f"Max:   {times.max() * 1000:8.2f} ms")
    print(f"Total: {times.sum():8.2f} s")
    print("=" * 50)


if __name__ == "__main__":
    main()
