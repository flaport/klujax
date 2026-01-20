"""Benchmark script for klujax dot performance testing."""

import time

import jax.numpy as jnp
import numpy as np
from jax import Array

import klujax
from klujax import coalesce


def generate_sparse_diagonally_dominant(
    n_col: int,
    density: float,
    n_lhs: int,
    dtype: type = np.float64,
    seed: int = 42,
) -> tuple[Array, Array, Array]:
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


def benchmark_dot(
    Ai: Array,
    Aj: Array,
    Ax: Array,
    x: Array,
    warmup: int = 2,
    iterations: int = 10,
) -> list[float]:
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


def main() -> None:
    """Run the benchmark."""
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
    times_arr = np.array(times)

    print(f"Mean:  {times_arr.mean() * 1000:8.2f} ms")
    print(f"Std:   {times_arr.std() * 1000:8.2f} ms")
    print(f"Min:   {times_arr.min() * 1000:8.2f} ms")
    print(f"Max:   {times_arr.max() * 1000:8.2f} ms")
    print(f"Total: {times_arr.sum():8.2f} s")
    print("=" * 50)


if __name__ == "__main__":
    main()
