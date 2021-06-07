import numpy as np

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
import jax.numpy as jnp

import klujax


def test_sparse_coo_matmul_f64():
    n_nz = 8
    n_col = 5
    n_rhs = 1
    Axkey, Aikey, Ajkey, bkey = jax.random.split(jax.random.PRNGKey(42), 4)
    Ax = jax.random.normal(Axkey, (n_nz,))
    Ai = jax.random.randint(Aikey, (n_nz,), 0, n_col, jnp.int32)
    Aj = jax.random.randint(Ajkey, (n_nz,), 0, n_col, jnp.int32)
    b = jax.random.normal(bkey, (n_col, n_rhs))
    coo_vec_mul = jax.jit(klujax.coo_vec_mul)
    x_sp = coo_vec_mul(Ax, Ai, Aj, b)

    A = jnp.zeros((n_col, n_col), dtype=jnp.float64).at[Ai, Aj].add(Ax)
    x = A @ b

    np.testing.assert_array_almost_equal(x_sp, x)

test_sparse_coo_matmul_f64()
