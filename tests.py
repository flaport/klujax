import jax; jax.config.update("jax_enable_x64", True)
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
    print(Ax)
    print(Ai)
    print(Aj)
    print(b)
    coo_vec_mul = jax.jit(klujax.coo_vec_mul)
    x = coo_vec_mul(Ax, Ai, Aj, b)


test_sparse_coo_matmul_f64()
