import numpy as np

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
import jax.numpy as jnp
import jax.scipy as jsp

import klujax


def test_solve_f64():
    n_nz = 8
    n_col = 5
    n_rhs = 1
    Axkey, Aikey, Ajkey, bkey = jax.random.split(jax.random.PRNGKey(33), 4)
    Ax = jax.random.normal(Axkey, (n_nz,))
    Ai = jax.random.randint(Aikey, (n_nz,), 0, n_col, jnp.int32)
    Aj = jax.random.randint(Ajkey, (n_nz,), 0, n_col, jnp.int32)
    b = jax.random.normal(bkey, (n_col, n_rhs))
    x_sp = klujax.solve(Ax, Ai, Aj, b)

    A = jnp.zeros((n_col, n_col), dtype=jnp.float64).at[Ai, Aj].add(Ax)
    x = jsp.linalg.inv(A) @ b

    print(x)
    print(x_sp)
    np.testing.assert_array_almost_equal(x_sp, x)


def test_solve_c128():
    n_nz = 8
    n_col = 5
    n_rhs = 1
    Axkey, Aikey, Ajkey, bkey = jax.random.split(jax.random.PRNGKey(33), 4)
    Ax_r, Ax_i = jax.random.normal(Axkey, (2, n_nz))
    Ax = Ax_r + 1j * Ax_i
    Ai = jax.random.randint(Aikey, (n_nz,), 0, n_col, jnp.int32)
    Aj = jax.random.randint(Ajkey, (n_nz,), 0, n_col, jnp.int32)
    b_r, b_i = jax.random.normal(bkey, (2, n_col, n_rhs))
    b = b_r + 1j * b_i
    x_sp = klujax.solve(Ax, Ai, Aj, b)

    A = jnp.zeros((n_col, n_col), dtype=jnp.complex128).at[Ai, Aj].add(Ax)
    x = jsp.linalg.inv(A) @ b

    print(x)
    print(x_sp)
    np.testing.assert_array_almost_equal(x_sp, x)


def test_coo_vec_mul_f64():
    n_nz = 8
    n_col = 5
    n_rhs = 1
    Axkey, Aikey, Ajkey, bkey = jax.random.split(jax.random.PRNGKey(33), 4)
    Ax = jax.random.normal(Axkey, (n_nz,))
    Ai = jax.random.randint(Aikey, (n_nz,), 0, n_col, jnp.int32)
    Aj = jax.random.randint(Ajkey, (n_nz,), 0, n_col, jnp.int32)
    b = jax.random.normal(bkey, (n_col, n_rhs))
    x_sp = klujax.coo_vec_mul(Ax, Ai, Aj, b)

    A = jnp.zeros((n_col, n_col), dtype=jnp.float64).at[Ai, Aj].add(Ax)
    x = A @ b

    print(x)
    print(x_sp)
    np.testing.assert_array_almost_equal(x_sp, x)


def test_coo_vec_mul_c128():
    n_nz = 8
    n_col = 5
    n_rhs = 1
    Axkey, Aikey, Ajkey, bkey = jax.random.split(jax.random.PRNGKey(33), 4)
    Ax_r, Ax_i = jax.random.normal(Axkey, (2, n_nz))
    Ax = Ax_r + 1j * Ax_i
    Ai = jax.random.randint(Aikey, (n_nz,), 0, n_col, jnp.int32)
    Aj = jax.random.randint(Ajkey, (n_nz,), 0, n_col, jnp.int32)
    b_r, b_i = jax.random.normal(bkey, (2, n_col, n_rhs))
    b = b_r + 1j * b_i
    x_sp = klujax.coo_vec_mul(Ax, Ai, Aj, b)

    A = jnp.zeros((n_col, n_col), dtype=jnp.complex128).at[Ai, Aj].add(Ax)
    x = A @ b

    print(x)
    print(x_sp)
    np.testing.assert_array_almost_equal(x_sp, x)


if __name__ == "__main__":
    test_solve_f64()
    test_solve_c128()
    test_coo_vec_mul_f64()
    test_coo_vec_mul_c128()
