import sys
from functools import wraps

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
import jax.numpy as jnp
import jax.scipy as jsp
from jax import lax
from jaxlib.xla_extension import XlaRuntimeError

import klujax


def log_test_name(f):
    @wraps(f)
    def new(*args, **kwargs):
        print(f"\n{f.__name__}", file=sys.stderr)
        if args:
            print(f"args={args}", file=sys.stderr)
        if kwargs:
            print(f"kwargs={kwargs}", file=sys.stderr)
        return f(*args, **kwargs)

    return new


@log_test_name
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("ops", [(klujax.solve, jsp.linalg.solve), (klujax.coo_mul_vec, lax.dot)])  # fmt: skip
def test_1d(dtype, ops):
    op_sparse, op_dense = ops
    Ai, Aj, Ax, b = _get_rand_arrs_1d(8, (n_col := 5), dtype=dtype)
    x_sp = op_sparse(Ai, Aj, Ax, b)
    A = jnp.zeros((n_col, n_col), dtype=Ax.dtype).at[Ai, Aj].add(Ax)
    x = op_dense(A, b)
    _log_and_test_equality(x, x_sp)


@log_test_name
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("ops", [(klujax.solve, jsp.linalg.solve), (klujax.coo_mul_vec, lax.dot)])  # fmt: skip
def test_2d(dtype, ops):
    op_sparse, op_dense = ops
    Ai, Aj, Ax, b = _get_rand_arrs_2d((n_lhs := 3), 8, (n_col := 5), dtype=dtype)
    x_sp = op_sparse(Ai, Aj, Ax, b)
    A = jnp.zeros((n_lhs, n_col, n_col), dtype=dtype).at[:, Ai, Aj].add(Ax)
    x = jax.vmap(op_dense, (0, 0), 0)(A, b)
    _log_and_test_equality(x, x_sp)


@log_test_name
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("ops", [(klujax.solve, jsp.linalg.solve), (klujax.coo_mul_vec, lax.dot)])  # fmt: skip
def test_2d_vmap(dtype, ops):
    op_sparse, op_dense = ops
    Ai, Aj, Ax, b = _get_rand_arrs_2d((n_lhs := 3), 8, (n_col := 5), dtype=dtype)
    x_sp = jax.vmap(op_sparse, (None, None, 1, 1), 0)(Ai, Aj, Ax.T, b.T)
    A = jnp.zeros((n_lhs, n_col, n_col), dtype=dtype).at[:, Ai, Aj].add(Ax)
    x = jax.vmap(op_dense, (0, 0), 0)(A, b)
    _log_and_test_equality(x, x_sp)


@log_test_name
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("ops", [(klujax.solve, jsp.linalg.solve), (klujax.coo_mul_vec, lax.dot)])  # fmt: skip
def test_3d(dtype, ops):
    op_sparse, op_dense = ops
    Ai, Aj, Ax, b = _get_rand_arrs_3d((n_lhs := 3), 8, (n_col := 5), 2, dtype=dtype)
    x_sp = op_sparse(Ai, Aj, Ax, b)
    A = jnp.zeros((n_lhs, n_col, n_col), dtype=dtype).at[:, Ai, Aj].add(Ax)
    x = jax.vmap(op_dense, (0, 0), 0)(A, b)
    _log_and_test_equality(x, x_sp)


@log_test_name
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("ops", [(klujax.solve, jsp.linalg.solve), (klujax.coo_mul_vec, lax.dot)])  # fmt: skip
def test_3d_vmap(dtype, ops):
    op_sparse, op_dense = ops
    Ai, Aj, Ax, b = _get_rand_arrs_3d((n_lhs := 3), 8, (n_col := 5), 2, dtype=dtype)
    _log(Ai_shape=Ai.shape, Aj_shape=Aj.shape, Ax_shape=Ax.shape, b_shape=b.shape)
    x_sp = jax.vmap(op_sparse, (None, None, None, -1), -1)(Ai, Aj, Ax, b)
    A = jnp.zeros((n_lhs, n_col, n_col), dtype=dtype).at[:, Ai, Aj].add(Ax)
    x = jax.vmap(op_dense, (0, 0), 0)(A, b)
    _log_and_test_equality(x, x_sp)


@log_test_name
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("op", [klujax.solve, klujax.coo_mul_vec])
def test_4d(dtype, op):
    Ai, Aj, Ax, b = _get_rand_arrs_3d(3, 8, 5, 2, dtype=dtype)
    with pytest.raises(ValueError):
        op(Ai, Aj, Ax, b[None])

    with pytest.raises(ValueError):
        op(Ai, Aj, Ax[None], b)

    with pytest.raises(ValueError):
        op(Ai, Aj, Ax[None], b[None])


@log_test_name
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("op", [klujax.solve, klujax.coo_mul_vec])
def test_4d_vmap(dtype, op):
    Ai, Aj, Ax, b = _get_rand_arrs_3d(3, 8, 5, 2, dtype=dtype)
    jax.vmap(op, (None, None, None, 1), 0)(Ai, Aj, Ax, b[:, None])
    # TODO: compare with dense result


@log_test_name
@pytest.mark.skipif(sys.platform == "win32", reason="FIXME: known to still segfault on Windows!")  # fmt: skip
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("op", [klujax.solve, klujax.coo_mul_vec])
def test_vmap_fail(dtype, op):
    n_lhs = 23
    n_nz = 8
    n_col = 5
    n_rhs = 1

    Axkey, Aikey, Ajkey, bkey = jax.random.split(jax.random.PRNGKey(33), 4)
    Ax = jax.random.normal(Axkey, (n_nz,), dtype=dtype)
    Ai = jax.random.randint(Aikey, (n_nz,), 0, n_col, jnp.int32)
    Aj = jax.random.randint(Ajkey, (n_nz,), 0, n_col, jnp.int32)
    b = jax.random.normal(bkey, (n_lhs, n_col, n_rhs), dtype=dtype)
    with pytest.raises(XlaRuntimeError):
        jax.vmap(op, in_axes=(None, None, None, 0), out_axes=0)(Ai, Aj, Ax, b)


def _get_rand_arrs_1d(n_nz, n_col, *, dtype, seed=33):
    Axkey, Aikey, Ajkey, bkey = jax.random.split(jax.random.PRNGKey(seed), 4)
    Ai = jax.random.randint(Aikey, (n_nz,), 0, n_col, jnp.int32)
    Aj = jax.random.randint(Ajkey, (n_nz,), 0, n_col, jnp.int32)
    Ax = jax.random.normal(Axkey, (n_nz,), dtype=dtype)
    b = jax.random.normal(bkey, (n_col,), dtype=dtype)
    return Ai, Aj, Ax, b


def _get_rand_arrs_2d(n_lhs, n_nz, n_col, *, dtype, seed=33):
    Axkey, Aikey, Ajkey, bkey = jax.random.split(jax.random.PRNGKey(seed), 4)
    Ai = jax.random.randint(Aikey, (n_nz,), 0, n_col, jnp.int32)
    Aj = jax.random.randint(Ajkey, (n_nz,), 0, n_col, jnp.int32)
    Ax = jax.random.normal(
        Axkey,
        (
            n_lhs,
            n_nz,
        ),
        dtype=dtype,
    )
    b = jax.random.normal(bkey, (n_lhs, n_col), dtype=dtype)
    return Ai, Aj, Ax, b


def _get_rand_arrs_3d(n_lhs, n_nz, n_col, n_rhs, *, dtype, seed=33):
    Axkey, Aikey, Ajkey, bkey = jax.random.split(jax.random.PRNGKey(seed), 4)
    Ai = jax.random.randint(Aikey, (n_nz,), 0, n_col, jnp.int32)
    Aj = jax.random.randint(Ajkey, (n_nz,), 0, n_col, jnp.int32)
    Ax = jax.random.normal(
        Axkey,
        (
            n_lhs,
            n_nz,
        ),
        dtype=dtype,
    )
    b = jax.random.normal(bkey, (n_lhs, n_col, n_rhs), dtype=dtype)
    return Ai, Aj, Ax, b


def _log_and_test_equality(x, x_sp):
    print(f"\nx_sp=\n{x_sp}")
    print(f"\nx=\n{x}")
    print(f"\ndiff=\n{np.round(x_sp - x, 9)}")
    print(f"\nis_equal=\n{_is_almost_equal(x_sp, x)}")
    np.testing.assert_array_almost_equal(x_sp, x)


def _log(**kwargs):
    for k, v in kwargs.items():
        print(f"{k}={v}")


def _is_almost_equal(arr1, arr2):
    try:
        np.testing.assert_array_almost_equal(arr1, arr2)
        return True
    except AssertionError:
        return False
