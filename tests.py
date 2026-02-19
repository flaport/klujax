import sys
from functools import wraps

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import pytest
from jax import lax

import klujax
from klujax import COMPLEX_DTYPES, coalesce

OPS_DENSE = {  # sparse to dense
    klujax.dot: lax.dot,
    klujax.solve: jsp.linalg.solve,
}


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


def parametrize_dtypes(func):
    return pytest.mark.parametrize("dtype", [np.float64, np.complex128])(func)


def parametrize_ops(func):
    return pytest.mark.parametrize("op_sparse", [klujax.solve, klujax.dot])(func)


@log_test_name
@parametrize_dtypes
@parametrize_ops
def test_1d(dtype, op_sparse):
    op_dense = OPS_DENSE[op_sparse]
    Ai, Aj, Ax, b = _get_rand_arrs_1d(15, (n_col := 5), dtype=dtype)
    x_sp = op_sparse(Ai, Aj, Ax, b)
    A = jnp.zeros((n_col, n_col), dtype=Ax.dtype).at[Ai, Aj].add(Ax)
    x = op_dense(A, b)
    _log_and_test_equality(x, x_sp)


@log_test_name
@parametrize_dtypes
@parametrize_ops
def test_2d(dtype, op_sparse):
    op_dense = jax.vmap(OPS_DENSE[op_sparse], (0, 0), 0)
    Ai, Aj, Ax, b = _get_rand_arrs_2d((n_lhs := 3), 15, (n_col := 5), dtype=dtype)
    x_sp = op_sparse(Ai, Aj, Ax, b)
    A = jnp.zeros((n_lhs, n_col, n_col), dtype=dtype).at[:, Ai, Aj].add(Ax)
    x = op_dense(A, b)
    _log_and_test_equality(x, x_sp)


@log_test_name
@parametrize_dtypes
@parametrize_ops
def test_2d_vmap(dtype, op_sparse):
    op_dense = OPS_DENSE[op_sparse]
    Ai, Aj, Ax, b = _get_rand_arrs_2d((n_lhs := 3), 15, (n_col := 5), dtype=dtype)
    x_sp = jax.vmap(op_sparse, (None, None, 1, 1), 0)(Ai, Aj, Ax.T, b.T)
    A = jnp.zeros((n_lhs, n_col, n_col), dtype=dtype).at[:, Ai, Aj].add(Ax)
    x = jax.vmap(op_dense, (0, 0), 0)(A, b)
    _log_and_test_equality(x, x_sp)


@log_test_name
@parametrize_dtypes
@parametrize_ops
def test_3d(dtype, op_sparse):
    op_dense = jax.vmap(OPS_DENSE[op_sparse], (0, 0), 0)
    Ai, Aj, Ax, b = _get_rand_arrs_3d((n_lhs := 2), 8, (n_col := 3), 4, dtype=dtype)
    x_sp = op_sparse(Ai, Aj, Ax, b)
    A = jnp.zeros((n_lhs, n_col, n_col), dtype=dtype).at[:, Ai, Aj].add(Ax)
    x = op_dense(A, b)
    _log_and_test_equality(x, x_sp)


@log_test_name
@parametrize_dtypes
@parametrize_ops
def test_3d_jacfwd(dtype, op_sparse):
    op_dense = jax.vmap(OPS_DENSE[op_sparse], (0, 0), 0)
    Ai, Aj, Ax, b = _get_rand_arrs_3d((n_lhs := 3), 15, (n_col := 5), 2, dtype=dtype)
    holomorphic = dtype in COMPLEX_DTYPES

    # jacobian on b
    jac_sp = jax.jacfwd(op_sparse, 3, holomorphic=holomorphic)(Ai, Aj, Ax, b)
    A = jnp.zeros((n_lhs, n_col, n_col), dtype=dtype).at[:, Ai, Aj].add(Ax)
    jac = jax.jacfwd(op_dense, 1, holomorphic=holomorphic)(A, b)
    _log_and_test_equality(jac_sp, jac)

    # jacobian on A
    jac_sp = jax.jacfwd(op_sparse, 2, holomorphic=holomorphic)(Ai, Aj, Ax, b)
    A = jnp.zeros((n_lhs, n_col, n_col), dtype=dtype).at[:, Ai, Aj].add(Ax)
    jac = jax.jacfwd(op_dense, 0, holomorphic=holomorphic)(A, b)[..., Ai, Aj]
    _log_and_test_equality(jac_sp, jac)


@log_test_name
@parametrize_dtypes
@parametrize_ops
def test_3d_jacrev(dtype, op_sparse):
    op_dense = jax.vmap(OPS_DENSE[op_sparse], (0, 0), 0)
    Ai, Aj, Ax, b = _get_rand_arrs_3d((n_lhs := 3), 15, (n_col := 5), 2, dtype=dtype)
    holomorphic = dtype in COMPLEX_DTYPES

    # jacobian on b
    jac_sp = jax.jacrev(op_sparse, 3, holomorphic=holomorphic)(Ai, Aj, Ax, b)
    A = jnp.zeros((n_lhs, n_col, n_col), dtype=dtype).at[:, Ai, Aj].add(Ax)
    jac = jax.jacrev(op_dense, 1, holomorphic=holomorphic)(A, b)
    _log_and_test_equality(jac_sp, jac)

    # jacobian on A
    jac_sp = jax.jacrev(op_sparse, 2, holomorphic=holomorphic)(Ai, Aj, Ax, b)
    A = jnp.zeros((n_lhs, n_col, n_col), dtype=dtype).at[:, Ai, Aj].add(Ax)
    jac = jax.jacrev(op_dense, 0, holomorphic=holomorphic)(A, b)[..., Ai, Aj]
    _log_and_test_equality(jac_sp, jac)


@log_test_name
@parametrize_dtypes
@parametrize_ops
def test_3d_vmap(dtype, op_sparse):
    op_dense = OPS_DENSE[op_sparse]
    Ai, Aj, Ax, b = _get_rand_arrs_3d((n_lhs := 3), 15, (n_col := 5), 2, dtype=dtype)
    _log(Ai_shape=Ai.shape, Aj_shape=Aj.shape, Ax_shape=Ax.shape, b_shape=b.shape)
    x_sp = jax.vmap(op_sparse, (None, None, None, -1), -1)(Ai, Aj, Ax, b)
    A = jnp.zeros((n_lhs, n_col, n_col), dtype=dtype).at[:, Ai, Aj].add(Ax)
    x = jax.vmap(op_dense, (0, 0), 0)(A, b)
    _log_and_test_equality(x, x_sp)


@log_test_name
@parametrize_dtypes
@parametrize_ops
def test_4d(dtype, op_sparse):
    Ai, Aj, Ax, b = _get_rand_arrs_3d(3, 15, 5, 2, dtype=dtype)
    with pytest.raises(ValueError):  # noqa: PT011
        op_sparse(Ai, Aj, Ax, b[None])

    with pytest.raises(ValueError):  # noqa: PT011
        op_sparse(Ai, Aj, Ax[None], b)

    with pytest.raises(ValueError):  # noqa: PT011
        op_sparse(Ai, Aj, Ax[None], b[None])


@log_test_name
@parametrize_dtypes
@parametrize_ops
def test_4d_vmap(dtype, op_sparse):
    Ai, Aj, Ax, b = _get_rand_arrs_3d(3, 8, 5, 2, dtype=dtype)
    bb = np.stack([b, (b2 := np.random.RandomState(seed=42).rand(*b.shape))], axis=1)
    r = jax.vmap(op_sparse, (None, None, None, 1), 0)(Ai, Aj, Ax, bb)
    r1 = op_sparse(Ai, Aj, Ax, b)
    r2 = op_sparse(Ai, Aj, Ax, b2)
    _log_and_test_equality(r[0], r1)
    _log_and_test_equality(r[1], r2)


@log_test_name
def test_analyze():
    Ai, Aj, Ax, b = _get_rand_arrs_1d(15, (n_col := 5), dtype=np.float64)

    # 1. Test Eager Analysis
    symbolic = klujax.analyze(Ai, Aj, n_col)
    assert isinstance(symbolic, klujax.KLUHandleManager)
    assert symbolic._owner is True
    assert symbolic.handle.dtype == jnp.uint64
    
    # Manually free to be clean before next step
    klujax.free_symbolic(symbolic)
    assert symbolic._freed is True

    @jax.jit
    def jit_analyze_and_solve(Ai, Aj, Ax, b):
        # Create handle inside JIT
        sym = klujax.analyze(Ai, Aj, 5) # Inside JIT, this is a Tracer

        x = klujax.solve_with_symbol(Ai, Aj, Ax, b, sym)
        
        klujax.free_symbolic(sym, dependency=x)
        return x

    x = jit_analyze_and_solve(Ai, Aj, Ax, b)
    assert x.shape == (n_col,)


@log_test_name
@parametrize_dtypes
def test_solve_with_symbol(dtype):
    Ai, Aj, Ax, b = _get_rand_arrs_1d(15, (n_col := 5), dtype=dtype)
    symbolic = klujax.analyze(Ai, Aj, n_col)

    x_sp = klujax.solve_with_symbol(Ai, Aj, Ax, b, symbolic)

    A = jnp.zeros((n_col, n_col), dtype=dtype).at[Ai, Aj].add(Ax)
    x = jsp.linalg.solve(A, b)
    _log_and_test_equality(x, x_sp)

    # Test JIT
    x_sp_jit = jax.jit(klujax.solve_with_symbol)(Ai, Aj, Ax, b, symbolic)
    _log_and_test_equality(x, x_sp_jit)

    klujax.free_symbolic(symbolic)


@log_test_name
@parametrize_dtypes
def test_solve_with_symbol_batched(dtype):
    Ai, Aj, Ax, b = _get_rand_arrs_2d((n_lhs := 3), 15, (n_col := 5), dtype=dtype)
    symbolic = klujax.analyze(Ai, Aj, n_col)

    x_sp = klujax.solve_with_symbol(Ai, Aj, Ax, b, symbolic)

    # Dense verification
    op_dense = jax.vmap(jsp.linalg.solve, (0, 0), 0)
    A = jnp.zeros((n_lhs, n_col, n_col), dtype=dtype).at[:, Ai, Aj].add(Ax)
    x = op_dense(A, b)
    _log_and_test_equality(x, x_sp)

    klujax.free_symbolic(symbolic)

@log_test_name
@parametrize_dtypes
def test_solve_with_numeric(dtype):
    Ai, Aj, Ax, b = _get_rand_arrs_1d(15, (n_col := 5), dtype=dtype)
    symbolic = klujax.analyze(Ai, Aj, n_col)
    numeric = klujax.factor(Ai, Aj, Ax, symbolic)

    x_sp = klujax.solve_with_numeric(numeric, b, symbolic)

    A = jnp.zeros((n_col, n_col), dtype=dtype).at[Ai, Aj].add(Ax)
    x = jsp.linalg.solve(A, b)
    _log_and_test_equality(x, x_sp)

    klujax.free_numeric(numeric)
    klujax.free_symbolic(symbolic)


@log_test_name
@parametrize_dtypes
def test_solve_with_numeric_batched(dtype):
    Ai, Aj, Ax, b = _get_rand_arrs_2d((n_lhs := 3), 15, (n_col := 5), dtype=dtype)
    symbolic = klujax.analyze(Ai, Aj, n_col)
    numeric = klujax.factor(Ai, Aj, Ax, symbolic)

    x_sp = klujax.solve_with_numeric(numeric, b, symbolic)

    # Dense verification
    op_dense = jax.vmap(jsp.linalg.solve, (0, 0), 0)
    A = jnp.zeros((n_lhs, n_col, n_col), dtype=dtype).at[:, Ai, Aj].add(Ax)
    x = op_dense(A, b)
    _log_and_test_equality(x, x_sp)

    klujax.free_numeric(numeric)
    klujax.free_symbolic(symbolic)

def _get_rand_arrs_1d(n_nz, n_col, *, dtype, seed=33):
    Axkey, Aikey, Ajkey, bkey = jax.random.split(jax.random.PRNGKey(seed), 4)
    Ai = jax.random.randint(Aikey, (n_nz,), 0, n_col, jnp.int32)
    Aj = jax.random.randint(Ajkey, (n_nz,), 0, n_col, jnp.int32)
    Ax = jax.random.normal(Axkey, (n_nz,), dtype=dtype)
    # Add diagonal to ensure matrix is invertible
    diag_i = jnp.arange(n_col, dtype=jnp.int32)
    diag_x = jnp.ones(n_col, dtype=dtype) * 10.0
    Ai = jnp.concatenate([Ai, diag_i])
    Aj = jnp.concatenate([Aj, diag_i])
    Ax = jnp.concatenate([Ax, diag_x])
    Ai, Aj, Ax = coalesce(Ai, Aj, Ax)
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
    # Add diagonal to ensure matrix is invertible
    diag_i = jnp.arange(n_col, dtype=jnp.int32)
    diag_x = jnp.ones((n_lhs, n_col), dtype=dtype) * 10.0
    Ai = jnp.concatenate([Ai, diag_i])
    Aj = jnp.concatenate([Aj, diag_i])
    Ax = jnp.concatenate([Ax, diag_x], axis=1)
    Ai, Aj, Ax = coalesce(Ai, Aj, Ax)
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
    # Add diagonal to ensure matrix is invertible
    diag_i = jnp.arange(n_col, dtype=jnp.int32)
    diag_x = jnp.ones((n_lhs, n_col), dtype=dtype) * 10.0
    Ai = jnp.concatenate([Ai, diag_i])
    Aj = jnp.concatenate([Aj, diag_i])
    Ax = jnp.concatenate([Ax, diag_x], axis=1)
    Ai, Aj, Ax = coalesce(Ai, Aj, Ax)
    b = jax.random.normal(bkey, (n_lhs, n_col, n_rhs), dtype=dtype)
    return Ai, Aj, Ax, b


def _log_and_test_equality(x, x_sp):
    print(f"\nx=\n{x}")
    print(f"\nx_sp=\n{x_sp}")
    print(f"\ndiff=\n{np.round(x_sp - x, 9)}")
    print(f"\nis_equal=\n{_is_almost_equal(x, x_sp)}")
    np.testing.assert_array_almost_equal(x, x_sp)


def _log(**kwargs):
    for k, v in kwargs.items():
        print(f"{k}={v}")


def _is_almost_equal(arr1, arr2):
    try:
        np.testing.assert_array_almost_equal(arr1, arr2)
    except AssertionError:
        return False
    else:
        return True

def test_solve_with_symbol_jvp():

    Ai = jnp.array([0, 1], dtype=jnp.int32)
    Aj = jnp.array([0, 1], dtype=jnp.int32)
    Ax = jnp.array([2.0, 4.0], dtype=jnp.float64) # Values
    b = jnp.array([10.0, 20.0], dtype=jnp.float64)

    # Pre-compute symbolic factorization
    n_col = 2
    symbolic = klujax.analyze(Ai, Aj, n_col)

    def solve_step(Ax_vals, b_vec):
        return klujax.solve_with_symbol(Ai, Aj, Ax_vals, b_vec, symbolic)

    # Tangents (perturbations)
    tangent_Ax = jnp.array([0.1, 0.1], dtype=jnp.float64)
    tangent_b = jnp.array([0.0, 0.0], dtype=jnp.float64)

    # This triggers the JVP rule lookup
    primals, tangents = jax.jvp(solve_step, (Ax, b), (tangent_Ax, tangent_b))
    assert primals.shape == (2,)  # noqa: S101
    assert tangents.shape == (2,)  # noqa: S101


# KLUHandleManager testing

def use_handle(manager, x):
    """Simulates any function requiring a concrete handle (e.g. a C pointer)."""
    assert not isinstance(manager.handle, jax.core.Tracer), (
        "Handle was traced! Got a Tracer instead of a concrete value."
    )
    return x * 2.0

def test_registration_traces_handle():

    manager = klujax.KLUHandleManager(jnp.array(0xDEADBEEF, dtype=jnp.int64),
                                      free_callable=lambda x: None)

    fn = jax.jit(use_handle)
    result = fn(manager, jnp.array(1.0))
    assert jnp.allclose(result, 2.0)

def test_registration_handle_concrete_under_grad():
    manager = klujax.KLUHandleManager(jnp.array(0xDEADBEEF, dtype=jnp.int64),
                                      free_callable=lambda x: None)
    fn = jax.grad(lambda x: use_handle(manager, x))
    fn(jnp.array(1.0))

def test_handle_survives_pytree_roundtrip():
    handle_val = jnp.array(0xDEADBEEF, dtype=jnp.int64)
    manager = klujax.KLUHandleManager(handle_val, free_callable=lambda x: None)

    leaves, treedef = jax.tree_util.tree_flatten(manager)
    reconstructed = treedef.unflatten(leaves)

    assert leaves == []
    assert int(reconstructed.handle) == int(handle_val)

def test_registration_handle_concrete_under_vmap():
    manager = klujax.KLUHandleManager(jnp.array(0xDEADBEEF, dtype=jnp.int64),
                                      free_callable=lambda x: None)
    fn = jax.vmap(lambda x: use_handle(manager, x))
    result = fn(jnp.ones((4,)))
    assert jnp.allclose(result, 2.0)