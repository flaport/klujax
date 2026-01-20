import sys
from functools import wraps

import jax
import jax.ffi
import jax.numpy as jnp
import jax.scipy as jsp
import klujax_cpp
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


# =============================================================================
# Tests for low-level split-solve API (klujax_cpp)
# =============================================================================


def _register_ffi_targets():
    """Register FFI targets for low-level API tests."""
    import jax.ffi

    # Only register if not already registered
    try:
        jax.ffi.register_ffi_target("test_coo_to_csc", klujax_cpp.coo_to_csc())
        jax.ffi.register_ffi_target("test_solve_csc_f64", klujax_cpp.solve_csc_f64())
        jax.ffi.register_ffi_target("test_solve_csc_c128", klujax_cpp.solve_csc_c128())
    except ValueError:
        pass  # Already registered


@log_test_name
def test_coo_to_csc():
    """Test that coo_to_csc correctly converts COO to CSC format."""
    import jax.ffi

    _register_ffi_targets()

    # Simple 3x3 matrix in COO format
    # [[1, 0, 2],
    #  [0, 3, 0],
    #  [4, 0, 5]]
    Ai = jnp.array([0, 0, 1, 2, 2], dtype=jnp.int32)  # row indices
    Aj = jnp.array([0, 2, 1, 0, 2], dtype=jnp.int32)  # col indices
    n_col = 3
    n_nz = len(Ai)

    # Call coo_to_csc via FFI
    coo_to_csc_fn = jax.ffi.ffi_call(
        "test_coo_to_csc",
        (
            jax.ShapeDtypeStruct((n_col + 1,), jnp.int32),  # Bp
            jax.ShapeDtypeStruct((n_nz,), jnp.int32),  # Bi
            jax.ShapeDtypeStruct((n_nz,), jnp.int32),  # Bk
        ),
        vmap_method="sequential",
    )
    Bp, Bi, Bk = coo_to_csc_fn(Ai, Aj, jnp.array([n_col], dtype=jnp.int32))

    # Verify CSC format is correct
    # Column 0: rows 0, 2 (values at COO indices 0, 3)
    # Column 1: row 1 (value at COO index 2)
    # Column 2: rows 0, 2 (values at COO indices 1, 4)
    assert Bp.shape == (n_col + 1,)
    assert Bi.shape == (n_nz,)
    assert Bk.shape == (n_nz,)

    # Bp should be column pointers: [0, 2, 3, 5]
    np.testing.assert_array_equal(np.array(Bp), [0, 2, 3, 5])


@log_test_name
@parametrize_dtypes
def test_solve_csc_matches_solve(dtype):
    """Test that solve_csc produces the same result as klujax.solve."""
    import jax.ffi

    _register_ffi_targets()

    # Generate test data
    Ai, Aj, Ax, b = _get_rand_arrs_3d(n_lhs=2, n_nz=8, n_col=5, n_rhs=3, dtype=dtype)
    n_col = b.shape[1]
    n_nz = len(Ai)

    # Get reference solution using high-level API
    x_ref = klujax.solve(Ai, Aj, Ax, b)

    # Convert COO to CSC
    coo_to_csc_fn = jax.ffi.ffi_call(
        "test_coo_to_csc",
        (
            jax.ShapeDtypeStruct((n_col + 1,), jnp.int32),
            jax.ShapeDtypeStruct((n_nz,), jnp.int32),
            jax.ShapeDtypeStruct((n_nz,), jnp.int32),
        ),
        vmap_method="sequential",
    )
    Bp, Bi, Bk = coo_to_csc_fn(Ai, Aj, jnp.array([n_col], dtype=jnp.int32))

    # Solve using CSC format
    if dtype == np.float64:
        target_name = "test_solve_csc_f64"
    else:
        target_name = "test_solve_csc_c128"

    solve_csc_fn = jax.ffi.ffi_call(
        target_name,
        jax.ShapeDtypeStruct(b.shape, dtype),
        vmap_method="sequential",
    )
    x_csc = solve_csc_fn(Bp, Bi, Bk, Ax, b)

    # Results should match
    _log_and_test_equality(x_ref, x_csc)


@log_test_name
@parametrize_dtypes
def test_solve_csc_reuse(dtype):
    """Test that CSC structure can be reused for multiple solves with different values."""
    import jax.ffi

    _register_ffi_targets()

    # Generate test data with same sparsity pattern but different values
    Ai, Aj, Ax1, b1 = _get_rand_arrs_3d(
        n_lhs=2, n_nz=8, n_col=5, n_rhs=2, dtype=dtype, seed=42
    )
    _, _, Ax2, b2 = _get_rand_arrs_3d(
        n_lhs=2, n_nz=8, n_col=5, n_rhs=2, dtype=dtype, seed=123
    )

    n_col = b1.shape[1]
    n_nz = len(Ai)

    # Convert COO to CSC once
    coo_to_csc_fn = jax.ffi.ffi_call(
        "test_coo_to_csc",
        (
            jax.ShapeDtypeStruct((n_col + 1,), jnp.int32),
            jax.ShapeDtypeStruct((n_nz,), jnp.int32),
            jax.ShapeDtypeStruct((n_nz,), jnp.int32),
        ),
        vmap_method="sequential",
    )
    Bp, Bi, Bk = coo_to_csc_fn(Ai, Aj, jnp.array([n_col], dtype=jnp.int32))

    if dtype == np.float64:
        target_name = "test_solve_csc_f64"
    else:
        target_name = "test_solve_csc_c128"

    solve_csc_fn = jax.ffi.ffi_call(
        target_name,
        jax.ShapeDtypeStruct(b1.shape, dtype),
        vmap_method="sequential",
    )

    # Solve first system
    x1_csc = solve_csc_fn(Bp, Bi, Bk, Ax1, b1)
    x1_ref = klujax.solve(Ai, Aj, Ax1, b1)
    _log_and_test_equality(x1_ref, x1_csc)

    # Solve second system (reusing CSC structure)
    x2_csc = solve_csc_fn(Bp, Bi, Bk, Ax2, b2)
    x2_ref = klujax.solve(Ai, Aj, Ax2, b2)
    _log_and_test_equality(x2_ref, x2_csc)
