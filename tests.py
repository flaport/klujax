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
    Bp, Bi, Bk = coo_to_csc_fn(Ai, Aj, jnp.array([n_col], dtype=jnp.int32))  # type: ignore[misc]

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
    Bp, Bi, Bk = coo_to_csc_fn(Ai, Aj, jnp.array([n_col], dtype=jnp.int32))  # type: ignore[misc]

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
    x_csc = solve_csc_fn(Bp, Bi, Bk, Ax, b)  # type: ignore[misc]

    # Results should match
    _log_and_test_equality(x_ref, x_csc)


@log_test_name
@parametrize_dtypes
def test_solve_csc_reuse(dtype):
    """Test that CSC structure can be reused for multiple solves with different values."""
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
    Bp, Bi, Bk = coo_to_csc_fn(Ai, Aj, jnp.array([n_col], dtype=jnp.int32))  # type: ignore[misc]

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
    x1_csc = solve_csc_fn(Bp, Bi, Bk, Ax1, b1)  # type: ignore[misc]
    x1_ref = klujax.solve(Ai, Aj, Ax1, b1)
    _log_and_test_equality(x1_ref, x1_csc)

    # Solve second system (reusing CSC structure)
    x2_csc = solve_csc_fn(Bp, Bi, Bk, Ax2, b2)  # type: ignore[misc]
    x2_ref = klujax.solve(Ai, Aj, Ax2, b2)
    _log_and_test_equality(x2_ref, x2_csc)


# =============================================================================
# Tests for true split-solve API (klu_analyze, klu_factor, klu_solve)
# =============================================================================


def _coo_to_csc_numpy(n_col, Ai, Aj, Ax):
    """Convert COO format to CSC format using numpy."""
    Ai = np.array(Ai)
    Aj = np.array(Aj)
    Ax = np.array(Ax)
    n_nz = len(Ai)

    # Count non-zeros per column
    Bp = np.zeros(n_col + 1, dtype=np.int32)
    for j in Aj:
        Bp[j + 1] += 1
    Bp = np.cumsum(Bp)

    # Fill CSC arrays
    Bi = np.zeros(n_nz, dtype=np.int32)
    Bx = np.zeros(n_nz, dtype=Ax.dtype)
    col_counts = np.zeros(n_col, dtype=np.int32)

    for k in range(n_nz):
        col = Aj[k]
        dest = Bp[col] + col_counts[col]
        Bi[dest] = Ai[k]
        Bx[dest] = Ax[k]
        col_counts[col] += 1

    return Bp, Bi, Bx


@log_test_name
@parametrize_dtypes
def test_klu_split_solve_basic(dtype):
    """Test basic klu_analyze, klu_factor, klu_solve workflow."""
    # Create a simple 3x3 sparse matrix in COO format
    # A = [[4, 1, 0], [1, 3, 1], [0, 1, 2]]
    Ai = np.array([0, 0, 1, 1, 1, 2, 2], dtype=np.int32)
    Aj = np.array([0, 1, 0, 1, 2, 1, 2], dtype=np.int32)
    Ax = np.array([4, 1, 1, 3, 1, 1, 2], dtype=dtype)
    n_col = 3

    # Convert to CSC
    Bp, Bi, Bx = _coo_to_csc_numpy(n_col, Ai, Aj, Ax)

    # Step 1: Symbolic analysis
    sym_handle = klujax_cpp.klu_analyze(Bp, Bi)
    assert sym_handle > 0, "Expected positive symbolic handle"

    # Step 2: Numeric factorization
    if np.issubdtype(dtype, np.complexfloating):
        num_handle = klujax_cpp.klu_factor_c128(sym_handle, Bp, Bi, Bx)
    else:
        num_handle = klujax_cpp.klu_factor_f64(sym_handle, Bp, Bi, Bx)
    assert num_handle > 0, "Expected positive numeric handle"

    # Step 3: Solve Ax = b
    # Create b such that x = [1, 1, 1]
    A = np.array([[4, 1, 0], [1, 3, 1], [0, 1, 2]], dtype=dtype)
    x_expected = np.array([[1], [1], [1]], dtype=dtype)
    b = A @ x_expected

    if np.issubdtype(dtype, np.complexfloating):
        x = klujax_cpp.klu_solve_c128(sym_handle, num_handle, b)
    else:
        x = klujax_cpp.klu_solve_f64(sym_handle, num_handle, b)

    np.testing.assert_array_almost_equal(x, x_expected)

    # Cleanup
    klujax_cpp.klu_free_numeric(num_handle)
    klujax_cpp.klu_free_symbolic(sym_handle)


@log_test_name
@parametrize_dtypes
def test_klu_split_solve_reuse_symbolic(dtype):
    """Test reusing symbolic analysis with different numeric values."""
    # Create sparse matrix structure
    Ai = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
    Aj = np.array([0, 1, 0, 1, 1, 2], dtype=np.int32)
    n_col = 3

    # Two different value sets for same sparsity
    Ax1 = np.array([4, 1, 1, 3, 1, 2], dtype=dtype)
    Ax2 = np.array([5, 2, 2, 4, 2, 3], dtype=dtype)

    # Convert to CSC (structure only depends on Ai, Aj)
    Bp, Bi, Bx1 = _coo_to_csc_numpy(n_col, Ai, Aj, Ax1)
    _, _, Bx2 = _coo_to_csc_numpy(n_col, Ai, Aj, Ax2)

    # Step 1: Symbolic analysis (done once)
    sym_handle = klujax_cpp.klu_analyze(Bp, Bi)

    # Build dense matrices for verification
    A1 = np.zeros((n_col, n_col), dtype=dtype)
    A2 = np.zeros((n_col, n_col), dtype=dtype)
    for k, (i, j) in enumerate(zip(Ai, Aj)):
        A1[i, j] = Ax1[k]
        A2[i, j] = Ax2[k]

    # Test with first values
    factor_fn = (
        klujax_cpp.klu_factor_c128
        if np.issubdtype(dtype, np.complexfloating)
        else klujax_cpp.klu_factor_f64
    )
    solve_fn = (
        klujax_cpp.klu_solve_c128
        if np.issubdtype(dtype, np.complexfloating)
        else klujax_cpp.klu_solve_f64
    )

    num_handle1 = factor_fn(sym_handle, Bp, Bi, Bx1)
    b1 = A1 @ np.array([[1], [2], [3]], dtype=dtype)
    x1 = solve_fn(sym_handle, num_handle1, b1)
    np.testing.assert_array_almost_equal(x1, [[1], [2], [3]])

    # Test with second values (reusing symbolic)
    num_handle2 = factor_fn(sym_handle, Bp, Bi, Bx2)
    b2 = A2 @ np.array([[3], [2], [1]], dtype=dtype)
    x2 = solve_fn(sym_handle, num_handle2, b2)
    np.testing.assert_array_almost_equal(x2, [[3], [2], [1]])

    # Cleanup
    klujax_cpp.klu_free_numeric(num_handle1)
    klujax_cpp.klu_free_numeric(num_handle2)
    klujax_cpp.klu_free_symbolic(sym_handle)


@log_test_name
def test_klu_split_solve_multiple_rhs():
    """Test solving with multiple right-hand sides.

    Note: The low-level KLU functions operate on raw memory and expect
    column-major (Fortran) layout for b and x matrices.
    """
    # 3x3 matrix
    Ai = np.array([0, 1, 0, 1, 2, 2], dtype=np.int32)
    Aj = np.array([0, 0, 1, 1, 1, 2], dtype=np.int32)
    Ax = np.array([4.0, 1.0, 1.0, 3.0, 1.0, 2.0], dtype=np.float64)
    n_col = 3
    n_rhs = 3

    Bp, Bi, Bx = _coo_to_csc_numpy(n_col, Ai, Aj, Ax)

    sym_handle = klujax_cpp.klu_analyze(Bp, Bi)
    num_handle = klujax_cpp.klu_factor_f64(sym_handle, Bp, Bi, Bx)

    # Build dense matrix for reference
    A = np.zeros((n_col, n_col), dtype=np.float64)
    for k, (i, j) in enumerate(zip(Ai, Aj)):
        A[i, j] = Ax[k]

    # Solve column by column to verify
    for rhs_idx in range(n_rhs):
        x_expected = np.array(
            [[1 + rhs_idx], [2 + rhs_idx], [3 + rhs_idx]], dtype=np.float64
        )
        b = A @ x_expected
        x = klujax_cpp.klu_solve_f64(sym_handle, num_handle, b)
        np.testing.assert_array_almost_equal(x, x_expected)

    klujax_cpp.klu_free_numeric(num_handle)
    klujax_cpp.klu_free_symbolic(sym_handle)
