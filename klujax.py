""" klujax: a KLU solver for JAX """

__version__ = "0.2.10"
__author__ = "Floris Laporte"
__all__ = ["solve", "coo_mul_vec"]


# Imports =============================================================================

# stdlib
from functools import partial

# 3rd party
import jax
import jax.extend as jex
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.interpreters import ad, batching

# this lib
import klujax_cpp

# Config ==============================================================================


jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


# Constants ===========================================================================


COMPLEX_DTYPES = (
    np.complex64,
    np.complex128,
    # np.complex256,
    jnp.complex64,
    jnp.complex128,
)


# Helper Decorators ===================================================================


def ad_register(primitive):
    def decorator(fun):
        ad.primitive_jvps[primitive] = fun
        return fun

    return decorator


def transpose_register(primitive):
    def decorator(fun):
        ad.primitive_transposes[primitive] = fun
        return fun

    return decorator


def vmap_register(primitive, operation):
    def decorator(fun):
        batching.primitive_batchers[primitive] = partial(fun, operation)
        return fun

    return decorator


# Register XLA Extensions ==============================================================

jex.ffi.register_ffi_target(
    "solve_f64",
    klujax_cpp.solve_f64(),
    platform="cpu",
)

jex.ffi.register_ffi_target(
    "coo_mul_vec_f64",
    klujax_cpp.coo_mul_vec_f64(),
    platform="cpu",
)

jex.ffi.register_ffi_target(
    "solve_c128",
    klujax_cpp.solve_c128(),
    platform="cpu",
)

jex.ffi.register_ffi_target(
    "coo_mul_vec_c128",
    klujax_cpp.coo_mul_vec_c128(),
    platform="cpu",
)

# The Main Functions ==================================================================


def solve(Ai, Aj, Ax, b):
    Ai = jnp.asarray(Ai)
    Aj = jnp.asarray(Aj)
    Ax = jnp.asarray(Ax)
    b = jnp.asarray(b)
    shape = b.shape

    if b.ndim < 2:
        b = jnp.atleast_2d(b).T

    if any(x.dtype in COMPLEX_DTYPES for x in (Ax, b)):
        result = solve_c128(
            Ai.astype(jnp.int32),
            Aj.astype(jnp.int32),
            Ax.astype(jnp.complex128),
            b.astype(jnp.complex128),
        )
    else:
        result = solve_f64(
            Ai.astype(jnp.int32),
            Aj.astype(jnp.int32),
            Ax.astype(jnp.float64),
            b.astype(jnp.float64),
        )

    return result.reshape(*shape)


def coo_mul_vec(Ai, Aj, Ax, x):
    Ai = jnp.asarray(Ai)
    Aj = jnp.asarray(Aj)
    Ax = jnp.asarray(Ax)
    x = jnp.asarray(x)
    shape = x.shape

    if x.ndim < 2:
        x = jnp.atleast_2d(x).T

    if any(x.dtype in COMPLEX_DTYPES for x in (Ax, x)):
        result = coo_mul_vec_c128(
            Ai.astype(jnp.int32),
            Aj.astype(jnp.int32),
            Ax.astype(jnp.complex128),
            x.astype(jnp.complex128),
        )
    else:
        result = coo_mul_vec_f64(
            Ai.astype(jnp.int32),
            Aj.astype(jnp.int32),
            Ax.astype(jnp.float64),
            x.astype(jnp.float64),
        )
    return result.reshape(*shape)


def solve_f64(Ai, Aj, Ax, b):
    *_, n_col, n_rhs = b.shape
    _b = b.reshape(-1, n_col, n_rhs)

    *ns_lhs, n_nz = Ax.shape
    _Ax = Ax.reshape(-1, n_nz)
    n_lhs, _ = _Ax.shape

    call = jex.ffi.ffi_call(
        "solve_f64",
        jax.ShapeDtypeStruct(_b.shape, _b.dtype),
        vmap_method="broadcast_all",
    )
    b = call(  # type: ignore
        Ai,
        Aj,
        _Ax,
        _b,
        n_col=np.int32(n_col),
        n_rhs=np.int32(n_rhs),
        n_lhs=np.int32(n_lhs),
        n_nz=np.int32(n_nz),
    )
    return b.reshape(*ns_lhs, n_col, n_rhs)  # type: ignore


def coo_mul_vec_f64(Ai, Aj, Ax, x):
    *_, n_col, n_rhs = x.shape
    _x = x.reshape(-1, n_col, n_rhs)

    *ns_lhs, n_nz = Ax.shape
    _Ax = Ax.reshape(-1, n_nz)
    n_lhs, _ = _Ax.shape

    call = jex.ffi.ffi_call(
        "coo_mul_vec_f64",
        jax.ShapeDtypeStruct(_x.shape, _x.dtype),
        vmap_method="broadcast_all",
    )
    b = call(  # type: ignore
        Ai,
        Aj,
        _Ax,
        _x,
        n_col=np.int32(n_col),
        n_rhs=np.int32(n_rhs),
        n_lhs=np.int32(n_lhs),
        n_nz=np.int32(n_nz),
    )
    return b.reshape(*ns_lhs, n_col, n_rhs)  # type: ignore


def solve_c128(Ai, Aj, Ax, b):
    *_, n_col, n_rhs = b.shape
    _b = b.reshape(-1, n_col, n_rhs)

    *ns_lhs, n_nz = Ax.shape
    _Ax = Ax.reshape(-1, n_nz)
    n_lhs, _ = _Ax.shape

    _Ax = _Ax.view(np.float64)
    _b = _b.view(np.float64)

    call = jex.ffi.ffi_call(
        "solve_c128",
        jax.ShapeDtypeStruct(_b.shape, _b.dtype),
        vmap_method="broadcast_all",
    )
    x = call(  # type: ignore
        Ai,
        Aj,
        _Ax,
        _b,
        n_col=np.int32(n_col),
        n_rhs=np.int32(n_rhs),
        n_lhs=np.int32(n_lhs),
        n_nz=np.int32(n_nz),
    )
    return x.view(b.dtype).reshape(*ns_lhs, n_col, n_rhs)  # type: ignore


def coo_mul_vec_c128(Ai, Aj, Ax, x):
    *_, n_col, n_rhs = x.shape
    _x = x.reshape(-1, n_col, n_rhs)

    *ns_lhs, n_nz = Ax.shape
    _Ax = Ax.reshape(-1, n_nz)
    n_lhs, _ = _Ax.shape

    _Ax = _Ax.view(np.float64)
    _x = _x.view(np.float64)
    call = jex.ffi.ffi_call(
        "coo_mul_vec_c128",
        jax.ShapeDtypeStruct(_x.shape, _x.dtype),
        vmap_method="broadcast_all",
    )
    y = call(  # type: ignore
        Ai,
        Aj,
        _Ax,
        _x,
        n_col=np.int32(n_col),
        n_rhs=np.int32(n_rhs),
        n_lhs=np.int32(n_lhs),
        n_nz=np.int32(n_nz),
    )
    return y.view(x.dtype).reshape(*ns_lhs, n_col, n_rhs)  # type: ignore


# Forward Gradients ===================================================================


@ad_register(solve_f64)
@ad_register(solve_c128)
def solve_value_and_jvp(arg_values, arg_tangents):
    Ai, Aj, Ax, b = arg_values
    dAi, dAj, dAx, db = arg_tangents
    dAx = dAx if not isinstance(dAx, ad.Zero) else lax.zeros_like_array(Ax)
    dAi = dAi if not isinstance(dAi, ad.Zero) else lax.zeros_like_array(Ai)
    dAj = dAj if not isinstance(dAj, ad.Zero) else lax.zeros_like_array(Aj)
    db = db if not isinstance(db, ad.Zero) else lax.zeros_like_array(b)
    x = solve(Ai, Aj, Ax, b)
    dA_x = coo_mul_vec(Ai, Aj, dAx, x)
    invA_dA_x = solve(Ai, Aj, Ax, dA_x)
    invA_db = solve(Ai, Aj, Ax, db)
    return x, -invA_dA_x + invA_db


@ad_register(coo_mul_vec_f64)
@ad_register(coo_mul_vec_c128)
def coo_mul_vec_value_and_jvp(arg_values, arg_tangents):
    Ai, Aj, Ax, b = arg_values
    dAi, dAj, dAx, db = arg_tangents
    dAx = dAx if not isinstance(dAx, ad.Zero) else lax.zeros_like_array(Ax)
    dAi = dAi if not isinstance(dAi, ad.Zero) else lax.zeros_like_array(Ai)
    dAj = dAj if not isinstance(dAj, ad.Zero) else lax.zeros_like_array(Aj)
    db = db if not isinstance(db, ad.Zero) else lax.zeros_like_array(b)
    x = coo_mul_vec(Ai, Aj, Ax, b)
    dA_b = coo_mul_vec(Ai, Aj, dAx, b)
    A_db = coo_mul_vec(Ai, Aj, Ax, db)
    return x, dA_b + A_db


# Backward Gradients through Transposition ============================================


@transpose_register(solve_f64)
@transpose_register(solve_c128)
def solve_transpose(ct, Ai, Aj, Ax, b):
    assert not ad.is_undefined_primal(Ai)
    assert not ad.is_undefined_primal(Aj)
    assert not ad.is_undefined_primal(Ax)
    assert not ad.is_undefined_primal(Ax)
    assert ad.is_undefined_primal(b)
    return None, None, None, solve(Aj, Ai, Ax.conj(), ct)  # = inv(A).H@ct [= ct@inv(A)]


@transpose_register(coo_mul_vec_f64)
@transpose_register(coo_mul_vec_c128)
def coo_mul_vec_transpose(ct, Ai, Aj, Ax, b):
    assert not ad.is_undefined_primal(Ai)
    assert not ad.is_undefined_primal(Aj)
    assert ad.is_undefined_primal(Ax) != ad.is_undefined_primal(b)  # xor

    if ad.is_undefined_primal(b):
        return None, None, None, coo_mul_vec(Aj, Ai, Ax.conj(), ct)  # = A.T@ct [= ct@A]
    else:
        dA = ct[Ai] * b[Aj]
        dA = dA.reshape(dA.shape[0], -1).sum(-1)  # not sure about this...
        return None, None, dA, None


# Vectorization (vmap) ================================================================


@vmap_register(solve_f64, solve)
@vmap_register(solve_c128, solve)
@vmap_register(coo_mul_vec_f64, coo_mul_vec)
@vmap_register(coo_mul_vec_c128, coo_mul_vec)
def coo_vec_operation_vmap(operation, vector_arg_values, batch_axes):
    aAi, aAj, aAx, ab = batch_axes
    Ai, Aj, Ax, b = vector_arg_values

    assert aAi is None, "Ai cannot be vectorized."
    assert aAj is None, "Aj cannot be vectorized."

    if aAx is not None and ab is not None:
        assert isinstance(aAx, int) and isinstance(ab, int)
        n_lhs = Ax.shape[aAx]
        if ab != 0:
            Ax = jnp.moveaxis(Ax, aAx, 0)
        if ab != 0:
            b = jnp.moveaxis(b, ab, 0)
        result = operation(Ai, Aj, Ax, b)
        return result, 0

    if ab is None:
        assert isinstance(aAx, int)
        n_lhs = Ax.shape[aAx]
        if aAx != 0:
            Ax = jnp.moveaxis(Ax, aAx, 0)
        b = jnp.broadcast_to(b[None], (Ax.shape[0], *b.shape))
        result = operation(Ai, Aj, Ax, b)
        return result, 0

    if aAx is None:
        assert isinstance(ab, int)
        if ab != 0:
            b = jnp.moveaxis(b, ab, 0)
        n_lhs, n_col, *n_rhs_list = b.shape
        n_rhs = np.prod(np.array(n_rhs_list, dtype=np.int32))
        b = b.reshape(n_lhs, n_col, n_rhs).transpose((1, 0, 2)).reshape(n_col, -1)
        result = operation(Ai, Aj, Ax, b)
        result = result.reshape(n_col, n_lhs, *n_rhs_list)
        return result, 1

    raise ValueError("invalid arguments for vmap")

