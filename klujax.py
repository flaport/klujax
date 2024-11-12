""" klujax: a KLU solver for JAX """

# Metadata ============================================================================

__version__ = "0.3.0"
__author__ = "Floris Laporte"
__all__ = ["solve", "coo_mul_vec"]

# Imports =============================================================================

import sys
from functools import partial

import jax
import jax.extend
import jax.numpy as jnp
import numpy as np
from jax import core, lax
from jax.core import ShapedArray
from jax.interpreters import ad, batching, mlir
from jaxtyping import Array

import klujax_cpp

# Config ==============================================================================

DEBUG = False
_log = lambda s: None if not DEBUG else print(s, file=sys.stderr)  # noqa: E731

# The main functions ==================================================================


@jax.jit
def solve(Ai: Array, Aj: Array, Ax: Array, b: Array) -> Array:
    """Solve for x in the sparse linear system Ax=b.

    Args:
        Ai: [n_nz; int32]: the row indices of the sparse matrix A
        Aj: [n_nz; int32]: the column indices of the sparse matrix A
        Ax: [n_lhs? x n_nz; float64|complex128]: the values of the sparse matrix A
        b:  [n_lhs? x n_col x n_rhs?; float64|complex128]: the target vector

    Returns:
        x: the result (x≈A^-1b)
    """
    if any(x.dtype in COMPLEX_DTYPES for x in (Ax, b)):
        result = solve_c128.bind(
            Ai.astype(jnp.int32),
            Aj.astype(jnp.int32),
            Ax.astype(jnp.complex128),
            b.astype(jnp.complex128),
        )
    else:
        result = solve_f64.bind(
            Ai.astype(jnp.int32),
            Aj.astype(jnp.int32),
            Ax.astype(jnp.float64),
            b.astype(jnp.float64),
        )

    return result  # type: ignore


@jax.jit
def coo_mul_vec(Ai: Array, Aj: Array, Ax: Array, x: Array) -> Array:
    """Multiply a sparse matrix with a vector: Ax=b

    Args:
        Ai: [n_nz; int32]: the row indices of the sparse matrix A
        Aj: [n_nz; int32]: the column indices of the sparse matrix A
        Ax: [n_lhs? x n_nz; float64|complex128]: the values of the sparse matrix A
        x:  [n_lhs? x n_col x n_rhs?; float64|complex128]: the vector that's being multiplied by A

    Returns:
        b: the result of the multiplication (b=Ax)
    """
    if any(x.dtype in COMPLEX_DTYPES for x in (Ax, x)):
        result = coo_mul_vec_c128.bind(
            Ai.astype(jnp.int32),
            Aj.astype(jnp.int32),
            Ax.astype(jnp.complex128),
            x.astype(jnp.complex128),
        )
    else:
        result = coo_mul_vec_f64.bind(
            Ai.astype(jnp.int32),
            Aj.astype(jnp.int32),
            Ax.astype(jnp.float64),
            x.astype(jnp.float64),
        )
    return result  # type: ignore


# Constants ===========================================================================

COMPLEX_DTYPES = (
    np.complex64,
    np.complex128,
    # np.complex256,
    jnp.complex64,
    jnp.complex128,
)

# Primitives ==========================================================================

solve_f64 = core.Primitive("solve_f64")
solve_c128 = core.Primitive("solve_c128")
coo_mul_vec_f64 = core.Primitive("coo_mul_vec_f64")
coo_mul_vec_c128 = core.Primitive("coo_mul_vec_c128")

# Register XLA extensions ==============================================================

jax.extend.ffi.register_ffi_target(
    "_solve_f64",
    klujax_cpp.solve_f64(),
    platform="cpu",
)

jax.extend.ffi.register_ffi_target(
    "_coo_mul_vec_f64",
    klujax_cpp.coo_mul_vec_f64(),
    platform="cpu",
)

jax.extend.ffi.register_ffi_target(
    "_solve_c128",
    klujax_cpp.solve_c128(),
    platform="cpu",
)

jax.extend.ffi.register_ffi_target(
    "_coo_mul_vec_c128",
    klujax_cpp.coo_mul_vec_c128(),
    platform="cpu",
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


# Implementations =====================================================================


@solve_f64.def_impl
def solve_f64_impl(Ai, Aj, Ax, b) -> jnp.ndarray:
    Ai, Aj, Ax, b, shape = _prepare_arguments(Ai, Aj, Ax, b)

    _b = b.transpose(0, 2, 1)
    call = jax.extend.ffi.ffi_call(
        "_solve_f64",
        jax.ShapeDtypeStruct(_b.shape, _b.dtype),
        vmap_method="broadcast_all",
    )
    result = call(  # type: ignore
        Ai,
        Aj,
        Ax,
        _b,
    )
    return result.transpose(0, 2, 1).reshape(*shape)  # type: ignore


@solve_c128.def_impl
def solve_c128_impl(Ai, Aj, Ax, b) -> jnp.ndarray:
    Ai, Aj, Ax, b, shape = _prepare_arguments(Ai, Aj, Ax, b)

    _b = b.transpose(0, 2, 1)
    call = jax.extend.ffi.ffi_call(
        "_solve_c128",
        jax.ShapeDtypeStruct(_b.shape, _b.dtype),
        vmap_method="broadcast_all",
    )
    result = call(  # type: ignore
        Ai,
        Aj,
        Ax,
        _b,
    )
    return result.transpose(0, 2, 1).reshape(*shape)  # type: ignore


@coo_mul_vec_f64.def_impl
def coo_mul_vec_f64_impl(Ai, Aj, Ax, x) -> jnp.ndarray:
    Ai, Aj, Ax, x, shape = _prepare_arguments(Ai, Aj, Ax, x)
    _x = x.transpose(0, 2, 1)
    call = jax.extend.ffi.ffi_call(
        "_coo_mul_vec_f64",
        jax.ShapeDtypeStruct(_x.shape, _x.dtype),
        vmap_method="broadcast_all",
    )
    result = call(  # type: ignore
        Ai,
        Aj,
        Ax,
        _x,
    )
    return result.transpose(0, 2, 1).reshape(*shape)  # type: ignore


@coo_mul_vec_c128.def_impl
def coo_mul_vec_c128_impl(Ai, Aj, Ax, x) -> jnp.ndarray:
    Ai, Aj, Ax, x, shape = _prepare_arguments(Ai, Aj, Ax, x)
    _x = x.transpose(0, 2, 1)
    call = jax.extend.ffi.ffi_call(
        "_coo_mul_vec_c128",
        jax.ShapeDtypeStruct(_x.shape, _x.dtype),
        vmap_method="broadcast_all",
    )
    result = call(  # type: ignore
        Ai,
        Aj,
        Ax,
        _x,
    )
    return result.transpose(0, 2, 1).reshape(*shape)  # type: ignore


def _prepare_arguments(Ai, Aj, Ax, x):
    Ai = jnp.asarray(Ai)
    Aj = jnp.asarray(Aj)
    Ax = jnp.asarray(Ax)
    x = jnp.asarray(x)
    shape = x.shape

    _log(f"{Ai.dtype=}")
    _log(f"{Aj.dtype=}")
    _log(f"{Ai.max()=}")
    _log(f"{Aj.max()=}")
    _log(f"{Ax.dtype=}")
    _log(f"{x.dtype=}")
    _log(f"{Ax.shape=}")
    _log(f"{x.shape=}")

    prefer_x_rhs_over_lhs = (Ax.ndim < 2) or (Ax.shape[0] != x.shape[0])
    _log(f"{prefer_x_rhs_over_lhs=}")

    if Ax.ndim > 2:
        raise ValueError(
            f"Ax should be at most 2D with shape: (n_lhs, n_nz). Got: {Ax.shape}. "
            "Note: jax.vmap is supported. Use it if needed."
        )
    else:
        Ax = jnp.atleast_2d(Ax)

    a_n_lhs, n_nz = Ax.shape
    Ax = Ax.reshape(-1, n_nz)
    _log(f"{Ax.shape=}")
    _log(f"{n_nz=}")

    if x.ndim == 0:
        x = x[None, None, None]
        x_n_lhs, n_col, n_rhs = x.shape
        shape_includes_lhs = False
    elif x.ndim == 1:
        x = x[None, :, None]
        x_n_lhs, n_col, n_rhs = x.shape
        shape_includes_lhs = False
    elif x.ndim == 2 and prefer_x_rhs_over_lhs:
        x = x[None, :, :]
        x_n_lhs, n_col, n_rhs = x.shape
        shape_includes_lhs = False
    elif x.ndim == 2 and not prefer_x_rhs_over_lhs:
        x = x[:, :, None]
        x_n_lhs, n_col, n_rhs = x.shape
        shape_includes_lhs = True
    elif x.ndim == 3:
        x_n_lhs, n_col, n_rhs = x.shape
        shape_includes_lhs = True
    else:
        raise ValueError(
            f"x should be at most 3D with shape: (n_lhs, n_col, n_rhs). Got: {x.shape}. "
            "Note: jax.vmap is supported. Use it if needed."
        )
    _log(f"{x.shape=}")
    _log(f"{n_col=}")
    _log(f"{n_rhs=}")

    if a_n_lhs == x_n_lhs:
        n_lhs = a_n_lhs
    elif a_n_lhs > x_n_lhs:
        if not x_n_lhs == 1:
            raise ValueError(
                f"Cannot broadcast n_lhs for x into n_lhs for Ax. "
                f"Got: n_lhs[x]={x_n_lhs}; n_lhs[Ax]={a_n_lhs}."
            )
        n_lhs = a_n_lhs
        if shape_includes_lhs:
            shape = (n_lhs,) + shape[1:]
        else:
            shape = (n_lhs,) + shape
    else:
        if not a_n_lhs == 1:
            raise ValueError(
                f"Cannot broadcast n_lhs for Ax into n_lhs for x. "
                f"Got: n_lhs[x]={x_n_lhs}; n_lhs[Ax]={a_n_lhs}."
            )
        n_lhs = x_n_lhs
    _log(f"{n_lhs=}")

    Ax = jnp.broadcast_to(Ax, (n_lhs, n_nz))
    x = jnp.broadcast_to(x, (n_lhs, n_col, n_rhs))

    _log(f"{Ax.shape=}")
    _log(f"{x.shape=}")

    # We retain the old shape of b so the result of the primitive can be
    # reshaped to the expected shape.
    return Ai, Aj, Ax, x, shape


# Lowerings ===========================================================================

solve_f64_lowering = mlir.lower_fun(solve_f64_impl, multiple_results=False)
mlir.register_lowering(solve_f64, solve_f64_lowering)

solve_c128_lowering = mlir.lower_fun(solve_c128_impl, multiple_results=False)
mlir.register_lowering(solve_c128, solve_c128_lowering)

coo_mul_vec_f64_lowering = mlir.lower_fun(coo_mul_vec_f64_impl, multiple_results=False)
mlir.register_lowering(coo_mul_vec_f64, coo_mul_vec_f64_lowering)

coo_mul_vec_c128_lowering = mlir.lower_fun(
    coo_mul_vec_c128_impl, multiple_results=False
)
mlir.register_lowering(coo_mul_vec_c128, coo_mul_vec_c128_lowering)

# Abstract Evaluations ================================================================


@solve_f64.def_abstract_eval
@solve_c128.def_abstract_eval
@coo_mul_vec_f64.def_abstract_eval
@coo_mul_vec_c128.def_abstract_eval
def coo_vec_operation_abstract_eval(Ai, Aj, Ax, b):
    return ShapedArray(b.shape, b.dtype)


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
        Ax = jnp.moveaxis(Ax, aAx, 0)  # treat as lhs
        b = jnp.moveaxis(b, ab, 0)  # treat as lhs
        result = operation(Ai, Aj, Ax, b)
        return result, 0

    if ab is None:
        assert isinstance(aAx, int)
        Ax = jnp.moveaxis(Ax, aAx, 0)  # treat as lhs
        b = jnp.broadcast_to(b[None], (Ax.shape[0], *b.shape))
        result = operation(Ai, Aj, Ax, b)
        return result, 0

    if aAx is None:
        assert isinstance(ab, int)
        _log(f"vmap: {b.shape=}")
        b = jnp.moveaxis(b, ab, -1)  # treat as rhs
        _log(f"vmap: {b.shape=}")
        shape = b.shape
        if b.ndim == 0:
            b = b[None, None, None]
        elif b.ndim == 1:
            b = b[None, None, :]
        elif b.ndim == 2:
            b = b[None, :, :]
        elif b.ndim == 3:
            b = b[:, :, :]

        b = b.reshape(b.shape[0], b.shape[1], -1)

        _log(f"vmap: {b.shape=}")
        # b is now guaranteed to have shape (n_lhs, n_col, n_rhs)
        result = operation(Ai, Aj, Ax, b)
        result = result.reshape(*shape)
        return result, -1

    raise ValueError("invalid arguments for vmap")
