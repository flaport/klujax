""" klujax: a KLU solver for JAX """

# Metadata ============================================================================

__version__ = "0.2.10"
__author__ = "Floris Laporte"
__all__ = ["solve", "coo_mul_vec"]

# Imports =============================================================================

from functools import partial

import jax
import jax.extend
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jax import core, lax
from jax._src.lib.mlir.dialects import hlo
from jax.core import ShapedArray
from jax.interpreters import ad, batching, mlir

import klujax_cpp

# Config ==============================================================================

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

# The main functions ==================================================================


def solve(Ai, Aj, Ax, b) -> jnp.ndarray:
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


def coo_mul_vec(Ai, Aj, Ax, b) -> jnp.ndarray:
    if any(x.dtype in COMPLEX_DTYPES for x in (Ax, b)):
        result = coo_mul_vec_c128.bind(
            Ai.astype(jnp.int32),
            Aj.astype(jnp.int32),
            Ax.astype(jnp.complex128),
            b.astype(jnp.complex128),
        )
    else:
        result = coo_mul_vec_f64.bind(
            Ai.astype(jnp.int32),
            Aj.astype(jnp.int32),
            Ax.astype(jnp.float64),
            b.astype(jnp.float64),
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

    # TODO: make expected shape consistent with expected shape for coo_mul_vec.
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

    # TODO: make expected shape consistent with expected shape for coo_mul_vec.
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
    call = jax.extend.ffi.ffi_call(
        "_coo_mul_vec_f64",
        jax.ShapeDtypeStruct(x.shape, x.dtype),
        vmap_method="broadcast_all",
    )
    result = call(  # type: ignore
        Ai,
        Aj,
        Ax,
        x,
    )
    return result.reshape(*shape)  # type: ignore


@coo_mul_vec_c128.def_impl
def coo_mul_vec_c128_impl(Ai, Aj, Ax, x) -> jnp.ndarray:
    Ai, Aj, Ax, x, shape = _prepare_arguments(Ai, Aj, Ax, x)
    call = jax.extend.ffi.ffi_call(
        "_coo_mul_vec_c128",
        jax.ShapeDtypeStruct(x.shape, x.dtype),
        vmap_method="broadcast_all",
    )
    result = call(  # type: ignore
        Ai,
        Aj,
        Ax,
        x,
    )
    return result.reshape(*shape)  # type: ignore


def _prepare_arguments(Ai, Aj, Ax, x):
    Ai = jnp.asarray(Ai)
    Aj = jnp.asarray(Aj)
    Ax = jnp.asarray(Ax)
    x = jnp.asarray(x)
    shape = x.shape

    prefer_x_rhs_over_lhs = (Ax.ndim < 2) or (Ax.shape[0] != x.shape[0])

    if Ax.ndim > 2:
        raise ValueError(
            f"Ax should be at most 2D with shape: (n_lhs, n_nz). Got: {Ax.shape}. "
            "Note: jax.vmap is supported. Use it if needed."
        )
    else:
        Ax = jnp.atleast_2d(Ax)

    a_n_lhs, n_nz = Ax.shape
    Ax = Ax.reshape(-1, n_nz)

    if x.ndim == 0:
        x = x[None, None, None]
        x_n_lhs, n_col, n_rhs = x.shape
    elif x.ndim == 1:
        x = x[None, :, None]
        x_n_lhs, n_col, n_rhs = x.shape
    elif x.ndim == 2 and prefer_x_rhs_over_lhs:
        x = x[None, :, :]
        x_n_lhs, n_col, n_rhs = x.shape
    elif x.ndim == 2 and not prefer_x_rhs_over_lhs:
        x = x[:, :, None]
        x_n_lhs, n_col, n_rhs = x.shape
    elif x.ndim == 3:
        x_n_lhs, n_col, n_rhs = x.shape
    else:
        raise ValueError(
            f"x should be at most 2D with shape: (n_lhs, n_col, n_rhs). Got: {x.shape}. "
            "Note: jax.vmap is supported. Use it if needed."
        )

    n_lhs = max(a_n_lhs, x_n_lhs)
    Ax = jnp.broadcast_to(Ax, (n_lhs, n_nz))
    x = jnp.broadcast_to(x, (n_lhs, n_col, n_rhs))

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
        n_lhs = Ax.shape[aAx]
        if aAx != 0:
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
        print(b.shape)
        n_lhs, n_col, *n_rhs_list = b.shape
        n_rhs = np.prod(np.array(n_rhs_list, dtype=np.int32))
        b = b.reshape(n_lhs, n_col, n_rhs).transpose((1, 0, 2)).reshape(n_col, -1)[None]
        result = operation(Ai, Aj, Ax, b)
        result = result.reshape(n_col, n_lhs, *n_rhs_list)
        return result, 1

    raise ValueError("invalid arguments for vmap")


# Quick Tests =========================================================================

if __name__ == "__main__":
    n_nz = 8
    n_col = 5
    n_rhs = 1
    n_lhs = 3  # only used in batched & vmap
    dtype = np.complex128

    ## SINGLE =========================================================================

    if True:
        print("single")
        Axkey, Aikey, Ajkey, bkey = jax.random.split(jax.random.PRNGKey(33), 4)
        Ax = jax.random.normal(Axkey, (n_nz,), dtype=dtype)
        Ai = jax.random.randint(Aikey, (n_nz,), 0, n_col, jnp.int32)
        Aj = jax.random.randint(Ajkey, (n_nz,), 0, n_col, jnp.int32)
        b = jax.random.normal(bkey, (n_col, n_rhs), dtype=dtype)
        x_sp = jax.jit(solve)(Ai, Aj, Ax, b)

        A = jnp.zeros((n_col, n_col), dtype=dtype).at[Ai, Aj].add(Ax)
        x = jsp.linalg.solve(A, b)

        print(x_sp - x)
        print(((x_sp - x) < 1e-9).all())

    ## BATCHED ========================================================================

    if True:
        print("batched")
        Axkey, Aikey, Ajkey, bkey = jax.random.split(jax.random.PRNGKey(33), 4)
        Ax = jax.random.normal(Axkey, (n_lhs, n_nz), dtype=dtype)
        Ai = jax.random.randint(Aikey, (n_nz,), 0, n_col, jnp.int32)
        Aj = jax.random.randint(Ajkey, (n_nz,), 0, n_col, jnp.int32)
        b = jax.random.normal(bkey, (n_lhs, n_col, n_rhs), dtype=dtype)
        x_sp = solve(Ai, Aj, Ax, b)

        A = jnp.zeros((n_lhs, n_col, n_col), dtype=dtype).at[:, Ai, Aj].add(Ax)
        x = jax.vmap(jsp.linalg.solve, (0, 0), 0)(A, b)

        print(x_sp - x)
        print(((x_sp - x) < 1e-9).all())

    ## RHS ============================================================================

    if True:
        print("rhs")
        Axkey, Aikey, Ajkey, bkey = jax.random.split(jax.random.PRNGKey(33), 4)
        Ax = jax.random.normal(Axkey, (n_nz,), dtype=dtype)
        Ai = jax.random.randint(Aikey, (n_nz,), 0, n_col, jnp.int32)
        Aj = jax.random.randint(Ajkey, (n_nz,), 0, n_col, jnp.int32)
        b = jax.random.normal(bkey, (n_col, 2), dtype=dtype)

        x_sp = solve(Ai, Aj, Ax, b[None])
        x_sp1 = solve(Ai, Aj, Ax, b[None, :, 0:1])
        x_sp2 = solve(Ai, Aj, Ax, b[None, :, 1:2])

        A = jnp.zeros((n_col, n_col), dtype=dtype).at[Ai, Aj].add(Ax)
        x = np.linalg.solve(A, b)

        print(b)
        print(x_sp)
        print(x_sp1)
        print(x_sp2)

    ## VMAP A & b =====================================================================

    if True:
        print("vmap A & b")
        Axkey, Aikey, Ajkey, bkey = jax.random.split(jax.random.PRNGKey(33), 4)
        Ax = jax.random.normal(Axkey, (n_lhs, n_nz), dtype=dtype).T
        Ai = jax.random.randint(Aikey, (n_nz,), 0, n_col, jnp.int32)
        Aj = jax.random.randint(Ajkey, (n_nz,), 0, n_col, jnp.int32)
        b = jax.random.normal(bkey, (n_col, n_rhs, n_lhs), dtype=dtype)
        x_sp = jax.vmap(solve, (None, None, 1, 2), 0)(Ai, Aj, Ax, b)

        A = jnp.zeros((n_lhs, n_col, n_col), dtype=dtype).at[:, Ai, Aj].add(Ax.T)
        x = jax.vmap(jsp.linalg.solve, (0, 0), 0)(A, b.transpose(2, 0, 1))

        print(x_sp - x)
        print(((x_sp - x) < 1e-9).all())

    ## VMAP A =========================================================================

    if True:
        print("vmap A")
        Axkey, Aikey, Ajkey, bkey = jax.random.split(jax.random.PRNGKey(33), 4)
        Ax = jax.random.normal(Axkey, (n_lhs, n_nz), dtype=dtype)
        Ai = jax.random.randint(Aikey, (n_nz,), 0, n_col, jnp.int32)
        Aj = jax.random.randint(Ajkey, (n_nz,), 0, n_col, jnp.int32)
        b = jax.random.normal(bkey, (n_col, n_rhs), dtype=dtype)

        vsolve = jax.vmap(solve, in_axes=(None, None, 0, None), out_axes=0)
        x_sp = vsolve(Ai, Aj, Ax, b)

        A = jnp.zeros((n_lhs, n_col, n_col), dtype=dtype).at[:, Ai, Aj].add(Ax)
        x = jsp.linalg.solve(A, b[None])

        print(x_sp - x)
        print(((x_sp - x) < 1e-9).all())

    ## VMAP b =========================================================================

    if True:
        print("vmap b")
        Axkey, Aikey, Ajkey, bkey = jax.random.split(jax.random.PRNGKey(33), 4)
        Ax = jax.random.normal(Axkey, (n_nz,), dtype=dtype)
        Ai = jax.random.randint(Aikey, (n_nz,), 0, n_col, jnp.int32)
        Aj = jax.random.randint(Ajkey, (n_nz,), 0, n_col, jnp.int32)
        b = jax.random.normal(bkey, (n_col, 2), dtype=dtype)

        x_sp = jax.vmap(solve, (None, None, None, 1), 1)(Ai, Aj, Ax, b)
        x_sp2 = jnp.stack([solve(Ai, Aj, Ax, b[:, i]) for i in range(2)], axis=1)

        print(x_sp)
        print(x_sp2)
        print(((x_sp - x_sp2) < 1e-9).all())
