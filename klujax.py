""" klujax: a KLU solver for JAX """

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

# Register XLA Extensions ==============================================================

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


# The Functions =======================================================================


def solve(Ai, Aj, Ax, b):
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
    return result


def coo_mul_vec(Ai, Aj, Ax, b):
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
    return result


# Implementations =====================================================================


@solve_f64.def_impl
def solve_f64_impl(Ai, Aj, Ax, b):
    Ai, Aj, Ax, b, shape, n_lhs, n_nz, n_col, n_rhs = _prepare_arguments(Ai, Aj, Ax, b)
    call = jax.extend.ffi.ffi_call(
        "_solve_f64",
        jax.ShapeDtypeStruct(b.shape, b.dtype),
        vmap_method="broadcast_all",
    )
    result = call(  # type: ignore
        Ai,
        Aj,
        Ax,
        b,
        n_col=np.int32(n_col),
        n_rhs=np.int32(n_rhs),
        n_lhs=np.int32(n_lhs),
        n_nz=np.int32(n_nz),
    )
    return result.reshape(*shape)  # type: ignore


@solve_c128.def_impl
def solve_c128_impl(Ai, Aj, Ax, b):
    Ai, Aj, Ax, b, shape, n_lhs, n_nz, n_col, n_rhs = _prepare_arguments(Ai, Aj, Ax, b)
    _b = b.view(np.float64).reshape(n_lhs, n_col, n_rhs, 2)
    call = jax.extend.ffi.ffi_call(
        "_solve_c128",
        jax.ShapeDtypeStruct(_b.shape, _b.dtype),
        vmap_method="broadcast_all",
    )
    result = call(  # type: ignore
        Ai,
        Aj,
        Ax.view(np.float64),
        _b,
        n_col=np.int32(n_col),
        n_rhs=np.int32(n_rhs),
        n_lhs=np.int32(n_lhs),
        n_nz=np.int32(n_nz),
    )
    return result.view(np.complex128).reshape(*shape)  # type: ignore


@coo_mul_vec_f64.def_impl
def coo_mul_vec_f64_impl(Ai, Aj, Ax, x):
    Ai, Aj, Ax, x, shape, n_lhs, n_nz, n_col, n_rhs = _prepare_arguments(Ai, Aj, Ax, x)
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
        n_col=np.int32(n_col),
        n_rhs=np.int32(n_rhs),
        n_lhs=np.int32(n_lhs),
        n_nz=np.int32(n_nz),
    )
    return result.reshape(*shape)  # type: ignore


@coo_mul_vec_c128.def_impl
def coo_mul_vec_c128_impl(Ai, Aj, Ax, x):
    Ai, Aj, Ax, x, shape, n_lhs, n_nz, n_col, n_rhs = _prepare_arguments(Ai, Aj, Ax, x)
    _x = x.view(np.float64).reshape(n_lhs, n_col, n_rhs, 2)
    call = jax.extend.ffi.ffi_call(
        "_coo_mul_vec_c128",
        jax.ShapeDtypeStruct(_x.shape, _x.dtype),
        vmap_method="broadcast_all",
    )
    result = call(  # type: ignore
        Ai,
        Aj,
        Ax.view(np.float64),
        _x,
        n_col=np.int32(n_col),
        n_rhs=np.int32(n_rhs),
        n_lhs=np.int32(n_lhs),
        n_nz=np.int32(n_nz),
    )
    return result.view(np.complex128).reshape(*shape)  # type: ignore


def _prepare_arguments(Ai, Aj, Ax, x):
    Ai = jnp.asarray(Ai)
    Aj = jnp.asarray(Aj)
    Ax = jnp.asarray(Ax)
    x = jnp.asarray(x)
    shape = x.shape

    if x.ndim < 2:
        x = jnp.atleast_2d(x).T

    n_col, n_rhs, *_ = x.shape[Ax.ndim - 1 :] + (1,)

    *_, n_nz = Ax.shape
    Ax = Ax.reshape(-1, n_nz)
    n_lhs, _ = Ax.shape

    x = x.reshape(-1, n_col, n_rhs)
    return Ai, Aj, Ax, x, shape, n_lhs, n_nz, n_col, n_rhs


# # Lowerings ===========================================================================
#
#
# def solve_f64_lowering(ctx, Ai, Aj, Ax, b):
#     result_avals = ctx.avals_out if ctx.avals_out is not None else ()
#     result_types = list(
#         mlir.flatten_ir_types([mlir.aval_to_ir_type(aval) for aval in result_avals])
#     )
#     custom_call = hlo.CustomCallOp(
#         result_types,
#         operands,
#         call_target_name=ir.StringAttr.get("tf.call_tf_function"),
#         has_side_effect=False,,
#         api_version=1,
#         called_computations=ir.ArrayAttr.get([]),
#         backend_config=ir.StringAttr.get(""),
#     )
#
#     print(ctx)
#
#
# mlir.register_lowering(solve_f64, solve_f64_lowering, "cpu")

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
    print(aAi, aAj, aAx, ab)
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
        print(Ai.shape, Aj.shape, Ax.shape, b.shape)
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


# Quick Tests =========================================================================


if __name__ == "__main__":
    n_nz = 8
    n_col = 5
    n_rhs = 1
    n_lhs = 3  # only used in batched & vmap
    dtype = np.float64

    ## SINGLE =========================================================================
    Axkey, Aikey, Ajkey, bkey = jax.random.split(jax.random.PRNGKey(33), 4)
    Ax = jax.random.normal(Axkey, (n_nz,), dtype=dtype)
    Ai = jax.random.randint(Aikey, (n_nz,), 0, n_col, jnp.int32)
    Aj = jax.random.randint(Ajkey, (n_nz,), 0, n_col, jnp.int32)
    b = jax.random.normal(bkey, (n_col, n_rhs), dtype=dtype)
    x_sp = jax.jit(solve)(Ai, Aj, Ax, b)
    print(x_sp.shape)

    A = jnp.zeros((n_col, n_col), dtype=dtype).at[Ai, Aj].add(Ax)
    x = jsp.linalg.solve(A, b)
    print(x.shape)

    print(x_sp - x)

    ## BATCHED ========================================================================
    # Axkey, Aikey, Ajkey, bkey = jax.random.split(jax.random.PRNGKey(33), 4)
    # Ax = jax.random.normal(Axkey, (n_lhs, n_nz), dtype=dtype)
    # Ai = jax.random.randint(Aikey, (n_nz,), 0, n_col, jnp.int32)
    # Aj = jax.random.randint(Ajkey, (n_nz,), 0, n_col, jnp.int32)
    # b = jax.random.normal(bkey, (n_lhs, n_col, n_rhs), dtype=dtype)
    # x_sp = solve(Ai, Aj, Ax, b)

    # A = jnp.zeros((n_lhs, n_col, n_col), dtype=dtype).at[:, Ai, Aj].add(Ax)
    # x = jax.vmap(jsp.linalg.solve, (0, 0), 0)(A, b)

    # print(x_sp - x)

    ## VMAP ========================================================================

    # Axkey, Aikey, Ajkey, bkey = jax.random.split(jax.random.PRNGKey(33), 4)
    # Ax = jax.random.normal(Axkey, (n_lhs, n_nz), dtype=dtype).T
    # Ai = jax.random.randint(Aikey, (n_nz,), 0, n_col, jnp.int32)
    # Aj = jax.random.randint(Ajkey, (n_nz,), 0, n_col, jnp.int32)
    # b = jax.random.normal(bkey, (n_col, n_rhs, n_lhs), dtype=dtype)
    # x_sp = jax.vmap(solve, (None, None, 1, 2), 0)(Ai, Aj, Ax, b)

    # A = jnp.zeros((n_lhs, n_col, n_col), dtype=dtype).at[:, Ai, Aj].add(Ax.T)
    # x = jax.vmap(jsp.linalg.solve, (0, 0), 0)(A, b.transpose(2, 0, 1))

    # print(x_sp - x)
