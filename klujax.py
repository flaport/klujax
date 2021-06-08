""" klujax: a KLU solver for JAX """

__all__ = ["solve", "mul_coo_vec"]

## IMPORTS

from time import time

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp
from jax import abstract_arrays, core, lax
from jax.interpreters import ad, batching, xla
from jax.lib import xla_client

import klujax_cpp

## CONSTANTS

COMPLEX_DTYPES = (
    np.complex64,
    np.complex128,
    np.complex256,
    jnp.complex64,
    jnp.complex128,
)

## PRIMITIVES

solve_f64 = core.Primitive("solve_f64")
solve_c128 = core.Primitive("solve_c128")
mul_coo_vec_f64 = core.Primitive("mul_coo_vec_f64")
mul_coo_vec_c128 = core.Primitive("mul_coo_vec_c128")


## EXTRA DECORATORS


def xla_register_cpu(primitive, cpp_fun):
    name = primitive.name.encode()

    def decorator(fun):
        xla_client.register_cpu_custom_call_target(
            name,
            cpp_fun(),
        )
        xla.backend_specific_translations["cpu"][primitive] = fun
        return fun

    return decorator


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


def vmap_register(primitive):
    def decorator(fun):
        batching.primitive_batchers[primitive] = fun
        return fun

    return decorator


## IMPLEMENTATIONS


@solve_f64.def_impl
def solve_f64_impl(Ai, Aj, Ax, b):
    raise NotImplementedError


@solve_c128.def_impl
def solve_c128_impl(Ai, Aj, Ax, b):
    raise NotImplementedError


@mul_coo_vec_f64.def_impl
def mul_coo_vec_f64_impl(Ai, Aj, Ax, b):
    raise NotImplementedError


@mul_coo_vec_c128.def_impl
def mul_coo_vec_c128_impl(Ai, Aj, Ax, b):
    raise NotImplementedError


## ABSTRACT EVALUATIONS


@solve_f64.def_abstract_eval
def solve_f64_abstract_eval(Ai, Aj, Ax, b):
    return abstract_arrays.ShapedArray(b.shape, b.dtype)


@solve_c128.def_abstract_eval
def solve_c128_abstract_eval(Ai, Aj, Ax, b):
    return abstract_arrays.ShapedArray(b.shape, b.dtype)


@mul_coo_vec_f64.def_abstract_eval
def mul_coo_vec_f64_abstract_eval(Ai, Aj, Ax, b):
    return abstract_arrays.ShapedArray(b.shape, b.dtype)


@mul_coo_vec_c128.def_abstract_eval
def mul_coo_vec_c128_abstract_eval(Ai, Aj, Ax, b):
    return abstract_arrays.ShapedArray(b.shape, b.dtype)


# ENABLE JIT


def _xla_coo_vec_operation_f64(c, Ai, Aj, Ax, b, name):
    if isinstance(name, str):
        name = name.encode()
    Ax_shape = c.get_shape(Ax)
    Ai_shape = c.get_shape(Ai)
    Aj_shape = c.get_shape(Aj)
    b_shape = c.get_shape(b)
    (_Anz,) = Ax_shape.dimensions()
    _n_col, *_n_rhs_list = orig_b_shape = b_shape.dimensions()
    _n_rhs = np.prod([1] + _n_rhs_list)
    b = xla_client.ops.Reshape(b, (_n_col, _n_rhs))
    b = xla_client.ops.Transpose(b, (1, 0))
    b = xla_client.ops.Reshape(b, (_n_rhs * _n_col,))
    b_shape = c.get_shape(b)
    Anz = xla_client.ops.ConstantLiteral(c, np.int32(_Anz))
    n_col = xla_client.ops.ConstantLiteral(c, np.int32(_n_col))
    n_rhs = xla_client.ops.ConstantLiteral(c, np.int32(_n_rhs))
    Anz_shape = xla_client.Shape.array_shape(np.dtype(np.int32), (), ())
    n_col_shape = xla_client.Shape.array_shape(np.dtype(np.int32), (), ())
    n_rhs_shape = xla_client.Shape.array_shape(np.dtype(np.int32), (), ())
    result = xla_client.ops.CustomCallWithLayout(
        c,
        name,
        operands=(n_col, n_rhs, Anz, Ai, Aj, Ax, b),
        operand_shapes_with_layout=(
            n_col_shape,
            n_rhs_shape,
            Anz_shape,
            Ai_shape,
            Aj_shape,
            Ax_shape,
            b_shape,
        ),
        shape_with_layout=b_shape,
    )
    result = xla_client.ops.Reshape(result, (_n_rhs, _n_col))
    result = xla_client.ops.Transpose(result, (1, 0))
    result = xla_client.ops.Reshape(result, orig_b_shape)
    return result


def _xla_coo_vec_operation_c128(c, Ai, Aj, Ax, b, name):
    if isinstance(name, str):
        name = name.encode()

    # Ax = jnp.stack([jnp.real(Ax), jnp.imag(Ax)], 1).ravel()
    (_Anz,) = c.get_shape(Ax).dimensions()
    rAx = xla_client.ops.Real(Ax)
    iAx = xla_client.ops.Imag(Ax)
    rAx = xla_client.ops.BroadcastInDim(rAx, [_Anz, 1], [0])  # rAx[:, None]
    iAx = xla_client.ops.BroadcastInDim(iAx, [_Anz, 1], [0])  # iAx[:, None]
    Ax = xla_client.ops.ConcatInDim(c, [rAx, iAx], 1)
    Ax = xla_client.ops.Reshape(Ax, [2 * _Anz])

    # b = jnp.stack([jnp.real(b), jnp.imag(b)], 1).reshape(-1, *b.shape[1:])
    _n_col, *_n_rhs_list = c.get_shape(b).dimensions()
    rb = xla_client.ops.Real(b)
    ib = xla_client.ops.Imag(b)
    new_shape = [_n_col, 1, *_n_rhs_list]
    broadcast_dims = [i for i in range(len(new_shape)) if i != 1]
    rb = xla_client.ops.BroadcastInDim(rb, new_shape, broadcast_dims)  # rb[:, None]
    ib = xla_client.ops.BroadcastInDim(ib, new_shape, broadcast_dims)  # ib[:, None]
    b = xla_client.ops.ConcatInDim(c, [rb, ib], 1)
    b = xla_client.ops.Reshape(b, [2 * _n_col, *_n_rhs_list])

    Ax_shape = c.get_shape(Ax)
    Ai_shape = c.get_shape(Ai)
    Aj_shape = c.get_shape(Aj)
    b_shape = c.get_shape(b)
    _n_rhs = np.prod([1] + _n_rhs_list)
    b = xla_client.ops.Reshape(b, (2 * _n_col, _n_rhs))
    b = xla_client.ops.Transpose(b, (1, 0))
    b = xla_client.ops.Reshape(b, (2 * _n_rhs * _n_col,))
    b_shape = c.get_shape(b)
    Anz = xla_client.ops.ConstantLiteral(c, np.int32(_Anz))
    n_col = xla_client.ops.ConstantLiteral(c, np.int32(_n_col))
    n_rhs = xla_client.ops.ConstantLiteral(c, np.int32(_n_rhs))
    Anz_shape = xla_client.Shape.array_shape(np.dtype(np.int32), (), ())
    n_col_shape = xla_client.Shape.array_shape(np.dtype(np.int32), (), ())
    n_rhs_shape = xla_client.Shape.array_shape(np.dtype(np.int32), (), ())
    result = xla_client.ops.CustomCallWithLayout(
        c,
        name,
        operands=(n_col, n_rhs, Anz, Ai, Aj, Ax, b),
        operand_shapes_with_layout=(
            n_col_shape,
            n_rhs_shape,
            Anz_shape,
            Ai_shape,
            Aj_shape,
            Ax_shape,
            b_shape,
        ),
        shape_with_layout=b_shape,
    )
    result = xla_client.ops.Reshape(result, (_n_rhs, _n_col, 2))
    result = xla_client.ops.Transpose(result, (2, 1, 0))
    result = xla_client.ops.Reshape(result, [2, _n_col, *_n_rhs_list])
    result_r = xla_client.ops.Slice(
        result,
        [0, 0, *(0 for _ in _n_rhs_list)],
        [1, _n_col, *_n_rhs_list],
        [1, 1, *(1 for _ in _n_rhs_list)],
    )
    result_i = xla_client.ops.Slice(
        result,
        [1, 0, *(0 for _ in _n_rhs_list)],
        [2, _n_col, *_n_rhs_list],
        [1, 1, *(1 for _ in _n_rhs_list)],
    )
    result = xla_client.ops.Complex(result_r, result_i)
    result = xla_client.ops.Reshape(result, [_n_col, *_n_rhs_list])
    return result


@xla_register_cpu(solve_f64, klujax_cpp.solve_f64)
def solve_f64_xla_translation(c, Ai, Aj, Ax, b):
    return _xla_coo_vec_operation_f64(c, Ai, Aj, Ax, b, "solve_f64")


@xla_register_cpu(solve_c128, klujax_cpp.solve_c128)
def solve_c128_xla_translation(c, Ai, Aj, Ax, b):
    return _xla_coo_vec_operation_c128(c, Ai, Aj, Ax, b, "solve_c128")


@xla_register_cpu(mul_coo_vec_f64, klujax_cpp.mul_coo_vec_f64)
def mul_coo_vec_f64_xla_translation(c, Ai, Aj, Ax, b):
    return _xla_coo_vec_operation_f64(c, Ai, Aj, Ax, b, "mul_coo_vec_f64")


@xla_register_cpu(mul_coo_vec_c128, klujax_cpp.mul_coo_vec_c128)
def mul_coo_vec_c128_xla_translation(c, Ai, Aj, Ax, b):
    return _xla_coo_vec_operation_c128(c, Ai, Aj, Ax, b, "mul_coo_vec_c128")


# ENABLE FORWARD GRAD


@ad_register(solve_f64)
def solve_f64_value_and_jvp(arg_values, arg_tangents):
    # A x - b = 0
    # ∂A x + A ∂x - ∂b = 0
    # ∂x = A^{-1} (∂b - ∂A x)
    Ai, Aj, Ax, b = arg_values
    dAi, dAj, dAx, db = arg_tangents
    dAx = dAx if not isinstance(dAx, ad.Zero) else lax.zeros_like_array(Ax)
    dAi = dAi if not isinstance(dAi, ad.Zero) else lax.zeros_like_array(Ai)
    dAj = dAj if not isinstance(dAj, ad.Zero) else lax.zeros_like_array(Aj)
    db = db if not isinstance(db, ad.Zero) else lax.zeros_like_array(b)

    x = solve(Ai, Aj, Ax, b)
    dA_x = mul_coo_vec(Ai, Aj, dAx, x)
    dx = solve(Ai, Aj, Ax, db)  # - dA_x)

    return x, dx


# ENABLE BACKWARD GRAD


@transpose_register(solve_f64)
def solve_f64_transpose(ct, Ai, Aj, Ax, b):
    assert ad.is_undefined_primal(b)
    ct_b = solve(Ai, Aj, Ax, ct)  # probably not correct...
    return None, None, None, ct_b


# ENABLE VMAP


def _coo_vec_operation_vmap(operation, vector_arg_values, batch_axes):
    aAi, aAj, aAx, ab = batch_axes
    Ai, Aj, Ax, b = vector_arg_values

    assert aAi is None, "Ai cannot be vectorized."
    assert aAj is None, "Aj cannot be vectorized."

    if aAx is not None and ab is not None:
        assert isinstance(aAx, int) and isinstance(ab, int)
        n_lhs = Ax.shape[aAx]
        assert (
            b.shape[ab] == n_lhs
        ), f"axis {aAx} of Ax and axis {ab} of b differ in size ({n_lhs} != {b.shape[ab]})"
        if ab != 0:
            Ax = jnp.moveaxis(Ax, aAx, 0)
        if ab != 0:
            b = jnp.moveaxis(b, ab, 0)
        x = jnp.stack([operation(Ai, Aj, Ax[i], b[i]) for i in range(n_lhs)], 0)
        return x, 0

    if aAx is None:
        assert isinstance(ab, int)
        n_lhs = b.shape[ab]
        if ab != 0:
            b = jnp.moveaxis(b, ab, 0)
        x = jnp.stack([operation(Ai, Aj, Ax, b[i]) for i in range(n_lhs)], 0)
        return x, 0

    if ab is None:
        assert isinstance(aAx, int)
        n_lhs = Ax.shape[aAx]
        if aAx != 0:
            Ax = jnp.moveaxis(Ax, aAx, 0)
        x = jnp.stack([operation(Ai, Aj, Ax[i], b) for i in range(n_lhs)], 0)
        return x, 0


# @vmap_register(solve_c128) # this segfaults...
@vmap_register(solve_f64)
def solve_vmap(vector_arg_values, batch_axes):
    return _coo_vec_operation_vmap(solve, vector_arg_values, batch_axes)


# @vmap_register(mul_coo_vec_c128) # this segfaults...
@vmap_register(mul_coo_vec_f64)
def mul_coo_vec_vmap(vector_arg_values, batch_axes):
    return _coo_vec_operation_vmap(mul_coo_vec, vector_arg_values, batch_axes)


## THE FUNCTIONS


@jax.jit  # jitting by default allows for empty implementation definitions
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


@jax.jit  # jitting by default allows for empty implementation definitions
def mul_coo_vec(Ai, Aj, Ax, b):
    if any(x.dtype in COMPLEX_DTYPES for x in (Ax, b)):
        result = mul_coo_vec_c128.bind(
            Ai.astype(jnp.int32),
            Aj.astype(jnp.int32),
            Ax.astype(jnp.complex128),
            b.astype(jnp.complex128),
        )
    else:
        result = mul_coo_vec_f64.bind(
            Ai.astype(jnp.int32),
            Aj.astype(jnp.int32),
            Ax.astype(jnp.float64),
            b.astype(jnp.float64),
        )
    return result


# TEST SOME STUFF

if __name__ == "__main__":
    A = jnp.array(
        [
            [2 + 3j, 3, 0, 0, 0],
            [3, 0, 4, 0, 6],
            [0, -1, -3, 2, 0],
            [0, 0, 1, 0, 0],
            [0, 4, 2, 0, 1],
        ],
        dtype=jnp.complex128,
    )
    A = jnp.array(
        [
            [2, 3, 0, 0, 0],
            [3, 0, 4, 0, 6],
            [0, -1, -3, 2, 0],
            [0, 0, 1, 0, 0],
            [0, 4, 2, 0, 1],
        ],
        dtype=jnp.float64,
    )
    b = jnp.array([[8], [45], [-3], [3], [19]], dtype=jnp.float64)
    b = jnp.array([[8, 7], [45, 44], [-3, -4], [3, 2], [19, 18]], dtype=jnp.float64)
    b = jnp.array([3 + 8j, 8 + 45j, 23 + -3j, -7 - 3j, 13 + 19j], dtype=jnp.complex128)
    b = jnp.array([8, 45, -3, 3, 19], dtype=jnp.float64)
    Ai, Aj = jnp.where(abs(A) > 0)
    Ax = A[Ai, Aj]

    t = time()
    result = solve(Ai, Aj, Ax, b)
    print(f"{time()-t:.3e}", result)

    t = time()
    result = solve(Ai, Aj, Ax, b)
    print(f"{time()-t:.3e}", result)

    t = time()
    result = solve(Ai, Aj, Ax, b)
    print(f"{time()-t:.3e}", result)

    def solve_sum(Ai, Aj, Ax, b):
        return solve(Ai, Aj, Ax, b).sum()

    solve_sum_grad = jax.grad(solve_sum, 2)
    t = time()
    result = solve_sum_grad(Ai, Aj, Ax, b)
    print(f"{time()-t:.3e}", result)
