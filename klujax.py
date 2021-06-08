""" klujax: a KLU solver for JAX """

__all__ = ["solve", "mul_coo_vec"]

## IMPORTS

from time import time

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

from jax import lax
from jax import core
from jax import abstract_arrays

from jax.interpreters import ad

from jax.lib import xla_client
from jax.interpreters import xla

import jax.numpy as jnp
import jax.scipy as jsp

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
    Ax_shape = c.get_shape(Ax)
    Ai_shape = c.get_shape(Ai)
    Aj_shape = c.get_shape(Aj)
    b_shape = c.get_shape(b)
    (_Anz,) = Ax_shape.dimensions()
    _Anz = _Anz // 2
    _n_col, *_n_rhs_list = orig_b_shape = b_shape.dimensions()
    _n_col = _n_col // 2
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
    result = xla_client.ops.Reshape(result, (_n_rhs, 2 * _n_col))
    result = xla_client.ops.Transpose(result, (1, 0))
    result = xla_client.ops.Reshape(result, orig_b_shape)
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


## THE FUNCTIONS


@jax.jit  # jitting by default allows for empty implementation definitions
def solve(Ai, Aj, Ax, b):
    if any(x.dtype in COMPLEX_DTYPES for x in (Ax, b)):
        Ax = jnp.stack([jnp.real(Ax), jnp.imag(Ax)], 1).ravel()
        b = jnp.stack([jnp.real(b), jnp.imag(b)], 1).reshape(-1, *b.shape[1:])
        result = solve_c128.bind(
            Ai.astype(jnp.int32),
            Aj.astype(jnp.int32),
            Ax.astype(jnp.float64),
            b.astype(jnp.float64),
        )
        result = result[::2] + 1j * result[1::2]
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
        Ax = jnp.stack([jnp.real(Ax), jnp.imag(Ax)], 1).ravel()
        b = jnp.stack([jnp.real(b), jnp.imag(b)], 1).reshape(-1, *b.shape[1:])
        result = mul_coo_vec_c128.bind(
            Ai.astype(jnp.int32),
            Aj.astype(jnp.int32),
            Ax.astype(jnp.float64),
            b.astype(jnp.float64),
        )
        result = result[::2] + 1j * result[1::2]
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
