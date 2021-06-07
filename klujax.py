from time import time

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)

from jax import lax
from jax import core
from jax import abstract_arrays

from jax.interpreters import ad

from jax.lib import xla_client
from jax.interpreters import xla

import jax.numpy as jnp
import jax.scipy as jsp

import klujax_cpp

COMPLEX_DTYPES = (
    np.complex64,
    np.complex128,
    np.complex256,
    jnp.complex64,
    jnp.complex128,
)

## PRIMITIVES

solve_f64 = core.Primitive("klu_solve_f64")
solve_c128 = core.Primitive("klu_solve_c128")
coo_vec_mul_f64 = core.Primitive("coo_vec_mul_f64")


def solve(Ax, Ai, Aj, b):
    if any(x.dtype in COMPLEX_DTYPES for x in (Ax, b)):
        Ax = jnp.stack([jnp.real(Ax), jnp.imag(Ax)], 1).ravel()
        b = jnp.stack([jnp.real(b), jnp.imag(b)], 1).ravel()
        result = solve_c128.bind(
            Ax.astype(jnp.float64),
            Ai.astype(jnp.int32),
            Aj.astype(jnp.int32),
            b.astype(jnp.float64),
        )
        result = result[::2] + 1j * result[1::2]
    else:
        result = solve_f64.bind(
            Ax.astype(jnp.float64),
            Ai.astype(jnp.int32),
            Aj.astype(jnp.int32),
            b.astype(jnp.float64),
        )
    return result


def coo_vec_mul(Ax, Ai, Aj, b):
    result = coo_vec_mul_f64.bind(
        Ax.astype(jnp.float64),
        Ai.astype(jnp.int32),
        Aj.astype(jnp.int32),
        b.astype(jnp.float64),
    )
    return result


## IMPLEMENTATIONS


@solve_f64.def_impl
def solve_f64_impl(Ax, Ai, Aj, b):
    A = jnp.zeros((b.shape[0], b.shape[0]), dtype=jnp.float64).at[Ai, Aj].set(Ax)
    inv_A = jsp.linalg.inv(A)
    return inv_A @ b.astype(jnp.float64)


@solve_c128.def_impl
def solve_c128_impl(Ax, Ai, Aj, b):
    A = jnp.zeros((b.shape[0] // 2, b.shape[0] // 2), dtype=jnp.complex128)
    A = A.at[Ai, Aj].set(Ax[::2]) + 1j * A.at[Ai, Aj].set(Ax[1::2])
    b = b[::2] + 1j * b[1::2]
    inv_A = jsp.linalg.inv(A)
    result = inv_A @ b.astype(jnp.complex128)
    result = jnp.stack([jnp.real(result), jnp.imag(result)], 1).ravel()
    return result


@coo_vec_mul_f64.def_impl
def coo_vec_mul_f64_impl(Ax, Ai, Aj, b):
    y = jnp.zeros_like(b)
    return y.at[Ai].set(Ax * b[Aj])  # todo: make this work for non-coalesced A


## ABSTRACT EVALUATIONS


@solve_f64.def_abstract_eval
def solve_f64_abstract_eval(Ax, Ai, Aj, b):
    return abstract_arrays.ShapedArray(b.shape, b.dtype)


@solve_c128.def_abstract_eval
def solve_c128_abstract_eval(Ax, Ai, Aj, b):
    return abstract_arrays.ShapedArray(b.shape, b.dtype)


@coo_vec_mul_f64.def_abstract_eval
def coo_vec_mul_f64_abstract_eval(Ax, Ai, Aj, b):
    return abstract_arrays.ShapedArray(b.shape, b.dtype)


# ENABLE JIT

xla_client.register_cpu_custom_call_target(
    b"solve_f64",
    klujax_cpp.solve_f64(),
)


def solve_f64_xla_translation(c, Ax, Ai, Aj, b):
    Ax_shape = c.get_shape(Ax)
    Ai_shape = c.get_shape(Ai)
    Aj_shape = c.get_shape(Aj)
    b_shape = c.get_shape(b)
    (_n_nz,) = Ax_shape.dimensions()
    _n_col, *_n_rhs_list = orig_b_shape = b_shape.dimensions()
    _n_rhs = np.prod([1] + _n_rhs_list)
    b = xla_client.ops.Reshape(b, (_n_col, _n_rhs))
    b = xla_client.ops.Transpose(b, (1, 0))
    b = xla_client.ops.Reshape(b, (_n_rhs * _n_col,))
    b_shape = c.get_shape(b)
    n_nz = xla_client.ops.ConstantLiteral(c, np.int32(_n_nz))
    n_col = xla_client.ops.ConstantLiteral(c, np.int32(_n_col))
    n_rhs = xla_client.ops.ConstantLiteral(c, np.int32(_n_rhs))
    n_nz_shape = xla_client.Shape.array_shape(np.dtype(np.int32), (), ())
    n_col_shape = xla_client.Shape.array_shape(np.dtype(np.int32), (), ())
    n_rhs_shape = xla_client.Shape.array_shape(np.dtype(np.int32), (), ())
    result = xla_client.ops.CustomCallWithLayout(
        c,
        b"solve_f64",
        operands=(n_nz, n_col, n_rhs, Ax, Ai, Aj, b),
        operand_shapes_with_layout=(
            n_nz_shape,
            n_col_shape,
            n_rhs_shape,
            Ax_shape,
            Ai_shape,
            Aj_shape,
            b_shape,
        ),
        shape_with_layout=b_shape,
    )
    result = xla_client.ops.Reshape(result, (_n_rhs, _n_col))
    result = xla_client.ops.Transpose(result, (1, 0))
    result = xla_client.ops.Reshape(result, orig_b_shape)
    return result


xla.backend_specific_translations["cpu"][solve_f64] = solve_f64_xla_translation


xla_client.register_cpu_custom_call_target(
    b"solve_c128",
    klujax_cpp.solve_c128(),
)


def solve_c128_xla_translation(c, Ax, Ai, Aj, b):
    Ax_shape = c.get_shape(Ax)
    Ai_shape = c.get_shape(Ai)
    Aj_shape = c.get_shape(Aj)
    b_shape = c.get_shape(b)
    (_n_nz,) = Ax_shape.dimensions()
    _n_nz = _n_nz // 2
    _n_col, *_n_rhs_list = orig_b_shape = b_shape.dimensions()
    _n_col = _n_col // 2
    _n_rhs = np.prod([1] + _n_rhs_list)
    b = xla_client.ops.Reshape(b, (2 * _n_col, _n_rhs))
    b = xla_client.ops.Transpose(b, (1, 0))
    b = xla_client.ops.Reshape(b, (2 * _n_rhs * _n_col,))
    b_shape = c.get_shape(b)
    n_nz = xla_client.ops.ConstantLiteral(c, np.int32(_n_nz))
    n_col = xla_client.ops.ConstantLiteral(c, np.int32(_n_col))
    n_rhs = xla_client.ops.ConstantLiteral(c, np.int32(_n_rhs))
    n_nz_shape = xla_client.Shape.array_shape(np.dtype(np.int32), (), ())
    n_col_shape = xla_client.Shape.array_shape(np.dtype(np.int32), (), ())
    n_rhs_shape = xla_client.Shape.array_shape(np.dtype(np.int32), (), ())
    result = xla_client.ops.CustomCallWithLayout(
        c,
        b"solve_c128",
        operands=(n_nz, n_col, n_rhs, Ax, Ai, Aj, b),
        operand_shapes_with_layout=(
            n_nz_shape,
            n_col_shape,
            n_rhs_shape,
            Ax_shape,
            Ai_shape,
            Aj_shape,
            b_shape,
        ),
        shape_with_layout=b_shape,
    )
    result = xla_client.ops.Reshape(result, (_n_rhs, 2 * _n_col))
    result = xla_client.ops.Transpose(result, (1, 0))
    result = xla_client.ops.Reshape(result, orig_b_shape)
    return result


xla.backend_specific_translations["cpu"][solve_c128] = solve_c128_xla_translation


xla_client.register_cpu_custom_call_target(
    b"coo_vec_mul_f64",
    klujax_cpp.coo_vec_mul_f64(),
)


def coo_vec_mul_f64_xla_translation(c, Ax, Ai, Aj, b):
    Ax_shape = c.get_shape(Ax)
    Ai_shape = c.get_shape(Ai)
    Aj_shape = c.get_shape(Aj)
    b_shape = c.get_shape(b)
    (_n_nz,) = Ax_shape.dimensions()
    _n_col, *_n_rhs_list = orig_b_shape = b_shape.dimensions()
    _n_rhs = np.prod([1] + _n_rhs_list)
    b = xla_client.ops.Reshape(b, (_n_col, _n_rhs))
    b = xla_client.ops.Transpose(b, (1, 0))
    b = xla_client.ops.Reshape(b, (_n_rhs * _n_col,))
    b_shape = c.get_shape(b)
    n_nz = xla_client.ops.ConstantLiteral(c, np.int32(_n_nz))
    n_col = xla_client.ops.ConstantLiteral(c, np.int32(_n_col))
    n_rhs = xla_client.ops.ConstantLiteral(c, np.int32(_n_rhs))
    n_nz_shape = xla_client.Shape.array_shape(np.dtype(np.int32), (), ())
    n_col_shape = xla_client.Shape.array_shape(np.dtype(np.int32), (), ())
    n_rhs_shape = xla_client.Shape.array_shape(np.dtype(np.int32), (), ())
    result = xla_client.ops.CustomCallWithLayout(
        c,
        b"coo_vec_mul_f64",
        operands=(n_nz, n_col, n_rhs, Ax, Ai, Aj, b),
        operand_shapes_with_layout=(
            n_nz_shape,
            n_col_shape,
            n_rhs_shape,
            Ax_shape,
            Ai_shape,
            Aj_shape,
            b_shape,
        ),
        shape_with_layout=b_shape,
    )
    result = xla_client.ops.Reshape(result, (_n_rhs, _n_col))
    result = xla_client.ops.Transpose(result, (1, 0))
    result = xla_client.ops.Reshape(result, orig_b_shape)
    return result


xla.backend_specific_translations["cpu"][
    coo_vec_mul_f64
] = coo_vec_mul_f64_xla_translation

# ENABLE GRAD


def solve_f64_value_and_jvp(arg_values, arg_tangents):
    # A x - b = 0
    # ∂A x + A ∂x - ∂b = 0
    # ∂x = A^{-1} (∂b - ∂A x)
    Ax, Ai, Aj, b = arg_values
    dAx, dAi, dAj, db = arg_tangents
    dAx = dAx if not isinstance(dAx, ad.Zero) else lax.zeros_like_array(Ax)
    dAi = dAi if not isinstance(dAi, ad.Zero) else lax.zeros_like_array(Ai)
    dAj = dAj if not isinstance(dAj, ad.Zero) else lax.zeros_like_array(Aj)
    db = db if not isinstance(db, ad.Zero) else lax.zeros_like_array(b)

    x = solve(Ax, Ai, Aj, b)
    dA_x = coo_vec_mul(dAx, Ai, Aj, x)
    dx = solve(Ax, Ai, Aj, db - dA_x)

    return x, dx


ad.primitive_jvps[solve_f64] = solve_f64_value_and_jvp


def solve_f64_transpose(ct, Ax, Ai, Aj, b):
    assert ad.is_undefined_primal(b)
    ct_b = solve(Ax, Ai, Aj, ct)  # probably not correct...
    return None, None, None, ct_b


ad.primitive_transposes[solve_f64] = solve_f64_transpose


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
    result = solve(Ax, Ai, Aj, b)
    print(f"{time()-t:.3e}", result)

    solve = jax.jit(solve)
    t = time()
    result = solve(Ax, Ai, Aj, b)
    print(f"{time()-t:.3e}", result)

    t = time()
    result = solve(Ax, Ai, Aj, b)
    print(f"{time()-t:.3e}", result)

    # def solve_sum(Ax, Ai, Aj, b):
    #     return solve(Ax, Ai, Aj, b).sum()

    # solve_sum_grad = jax.grad(solve_sum)
    # t = time()
    # result = solve_sum_grad(Ax, Ai, Aj, b)
    # print(f"{time()-t:.3e}", result)
