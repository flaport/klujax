from time import time
import numpy as np

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.scipy as jsp

import klujax_cpp

solve_prim = jax.core.Primitive("solve")


def solve(Ax, Ai, Aj, b):
    return solve_prim.bind(
        Ax.astype(jnp.float64),
        Ai.astype(jnp.int32),
        Aj.astype(jnp.int32),
        b.astype(jnp.float64),
    )


def solve_impl(Ax, Ai, Aj, b):
    A = jnp.zeros((b.shape[0], b.shape[0])).at[Ai, Aj].set(Ax)
    inv_A = jsp.linalg.inv(A)
    return inv_A @ b


solve_prim.def_impl(solve_impl)


def solve_abstract_eval(Ax, Ai, Aj, b):
    return jax.abstract_arrays.ShapedArray(b.shape, b.dtype)


solve_prim.def_abstract_eval(solve_abstract_eval)


# make jittable
from jax.lib import xla_client
from jax.interpreters import xla

xla_client.register_cpu_custom_call_target(
    b"solve",
    klujax_cpp.solve_f64(),
)


def solve_xla_translation(c, Ax, Ai, Aj, b):
    Ax_shape = c.get_shape(Ax)
    Ai_shape = c.get_shape(Ai)
    Aj_shape = c.get_shape(Aj)
    b_shape = c.get_shape(b)
    nnz = xla_client.ops.ConstantLiteral(c, np.int32(Ax_shape.dimensions()[0]))
    n_col = xla_client.ops.ConstantLiteral(c, np.int32(b_shape.dimensions()[0]))
    nnz_shape = xla_client.Shape.array_shape(np.dtype(np.int32), (), ())
    n_col_shape = xla_client.Shape.array_shape(np.dtype(np.int32), (), ())
    return xla_client.ops.CustomCallWithLayout(
        c,
        b"solve",
        operands=(nnz, n_col, Ax, Ai, Aj, b),
        operand_shapes_with_layout=(
            nnz_shape,
            n_col_shape,
            Ax_shape,
            Ai_shape,
            Aj_shape,
            b_shape,
        ),
        shape_with_layout=b_shape,
    )


xla.backend_specific_translations["cpu"][solve_prim] = solve_xla_translation


# make differentiable
# we'll work on this later...
#
# from jax.interpreters import ad
#
#
# def solve_value_and_jvp(arg_values, arg_tangents):
#     A, b = arg_values
#     At, bt = arg_tangents
#     primal_out = solve(A, b)
#
#     At = jnp.zeros_like(A) if isinstance(At, ad.Zero) else At
#     bt = jnp.zeros_like(b) if isinstance(bt, ad.Zero) else bt
#     output_tan = -b * At / A ** 2 + bt / A
#     return primal_out, output_tan
#
# ad.primitive_jvps[solve_prim] = solve_value_and_jvp

if __name__ == "__main__":
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
    b = jnp.array([8, 45, -3, 3, 19], dtype=jnp.float64)
    Ai, Aj = jnp.where(A > 0)
    Ax = A[Ai, Aj]

    t = time()
    result = solve(Ax, Ai, Aj, b)
    print(f"{time()-t:.3e}", result[:5])

    solve = jax.jit(solve)
    t = time()
    result = solve(Ax, Ai, Aj, b)
    print(f"{time()-t:.3e}", result[:5])

    t = time()
    result = solve(Ax, Ai, Aj, b)
    print(f"{time()-t:.3e}", result[:5])
