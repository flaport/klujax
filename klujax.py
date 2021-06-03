from time import time
import numpy as np

import jax
import jax.numpy as jnp

import klujax_cpp

solve_prim = jax.core.Primitive("solve")


def solve(Ax, Ai, Aj, b):
    return solve_prim.bind(Ax, Ai, Aj, b)


def solve_impl(Ax, Ai, Aj, b):
    return b / Ax


solve_prim.def_impl(solve_impl)


def solve_abstract_eval(Ax, Ai, Aj, b):
    return jax.abstract_arrays.ShapedArray(b.shape, b.dtype)


solve_prim.def_abstract_eval(solve_abstract_eval)


# make jittable
from jax.lib import xla_client
from jax.interpreters import xla

xla_client.register_cpu_custom_call_target(
    b"solve",
    klujax_cpp.solve_f32(),
)


def solve_xla_translation(c, Ax, Ai, Aj, b):
    Ax_shape = c.get_shape(Ax)
    Ai_shape = c.get_shape(Ai)
    Aj_shape = c.get_shape(Aj)
    b_shape = c.get_shape(b)
    N = b_shape.dimensions()[0]
    return xla_client.ops.CustomCallWithLayout(
        c,
        b"solve",
        operands=(xla_client.ops.ConstantLiteral(c, N), Ax, Ai, Aj, b),
        operand_shapes_with_layout=(
            xla_client.Shape.array_shape(np.dtype(np.int64), (), ()),
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
    A_key, b_key = jax.random.split(jax.random.PRNGKey(42), 2)
    Ax = jax.random.normal(A_key, (3000,))
    b = jax.random.normal(b_key, (3000,))
    Ai = jnp.arange(Ax.shape[0], dtype=jnp.int32)
    Aj = jnp.arange(Ax.shape[0], dtype=jnp.int32)

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

