from time import time

import jax
import jax.numpy as jnp

import klujax_cpp

solve_prim = jax.core.Primitive("solve")


def solve(A, b):
    return solve_prim.bind(A, b)


def solve_impl(A, b):
    return b / A


solve_prim.def_impl(solve_impl)


def solve_abstract_eval(A, b):
    return jax.abstract_arrays.ShapedArray(A.shape, A.dtype)


solve_prim.def_abstract_eval(solve_abstract_eval)


# make jittable
from jax.lib import xla_client
from jax.interpreters import xla

xla_client.register_cpu_custom_call_target(
    b"solve",
    klujax_cpp.solve_f32(),
)


def solve_xla_translation(c, Ac, bc):
    shape = c.get_shape(Ac)
    return xla_client.ops.CustomCallWithLayout(
        c,
        b"solve",
        operands=(Ac, bc),
        operand_shapes_with_layout=(
            shape,
            shape,
        ),
        shape_with_layout=shape,
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
    a = jnp.array([3.0, 5.0])
    b = jnp.array([7.0, 11.0])
    t1 = time()
    print(solve(a, b))
    t0, t1 = t1, time()
    print(t1 - t0)
    print(jax.jit(solve)(a, b))
    t0, t1 = t1, time()
