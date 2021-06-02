import jax

from klujax_cpp import solve as solve_cpp

solve_prim = jax.core.Primitive("solve")

def solve(A, b):
    return solve_prim.bind(A, b)

solve_prim.def_impl(solve_cpp)

def solve_abstract_eval(A, b):
    return jax.abstract_arrays.ShapedArray((), A.dtype)

solve_prim.def_abstract_eval(solve_abstract_eval)


# make jittable
# This should be replaced by c++-registration...
from jax.lib import xla_client
def solve_xla_translation(c, Ac, bc):
    return xla_client.ops.Div(bc, Ac)
from jax.interpreters import xla
xla.backend_specific_translations['cpu'][solve_prim] = solve_xla_translation


# make differentiable
from jax.interpreters import ad
def solve_value_and_jvp(arg_values, arg_tangents):
    A, b = arg_values
    At, bt = arg_tangents
    primal_out = solve(A, b)

    At = 0.0 if isinstance(At, ad.Zero) else At
    bt = 0.0 if isinstance(bt, ad.Zero) else bt
    output_tan = -b*At/A**2 + bt/A
    return primal_out, output_tan

ad.primitive_jvps[solve_prim] = solve_value_and_jvp

if __name__ == "__main__":
    a = 3.0
    b = 4.0
    print(solve(a, b))
    print(jax.jit(solve)(a, b))
    print(jax.grad(solve)(a, b))
    print(jax.grad(jax.grad(solve))(a, b))

