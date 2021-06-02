import jax

from klujax_cpp import solve as solve_cpp

solve_prim = jax.core.Primitive("solve")

def solve(A, b):
    return solve_prim.bind(A, b)

solve_prim.def_impl(solve_cpp)

def solve_abstract_eval(A, b):
    return jax.abstract_arrays.ShapedArray((), A.dtype)

solve_prim.def_abstract_eval(solve_abstract_eval)


## This should be replaced by c++-registration...
from jax.lib import xla_client
def solve_xla_translation(c, Ac, bc):
    return xla_client.ops.Div(bc, Ac)
from jax.interpreters import xla
xla.backend_specific_translations['cpu'][solve_prim] = solve_xla_translation

if __name__ == "__main__":
    print(solve(3.0, 4.0))
    print(jax.jit(solve)(3.0, 4.0))

