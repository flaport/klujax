import jax

from klujax_cpp import solve as solve_cpp

solve_prim = jax.core.Primitive("solve")

def solve(A, b):
    return solve_prim.bind(A, b)

solve_prim.def_impl(solve_cpp)

def solve_abstract_eval(A, b):
    return jax.abstract_arrays.ShapedArray((), A.dtype)

solve_prim.def_abstract_eval(solve_abstract_eval)

if __name__ == "__main__":
    print(solve(3.0, 4.0))
    print(jax.jit(solve)(3.0, 4.0))

