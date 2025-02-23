"""klujax: a KLU solver for JAX."""

# Metadata ============================================================================

__version__ = "0.3.1"
__author__ = "Floris Laporte"
__all__ = []

# Imports =============================================================================

import sys

import jax
import jax.extend.core
import jax.numpy as jnp
import klujax_cpp
import numpy as np
from jax.core import ShapedArray
from jax.interpreters import mlir
from jaxtyping import Array

# Config ==============================================================================

DEBUG = False
jax.config.update(name="jax_enable_x64", val=True)
jax.config.update(name="jax_platform_name", val="cpu")
_log = lambda s: None if not DEBUG else print(s, file=sys.stderr)  # noqa: E731,T201

# Constants ===========================================================================

COMPLEX_DTYPES = (
    np.complex64,
    np.complex128,
    jnp.complex64,
    jnp.complex128,
)

# Main Functions ======================================================================


@jax.jit
def dot(Ai: Array, Aj: Array, Ax: Array, x: Array) -> Array:
    """
    Multiply a sparse matrix with a vector: Ax=b.

    Args:
        Ai: [n_nz; int32]: the row indices of the sparse matrix A
        Aj: [n_nz; int32]: the column indices of the sparse matrix A
        Ax: [n_lhs? x n_nz; float64|complex128]: the values of the sparse matrix A
        x:  [n_lhs? x n_col x n_rhs?; float64|complex128]: the vector multiplied by A

    Returns:
        b: the result of the multiplication (b=Ax)

    """
    if any(x.dtype in COMPLEX_DTYPES for x in (Ax, x)):
        _log("COMPLEX DOT")
        result = dot_f64.bind(
            Ai.astype(jnp.int32),
            Aj.astype(jnp.int32),
            Ax.astype(jnp.complex128),
            x.astype(jnp.complex128),
        )
    else:
        _log("FLOAT DOT")
        result = dot_f64.bind(
            Ai.astype(jnp.int32),
            Aj.astype(jnp.int32),
            Ax.astype(jnp.float64),
            x.astype(jnp.float64),
        )
    return result


# Primitives ==========================================================================

dot_f64 = jax.extend.core.Primitive("dot_f64")

# Implementations =====================================================================


@dot_f64.def_impl
def dot_f64_impl(Ai: Array, Aj: Array, Ax: Array, x: Array) -> Array:
    call = jax.ffi.ffi_call(
        "dot_f64",
        jax.ShapeDtypeStruct(x.shape, x.dtype),
        vmap_method="broadcast_all",
    )
    if not callable(call):
        msg = "jax.ffi.ffi_call did not return a callable."
        raise RuntimeError(msg)  # noqa: TRY004
    return call(Ai, Aj, Ax, x)


# Lowerings ===========================================================================

jax.ffi.register_ffi_target(
    "dot_f64",
    klujax_cpp.dot_f64(),
    platform="cpu",
)

solve_f64_low = mlir.lower_fun(dot_f64_impl, multiple_results=False)
mlir.register_lowering(dot_f64, solve_f64_low)


# Abstract Evals ======================================================================


@dot_f64.def_abstract_eval
def op_eval(Ai: Array, Aj: Array, Ax: Array, b: Array) -> ShapedArray:  # noqa: ARG001
    return ShapedArray(b.shape, b.dtype)
