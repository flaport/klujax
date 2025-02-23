"""klujax: a KLU solver for JAX."""

# Metadata ============================================================================

__version__ = "0.3.1"
__author__ = "Floris Laporte"
__all__ = []

# Imports =============================================================================

import os
import sys

import jax
import jax.extend.core
import jax.numpy as jnp
import klujax_cpp
import numpy as np
from jax import lax
from jax.core import ShapedArray
from jax.interpreters import ad, batching, mlir
from jaxtyping import Array

# Config ==============================================================================

DEBUG = os.environ.get("KLUJAX_DEBUG", False)
jax.config.update(name="jax_enable_x64", val=True)
jax.config.update(name="jax_platform_name", val="cpu")
log = lambda s: None if not DEBUG else print(s, file=sys.stderr)  # noqa: E731,T201

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
        log("COMPLEX DOT")
        result = dot_f64.bind(
            Ai.astype(jnp.int32),
            Aj.astype(jnp.int32),
            Ax.astype(jnp.complex128),
            x.astype(jnp.complex128),
        )
    else:
        log("FLOAT DOT")
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
def general_abstract_eval(Ai: Array, Aj: Array, Ax: Array, b: Array) -> ShapedArray:  # noqa: ARG001
    return ShapedArray(b.shape, b.dtype)


# Forward Differentiation =============================================================


def dot_f64_value_and_jvp(
    arg_values: tuple[Array, Array, Array, Array],
    arg_tangents: tuple[Array, Array, Array, Array],
) -> tuple[Array, Array]:
    Ai, Aj, Ax, b = arg_values
    dAi, dAj, dAx, db = arg_tangents
    dAx = dAx if not isinstance(dAx, ad.Zero) else lax.zeros_like_array(Ax)
    dAi = dAi if not isinstance(dAi, ad.Zero) else lax.zeros_like_array(Ai)
    dAj = dAj if not isinstance(dAj, ad.Zero) else lax.zeros_like_array(Aj)
    db = db if not isinstance(db, ad.Zero) else lax.zeros_like_array(b)
    x = dot_f64.bind(Ai, Aj, Ax, b)
    dA_b = dot_f64.bind(Ai, Aj, dAx, b)
    A_db = dot_f64.bind(Ai, Aj, Ax, db)
    return x, dA_b + A_db


ad.primitive_jvps[dot_f64] = dot_f64_value_and_jvp


# Batching (vmap) =====================================================================


def dot_f64_vmap(
    vector_arg_values: tuple[Array, Array, Array, Array],
    batch_axes: tuple[int | None, int | None, int | None, int | None],
) -> tuple[Array, int]:
    Ai, Aj, Ax, x = vector_arg_values
    aAi, aAj, aAx, ax = batch_axes

    if aAi is not None:
        msg = "Ai cannot be vectorized."
        raise ValueError(msg)

    if aAj is not None:
        msg = "Aj cannot be vectorized."
        raise ValueError(msg)

    if aAx is not None and ax is not None:
        if Ax.ndim != 3 or x.ndim != 4:
            msg = (
                "Ax and x should be 3D and 4D respectively when vectorizing "
                f"over them simultaneously. Got: {Ax.shape=}; {x.shape=}."
            )
            raise ValueError(msg)
        # vectorize over n_lhs
        Ax = jnp.moveaxis(Ax, aAx, 0)
        x = jnp.moveaxis(x, ax, 0)
        shape = x.shape
        Ax = Ax.reshape(Ax.shape[0] * Ax.shape[1], Ax.shape[2])
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        return dot_f64.bind(Ai, Aj, Ax, x).reshape(*shape), 0
    if aAx is not None:
        if Ax.ndim != 3 or x.ndim != 3:
            msg = (
                "Ax and x should both be 3D when vectorizing "
                f"over Ax. Got: {Ax.shape=}; {x.shape=}."
            )
            raise ValueError(msg)
        # vectorize over n_lhs
        ax = 0
        Ax = jnp.moveaxis(Ax, aAx, 0)
        x = jnp.broadcast_to(x[None], (Ax.shape[0], x.shape[0], x.shape[1], x.shape[2]))
        shape = x.shape
        Ax = Ax.reshape(Ax.shape[0] * Ax.shape[1], Ax.shape[2])
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        return dot_f64.bind(Ai, Aj, Ax, x).reshape(*shape), 0
    if ax is not None:
        if Ax.ndim != 2 or x.ndim != 4:
            msg = (
                "Ax and x should both be 2D and 4D respectively when vectorizing "
                f"over x. Got: {Ax.shape=}; {x.shape=}."
            )
            raise ValueError(msg)
        # vectorize over n_rhs
        x = jnp.moveaxis(x, ax, 3)
        shape = x.shape
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        return dot_f64.bind(Ai, Aj, Ax, x).reshape(*shape), 3
    msg = "vmap failed. Please select an axis to vectorize over."
    raise ValueError(msg)


batching.primitive_batchers[dot_f64] = dot_f64_vmap

# Transposition =======================================================================


def dot_f64_transpose(
    ct: Array,
    Ai: Array,
    Aj: Array,
    Ax: Array,
    x: Array,
) -> tuple[Array, Array, Array, Array]:
    if ad.is_undefined_primal(Ai):
        msg = "Found undefined primal for Ai."
        raise ValueError(msg)

    if ad.is_undefined_primal(Aj):
        msg = "Found undefined primal for Aj."
        raise ValueError(msg)

    if ad.is_undefined_primal(x):
        # replace x by ct
        return Aj, Ai, Ax, dot_f64.bind(Aj, Ai, Ax, ct)

    if ad.is_undefined_primal(Ax):
        # replace Ax by ct
        # not really sure what I'm doing here, but this makes test_3d_jacrev pass.
        return Ai, Aj, (ct[:, Ai] * x[:, Aj, :]).sum(-1), x

    msg = "No undefined primals in transpose."
    raise ValueError(msg)


ad.primitive_transposes[dot_f64] = dot_f64_transpose
