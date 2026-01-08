"""klujax: a KLU solver for JAX."""

# Metadata ============================================================================

__version__ = "0.4.5"
__author__ = "Floris Laporte"
__all__ = ["coalesce", "dot", "solve"]

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
debug = lambda s: None if not DEBUG else print(s, file=sys.stderr)  # noqa: E731,T201
debug("KLUJAX DEBUG MODE.")

# Constants ===========================================================================

COMPLEX_DTYPES = (
    np.complex64,
    np.complex128,
    jnp.complex64,
    jnp.complex128,
)

# Main Functions ======================================================================


@jax.jit
def solve(Ai: Array, Aj: Array, Ax: Array, b: Array) -> Array:
    """Solve for x in the sparse linear system Ax=b.

    Args:
        Ai: [n_nz; int32]: the row indices of the sparse matrix A
        Aj: [n_nz; int32]: the column indices of the sparse matrix A
        Ax: [n_lhs? x n_nz; float64|complex128]: the values of the sparse matrix A
        b:  [n_lhs? x n_col x n_rhs?; float64|complex128]: the target vector

    Returns:
        x: the result (x≈A^-1b)

    """
    debug("solve")
    Ai, Aj, Ax, b, shape = validate_args(Ai, Aj, Ax, b, x_name="b")
    if any(x.dtype in COMPLEX_DTYPES for x in (Ax, b)):
        debug("solve-complex128")
        x = solve_c128.bind(
            Ai.astype(jnp.int32),
            Aj.astype(jnp.int32),
            Ax.astype(jnp.complex128),
            b.astype(jnp.complex128),
        )
    else:
        debug("solve-float64")
        x = solve_f64.bind(
            Ai.astype(jnp.int32),
            Aj.astype(jnp.int32),
            Ax.astype(jnp.float64),
            b.astype(jnp.float64),
        )

    return x.reshape(*shape)


@jax.jit
def dot(Ai: Array, Aj: Array, Ax: Array, x: Array) -> Array:
    """Multiply a sparse matrix with a vector: Ax=b.

    Args:
        Ai: [n_nz; int32]: the row indices of the sparse matrix A
        Aj: [n_nz; int32]: the column indices of the sparse matrix A
        Ax: [n_lhs? x n_nz; float64|complex128]: the values of the sparse matrix A
        x:  [n_lhs? x n_col x n_rhs?; float64|complex128]: the vector multiplied by A

    Returns:
        b: the result (b=A@x)

    """
    debug("dot")
    Ai, Aj, Ax, x, shape = validate_args(Ai, Aj, Ax, x, x_name="x")
    if any(x.dtype in COMPLEX_DTYPES for x in (Ax, x)):
        debug("dot-complex128")
        b = dot_c128.bind(
            Ai.astype(jnp.int32),
            Aj.astype(jnp.int32),
            Ax.astype(jnp.complex128),
            x.astype(jnp.complex128),
        )
    else:
        debug("dot-float64")
        b = dot_f64.bind(
            Ai.astype(jnp.int32),
            Aj.astype(jnp.int32),
            Ax.astype(jnp.float64),
            x.astype(jnp.float64),
        )
    return b.reshape(*shape)


def coalesce(
    Ai: jax.Array,
    Aj: jax.Array,
    Ax: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Coalesce a sparse matrix by summing duplicate indices.

    Args:
        Ai: [n_nz; int32]: the row indices of the sparse matrix A
        Aj: [n_nz; int32]: the column indices of the sparse matrix A
        Ax: [... x n_nz; float64|complex128]: the values of the sparse matrix A

    Returns:
        coalesced Ai, Aj, Ax

    """
    with jax.ensure_compile_time_eval():
        shape = Ax.shape

        order = jnp.lexsort((Aj, Ai))
        Ai = Ai[order]
        Aj = Aj[order]

        # Compute unique indices
        unique_mask = jnp.concatenate(
            [jnp.array([True]), (Ai[1:] != Ai[:-1]) | (Aj[1:] != Aj[:-1])],
        )
        unique_idxs = jnp.where(unique_mask)[0]

        # Assign each entry to a unique group
        groups = jnp.cumsum(unique_mask) - 1

        # Sum Ax values over groups
        Ai = Ai[unique_idxs]
        Aj = Aj[unique_idxs]

    Ax = Ax.reshape(-1, shape[-1])
    Ax = Ax[:, order]
    Ax = jax.vmap(jax.ops.segment_sum, [0, None], 0)(Ax, groups)

    return Ai, Aj, Ax.reshape(*shape[:-1], -1)


# Primitives ==========================================================================

dot_f64 = jax.extend.core.Primitive("dot_f64")
dot_c128 = jax.extend.core.Primitive("dot_c128")
solve_f64 = jax.extend.core.Primitive("solve_f64")
solve_c128 = jax.extend.core.Primitive("solve_c128")

# Implementations =====================================================================


@dot_f64.def_impl
def dot_f64_impl(Ai: Array, Aj: Array, Ax: Array, x: Array) -> Array:
    return general_impl("dot_f64", Ai, Aj, Ax, x)


@dot_c128.def_impl
def dot_c128_impl(Ai: Array, Aj: Array, Ax: Array, x: Array) -> Array:
    return general_impl("dot_c128", Ai, Aj, Ax, x)


@solve_f64.def_impl
def solve_f64_impl(Ai: Array, Aj: Array, Ax: Array, x: Array) -> Array:
    return general_impl("solve_f64", Ai, Aj, Ax, x)


@solve_c128.def_impl
def solve_c128_impl(Ai: Array, Aj: Array, Ax: Array, x: Array) -> Array:
    return general_impl("solve_c128", Ai, Aj, Ax, x)


def general_impl(name: str, Ai: Array, Aj: Array, Ax: Array, x: Array) -> Array:
    call = jax.ffi.ffi_call(
        name,
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

dot_f64_low = mlir.lower_fun(dot_f64_impl, multiple_results=False)
mlir.register_lowering(dot_f64, dot_f64_low)

jax.ffi.register_ffi_target(
    "dot_c128",
    klujax_cpp.dot_c128(),
    platform="cpu",
)

dot_c128_low = mlir.lower_fun(dot_c128_impl, multiple_results=False)
mlir.register_lowering(dot_c128, dot_c128_low)

jax.ffi.register_ffi_target(
    "solve_f64",
    klujax_cpp.solve_f64(),
    platform="cpu",
)

solve_f64_low = mlir.lower_fun(solve_f64_impl, multiple_results=False)
mlir.register_lowering(solve_f64, solve_f64_low)

jax.ffi.register_ffi_target(
    "solve_c128",
    klujax_cpp.solve_c128(),
    platform="cpu",
)

solve_c128_low = mlir.lower_fun(solve_c128_impl, multiple_results=False)
mlir.register_lowering(solve_c128, solve_c128_low)

# Abstract Evals ======================================================================


@dot_f64.def_abstract_eval
@dot_c128.def_abstract_eval
@solve_f64.def_abstract_eval
@solve_c128.def_abstract_eval
def general_abstract_eval(Ai: Array, Aj: Array, Ax: Array, b: Array) -> ShapedArray:  # noqa: ARG001
    return ShapedArray(b.shape, b.dtype)


# Forward Differentiation =============================================================


def dot_f64_value_and_jvp(
    arg_values: tuple[Array, Array, Array, Array],
    arg_tangents: tuple[Array, Array, Array, Array],
) -> tuple[Array, Array]:
    return dot_value_and_jvp(dot_f64, arg_values, arg_tangents)


ad.primitive_jvps[dot_f64] = dot_f64_value_and_jvp


def dot_c128_value_and_jvp(
    arg_values: tuple[Array, Array, Array, Array],
    arg_tangents: tuple[Array, Array, Array, Array],
) -> tuple[Array, Array]:
    return dot_value_and_jvp(dot_c128, arg_values, arg_tangents)


ad.primitive_jvps[dot_c128] = dot_c128_value_and_jvp


def solve_f64_value_and_jvp(
    arg_values: tuple[Array, Array, Array, Array],
    arg_tangents: tuple[Array, Array, Array, Array],
) -> tuple[Array, Array]:
    return solve_value_and_jvp(solve_f64, dot_f64, arg_values, arg_tangents)


ad.primitive_jvps[solve_f64] = solve_f64_value_and_jvp


def solve_c128_value_and_jvp(
    arg_values: tuple[Array, Array, Array, Array],
    arg_tangents: tuple[Array, Array, Array, Array],
) -> tuple[Array, Array]:
    return solve_value_and_jvp(solve_c128, dot_c128, arg_values, arg_tangents)


ad.primitive_jvps[solve_c128] = solve_c128_value_and_jvp


def solve_value_and_jvp(
    prim_solve: jax.extend.core.Primitive,
    prim_dot: jax.extend.core.Primitive,
    arg_values: tuple[Array, Array, Array, Array],
    arg_tangents: tuple[Array, Array, Array, Array],
) -> tuple[Array, Array]:
    Ai, Aj, Ax, b = arg_values
    dAi, dAj, dAx, db = arg_tangents
    dAx = dAx if not isinstance(dAx, ad.Zero) else lax.zeros_like_array(Ax)
    dAi = dAi if not isinstance(dAi, ad.Zero) else lax.zeros_like_array(Ai)
    dAj = dAj if not isinstance(dAj, ad.Zero) else lax.zeros_like_array(Aj)
    db = db if not isinstance(db, ad.Zero) else lax.zeros_like_array(b)
    x = prim_solve.bind(Ai, Aj, Ax, b)
    dA_x = prim_dot.bind(Ai, Aj, dAx, x)
    invA_dA_x = prim_solve.bind(Ai, Aj, Ax, dA_x)
    invA_db = prim_solve.bind(Ai, Aj, Ax, db)
    return x, -invA_dA_x + invA_db


def dot_value_and_jvp(
    prim: jax.extend.core.Primitive,
    arg_values: tuple[Array, Array, Array, Array],
    arg_tangents: tuple[Array, Array, Array, Array],
) -> tuple[Array, Array]:
    Ai, Aj, Ax, b = arg_values
    dAi, dAj, dAx, db = arg_tangents
    dAx = dAx if not isinstance(dAx, ad.Zero) else lax.zeros_like_array(Ax)
    dAi = dAi if not isinstance(dAi, ad.Zero) else lax.zeros_like_array(Ai)
    dAj = dAj if not isinstance(dAj, ad.Zero) else lax.zeros_like_array(Aj)
    db = db if not isinstance(db, ad.Zero) else lax.zeros_like_array(b)
    x = prim.bind(Ai, Aj, Ax, b)
    dA_b = prim.bind(Ai, Aj, dAx, b)
    A_db = prim.bind(Ai, Aj, Ax, db)
    return x, dA_b + A_db


# Batching (vmap) =====================================================================


def dot_f64_vmap(
    vector_arg_values: tuple[Array, Array, Array, Array],
    batch_axes: tuple[int | None, int | None, int | None, int | None],
) -> tuple[Array, int]:
    return general_vmap(dot_f64, vector_arg_values, batch_axes)


batching.primitive_batchers[dot_f64] = dot_f64_vmap


def dot_c128_vmap(
    vector_arg_values: tuple[Array, Array, Array, Array],
    batch_axes: tuple[int | None, int | None, int | None, int | None],
) -> tuple[Array, int]:
    return general_vmap(dot_c128, vector_arg_values, batch_axes)


batching.primitive_batchers[dot_c128] = dot_c128_vmap


def solve_f64_vmap(
    vector_arg_values: tuple[Array, Array, Array, Array],
    batch_axes: tuple[int | None, int | None, int | None, int | None],
) -> tuple[Array, int]:
    return general_vmap(solve_f64, vector_arg_values, batch_axes)


batching.primitive_batchers[solve_f64] = solve_f64_vmap


def solve_c128_vmap(
    vector_arg_values: tuple[Array, Array, Array, Array],
    batch_axes: tuple[int | None, int | None, int | None, int | None],
) -> tuple[Array, int]:
    return general_vmap(solve_c128, vector_arg_values, batch_axes)


batching.primitive_batchers[solve_c128] = solve_c128_vmap


def general_vmap(
    prim: jax.extend.core.Primitive,
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
        return prim.bind(Ai, Aj, Ax, x).reshape(*shape), 0
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
        return prim.bind(Ai, Aj, Ax, x).reshape(*shape), 0
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
        return prim.bind(Ai, Aj, Ax, x).reshape(*shape), 3
    msg = "vmap failed. Please select an axis to vectorize over."
    raise ValueError(msg)


# Transposition =======================================================================


def dot_f64_transpose(
    ct: Array,
    Ai: Array,
    Aj: Array,
    Ax: Array,
    x: Array,
) -> tuple[Array, Array, Array, Array]:
    return dot_transpose(dot_f64, ct, Ai, Aj, Ax, x)


ad.primitive_transposes[dot_f64] = dot_f64_transpose


def dot_c128_transpose(
    ct: Array,
    Ai: Array,
    Aj: Array,
    Ax: Array,
    x: Array,
) -> tuple[Array, Array, Array, Array]:
    return dot_transpose(dot_c128, ct, Ai, Aj, Ax, x)


ad.primitive_transposes[dot_c128] = dot_c128_transpose


def solve_f64_transpose(
    ct: Array,
    Ai: Array,
    Aj: Array,
    Ax: Array,
    b: Array,
) -> tuple[Array, Array, Array, Array]:
    return solve_transpose(solve_f64, ct, Ai, Aj, Ax, b)


ad.primitive_transposes[solve_f64] = solve_f64_transpose


def solve_c128_transpose(
    ct: Array,
    Ai: Array,
    Aj: Array,
    Ax: Array,
    b: Array,
) -> tuple[Array, Array, Array, Array]:
    return solve_transpose(solve_c128, ct, Ai, Aj, Ax, b)


ad.primitive_transposes[solve_c128] = solve_c128_transpose


def dot_transpose(
    prim: jax.extend.core.Primitive,
    ct: Array,
    Ai: Array,
    Aj: Array,
    Ax: Array,
    x: Array,
) -> tuple[Array, Array, Array, Array]:
    if ad.is_undefined_primal(Ai) or ad.is_undefined_primal(Aj):
        msg = "Sparse indices Ai and Aj should not require gradients."
        raise ValueError(msg)

    if ad.is_undefined_primal(x):
        # replace x by ct
        return Aj, Ai, Ax, prim.bind(Aj, Ai, Ax, ct)

    if ad.is_undefined_primal(Ax):
        # replace Ax by ct
        # not really sure what I'm doing here, but this makes test_3d_jacrev pass.
        return Ai, Aj, (ct[:, Ai] * x[:, Aj, :]).sum(-1), x

    msg = "No undefined primals in transpose."
    raise ValueError(msg)


def solve_transpose(
    prim: jax.extend.core.Primitive,
    ct: Array,
    Ai: Array,
    Aj: Array,
    Ax: Array,
    b: Array,
) -> tuple[Array, Array, Array, Array]:
    if ad.is_undefined_primal(Ai) or ad.is_undefined_primal(Aj):
        msg = "Sparse indices Ai and Aj should not require gradients."
        raise ValueError(msg)

    if ad.is_undefined_primal(b):
        b_bar = prim.bind(Aj, Ai, Ax, ct)
        return Ai, Aj, Ax, b_bar

    if ad.is_undefined_primal(Ax):
        Ax_bar = -(ct * prim.bind(Ai, Aj, Ax, b)).sum(-1)
        return Ai, Aj, Ax_bar, b

    msg = "No undefined primals in transpose."
    raise ValueError(msg)


# Validators ==========================================================================


def validate_args(  # noqa: C901,PLR0912
    Ai: Array, Aj: Array, Ax: Array, x: Array, x_name: str = "x"
) -> tuple[Array, Array, Array, Array, tuple[int, ...]]:
    # cases:
    # - (n_lhs, n_nz) x (n_lhs, n_col, n_rhs)
    # - (n_lhs, n_nz) x (n_lhs, n_col) --> (n_lhs, n_nz) x (n_lhs, n_col, 1)
    # - (n_nz,) x (n_lhs, n_col, n_rhs) --> (n_lhs, n_nz) x (n_lhs, n_col, n_rhs)
    # - (n_nz,) x (n_col, n_rhs) --> (1, n_nz) x (1, n_col, n_rhs)
    # - (n_nz,) x (n_col,) --> (1, n_nz) x (1, n_col, 1)
    if Ai.ndim != 1:
        msg = f"Ai should be 1D with shape (n_nz). Got: {Ai.shape=}."
        raise ValueError(msg)
    if Aj.ndim != 1:
        msg = f"Aj should be 1D with shape (n_nz,). Got: {Aj.shape=}."
        raise ValueError(msg)
    if Ax.ndim == 0 or Ax.ndim > 2:
        msg = (
            "Ax should be 1D with shape (n_nz,) "
            "or 2D with shape (n_lhs, n_nz). "
            f"Got: {Ax.shape=}."
        )
        raise ValueError(msg)
    if x.ndim == 0 or x.ndim > 3:
        msg = (
            f"{x_name} should be 1D with shape (n_col,) "
            "or 2D with shape (n_col, n_rhs) "
            "or 3D with shape (n_lhs, n_col, n_rhs). "
            f"Got: {x_name}.shape={x.shape}."
        )
        raise ValueError(msg)

    shape = x.shape

    if Ax.ndim == 1 and x.ndim == 1:  # expand Ax and b dims
        debug(f"assuming (n_nz:={Ax.shape[0]},) x (n_col:={x.shape[0]},)")
        Ax = Ax[None, :]
        x = x[None, :, None]
    elif Ax.ndim == 1 and x.ndim == 2:  # expand Ax and b dims
        debug(
            f"assuming (n_nz:={Ax.shape[0]},) x "
            f"(n_col:={x.shape[0]}, n_rhs:={x.shape[1]})"
        )
        Ax = Ax[None, :]
        x = x[None, :, :]
    elif Ax.ndim == 1 and x.ndim == 3:  # expand A dim (broadcast will happen in base)
        debug(
            f"assuming (n_nz:={Ax.shape[0]},) x "
            f"(n_lhs:={x.shape[0]}, n_col:={x.shape[1]}, n_rhs:={x.shape[2]})"
        )
        Ax = Ax[None, :]
    elif Ax.ndim == 2 and x.ndim == 1:  # expand dims to base case
        debug(
            f"assuming (n_lhs:={Ax.shape[0]}, n_nz:={Ax.shape[1]}) x "
            f"(n_col:={x.shape[1]},)"
        )
        x = x[None, :, None]
        shape = (Ax.shape[0], shape[0])  # we need to expand the shape here.
    elif Ax.ndim == 2 and x.ndim == 2:  # expand dims to base case
        debug(
            f"assuming (n_lhs:={Ax.shape[0]}, n_nz:={Ax.shape[1]}) x "
            f"(n_lhs:={x.shape[0]}, n_col:={x.shape[1]})"
        )
        if Ax.shape[0] != x.shape[0] and Ax.shape[0] != 1 and x.shape[0] != 1:
            msg = (
                f"Ax (2D) and {x_name} (2D) should have their first shape "
                f"index `n_lhs` match. Got: {Ax.shape=}; {x_name}.shape={x.shape}. "
                f"assuming (n_lhs:={Ax.shape[0]}, n_nz:={Ax.shape[1]}) x "
                f"(n_lhs:={x.shape[0]}, n_col:={x.shape[1]})"
            )
            raise ValueError(msg)
        x = x[:, :, None]
        if x.shape[0] == 1 and Ax.shape[0] > 0:
            shape = (Ax.shape[0], *shape[1:])

    if Ax.ndim != 2 or x.ndim != 3:
        msg = (
            f"Invalid shapes for Ax and {x_name}. "
            f"Got: {Ax.shape=}; {x_name}.shape={x.shape}. "
            f"Expected: Ax.shape=([n_lhs],n_nz); "
            f"{x_name}.shape=([n_lhs],n_col,[n_rhs])."
        )
        raise ValueError(msg)

    # base case
    debug(
        f"assuming (n_lhs:={Ax.shape[0]}, n_nz:={Ax.shape[1]}) x "
        f"(n_lhs:={x.shape[0]}, n_col:={x.shape[1]}, n_rhs:={x.shape[2]})"
    )
    if Ax.shape[0] != x.shape[0] and Ax.shape[0] != 1 and x.shape[0] != 1:
        msg = (
            f"Ax (2D) and {x_name} (3D) should have their first shape "
            f"index `n_lhs` match. Got: {Ax.shape=}; {x_name}.shape={x.shape}."
            f"assuming (n_lhs:={Ax.shape[0]}, n_nz:={Ax.shape[1]}) x "
            f"(n_lhs:={x.shape[0]}, n_col:={x.shape[1]}, n_rhs:={x.shape[2]})"
        )
        raise ValueError(msg)
    n_lhs = max(Ax.shape[0], x.shape[0])  # handle broadcastable 1-index
    Ax = jnp.broadcast_to(Ax, (n_lhs, Ax.shape[1]))
    x = jnp.broadcast_to(x, (n_lhs, x.shape[1], x.shape[2]))
    if len(shape) == 3 and shape[0] != x.shape[0]:
        shape = (Ax.shape[0], shape[1], shape[2])
    return Ai, Aj, Ax, x, shape
