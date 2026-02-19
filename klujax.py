"""klujax: a KLU solver for JAX."""

# Metadata ============================================================================

__version__ = "0.4.8"
__author__ = "Floris Laporte"
__all__ = ["analyze", "coalesce", "dot", "factor", "free_numeric", "free_symbolic", "solve", "solve_with_numeric", "solve_with_symbol"]

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
from typing import Any, Callable, Optional, Type, Tuple, Union
from types import TracebackType

# Config ==============================================================================

DEBUG = bool(int(os.environ.get("KLUJAX_DEBUG", "0")))
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
    Ai: Array,
    Aj: Array,
    Ax: Array,
) -> tuple[Array, Array, Array]:
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

# Split Solve pointer management =========================================================

class KLUHandleManager:
    """RAII wrapper for KLU handles. Handles are freed on __del__ or __exit__."""

    def __init__(self, handle: Array, free_callable: Callable, owner: bool = True) -> None:
        self.handle = handle
        self.free_callable = free_callable
        self._owner = owner
        self._freed = False

    def close(self) -> None:
        """Release the C++ resource if this instance is the owner."""
        # Safety check: If the interpreter is shutting down, 'jax' might be None.
        # If so, we can simply return, as the OS will reclaim the memory momentarily.
        if jax is None:
            return

        if self._freed or isinstance(self.handle, jax.core.Tracer):
            return

        if self._owner and self.free_callable:
            try:
                self.free_callable(self.handle)
            except Exception:
                pass
        self._freed = True

    def __enter__(self) -> "KLUHandleManager":
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.close()

    def __del__(self) -> None:
        if hasattr(self, "close"):
            self.close()

def _klu_flatten(obj: KLUHandleManager) -> tuple[tuple[()], tuple[Array, Callable]]:
    # No leaves — handle and callable are both static aux data
    return (), (obj.handle, obj.free_callable)


def _klu_unflatten(aux: tuple[Array, Callable], children: tuple[()]) -> KLUHandleManager:
    handle, free_callable = aux
    return KLUHandleManager(handle, free_callable=free_callable, owner=False)

jax.tree_util.register_pytree_node(KLUHandleManager, _klu_flatten, _klu_unflatten)


def free_symbolic(symbolic: Union[KLUHandleManager, Array], dependency: Any = None) -> Array:
    """Free the KLU symbolic analysis object.

    Args:
        symbolic: [KLUHandleManager|Array]: the symbolic analysis object or handle
        dependency: [Any]: optional dependency to enforce ordering in JIT

    Returns:
        result: [int32]: 0 if successful

    """
    if isinstance(symbolic, KLUHandleManager):
        symbolic.close()
        return jnp.array(0, dtype=jnp.int32)

    handle = getattr(symbolic, "handle", symbolic)
    if isinstance(handle, jax.core.Tracer) and dependency is not None:
        token = jax.tree_util.tree_leaves(dependency)[0]
        return lax.cond(
            jnp.array(True),
            lambda ops: free_symbolic_p.bind(ops[0]),
            lambda _: jnp.array(0, dtype=jnp.int32),
            operand=(handle, token),
        )
    return free_symbolic_p.bind(handle)

def free_numeric(numeric: Union[KLUHandleManager, Array], dependency: Any = None) -> Array:
    """Free the KLU numeric factorization object.

    Args:
        numeric: [KLUHandleManager|Array]: the numeric factorization object or handle
        dependency: [Any]: optional dependency to enforce ordering in JIT

    Returns:
        result: [int32]: 0 if successful

    """
    if isinstance(numeric, KLUHandleManager):
        numeric.close()
        return jnp.array(0, dtype=jnp.int32)

    handle = getattr(numeric, "handle", numeric)
    if isinstance(handle, jax.core.Tracer) and dependency is not None:
        token = jax.tree_util.tree_leaves(dependency)[0]
        return lax.cond(
            jnp.array(True),
            lambda ops: free_numeric_p.bind(ops[0]),
            lambda _: jnp.array(0, dtype=jnp.int32),
            operand=(handle, token),
        )
    return free_numeric_p.bind(handle)

# Split Solve routines =============================================================

def analyze(Ai: Array, Aj: Array, n_col: int) -> KLUHandleManager:
    """Analyze the sparsity pattern of a matrix A.

    Args:
        Ai: [n_nz; int32]: the row indices of the sparse matrix A
        Aj: [n_nz; int32]: the column indices of the sparse matrix A
        n_col: [int]: the number of columns in the sparse matrix A

    Returns:
        symbolic: [KLUHandleManager]: the symbolic analysis object

    """
    Ai = jnp.asarray(Ai, dtype=jnp.int32)
    Aj = jnp.asarray(Aj, dtype=jnp.int32)
    raw_symbol = analyze_p.bind(Ai, Aj, jnp.int32(n_col))
    return KLUHandleManager(raw_symbol, free_symbolic, owner=True)

def validate_numeric_solve(
    Ai: Array, Aj: Array, Ax: Array, b: Array
) -> tuple[Array, Array, Array, Array, tuple[int, ...]]:
    """Reduced set of validate_args for use with solve_with_symbol"""

    order = jnp.lexsort((Aj, Ai))
    Ai, Aj = Ai[order], Aj[order]
    Ax = Ax[..., order] if Ax.ndim == 2 else Ax[order]

    shape = b.shape

    # 2. Dimension expansion to base case: (n_lhs, n_nz) and (n_lhs, n_col, n_rhs)
    if Ax.ndim == 1 and b.ndim == 1:
        Ax, b = Ax[None, :], b[None, :, None]
    elif Ax.ndim == 1 and b.ndim == 2:
        Ax, b = Ax[None, :], b[None, :, :]
    elif Ax.ndim == 1 and b.ndim == 3:
        Ax = Ax[None, :]
    elif Ax.ndim == 2 and b.ndim == 1:
        b = b[None, :, None]
        shape = (Ax.shape[0], shape[0])
    elif Ax.ndim == 2 and b.ndim == 2:
        if Ax.shape[0] != b.shape[0] and Ax.shape[0] != 1 and b.shape[0] != 1:
            raise ValueError(f"Batch mismatch: {Ax.shape=} vs {b.shape=}")
        b = b[:, :, None]
        if b.shape[0] == 1 and Ax.shape[0] > 1:
            shape = (Ax.shape[0], *shape[1:])

    # 3. Final broadcasting for C++ FFI
    n_lhs = max(Ax.shape[0], b.shape[0])
    Ax = jnp.broadcast_to(Ax, (n_lhs, Ax.shape[1]))
    b = jnp.broadcast_to(b, (n_lhs, b.shape[1], b.shape[2]))

    if len(shape) == 3 and shape[0] != b.shape[0]:
        shape = (Ax.shape[0], shape[1], shape[2])

    return Ai, Aj, Ax, b, shape

@jax.jit
def _solve_with_symbol_jit(Ai: Array, Aj: Array, Ax: Array, b: Array, sym_h: Array) -> Array:
    # Use the robust validator
    Ai, Aj, Ax, b, out_shape = validate_numeric_solve(Ai, Aj, Ax, b)
    
    is_complex = any(x.dtype in COMPLEX_DTYPES for x in (Ax, b))
    prim = solve_with_symbol_c128 if is_complex else solve_with_symbol_f64
    
    # Pass standardized arrays to the C++ extension
    x = prim.bind(
        Ai, Aj,
        Ax.astype(jnp.complex128 if is_complex else jnp.float64),
        b.astype(jnp.complex128 if is_complex else jnp.float64),
        sym_h.astype(jnp.uint64)
    )
    
    return x.reshape(*out_shape)

def solve_with_symbol(Ai: Array, Aj: Array, Ax: Array, b: Array, symbolic: Union[KLUHandleManager, Array]) -> Array:
    """Solve Ax=b using a pre-computed symbolic analysis.

    Args:
        Ai: [n_nz; int32]: the row indices of the sparse matrix A
        Aj: [n_nz; int32]: the column indices of the sparse matrix A
        Ax: [n_lhs? x n_nz; float64|complex128]: the values of the sparse matrix A
        b:  [n_lhs? x n_col x n_rhs?; float64|complex128]: the target vector
        symbolic: [KLUHandleManager|Array]: the symbolic analysis object or handle

    Returns:
        x: the result (x≈A^-1b)

    """
    handle = getattr(symbolic, "handle", symbolic)
    return _solve_with_symbol_jit(Ai, Aj, Ax, b, handle)

@jax.jit
def _factor_jit(Ai: Array, Aj: Array, Ax: Array, sym_h: Array) -> Array:
    dummy_b = jnp.zeros((1,), dtype=Ax.dtype)
    Ai, Aj, Ax, _, _ = validate_args(Ai, Aj, Ax, dummy_b)
    prim = factor_c128 if Ax.dtype in COMPLEX_DTYPES else factor_f64
    return prim.bind(Ai, Aj, Ax, sym_h)

def factor(Ai: Array, Aj: Array, Ax: Array, symbolic: Union[KLUHandleManager, Array]) -> KLUHandleManager:
    """Compute the numeric factorization of a matrix A given its symbolic analysis.

    Args:
        Ai: [n_nz; int32]: the row indices of the sparse matrix A
        Aj: [n_nz; int32]: the column indices of the sparse matrix A
        Ax: [n_lhs? x n_nz; float64|complex128]: the values of the sparse matrix A
        symbolic: [KLUHandleManager|Array]: the symbolic analysis object or handle

    Returns:
        numeric: [KLUHandleManager]: the numeric factorization object

    """
    sym_h = getattr(symbolic, "handle", symbolic)
    raw_numeric = _factor_jit(Ai, Aj, Ax, sym_h)
    return KLUHandleManager(raw_numeric, free_numeric, owner=True)

@jax.jit
def _solve_with_numeric_jit(num_h: Array, b: Array, sym_h: Array) -> Array:
    prim = solve_with_numeric_c128 if b.dtype in COMPLEX_DTYPES else solve_with_numeric_f64
    return prim.bind(sym_h.astype(jnp.uint64), num_h.astype(jnp.uint64), b)

def solve_with_numeric(numeric: Union[KLUHandleManager, Array], b: Array, symbolic: Union[KLUHandleManager, Array]) -> Array:
    """Solve Ax=b using a pre-computed numeric factorization.

    Args:
        numeric: [KLUHandleManager|Array]: the numeric factorization object or handle
        b:  [n_lhs? x n_col x n_rhs?; float64|complex128]: the target vector
        symbolic: [KLUHandleManager|Array]: the symbolic analysis object or handle

    Returns:
        x: the result (x≈A^-1b)

    """
    num_h = getattr(numeric, "handle", numeric)
    sym_h = getattr(symbolic, "handle", symbolic)
    result = _solve_with_numeric_jit(num_h, b, sym_h)
    
    # Auto-cleanup if the handle was created inside a JIT block.
    if isinstance(num_h, jax.core.Tracer) and isinstance(numeric, KLUHandleManager):
        free_numeric(numeric, dependency=result)
    return result


# Primitives ==========================================================================

dot_f64 = jax.extend.core.Primitive("dot_f64")
dot_c128 = jax.extend.core.Primitive("dot_c128")
solve_f64 = jax.extend.core.Primitive("solve_f64")
solve_c128 = jax.extend.core.Primitive("solve_c128")
analyze_p = jax.extend.core.Primitive("analyze")
solve_with_symbol_f64 = jax.extend.core.Primitive("solve_with_symbol_f64")
solve_with_symbol_c128 = jax.extend.core.Primitive("solve_with_symbol_c128")
free_symbolic_p = jax.extend.core.Primitive("free_symbolic")
factor_f64 = jax.extend.core.Primitive("factor_f64")
factor_c128 = jax.extend.core.Primitive("factor_c128")
solve_with_numeric_f64 = jax.extend.core.Primitive("solve_with_numeric_f64")
solve_with_numeric_c128 = jax.extend.core.Primitive("solve_with_numeric_c128")
free_numeric_p = jax.extend.core.Primitive("free_numeric")

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


@solve_with_symbol_f64.def_impl
def solve_with_symbol_f64_impl(Ai: Array, Aj: Array, Ax: Array, b: Array, symbolic: Array) -> Array:
    return general_impl("solve_with_symbol_f64", Ai, Aj, Ax, b, symbolic)


@solve_with_symbol_c128.def_impl
def solve_with_symbol_c128_impl(Ai: Array, Aj: Array, Ax: Array, b: Array, symbolic: Array) -> Array:
    return general_impl("solve_with_symbol_c128", Ai, Aj, Ax, b, symbolic)


@analyze_p.def_impl
def analyze_impl(Ai: Array, Aj: Array, n_col: Array) -> Array:
    return jax.ffi.ffi_call("analyze", ShapedArray((), jnp.uint64))(Ai, Aj, n_col)


@factor_f64.def_impl
def factor_f64_impl(Ai, Aj, Ax, symbolic):
    n_lhs = Ax.shape[0]
    call = jax.ffi.ffi_call("factor_f64", jax.ShapeDtypeStruct((n_lhs,), jnp.uint64))
    return call(Ai, Aj, Ax, symbolic)

@factor_c128.def_impl
def factor_c128_impl(Ai, Aj, Ax, symbolic):
    n_lhs = Ax.shape[0]
    call = jax.ffi.ffi_call("factor_c128", jax.ShapeDtypeStruct((n_lhs,), jnp.uint64))
    return call(Ai, Aj, Ax, symbolic)

@solve_with_numeric_f64.def_impl
def solve_with_numeric_f64_impl(symbolic, numeric, b):
    call = jax.ffi.ffi_call("solve_with_numeric_f64", jax.ShapeDtypeStruct(b.shape, b.dtype))
    return call(symbolic, numeric, b)

@solve_with_numeric_c128.def_impl
def solve_with_numeric_c128_impl(symbolic, numeric, b):
    call = jax.ffi.ffi_call("solve_with_numeric_c128", jax.ShapeDtypeStruct(b.shape, b.dtype))
    return call(symbolic, numeric, b)


def general_impl(name: str, Ai: Array, Aj: Array, Ax: Array, x: Array, *args: Array) -> Array:
    call = jax.ffi.ffi_call(
        name,
        jax.ShapeDtypeStruct(x.shape, x.dtype),
    )
    if not callable(call):
        msg = "jax.ffi.ffi_call did not return a callable."
        raise RuntimeError(msg)  # noqa: TRY004
    return call(Ai, Aj, Ax, x, *args)


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

jax.ffi.register_ffi_target(
    "analyze",
    klujax_cpp.analyze(),
    platform="cpu",
)

analyze_low = mlir.lower_fun(analyze_impl, multiple_results=False)
mlir.register_lowering(analyze_p, analyze_low)

jax.ffi.register_ffi_target(
    "solve_with_symbol_f64",
    klujax_cpp.solve_with_symbol_f64(),
    platform="cpu",
)

solve_with_symbol_f64_low = mlir.lower_fun(solve_with_symbol_f64_impl, multiple_results=False)
mlir.register_lowering(solve_with_symbol_f64, solve_with_symbol_f64_low)

jax.ffi.register_ffi_target(
    "solve_with_symbol_c128",
    klujax_cpp.solve_with_symbol_c128(),
    platform="cpu",
)

solve_with_symbol_c128_low = mlir.lower_fun(solve_with_symbol_c128_impl, multiple_results=False)
mlir.register_lowering(solve_with_symbol_c128, solve_with_symbol_c128_low)

jax.ffi.register_ffi_target(
    "free_symbolic",
    klujax_cpp.free_symbolic(),
    platform="cpu",
)

jax.ffi.register_ffi_target(
    "free_numeric",
    klujax_cpp.free_numeric(),
    platform="cpu",
)

@free_numeric_p.def_impl
def free_numeric_impl(numeric):
    call = jax.ffi.ffi_call("free_numeric", jax.ShapeDtypeStruct((), jnp.int32))
    return call(numeric)


@free_symbolic_p.def_impl
def free_symbolic_impl(symbolic):
    call = jax.ffi.ffi_call("free_symbolic", jax.ShapeDtypeStruct((), jnp.int32))
    return call(symbolic)

@free_numeric_p.def_abstract_eval
def free_numeric_abstract_eval(numeric):
    return ShapedArray((), jnp.int32)

@free_symbolic_p.def_abstract_eval
def free_symbolic_abstract_eval(symbolic):
    return ShapedArray((), jnp.int32)

jax.ffi.register_ffi_target("factor_f64", klujax_cpp.factor_f64(), platform="cpu")
factor_f64_low = mlir.lower_fun(factor_f64_impl, multiple_results=False)
mlir.register_lowering(factor_f64, factor_f64_low)

jax.ffi.register_ffi_target("factor_c128", klujax_cpp.factor_c128(), platform="cpu")
factor_c128_low = mlir.lower_fun(factor_c128_impl, multiple_results=False)
mlir.register_lowering(factor_c128, factor_c128_low)

jax.ffi.register_ffi_target("solve_with_numeric_f64", klujax_cpp.solve_with_numeric_f64(), platform="cpu")
solve_with_numeric_f64_low = mlir.lower_fun(solve_with_numeric_f64_impl, multiple_results=False)
mlir.register_lowering(solve_with_numeric_f64, solve_with_numeric_f64_low)

jax.ffi.register_ffi_target("solve_with_numeric_c128", klujax_cpp.solve_with_numeric_c128(), platform="cpu")
solve_with_numeric_c128_low = mlir.lower_fun(solve_with_numeric_c128_impl, multiple_results=False)
mlir.register_lowering(solve_with_numeric_c128, solve_with_numeric_c128_low)

free_numeric_low = mlir.lower_fun(free_numeric_impl, multiple_results=False)
mlir.register_lowering(free_numeric_p, free_numeric_low)

free_symbolic_low = mlir.lower_fun(free_symbolic_impl, multiple_results=False)
mlir.register_lowering(free_symbolic_p, free_symbolic_low)

# Abstract Evals ======================================================================


@dot_f64.def_abstract_eval
@dot_c128.def_abstract_eval
@solve_f64.def_abstract_eval
@solve_c128.def_abstract_eval
@solve_with_symbol_f64.def_abstract_eval
@solve_with_symbol_c128.def_abstract_eval
def general_abstract_eval(Ai: Array, Aj: Array, Ax: Array, b: Array, *args: Array) -> ShapedArray:  # noqa: ARG001
    return ShapedArray(b.shape, b.dtype)


@analyze_p.def_abstract_eval
def analyze_abstract_eval(Ai: Array, Aj: Array, n_col: Array) -> ShapedArray:  # noqa: ARG001
    return ShapedArray((), jnp.uint64)


@free_symbolic_p.def_abstract_eval
def free_symbolic_abstract_eval(symbolic: Array) -> None:  # noqa: ARG001
    return None

@factor_f64.def_abstract_eval
@factor_c128.def_abstract_eval
def factor_abstract_eval(Ai, Aj, Ax, symbolic):
    return ShapedArray((Ax.shape[0],), jnp.uint64)

@solve_with_numeric_f64.def_abstract_eval
@solve_with_numeric_c128.def_abstract_eval
def solve_with_numeric_abstract_eval(symbolic, numeric, b):
    # Output has same shape as input b
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
    if not isinstance(dAi, ad.Zero) or not isinstance(dAj, ad.Zero):
        msg = "Sparse indices Ai and Aj should not require gradients."
        raise ValueError(msg)  # noqa: TRY004
    dAx = dAx if not isinstance(dAx, ad.Zero) else jnp.zeros_like(Ax)
    db = db if not isinstance(db, ad.Zero) else jnp.zeros_like(b)
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
    if not isinstance(dAi, ad.Zero) or not isinstance(dAj, ad.Zero):
        msg = "Sparse indices Ai and Aj should not require gradients."
        raise ValueError(msg)  # noqa: TRY004
    dAx = dAx if not isinstance(dAx, ad.Zero) else jnp.zeros_like(Ax)
    db = db if not isinstance(db, ad.Zero) else jnp.zeros_like(b)
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


def solve_with_symbol_f64_vmap(
    vector_arg_values: tuple[Array, Array, Array, Array, Array],
    batch_axes: tuple[int | None, int | None, int | None, int | None, int | None],
) -> tuple[Array, int]:
    return general_vmap_with_symbol(solve_with_symbol_f64, vector_arg_values, batch_axes)


batching.primitive_batchers[solve_with_symbol_f64] = solve_with_symbol_f64_vmap


def solve_with_symbol_c128_vmap(
    vector_arg_values: tuple[Array, Array, Array, Array, Array],
    batch_axes: tuple[int | None, int | None, int | None, int | None, int | None],
) -> tuple[Array, int]:
    return general_vmap_with_symbol(solve_with_symbol_c128, vector_arg_values, batch_axes)


batching.primitive_batchers[solve_with_symbol_c128] = solve_with_symbol_c128_vmap


def solve_with_numeric_f64_vmap(
    vector_arg_values: tuple[Array, Array, Array],
    batch_axes: tuple[int | None, int | None, int | None],
) -> tuple[Array, int]:
    return general_vmap_with_numeric(solve_with_numeric_f64, vector_arg_values, batch_axes)


batching.primitive_batchers[solve_with_numeric_f64] = solve_with_numeric_f64_vmap


def solve_with_numeric_c128_vmap(
    vector_arg_values: tuple[Array, Array, Array],
    batch_axes: tuple[int | None, int | None, int | None],
) -> tuple[Array, int]:
    return general_vmap_with_numeric(solve_with_numeric_c128, vector_arg_values, batch_axes)


batching.primitive_batchers[solve_with_numeric_c128] = solve_with_numeric_c128_vmap


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


def general_vmap_with_symbol(
    prim: jax.extend.core.Primitive,
    vector_arg_values: tuple[Array, Array, Array, Array, Array],
    batch_axes: tuple[int | None, int | None, int | None, int | None, int | None],
) -> tuple[Array, int]:
    Ai, Aj, Ax, x, symbolic = vector_arg_values
    aAi, aAj, aAx, ax, asymbolic = batch_axes

    if aAi is not None:
        msg = "Ai cannot be vectorized."
        raise ValueError(msg)

    if aAj is not None:
        msg = "Aj cannot be vectorized."
        raise ValueError(msg)
    
    if asymbolic is not None:
        msg = "symbolic handle cannot be vectorized."
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
        return prim.bind(Ai, Aj, Ax, x, symbolic).reshape(*shape), 0
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
        return prim.bind(Ai, Aj, Ax, x, symbolic).reshape(*shape), 0
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
        return prim.bind(Ai, Aj, Ax, x, symbolic).reshape(*shape), 3
    msg = "vmap failed. Please select an axis to vectorize over."
    raise ValueError(msg)


def general_vmap_with_numeric(
    prim: jax.extend.core.Primitive,
    vector_arg_values: tuple[Array, Array, Array],
    batch_axes: tuple[int | None, int | None, int | None],
) -> tuple[Array, int]:
    symbolic, numeric, b = vector_arg_values
    asymbolic, anumeric, ab = batch_axes
    
    if asymbolic is not None:
        msg = "symbolic handle cannot be vectorized."
        raise ValueError(msg)
    
    # numeric and b should batch together
    if anumeric is not None and ab is not None:
        # Both numeric and b are batched - they should batch along the same dimension
        if numeric.ndim != 2 or b.ndim != 3:
            msg = (
                "numeric and b should be 2D and 3D respectively when vectorizing "
                f"over them simultaneously. Got: {numeric.shape=}; {b.shape=}."
            )
            raise ValueError(msg)
        # Move batch axes to the front
        numeric = jnp.moveaxis(numeric, anumeric, 0)
        b = jnp.moveaxis(b, ab, 0)
        shape = b.shape
        # Flatten the batch dimension with the matrix dimension
        numeric = numeric.reshape(numeric.shape[0] * numeric.shape[1])
        b = b.reshape(b.shape[0] * b.shape[1], b.shape[2])
        result = prim.bind(symbolic, numeric, b)
        # Reshape back to include the batch dimension
        return result.reshape(*shape), 0
    
    if anumeric is not None:
        # Only numeric is batched
        if numeric.ndim != 2 or b.ndim != 2:
            msg = (
                "numeric and b should be 2D when vectorizing over numeric. "
                f"Got: {numeric.shape=}; {b.shape=}."
            )
            raise ValueError(msg)
        numeric = jnp.moveaxis(numeric, anumeric, 0)
        # Broadcast b to match the batch dimension
        b = jnp.broadcast_to(b[None], (numeric.shape[0], b.shape[0], b.shape[1]))
        shape = b.shape
        numeric = numeric.reshape(numeric.shape[0] * numeric.shape[1])
        b = b.reshape(b.shape[0] * b.shape[1], b.shape[2])
        return prim.bind(symbolic, numeric, b).reshape(*shape), 0
    
    if ab is not None:
        # Only b is batched
        if b.ndim != 3:
            msg = f"b should be 3D when vectorizing over it. Got: {b.shape=}."
            raise ValueError(msg)
        b = jnp.moveaxis(b, ab, 2)
        shape = b.shape
        b = b.reshape(b.shape[0], b.shape[1] * b.shape[2])
        return prim.bind(symbolic, numeric, b).reshape(*shape), 2
    
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
        # ∂L/∂Ax[m,n] = Σₖ ct[m, Ai[n], k] · x[m, Aj[n], k]
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
            f"(n_col:={x.shape[0]},)"
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


# Differentiation rules for solve_with_symbol =================================

def solve_with_symbol_value_and_jvp(
    prim_solve: jax.extend.core.Primitive,
    prim_dot: jax.extend.core.Primitive,
    arg_values: Tuple[Array, Array, Array, Array, Array],
    arg_tangents: Tuple[Any, Any, Any, Any, Any],
) -> Tuple[Array, Array]:
    """Jacobian-vector product rule for `solve_with_symbol`.

    The rule for the JVP of a linear solve `x = A^-1 * b` is:
    `dx = A^-1 * (db - dA * x)`

    This function implements this rule efficiently. It first computes the primal
    solution `x`. Then it computes the right-hand side of the tangent system,
    `rhs = db - dA * x`. If `dA` is not zero, `dA * x` is computed using the
    `dot` primitive. If the `rhs` is not zero, `dx` is found by solving the
    system `A * dx = rhs`, reusing the symbolic factorization of A for
    efficiency.

    Args:
        prim_solve: The `solve_with_symbol` primitive.
        prim_dot: The corresponding `dot` primitive.
        arg_values: Primal values for `(Ai, Aj, Ax, b, symbolic)`.
        arg_tangents: Tangent values for `(Ai, Aj, Ax, b, symbolic)`.

    Returns:
        A tuple containing the primal output `x` and the tangent output `dx`.
    """
    Ai, Aj, Ax, b, symbolic = arg_values
    t_Ai, t_Aj, t_Ax, t_b, t_symbolic = arg_tangents

    x = prim_solve.bind(Ai, Aj, Ax, b, symbolic)

    rhs = t_b

    if not isinstance(t_Ax, ad.Zero):
        dAx = prim_dot.bind(Ai, Aj, t_Ax, x)
        if isinstance(rhs, ad.Zero):
            rhs = -dAx
        else:
            rhs = rhs - dAx

    if isinstance(rhs, ad.Zero):
        dx = lax.zeros_like_array(x)
    else:
        dx = prim_solve.bind(Ai, Aj, Ax, rhs, symbolic)

    return x, dx

# Register JVP for Float64
def solve_with_symbol_f64_value_and_jvp(
    arg_values: Tuple[Array, Array, Array, Array, Array],
    arg_tangents: Tuple[Any, Any, Any, Any, Any],
) -> Tuple[Array, Array]:
    """JVP rule for float64 solve_with_symbol."""
    return solve_with_symbol_value_and_jvp(
        solve_with_symbol_f64,
        dot_f64,
        arg_values,
        arg_tangents,
    )

ad.primitive_jvps[solve_with_symbol_f64] = solve_with_symbol_f64_value_and_jvp

# Register JVP for Complex128
def solve_with_symbol_c128_value_and_jvp(
    arg_values: Tuple[Array, Array, Array, Array, Array],
    arg_tangents: Tuple[Any, Any, Any, Any, Any],
) -> Tuple[Array, Array]:
    """JVP rule for complex128 solve_with_symbol."""
    return solve_with_symbol_value_and_jvp(
        solve_with_symbol_c128,
        dot_c128,
        arg_values,
        arg_tangents,
    )

ad.primitive_jvps[solve_with_symbol_c128] = solve_with_symbol_c128_value_and_jvp


def solve_with_symbol_transpose(
    solve_prim: jax.extend.core.Primitive,
    ct: Array,
    Ai: Array,
    Aj: Array,
    Ax: Array,
    b: Array,
    symbolic: Array,
) -> tuple[Array, Array, Array, Array, None]:
    """Compute the transpose of solve_with_symbol.

    Note:
        For the reverse pass (A.T), we cannot reuse the 'symbolic' handle because
        it describes A, not A.T. Therefore, we fallback to the standard solve_transpose
        logic which effectively treats it like a fresh solve.

    Args:
        solve_prim: [Primitive]: the primitive to transpose
        ct: [Array]: the cotangent vector
        Ai: [n_nz; int32]: the row indices of the sparse matrix A
        Aj: [n_nz; int32]: the column indices of the sparse matrix A
        Ax: [n_lhs? x n_nz; float64|complex128]: the values of the sparse matrix A
        b:  [n_lhs? x n_col x n_rhs?; float64|complex128]: the target vector
        symbolic: [Array]: the symbolic analysis handle

    Returns:
        tangents: tuple of tangents for (Ai, Aj, Ax, b, symbolic)

    """
    if solve_prim is solve_with_symbol_f64:
        base_solve = solve_f64
    else:
        base_solve = solve_c128

    t_Ai, t_Aj, t_Ax, t_b = solve_transpose(base_solve, ct, Ai, Aj, Ax, b)

    return t_Ai, t_Aj, t_Ax, t_b, None

def solve_with_symbol_f64_transpose(ct, Ai, Aj, Ax, b, symbolic):
    return solve_with_symbol_transpose(solve_with_symbol_f64, ct, Ai, Aj, Ax, b, symbolic)

ad.primitive_transposes[solve_with_symbol_f64] = solve_with_symbol_f64_transpose

def solve_with_symbol_c128_transpose(ct, Ai, Aj, Ax, b, symbolic):
    return solve_with_symbol_transpose(solve_with_symbol_c128, ct, Ai, Aj, Ax, b, symbolic)

ad.primitive_transposes[solve_with_symbol_c128] = solve_with_symbol_c128_transpose