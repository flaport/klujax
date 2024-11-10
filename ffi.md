---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{code-cell} ipython3
%env JAX_ENABLE_X64=1
```

## Imports

```{code-cell} ipython3
from pathlib import Path

import jax
import jax.extend as jex
import jax.numpy as jnp
import jax.scipy as jsp
import klujax_cpp
import numpy as np
```

## solve_f64

```{code-cell} ipython3
jex.ffi.register_ffi_target(
    "solve_f64",
    klujax_cpp.solve_f64(),
    platform="cpu",
)
```

```{code-cell} ipython3
def solve_f64(Ai, Aj, Ax, b):
    *_, n_col, n_rhs = b.shape
    _b = b.reshape(-1, n_col, n_rhs)
    
    *ns_lhs, n_nz = Ax.shape
    _Ax = Ax.reshape(-1, n_nz)
    n_lhs, _ = _Ax.shape
    
    call = jex.ffi.ffi_call(
        "solve_f64",
        jax.ShapeDtypeStruct(_b.shape, _b.dtype),
        vmap_method="broadcast_all",
    )
    b = call(
        Ai,
        Aj,
        _Ax,
        _b,
        n_col=np.int32(n_col),
        n_rhs=np.int32(n_rhs),
        n_lhs=np.int32(n_lhs),
        n_nz=np.int32(n_nz),
    )
    return b.reshape(*ns_lhs, n_col, n_rhs)
```

### Test Single

```{code-cell} ipython3
n_nz = 8
n_col = 5
n_rhs = 1
Axkey, Aikey, Ajkey, bkey = jax.random.split(jax.random.PRNGKey(33), 4)
Ax = jax.random.normal(Axkey, (n_nz,))
Ai = jax.random.randint(Aikey, (n_nz,), 0, n_col, jnp.int32)
Aj = jax.random.randint(Ajkey, (n_nz,), 0, n_col, jnp.int32)
b = jax.random.normal(bkey, (n_col, n_rhs))
x_sp = solve_f64(Ai, Aj, Ax, b)

A = jnp.zeros((n_col, n_col), dtype=jnp.float64).at[Ai, Aj].add(Ax)
x = jsp.linalg.solve(A, b)

x_sp - x
```

### Test Batched

```{code-cell} ipython3
n_nz = 8
n_col = 5
n_rhs = 1
n_lhs = 3

Axkey, Aikey, Ajkey, bkey = jax.random.split(jax.random.PRNGKey(33), 4)
Ax = jax.random.normal(Axkey, (n_lhs, n_nz))
Ai = jax.random.randint(Aikey, (n_nz,), 0, n_col, jnp.int32)
Aj = jax.random.randint(Ajkey, (n_nz,), 0, n_col, jnp.int32)
b = jax.random.normal(bkey, (n_lhs, n_col, n_rhs))
x_sp = solve_f64(Ai, Aj, Ax, b)

A = jnp.zeros((n_lhs, n_col, n_col), dtype=jnp.complex128).at[:, Ai, Aj].add(Ax)
x = jax.vmap(jsp.linalg.solve, (0, 0), 0)(A, b)

x_sp - x
```

## coo_mul_vec_f64

```{code-cell} ipython3
jex.ffi.register_ffi_target(
    "coo_mul_vec_f64",
    klujax_cpp.coo_mul_vec_f64(),
    platform="cpu",
)
```

```{code-cell} ipython3
def coo_mul_vec_f64(Ai, Aj, Ax, x):
    *_, n_col, n_rhs = x.shape
    _x = x.reshape(-1, n_col, n_rhs)
    
    *ns_lhs, n_nz = Ax.shape
    _Ax = Ax.reshape(-1, n_nz)
    n_lhs, _ = _Ax.shape

    call = jex.ffi.ffi_call(
        "coo_mul_vec_f64",
        jax.ShapeDtypeStruct(_x.shape, _x.dtype),
        vmap_method="broadcast_all",
    )
    b = call(
        Ai,
        Aj,
        _Ax,
        _x,
        n_col=np.int32(n_col),
        n_rhs=np.int32(n_rhs),
        n_lhs=np.int32(n_lhs),
        n_nz=np.int32(n_nz),
    )
    return b.reshape(*ns_lhs, n_col, n_rhs)
```

### Test Single

```{code-cell} ipython3
n_nz = 8
n_col = 5
n_rhs = 1
Axkey, Aikey, Ajkey, bkey = jax.random.split(jax.random.PRNGKey(33), 4)
Ax = jax.random.normal(Axkey, (n_nz,))
Ai = jax.random.randint(Aikey, (n_nz,), 0, n_col, jnp.int32)
Aj = jax.random.randint(Ajkey, (n_nz,), 0, n_col, jnp.int32)
x = jax.random.normal(bkey, (n_col, n_rhs))
b_sp = coo_mul_vec_f64(Ai, Aj, Ax, x)

A = jnp.zeros((n_col, n_col), dtype=jnp.float64).at[Ai, Aj].add(Ax)
b = A @ x

b_sp - b
```

### Test Batched

```{code-cell} ipython3
n_nz = 8
n_col = 5
n_rhs = 1
n_lhs = 3
Axkey, Aikey, Ajkey, bkey = jax.random.split(jax.random.PRNGKey(33), 4)
Ax = jax.random.normal(Axkey, (n_lhs, n_nz))
Ai = jax.random.randint(Aikey, (n_nz,), 0, n_col, jnp.int32)
Aj = jax.random.randint(Ajkey, (n_nz,), 0, n_col, jnp.int32)
x = jax.random.normal(bkey, (n_lhs, n_col, n_rhs))

b_sp = coo_mul_vec_f64(Ai, Aj, Ax, x)

A = jnp.zeros((n_lhs, n_col, n_col), dtype=jnp.complex128).at[:, Ai, Aj].add(Ax)
b = jnp.einsum("bij,bjk->bik", A, x)
b_sp - b
```

## solve_c128

```{code-cell} ipython3
jex.ffi.register_ffi_target(
    "solve_c128",
    klujax_cpp.solve_c128(),
    platform="cpu",
)
```

```{code-cell} ipython3
def solve_c128(Ai, Aj, Ax, b):
    *_, n_col, n_rhs = b.shape
    _b = b.reshape(-1, n_col, n_rhs)
    
    *ns_lhs, n_nz = Ax.shape
    _Ax = Ax.reshape(-1, n_nz)
    n_lhs, _ = _Ax.shape

    _Ax = _Ax.view(np.float64)
    _b = _b.view(np.float64)
    
    call = jex.ffi.ffi_call(
        "solve_c128",
        jax.ShapeDtypeStruct(_b.shape, _b.dtype),
        vmap_method="broadcast_all",
    )
    x = call(
        Ai,
        Aj,
        _Ax,
        _b,
        n_col=np.int32(n_col),
        n_rhs=np.int32(n_rhs),
        n_lhs=np.int32(n_lhs),
        n_nz=np.int32(n_nz),
    )
    return x.view(b.dtype).reshape(*ns_lhs, n_col, n_rhs)
```

### Test Single

```{code-cell} ipython3
n_nz = 8
n_col = 5
n_rhs = 1
Axkey, Aikey, Ajkey, bkey = jax.random.split(jax.random.PRNGKey(33), 4)
Ax_r, Ax_i = jax.random.normal(Axkey, (2, n_nz))
Ax = Ax_r + 1j * Ax_i
Ai = jax.random.randint(Aikey, (n_nz,), 0, n_col, jnp.int32)
Aj = jax.random.randint(Ajkey, (n_nz,), 0, n_col, jnp.int32)
b_r, b_i = jax.random.normal(bkey, (2, n_col, n_rhs))
b = b_r + 1j * b_i
x_sp = solve_c128(Ai, Aj, Ax, b)

A = jnp.zeros((n_col, n_col), dtype=jnp.complex128).at[Ai, Aj].add(Ax)
x = jsp.linalg.solve(A, b)

x_sp - x
```

### Test Batched

```{code-cell} ipython3
n_nz = 8
n_col = 5
n_rhs = 1
n_lhs = 3

Axkey, Aikey, Ajkey, bkey = jax.random.split(jax.random.PRNGKey(33), 4)
Ax = jax.random.normal(Axkey, (n_lhs, n_nz), dtype=jnp.complex128)
Ai = jax.random.randint(Aikey, (n_nz,), 0, n_col, jnp.int32)
Aj = jax.random.randint(Ajkey, (n_nz,), 0, n_col, jnp.int32)
b = jax.random.normal(bkey, (n_lhs, n_col, n_rhs), dtype=jnp.complex128)
x_sp = solve_c128(Ai, Aj, Ax, b)

A = jnp.zeros((n_lhs, n_col, n_col), dtype=jnp.complex128).at[:, Ai, Aj].add(Ax)
x = jax.vmap(jsp.linalg.solve, (0, 0), 0)(A, b)

x_sp - x
```

## coo_mul_vec_c128

```{code-cell} ipython3
jex.ffi.register_ffi_target(
    "coo_mul_vec_c128",
    klujax_cpp.coo_mul_vec_c128(),
    platform="cpu",
)
```

```{code-cell} ipython3
def coo_mul_vec_c128(Ai, Aj, Ax, x):
    *_, n_col, n_rhs = x.shape
    _x = x.reshape(-1, n_col, n_rhs)
    
    *ns_lhs, n_nz = Ax.shape
    _Ax = Ax.reshape(-1, n_nz)
    n_lhs, _ = _Ax.shape

    _Ax = _Ax.view(np.float64)
    _x = _x.view(np.float64)
    call = jex.ffi.ffi_call(
        "coo_mul_vec_c128",
        jax.ShapeDtypeStruct(_x.shape, _x.dtype),
        vmap_method="broadcast_all",
    )
    y = call(
        Ai,
        Aj,
        _Ax,
        _x,
        n_col=np.int32(n_col),
        n_rhs=np.int32(n_rhs),
        n_lhs=np.int32(n_lhs),
        n_nz=np.int32(n_nz),
    )
    return y.view(x.dtype).reshape(*ns_lhs, n_col, n_rhs)
```

### Test Single

```{code-cell} ipython3
n_nz = 8
n_col = 5
n_rhs = 1
Axkey, Aikey, Ajkey, bkey = jax.random.split(jax.random.PRNGKey(33), 4)
Ax = jax.random.normal(Axkey, (n_nz,), dtype=jnp.complex128)
Ai = jax.random.randint(Aikey, (n_nz,), 0, n_col, jnp.int32)
Aj = jax.random.randint(Ajkey, (n_nz,), 0, n_col, jnp.int32)
x = jax.random.normal(bkey, (n_col, n_rhs), dtype=jnp.complex128)

b_sp = coo_mul_vec_c128(Ai, Aj, Ax, x)

A = jnp.zeros((n_col, n_col), dtype=jnp.complex128).at[Ai, Aj].add(Ax)
b = A @ x

b_sp - b
```

### Test Batched

```{code-cell} ipython3
n_nz = 8
n_col = 5
n_rhs = 1
n_lhs = 3
Axkey, Aikey, Ajkey, bkey = jax.random.split(jax.random.PRNGKey(33), 4)
Ax = jax.random.normal(Axkey, (n_lhs, n_nz), dtype=jnp.complex128)
Ai = jax.random.randint(Aikey, (n_nz,), 0, n_col, jnp.int32)
Aj = jax.random.randint(Ajkey, (n_nz,), 0, n_col, jnp.int32)
x = jax.random.normal(bkey, (n_lhs, n_col, n_rhs), dtype=jnp.complex128)

b_sp = coo_mul_vec_c128(Ai, Aj, Ax, x)

A = jnp.zeros((n_lhs, n_col, n_col), dtype=jnp.complex128).at[:, Ai, Aj].add(Ax)
b = jnp.einsum("bij,bjk->bik", A, x)
b_sp - b
```

```{code-cell} ipython3
np.atleast_2d([0, 0], ).shape
```
