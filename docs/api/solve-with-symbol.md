---
title: solve_with_symbol
summary: Solve using a pre-computed symbolic analysis
---

# solve_with_symbol

```python
klujax.solve_with_symbol(Ai, Aj, Ax, b, symbolic) -> Array
```

Solve **Ax = b** using a pre-computed symbolic analysis. This skips the expensive analyze step and only performs factorization + solve. Use this when the sparsity pattern is constant but the values and right-hand side change.

## Parameters

| Parameter  | Type                  | Shape                     | Description                       |
| ---------- | --------------------- | ------------------------- | --------------------------------- |
| `Ai`       | int32                 | `(n_nz,)`                 | Row indices                       |
| `Aj`       | int32                 | `(n_nz,)`                 | Column indices                    |
| `Ax`       | float64 or complex128 | `(n_lhs?, n_nz)`          | Matrix values                     |
| `b`        | float64 or complex128 | `(n_lhs?, n_col, n_rhs?)` | Right-hand side                   |
| `symbolic` | KLUHandleManager      | —                         | Handle from [analyze](analyze.md) |

## Returns

| Type  | Shape             | Description        |
| ----- | ----------------- | ------------------ |
| Array | Same shape as `b` | The solution **x** |

## How It Fits In

```mermaid
flowchart LR
    subgraph "Once (outside loop)"
        AN["analyze#40;Ai, Aj, n_col#41;"] --> SYM["symbolic"]
    end
    subgraph "Every iteration"
        SYM --> SWS["solve_with_symbol\n#40;Ai, Aj, Ax_t, b_t, symbolic#41;"]:::active
        AX["Ax_t"] --> SWS
        B["b_t"] --> SWS
        SWS --> X["x_t"]
    end

    classDef active fill:#10b981,color:#fff,stroke:none
```

## Example

```python
import jax
import klujax
import jax.numpy as jnp

Ai = jnp.array([0, 0, 1, 1, 2, 2], dtype=jnp.int32)
Aj = jnp.array([0, 1, 0, 1, 1, 2], dtype=jnp.int32)
n_col = 3

# Expensive analysis — done once
symbolic = klujax.analyze(Ai, Aj, n_col)

# Fast JIT-compiled solve — done many times
@jax.jit
def step(Ax, b, sym):
    return klujax.solve_with_symbol(Ai, Aj, Ax, b, sym)

for t in range(1000):
    x_t = step(Ax_values[t], b_values[t], symbolic)
```

## Performance Comparison

```mermaid
flowchart LR
    subgraph "solve (repeats everything)"
        direction LR
        A1["Analyze"] --> F1["Factor"] --> S1["Solve"]
    end
    subgraph "solve_with_symbol (skips analyze)"
        direction LR
        F2["Factor"] --> S2["Solve"]
    end

    style A1 fill:#ef4444,color:#fff,stroke:none
    style F1 fill:#6366f1,color:#fff,stroke:none
    style S1 fill:#10b981,color:#fff,stroke:none
    style F2 fill:#6366f1,color:#fff,stroke:none
    style S2 fill:#10b981,color:#fff,stroke:none
```

The analyze step is typically the most expensive part. Skipping it can give substantial speedups when solving many systems with the same sparsity pattern.

## JAX Features

| Feature      | Supported                 |
| ------------ | ------------------------- |
| `jax.jit`    | Yes                       |
| `jax.grad`   | Yes (w.r.t. `Ax` and `b`) |
| `jax.jacfwd` | Yes                       |
| `jax.vmap`   | Yes                       |
