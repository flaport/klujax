---
title: Performance Guide
summary: Choose the right solve strategy for your use case
---

# Performance Guide

klujax gives you control over which parts of the solve to run and when. Choosing the right strategy can make your code orders of magnitude faster.

## The Three Stages

Every sparse solve has three stages with very different costs:

```mermaid
flowchart LR
    A["Analyze\n#40;pattern#41;"]:::expensive --> F["Factor\n#40;LU decomp#41;"]:::moderate --> S["Solve\n#40;substitution#41;"]:::cheap

    classDef expensive fill:#ef4444,color:#fff,stroke:none
    classDef moderate fill:#f59e0b,color:#fff,stroke:none
    classDef cheap fill:#10b981,color:#fff,stroke:none
```

| Stage       | What It Does                                 | Cost      | Depends On          |
| ----------- | -------------------------------------------- | --------- | ------------------- |
| **Analyze** | Find optimal orderings from sparsity pattern | Expensive | Ai, Aj              |
| **Factor**  | LU decomposition (A = LU)                    | Moderate  | Ax (values)         |
| **Solve**   | Forward/backward substitution                | Cheap     | b (right-hand side) |

## Decision Tree

```mermaid
flowchart TD
    START["How many solves?"]
    START -->|"Just one"| ONE["Use klujax.solve\n#40;simplest#41;"]
    START -->|"Many"| MANY["Does the sparsity\npattern change?"]

    MANY -->|"Yes, every time"| SOLVE["Use klujax.solve\n#40;no shortcut possible#41;"]
    MANY -->|"No, pattern is fixed"| PATTERN["Do matrix values\nchange?"]

    PATTERN -->|"Yes, every solve"| SWS["analyze once →\nsolve_with_symbol\neach step"]
    PATTERN -->|"Occasionally"| RF["analyze once →\nfactor once →\nrefactor when needed →\nsolve_with_numeric"]
    PATTERN -->|"No, matrix is fixed"| SWN["analyze once →\nfactor once →\nsolve_with_numeric\n#40;fastest#41;"]

    style ONE fill:#94a3b8,color:#fff,stroke:none
    style SOLVE fill:#ef4444,color:#fff,stroke:none
    style SWS fill:#f59e0b,color:#fff,stroke:none
    style RF fill:#6366f1,color:#fff,stroke:none
    style SWN fill:#10b981,color:#fff,stroke:none
```

## Strategy 1: All-in-One (klujax.solve)

**When to use:** One-off solves, prototyping, or when the sparsity pattern changes every time.

```python
x = klujax.solve(Ai, Aj, Ax, b)
```

Runs: analyze → factor → solve. Every call.

## Strategy 2: Reuse Symbolic (analyze + solve_with_symbol)

**When to use:** The sparsity pattern is constant, but values and right-hand side change every iteration.

**Typical use case:** Transient simulations, time-stepping.

```python
symbolic = klujax.analyze(Ai, Aj, n_col)  # once

for t in range(steps):
    x = klujax.solve_with_symbol(Ai, Aj, Ax[t], b[t], symbolic)
```

Runs per step: factor → solve. Skips analyze.

## Strategy 3: Reuse Numeric (analyze + factor + solve_with_numeric)

**When to use:** The matrix is constant and only b changes.

**Typical use case:** Modified Newton-Raphson (reuse Jacobian), multiple load cases.

```python
symbolic = klujax.analyze(Ai, Aj, n_col)  # once
numeric = klujax.factor(Ai, Aj, Ax, symbolic)  # once

for i in range(iterations):
    x = klujax.solve_with_numeric(numeric, b[i], symbolic)
```

Runs per step: solve only. Skips analyze and factor.

## Strategy 4: Refactor (analyze + factor + refactor loop)

**When to use:** Matrix values change, but you want to update the factorization in-place rather than creating a new one.

**Typical use case:** Full Newton-Raphson with Jacobian updates.

```python
symbolic = klujax.analyze(Ai, Aj, n_col)
numeric = klujax.factor(Ai, Aj, Ax_initial, symbolic)

for t in range(steps):
    numeric = klujax.refactor(Ai, Aj, Ax[t], numeric, symbolic)
    x = klujax.solve_with_numeric(numeric, b[t], symbolic)
```

Runs per step: refactor → solve. Refactor is slightly faster than factor because it reuses allocated memory.

## Strategy Comparison

```mermaid
flowchart LR
    subgraph "Strategy 1: solve"
        direction LR
        S1A["Analyze"] --> S1F["Factor"] --> S1S["Solve"]
    end
    subgraph "Strategy 2: solve_with_symbol"
        direction LR
        S2F["Factor"] --> S2S["Solve"]
    end
    subgraph "Strategy 3: solve_with_numeric"
        direction LR
        S3S["Solve"]
    end
    subgraph "Strategy 4: refactor + solve_with_numeric"
        direction LR
        S4R["Refactor"] --> S4S["Solve"]
    end

    style S1A fill:#ef4444,color:#fff,stroke:none
    style S1F fill:#f59e0b,color:#fff,stroke:none
    style S1S fill:#10b981,color:#fff,stroke:none
    style S2F fill:#f59e0b,color:#fff,stroke:none
    style S2S fill:#10b981,color:#fff,stroke:none
    style S3S fill:#10b981,color:#fff,stroke:none
    style S4R fill:#f59e0b,color:#fff,stroke:none
    style S4S fill:#10b981,color:#fff,stroke:none
```

## Batching for Throughput

When solving many independent systems, use batching instead of loops:

```python
# BAD: Python loop
for i in range(100):
    x[i] = klujax.solve(Ai, Aj, Ax[i], b[i])

# GOOD: Batched (single kernel launch)
x = klujax.solve(Ai, Aj, Ax_batch, b_batch)

# ALSO GOOD: vmap (single kernel launch)
x = jax.vmap(klujax.solve, in_axes=(None, None, 0, 0))(Ai, Aj, Ax_batch, b_batch)
```

Batched operations launch a single optimized kernel instead of 100 separate ones.

## Tips

1. **Profile first.** Don't optimize until you know where time is spent.
2. **Coalesce early.** Call `klujax.coalesce` once during setup, not in your loop.
3. **Keep handles alive.** Don't let `symbolic` or `numeric` go out of scope if you're still using them.
4. **Use float64.** Lower precision gets auto-cast to float64 anyway — save the conversion cost by using float64 from the start.
5. **Batch when possible.** A batched solve is much faster than a loop of individual solves.
