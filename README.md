# KLUJAX

A sparse linear solver for JAX based on the
efficient [KLU algorithm](https://ufdcimages.uflib.ufl.edu/UF/E0/01/17/21/00001/palamadai_e.pdf).

## CPU & float64

This library is a wrapper around the [SuiteSparse](https://github.com/DrTimothyAldenDavis/SuiteSparse) KLU
algorithms. This means the algorithm is only implemented for
C-arrays and hence is **only available for CPU
arrays with double precision**, i.e. float64 or complex128.

Note that this will be enforced at import of `klujax`!

## Usage

The `klujax` library provides a single function `solve(A, b)`, which solves for `x` in
the linear system `Ax=b` A is a sparse tensor in COO-format with shape `mxm` and x and b
have shape `mxn`. Note that JAX does not have a native sparse matrix representation and
hence A should be represented as a tuple of two index arrays and a value
array: `(Ai, Aj, Ax)`.

```python
import jax.numpy as jnp
from klujax import solve

b = jnp.array([8, 45, -3, 3, 19], dtype=jnp.float64)
A_dense = jnp.array([[2, 3, 0, 0, 0],
                     [3, 0, 4, 0, 6],
                     [0, -1, -3, 2, 0],
                     [0, 0, 1, 0, 0],
                     [0, 4, 2, 0, 1]], dtype=jnp.float64)
Ai, Aj = jnp.where(jnp.abs(A_dense) > 0)
Ax = A_dense[Ai, Aj]

result_ref = jnp.linalg.inv(A_dense)@b
result = solve(Ai, Aj, Ax, b)

print(jnp.abs(result - result_ref) < 1e-12)
print(result)
```

```
[ True True True True True]
[1. 2. 3. 4. 5.]
```

## Installation

The library can be installed with `pip`:

```bash
pip install klujax
```

Please note that no pre-built wheels exist. This means that `pip` will
attempt to install the library from source. Make sure you have the
necessary (build-)dependencies installed.

```bash
conda install suitesparse pybind11
pip install jax
pip install torch_sparse_solve
```

## License & Credits

© Floris Laporte 2022, LGPL-2.1

This library was partly based on:

- [torch_sparse_solve](https://github.com/flaport/torch_sparse_solve), LGPL-2.1
- [SuiteSparse](https://github.com/DrTimothyAldenDavis/SuiteSparse), LGPL-2.1
- [kagami-c/PyKLU](https://github.com/kagami-c/PyKLU), LGPL-2.1
- [scipy.sparse](https://github.com/scipy/scipy/tree/master/scipy/sparse), BSD-3
