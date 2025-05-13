# KLUJAX

> version: 0.4.3

A sparse linear solver for JAX based on the efficient [KLU algorithm](https://ufdcimages.uflib.ufl.edu/UF/E0/01/17/21/00001/palamadai_e.pdf).

## CPU & float64

This library is a wrapper around the [SuiteSparse](https://github.com/DrTimothyAldenDavis/SuiteSparse) KLU
algorithms. This means the algorithm is only implemented for
C-arrays and hence is **only available for CPU
arrays with double precision**, i.e. float64 or complex128.

Note that `float32`/`complex64` arrays will be cast to `float64`/`complex128`!

## Usage

The `klujax` library provides a single function `solve(Ai, Aj, Ax, b)`, which solves for `x` in
the sparse linear system `Ax=b`, where `A` is explicitly given in COO-format (`Ai`, `Aj`, `Ax`).

> NOTE: the sparse matrix represented by (`Ai`, `Aj`, `Ax`) needs to be [coalesced](https://pytorch.org/docs/stable/sparse.html#uncoalesced-sparse-coo-tensors)!
> KLUJAX provides a `coalesce` function (which unfortunately is not jax-jittable).

Supported shapes (`?` suffix means optional):

- `Ai`: `(n_nz,)`
- `Aj`: `(n_nz,)`
- `Ax`: `(n_lhs?, n_nz)`
- `b`: `(n_lhs?, n_col, n_rhs?)`
- `A` (represented by (`Ai`, `Aj`, `Ax`)): (`n_lhs?`, `n_col`, `n_col`)

KLUJAX will automatically select a sensible way to act on underdefined dimensions of Ax
and b:

| dim(Ax) | dim(b) | assumed shape(Ax) | assumed shape(b)      |
| ------- | ------ | ----------------- | --------------------- |
| 1D      | 1D     | n_nz              | n_col                 |
| 1D      | 2D     | n_nz              | n_col x n_rhs         |
| 1D      | 3D     | n_nz              | n_lhs x n_col x n_rhs |
| 2D      | 1D     | n_lhs x n_nz      | n_col                 |
| 2D      | 2D     | n_lhs x n_nz      | n_lhs x n_col         |
| 2D      | 3D     | n_lhs x n_nz      | n_lhs x n_col x n_rhs |

Where the `A` is always acting on the `n_col` dimension of `b`. The `n_lhs` dim is a
shared batch dimension between `A` and `b`.

Additional dimensions can be added with `jax.vmap` (alternatively any higher dimensional
problem can be reduced to the one above by properly transposing and reshaping `Ax` and `b`).

> NOTE: JAX now has an experimental sparse library (`jax.experimental.sparse`). Using
> this natively in KLUJAX is not yet supported (but converting from `BCOO` or `COO` to
> `Ai`, `Aj`, `Ax` is trivial).

## Basic Example

Script:

```python
import klujax
import jax.numpy as jnp

b = jnp.array([8, 45, -3, 3, 19])
A_dense = jnp.array(
    [
        [2, 3, 0, 0, 0],
        [3, 0, 4, 0, 6],
        [0, -1, -3, 2, 0],
        [0, 0, 1, 0, 0],
        [0, 4, 2, 0, 1],
    ]
)
Ai, Aj = jnp.where(jnp.abs(A_dense) > 0)
Ax = A_dense[Ai, Aj]

result_ref = jnp.linalg.inv(A_dense) @ b
result = klujax.solve(Ai, Aj, Ax, b)

print(jnp.abs(result - result_ref) < 1e-12)
print(result)
```

Output:

```
[ True True True True True]
[1. 2. 3. 4. 5.]
```

## Installation

The library is statically linked to the SuiteSparse C++ library. It can be installed on
most platforms as follows:

```bash
pip install klujax
```

**There exist pre-built wheels for Linux and Windows (python 3.8+).** If no compatible
wheel is found, however, pip will attempt to install the library from source... make
sure you have the necessary build dependencies installed (see [Installing from Source](#installing-from-source))

## Installing from Source

> NOTE: Installing from source should only be necessary when developing the library. If
> you as the user experience an install from source please create an issue.

Before installing, clone the build dependencies:

```sh
git clone --depth 1 --branch v7.2.0 https://github.com/DrTimothyAldenDavis/SuiteSparse suitesparse
git clone --depth 1 --branch main https://github.com/openxla/xla xla
git clone --depth 1 --branch stable https://github.com/pybind/pybind11 pybind11
```

### Linux

On linux, you'll need `gcc` and `g++`, then inside the repo:

```sh
pip install .
```

### MacOs

On MacOS, you'll need `clang`, then inside the repo:

```sh
pip install .
```

### Windows

On Windows, installing from source is a bit more involved as typically the build
dependencies are not installed. To install those, download Visual Studio Community 2017
from [here](https://my.visualstudio.com/Downloads?q=visual%20studio%202017&wt.mc_id=o~msft~vscom~older-downloads). During installation, go to Workloads and select the following workloads:

- Desktop development with C++
- Python development

Then go to Individual Components and select the following additional items:

- C++/CLI support
- VC++ 2015.3 v14.00 (v140) toolset for desktop

Then, download and install Microsoft Visual C++ Redistributable from [here](https://aka.ms/vs/16/release/vc_redist.x64.exe).

After these installation steps, run the following commands inside a x64 Native Tools
Command Prompt for VS 2017:

```cmd
set DISTUTILS_USE_SDK=1
pip install .
```

## License & Credits

© Floris Laporte 2022, LGPL-2.1

This library was partly based on:

- [torch_sparse_solve](https://github.com/flaport/torch_sparse_solve), LGPL-2.1
- [SuiteSparse](https://github.com/DrTimothyAldenDavis/SuiteSparse), LGPL-2.1
- [kagami-c/PyKLU](https://github.com/kagami-c/PyKLU), LGPL-2.1
- [scipy.sparse](https://github.com/scipy/scipy/tree/master/scipy/sparse), BSD-3

This library vendors an unmodified version of the
[SuiteSparse](https://github.com/DrTimothyAldenDavis/SuiteSparse) libraries in its source
(.tar.gz) distribution to allow for static linking.
This is in accordance with their
[LGPL licence](https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/LICENSE.txt).
