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

The library is dynamically linked to the SuiteSparse C++ library. The easiest way to
install is as follows:

```bash
conda install pybind11 suitesparse
pip install klujax
```

**There exist pre-built wheels for Linux and Windows (python 3.8+).** If no compatible
wheel is found, however, pip will attempt to install the library from source... make
sure you have the necessary build dependencies installed.

### Linux

On linux, having `gcc` and `g++` available in your path should be sufficient to be able
to build the library from source.

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
Command Prompt for VS 2017, after activating your conda environment:

```
set DISTUTILS_USE_SDK=1
conda install pybind11 suitesparse
pip install klujax
```

## License & Credits

© Floris Laporte 2022, LGPL-2.1

This library was partly based on:

- [torch_sparse_solve](https://github.com/flaport/torch_sparse_solve), LGPL-2.1
- [SuiteSparse](https://github.com/DrTimothyAldenDavis/SuiteSparse), LGPL-2.1
- [kagami-c/PyKLU](https://github.com/kagami-c/PyKLU), LGPL-2.1
- [scipy.sparse](https://github.com/scipy/scipy/tree/master/scipy/sparse), BSD-3
