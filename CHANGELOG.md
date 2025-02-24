# Changelog

## 0.4.0

- Upgrade to jax>=0.5.0
- Fix issues in vmap
- Fix issues in jacfwd/jacrev
- No more hidden segfaults (hopefully)
- More consistent array shape broadcasting
- Drop support for x86 MacOS (not supported by jax>=0.5.0 either)

## 0.3.1

- Bugfixes

## 0.3.0

- Implement new FFI API for C++ extension
- Enhance C++ testing and error handling (proper error throwing instead of segfaulting)
- Enable JIT and `vmap` for optimized performance
- Improve shape handling for arrays
- Retrieve array sizes from C++ buffers
- Refactor workflows and remove deprecated implementations
- Remove old notebook files
- Upgrade dependencies and ensure compatibility with C++17 standard

## 0.2.10

- Run tests post wheel build
- Pin exact dependency versions
- Streamline GitHub workflows and update `setup.py`
- Fix issues with `vmap` over array `b`

## 0.2.8

- Refine CI/CD environment variable configuration for `cibuildwheel`

## 0.2.7

- Prevent memory leaks
- Introduce pre-commit configuration
- Update `cibuildwheel` configuration
- Clone specific SuiteSparse version

## 0.2.5

- Address deprecations in XLA translations

## 0.2.4

- Add support for Python 3.12
- Consolidate GitHub workflow files and update package metadata

## 0.2.0

- Vendor SuiteSparse library in source distribution
- Re-enable `PIP_FIND_LINKS`
- Update build recipes and dependencies
- Improve README with setup and build instructions

## 0.1.4

- Add support for Python 3.11

## 0.1.3

- Enable installation on macOS
- Fix issues with static linking on macOS (C++11 requirement)

## 0.1.1

- Publish release on PyPI and include tarball for distribution
- Add support for multiple Python versions and manylinux2014 wheels

## 0.1.0

- Enable custom JVP/VJP rules
- Improve differentiation features with forward-mode JVP and transposition
- Add `pyproject.toml` for better build configuration

## 0.0.6

- Add more library/include paths for builds
- Refine README and setup instructions
- Initial integration of complex value handling in `vmap`

## 0.0.4

- Add `bump2version` configuration for automated versioning
- Bugfix: Correct matrix-vector multiplication in COO format (`mul_coo_vec`)

## 0.0.3

- Set up core functionality with sparse matrix multiplication (COO format)
- Integrate `vmap` for float64 and complex128 arrays
- Initial setup with Makefile, test suites, and Docker configuration
- Begin development of XLA translations and gradient support
