// version: 0.3.1
// Imports

// include "klu.h"
#include <cmath>

#include "pybind11/pybind11.h"
#include "xla/ffi/api/ffi.h"
namespace py = pybind11;
namespace ffi = xla::ffi;

#include <cstring>  // for memset

ffi::Error validate_dot_f64_args(
    ffi::Buffer<ffi::DataType::S32> &Ai,
    ffi::Buffer<ffi::DataType::S32> &Aj,
    const ffi::AnyBuffer::Dimensions ds_Ax,
    const ffi::AnyBuffer::Dimensions ds_x) {
    int d_x = ds_x.size();
    if (d_x != 3) {
        return ffi::Error::InvalidArgument("x is not 3D.");
    }
    int n_lhs = (int)ds_x[0];
    int n_col = (int)ds_x[1];

    int d_Ax = ds_Ax.size();
    if (d_Ax != 2) {
        return ffi::Error::InvalidArgument("Ax is not 2D.");
    }
    int n_lhs_bis = (int)ds_Ax[0];
    int n_nz = (int)ds_Ax[1];

    if (n_lhs != n_lhs_bis) {
        return ffi::Error::InvalidArgument(
            "n_lhs mismatch: Ax.shape[0] != x.shape[0]: Got " + std::to_string(n_lhs_bis) + " != " + std::to_string(n_lhs));
    }

    auto ds_Ai = Ai.dimensions();
    int d_Ai = ds_Ai.size();
    if (d_Ai != 1) {
        return ffi::Error::InvalidArgument("Ai is not 1D.");
    }
    int n_nz_bis = (int)ds_Ai[0];
    if (n_nz != n_nz_bis) {
        return ffi::Error::InvalidArgument(
            "n_lhs mismatch: Ai.shape[0] != Ax.shape[1]: Got " + std::to_string(n_nz_bis) + " != " + std::to_string(n_nz));
    }

    auto ds_Aj = Aj.dimensions();
    int d_Aj = ds_Aj.size();
    if (d_Aj != 1) {
        return ffi::Error::InvalidArgument("Aj is not 1D.");
    }
    n_nz_bis = (int)ds_Aj[0];
    if (n_nz != n_nz_bis) {
        return ffi::Error::InvalidArgument(
            "n_lhs mismatch: Aj.shape[0] != Ax.shape[1]: Got " + std::to_string(n_nz_bis) + " != " + std::to_string(n_nz));
    }

    int i;
    int j;
    const int *_Ai = Ai.typed_data();
    const int *_Aj = Aj.typed_data();
    for (int n = 0; n < n_nz; n++) {
        i = _Ai[n];
        if (i >= n_col) {
            return ffi::Error::InvalidArgument("Ai.max() >= n_col");
        }
        j = _Aj[n];
        if (j >= n_col) {
            return ffi::Error::InvalidArgument("Aj.max() >= n_col");
        }
    }
    return ffi::Error::Success();
}

ffi::Error _dot_f64(
    ffi::Buffer<ffi::DataType::S32> Ai,
    ffi::Buffer<ffi::DataType::S32> Aj,
    ffi::Buffer<ffi::DataType::F64> Ax,
    ffi::Buffer<ffi::DataType::F64> x,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> b) {
    auto ds_x = x.dimensions();
    auto ds_Ax = Ax.dimensions();
    ffi::Error err = validate_dot_f64_args(Ai, Aj, ds_Ax, ds_x);
    if (err.failure()) {
        return err;
    }

    int n_lhs = (int)ds_x[0];
    int n_col = (int)ds_x[1];
    int n_rhs = (int)ds_x[2];
    int n_nz = (int)ds_Ax[1];
    const int *_Ai = Ai.typed_data();
    const int *_Aj = Aj.typed_data();
    const double *_Ax = Ax.typed_data();
    const double *_x = x.typed_data();
    double *_b = b->typed_data();

    // initialize empty result
    for (int i = 0; i < n_lhs * n_col * n_rhs; i++) {
        _b[i] = 0.0;
    }

    // fill result (all multi-dim arrays are row-major)
    // x_mik = A_mij × x_mjk (einsum)
    // sizes: m<n_lhs; i<n_col<--Ai; j<n_col<--Aj; k<n_rhs
    int i;
    int j;
    for (int n = 0; n < n_nz; n++) {
        i = _Ai[n];
        j = _Aj[n];
        for (int m = 0; m < n_lhs; m++) {
            for (int k = 0; k < n_rhs; k++) {
                _b[m * n_col * n_rhs + i * n_rhs + k] += _Ax[m * n_nz + n] * _x[m * n_col * n_rhs + j * n_rhs + k];
            }
        }
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(  // b = A x
    dot_f64, _dot_f64,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // Ai
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // Aj
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // Ax
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // x
        .Ret<ffi::Buffer<ffi::DataType::F64>>()  // b
);

ffi::Error _dot_c128(
    ffi::Buffer<ffi::DataType::S32> Ai,
    ffi::Buffer<ffi::DataType::S32> Aj,
    ffi::Buffer<ffi::DataType::C128> Ax,
    ffi::Buffer<ffi::DataType::C128> x,
    ffi::Result<ffi::Buffer<ffi::DataType::C128>> b) {
    auto ds_x = x.dimensions();
    auto ds_Ax = Ax.dimensions();
    ffi::Error err = validate_dot_f64_args(Ai, Aj, ds_Ax, ds_x);
    if (err.failure()) {
        return err;
    }
    int n_lhs = (int)ds_x[0];
    int n_col = (int)ds_x[1];
    int n_rhs = (int)ds_x[2];
    int n_nz = (int)ds_Ax[1];
    const int *_Ai = Ai.typed_data();
    const int *_Aj = Aj.typed_data();
    const double *_Ax = (double *)Ax.typed_data();
    const double *_x = (double *)x.typed_data();
    double *_b = (double *)b->typed_data();

    // initialize empty result
    for (int i = 0; i < 2 * n_lhs * n_col * n_rhs; i++) {
        _b[i] = 0.0;
    }

    // fill result (all multi-dim arrays are row-major)
    // x_mik = A_mij × x_mjk (einsum)
    // sizes: m<n_lhs; i<n_col<--Ai; j<n_col<--Aj; k<n_rhs
    int i;
    int j;
    for (int n = 0; n < n_nz; n++) {
        i = _Ai[n];
        j = _Aj[n];
        for (int m = 0; m < n_lhs; m++) {
            for (int k = 0; k < n_rhs; k++) {
                _b[2 * (m * n_col * n_rhs + i * n_rhs + k)] +=                                        // real
                    _Ax[2 * (m * n_nz + n)] * _x[2 * (m * n_col * n_rhs + j * n_rhs + k)]             // real*real
                    - _Ax[2 * (m * n_nz + n) + 1] * _x[2 * (m * n_col * n_rhs + j * n_rhs + k) + 1];  // imag*imag
                _b[2 * (m * n_col * n_rhs + i * n_rhs + k) + 1] +=                                    // imag
                    _Ax[2 * (m * n_nz + n)] * _x[2 * (m * n_col * n_rhs + j * n_rhs + k) + 1]         // real*imag
                    + _Ax[2 * (m * n_nz + n) + 1] * _x[2 * (m * n_col * n_rhs + j * n_rhs + k)];      // imag*real
            }
        }
    }

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(  // b = A x
    dot_c128, _dot_c128,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // Ai
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // Aj
        .Arg<ffi::Buffer<ffi::DataType::C128>>()  // Ax
        .Arg<ffi::Buffer<ffi::DataType::C128>>()  // x
        .Ret<ffi::Buffer<ffi::DataType::C128>>()  // b
);

// Python wrappers
PYBIND11_MODULE(klujax_cpp, m) {
    m.def("dot_f64",
          []() { return py::capsule((void *)&dot_f64); });
    m.def("dot_c128",
          []() { return py::capsule((void *)&dot_c128); });
}
