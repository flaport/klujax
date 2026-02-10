// version: 0.4.8
// Imports

#include <cmath>

#include <complex>
#include <cstdint>
#include "klu.h"
#include "pybind11/pybind11.h"
#include "xla/ffi/api/ffi.h"
namespace py = pybind11;
namespace ffi = xla::ffi;

#include <cstring>  // for memset
#include <memory>   // for unique_ptr

ffi::Error validate_args(
    const ffi::Buffer<ffi::DataType::S32>& Ai,
    const ffi::Buffer<ffi::DataType::S32>& Aj,
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
            "n_nz mismatch: Ai.shape[0] != Ax.shape[1]: Got " + std::to_string(n_nz_bis) + " != " + std::to_string(n_nz));
    }

    auto ds_Aj = Aj.dimensions();
    int d_Aj = ds_Aj.size();
    if (d_Aj != 1) {
        return ffi::Error::InvalidArgument("Aj is not 1D.");
    }
    n_nz_bis = (int)ds_Aj[0];
    if (n_nz != n_nz_bis) {
        return ffi::Error::InvalidArgument(
            "n_nz mismatch: Aj.shape[0] != Ax.shape[1]: Got " + std::to_string(n_nz_bis) + " != " + std::to_string(n_nz));
    }

    int i;
    int j;
    const int* _Ai = Ai.typed_data();
    const int* _Aj = Aj.typed_data();
    for (int n = 0; n < n_nz; n++) {
        i = _Ai[n];
        if (i < 0) {
            return ffi::Error::InvalidArgument("Ai contains negative index");
        }
        if (i >= n_col) {
            return ffi::Error::InvalidArgument("Ai.max() >= n_col");
        }
        j = _Aj[n];
        if (j < 0) {
            return ffi::Error::InvalidArgument("Aj contains negative index");
        }
        if (j >= n_col) {
            return ffi::Error::InvalidArgument("Aj.max() >= n_col");
        }
    }
    return ffi::Error::Success();
}

void coo_to_csc_analyze(
    const int n_col,
    const int n_nz,
    const int* Ai,
    const int* Aj,
    int* Bi,
    int* Bp,
    int* Bk) {
    // compute number of non-zero entries per row of A
    for (int n = 0; n < n_nz; n++) {
        Bp[Aj[n]] += 1;
    }

    // cumsum the n_nz per row to get Bp
    int cumsum = 0;
    int temp = 0;
    for (int j = 0; j <= n_col; j++) {
        temp = Bp[j];
        Bp[j] = cumsum;
        cumsum += temp;
    }

    // write Ai, Aj into Bi, Bk
    int col = 0;
    int dest = 0;
    for (int n = 0; n < n_nz; n++) {
        col = Aj[n];
        dest = Bp[col];
        Bi[dest] = Ai[n];
        Bk[dest] = n;
        Bp[col] += 1;
    }

    int last = 0;
    for (int i = 0; i <= n_col; i++) {
        temp = Bp[i];
        Bp[i] = last;
        last = temp;
    }
}

using Complex = std::complex<double>;

template <typename T>
struct KluTraits;

template <>
struct KluTraits<double> {
    static klu_numeric* factor(int* Ap, int* Ai, double* Ax, klu_symbolic* Symbolic, klu_common* Common) {
        return klu_factor(Ap, Ai, Ax, Symbolic, Common);
    }
    static int solve(klu_symbolic* Symbolic, klu_numeric* Numeric, int d, int nrhs, double* B, klu_common* Common) {
        return klu_solve(Symbolic, Numeric, d, nrhs, B, Common);
    }
};

template <>
struct KluTraits<Complex> {
    static klu_numeric* factor(int* Ap, int* Ai, Complex* Ax, klu_symbolic* Symbolic, klu_common* Common) {
        return klu_z_factor(Ap, Ai, reinterpret_cast<double*>(Ax), Symbolic, Common);
    }
    static int solve(klu_symbolic* Symbolic, klu_numeric* Numeric, int d, int nrhs, Complex* B, klu_common* Common) {
        return klu_z_solve(Symbolic, Numeric, d, nrhs, reinterpret_cast<double*>(B), Common);
    }
};

template <typename T>
ffi::Error dot_impl(
    const ffi::Buffer<ffi::DataType::S32>& Ai,
    const ffi::Buffer<ffi::DataType::S32>& Aj,
    const ffi::AnyBuffer::Dimensions& ds_Ax,
    const ffi::AnyBuffer::Dimensions& ds_x,
    const T* _Ax,
    const T* _x,
    T* _b) {
    ffi::Error err = validate_args(Ai, Aj, ds_Ax, ds_x);
    if (err.failure()) {
        return err;
    }

    int n_lhs = (int)ds_x[0];
    int n_col = (int)ds_x[1];
    int n_rhs = (int)ds_x[2];
    int n_nz = (int)ds_Ax[1];

    // initialize empty result
    for (int i = 0; i < n_lhs * n_col * n_rhs; i++) {
        _b[i] = 0.0;
    }

    // fill result (all multi-dim arrays are row-major)
    // x_mik = A_mij × x_mjk (einsum)
    // sizes: m<n_lhs; i<n_col<--Ai; j<n_col<--Aj; k<n_rhs
    // Loop order: m (batch) outer for better cache locality on Ax
    int i;
    int j;
    const int* _Ai = Ai.typed_data();
    const int* _Aj = Aj.typed_data();
    for (int m = 0; m < n_lhs; m++) {
        for (int n = 0; n < n_nz; n++) {
            i = _Ai[n];
            j = _Aj[n];
            for (int k = 0; k < n_rhs; k++) {
                _b[m * n_col * n_rhs + i * n_rhs + k] += _Ax[m * n_nz + n] * _x[m * n_col * n_rhs + j * n_rhs + k];
            }
        }
    }
    return ffi::Error::Success();
}

ffi::Error dot_f64(
    const ffi::Buffer<ffi::DataType::S32> Ai,
    const ffi::Buffer<ffi::DataType::S32> Aj,
    const ffi::Buffer<ffi::DataType::F64> Ax,
    const ffi::Buffer<ffi::DataType::F64> x,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> b) {
    return dot_impl<double>(Ai, Aj, Ax.dimensions(), x.dimensions(),
                            Ax.typed_data(), x.typed_data(), b->typed_data());
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(  // b = A x
    dot_f64_handler, dot_f64,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // Ai
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // Aj
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // Ax
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // x
        .Ret<ffi::Buffer<ffi::DataType::F64>>()  // b
);

ffi::Error dot_c128(
    const ffi::Buffer<ffi::DataType::S32> Ai,
    const ffi::Buffer<ffi::DataType::S32> Aj,
    const ffi::Buffer<ffi::DataType::C128> Ax,
    const ffi::Buffer<ffi::DataType::C128> x,
    ffi::Result<ffi::Buffer<ffi::DataType::C128>> b) {
    return dot_impl<Complex>(Ai, Aj, Ax.dimensions(), x.dimensions(),
                             reinterpret_cast<const Complex*>(Ax.typed_data()),
                             reinterpret_cast<const Complex*>(x.typed_data()),
                             reinterpret_cast<Complex*>(b->typed_data()));
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(  // b = A x
    dot_c128_handler, dot_c128,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // Ai
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // Aj
        .Arg<ffi::Buffer<ffi::DataType::C128>>()  // Ax
        .Arg<ffi::Buffer<ffi::DataType::C128>>()  // x
        .Ret<ffi::Buffer<ffi::DataType::C128>>()  // b
);

template <typename T>
ffi::Error solve_impl(
    const ffi::Buffer<ffi::DataType::S32>& Ai,
    const ffi::Buffer<ffi::DataType::S32>& Aj,
    const ffi::AnyBuffer::Dimensions& ds_Ax,
    const ffi::AnyBuffer::Dimensions& ds_b,
    const T* _Ax,
    const T* _b,
    T* _x) {
    ffi::Error err = validate_args(Ai, Aj, ds_Ax, ds_b);
    if (err.failure()) {
        return err;
    }

    int n_lhs = (int)ds_b[0];
    int n_col = (int)ds_b[1];
    int n_rhs = (int)ds_b[2];
    int n_nz = (int)ds_Ax[1];
    const int* _Ai = Ai.typed_data();
    const int* _Aj = Aj.typed_data();

    // get COO -> CSC transformation information (using RAII for automatic cleanup)
    auto _Bk = std::make_unique<int[]>(n_nz);  // Ax -> Bx transformation indices
    auto _Bi = std::make_unique<int[]>(n_nz);
    auto _Bp = std::make_unique<int[]>(n_col + 1);
    auto _Bx = std::make_unique<T[]>(n_nz);

    coo_to_csc_analyze(n_col, n_nz, _Ai, _Aj, _Bi.get(), _Bp.get(), _Bk.get());

    // copy _b into _x_temp and transpose the last two dimensions since KLU expects col-major layout
    // _b itself won't be used anymore. KLU works on _x_temp in-place.
    auto _x_temp = std::make_unique<T[]>(n_lhs * n_col * n_rhs);
    for (int m = 0; m < n_lhs; m++) {
        for (int n = 0; n < n_col; n++) {
            for (int p = 0; p < n_rhs; p++) {
                _x_temp[m * n_rhs * n_col + p * n_col + n] = _b[m * n_col * n_rhs + n * n_rhs + p];
            }
        }
    }

    // initialize KLU for given sparsity pattern
    klu_symbolic* Symbolic;
    klu_numeric* Numeric;
    klu_common Common;
    klu_defaults(&Common);
    Symbolic = klu_analyze(n_col, _Bp.get(), _Bi.get(), &Common);

    // solve for all elements in batch:
    // NOTE: same sparsity pattern for each element in batch assumed
    for (int i = 0; i < n_lhs; i++) {
        int m = i * n_nz;
        int n = i * n_rhs * n_col;

        // convert COO Ax to CSC Bx
        for (int k = 0; k < n_nz; k++) {
            _Bx[k] = _Ax[m + _Bk[k]];
        }

        // solve using KLU
        Numeric = KluTraits<T>::factor(_Bp.get(), _Bi.get(), _Bx.get(), Symbolic, &Common);
        if (Numeric == nullptr || Common.status < KLU_OK) {
            klu_free_symbolic(&Symbolic, &Common);
            return ffi::Error::InvalidArgument("klu_factor/z_factor failed (singular matrix?)");
        }
        KluTraits<T>::solve(Symbolic, Numeric, n_col, n_rhs, &_x_temp[n], &Common);
        if (Common.status < KLU_OK) {
            klu_free_numeric(&Numeric, &Common);
            klu_free_symbolic(&Symbolic, &Common);
            return ffi::Error::InvalidArgument("klu_solve/z_solve failed");
        }
        klu_free_numeric(&Numeric, &Common);
    }

    // copy _x_temp into _x and transpose the last two dimensions since JAX expects row-major layout
    // NOTE: it feels a bit weird to have to do all this copying and transposing here. This might actually be
    // pretty inefficient. Ideally I'd like to get rid of this transpose. Maybe just represent b/x in python
    // as n_lhs x n_rhs x n_col in stead of n_lhs x n_col x n_rhs?
    for (int m = 0; m < n_lhs; m++) {
        for (int n = 0; n < n_col; n++) {
            for (int p = 0; p < n_rhs; p++) {
                _x[m * n_col * n_rhs + n * n_rhs + p] = _x_temp[m * n_rhs * n_col + p * n_col + n];
            }
        }
    }

    klu_free_symbolic(&Symbolic, &Common);
    return ffi::Error::Success();
}

ffi::Error solve_f64(
    ffi::Buffer<ffi::DataType::S32> Ai,
    ffi::Buffer<ffi::DataType::S32> Aj,
    ffi::Buffer<ffi::DataType::F64> Ax,
    ffi::Buffer<ffi::DataType::F64> b,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> x) {
    return solve_impl<double>(Ai, Aj, Ax.dimensions(), b.dimensions(),
                              Ax.typed_data(), b.typed_data(), x->typed_data());
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(  // b = A x
    solve_f64_handler, solve_f64,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // Ai
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // Aj
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // Ax
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // b
        .Ret<ffi::Buffer<ffi::DataType::F64>>()  // x
);

ffi::Error solve_c128(
    const ffi::Buffer<ffi::DataType::S32> Ai,
    const ffi::Buffer<ffi::DataType::S32> Aj,
    const ffi::Buffer<ffi::DataType::C128> Ax,
    const ffi::Buffer<ffi::DataType::C128> b,
    ffi::Result<ffi::Buffer<ffi::DataType::C128>> x) {
    return solve_impl<Complex>(Ai, Aj, Ax.dimensions(), b.dimensions(),
                               reinterpret_cast<const Complex*>(Ax.typed_data()),
                               reinterpret_cast<const Complex*>(b.typed_data()),
                               reinterpret_cast<Complex*>(x->typed_data()));
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(  // b = A x
    solve_c128_handler, solve_c128,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // Ai
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // Aj
        .Arg<ffi::Buffer<ffi::DataType::C128>>()  // Ax
        .Arg<ffi::Buffer<ffi::DataType::C128>>()  // b
        .Ret<ffi::Buffer<ffi::DataType::C128>>()  // x
);

template <typename T>
ffi::Error solve_with_symbol_impl(
    const ffi::Buffer<ffi::DataType::S32>& Ai,
    const ffi::Buffer<ffi::DataType::S32>& Aj,
    const ffi::AnyBuffer::Dimensions& ds_Ax,
    const ffi::AnyBuffer::Dimensions& ds_b,
    const ffi::Buffer<ffi::DataType::U64>& symbolic,
    const T* _Ax,
    const T* _b,
    T* _x) {
    if (symbolic.element_count() != 1) return ffi::Error::InvalidArgument("symbolic must be scalar");
    uint64_t sym_addr = *symbolic.typed_data();
    if (sym_addr == 0) return ffi::Error::InvalidArgument("symbolic pointer is null");
    klu_symbolic* Symbolic = reinterpret_cast<klu_symbolic*>(sym_addr);

    ffi::Error err = validate_args(Ai, Aj, ds_Ax, ds_b);
    if (err.failure()) {
        return err;
    }

    int n_lhs = (int)ds_b[0];
    int n_col = (int)ds_b[1];
    int n_rhs = (int)ds_b[2];
    int n_nz = (int)ds_Ax[1];
    const int* _Ai = Ai.typed_data();
    const int* _Aj = Aj.typed_data();

    // get COO -> CSC transformation information (using RAII for automatic cleanup)
    auto _Bk = std::make_unique<int[]>(n_nz);  // Ax -> Bx transformation indices
    auto _Bi = std::make_unique<int[]>(n_nz);
    auto _Bp = std::make_unique<int[]>(n_col + 1);
    auto _Bx = std::make_unique<T[]>(n_nz);

    coo_to_csc_analyze(n_col, n_nz, _Ai, _Aj, _Bi.get(), _Bp.get(), _Bk.get());

    // copy _b into _x_temp and transpose the last two dimensions since KLU expects col-major layout
    auto _x_temp = std::make_unique<T[]>(n_lhs * n_col * n_rhs);
    for (int m = 0; m < n_lhs; m++) {
        for (int n = 0; n < n_col; n++) {
            for (int p = 0; p < n_rhs; p++) {
                _x_temp[m * n_rhs * n_col + p * n_col + n] = _b[m * n_col * n_rhs + n * n_rhs + p];
            }
        }
    }

    klu_common Common;
    klu_defaults(&Common);

    klu_numeric* Numeric;
    for (int i = 0; i < n_lhs; i++) {
        int m = i * n_nz;
        int n = i * n_rhs * n_col;

        // convert COO Ax to CSC Bx
        for (int k = 0; k < n_nz; k++) {
            _Bx[k] = _Ax[m + _Bk[k]];
        }

        // solve using KLU with provided Symbolic handle
        Numeric = KluTraits<T>::factor(_Bp.get(), _Bi.get(), _Bx.get(), Symbolic, &Common);
        if (Numeric == nullptr || Common.status < KLU_OK) {
            return ffi::Error::InvalidArgument("klu_factor/z_factor failed (singular matrix?)");
        }
        KluTraits<T>::solve(Symbolic, Numeric, n_col, n_rhs, &_x_temp[n], &Common);
        if (Common.status < KLU_OK) {
            klu_free_numeric(&Numeric, &Common);
            return ffi::Error::InvalidArgument("klu_solve/z_solve failed");
        }
        klu_free_numeric(&Numeric, &Common);
    }

    // copy _x_temp into _x and transpose
    for (int m = 0; m < n_lhs; m++) {
        for (int n = 0; n < n_col; n++) {
            for (int p = 0; p < n_rhs; p++) {
                _x[m * n_col * n_rhs + n * n_rhs + p] = _x_temp[m * n_rhs * n_col + p * n_col + n];
            }
        }
    }

    return ffi::Error::Success();
}

ffi::Error solve_with_symbol_f64(
    const ffi::Buffer<ffi::DataType::S32> Ai,
    const ffi::Buffer<ffi::DataType::S32> Aj,
    const ffi::Buffer<ffi::DataType::F64> Ax,
    const ffi::Buffer<ffi::DataType::F64> b,
    const ffi::Buffer<ffi::DataType::U64> symbolic,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> x) {
    return solve_with_symbol_impl<double>(Ai, Aj, Ax.dimensions(), b.dimensions(), symbolic,
                                          Ax.typed_data(), b.typed_data(), x->typed_data());
}

ffi::Error solve_with_symbol_c128(
    const ffi::Buffer<ffi::DataType::S32> Ai,
    const ffi::Buffer<ffi::DataType::S32> Aj,
    const ffi::Buffer<ffi::DataType::C128> Ax,
    const ffi::Buffer<ffi::DataType::C128> b,
    const ffi::Buffer<ffi::DataType::U64> symbolic,
    ffi::Result<ffi::Buffer<ffi::DataType::C128>> x) {
    return solve_with_symbol_impl<Complex>(Ai, Aj, Ax.dimensions(), b.dimensions(), symbolic,
                                           reinterpret_cast<const Complex*>(Ax.typed_data()),
                                           reinterpret_cast<const Complex*>(b.typed_data()),
                                           reinterpret_cast<Complex*>(x->typed_data()));
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    solve_with_symbol_f64_handler, solve_with_symbol_f64,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // Ai
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // Aj
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // Ax
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // b
        .Arg<ffi::Buffer<ffi::DataType::U64>>()  // symbolic
        .Ret<ffi::Buffer<ffi::DataType::F64>>()  // x
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    solve_with_symbol_c128_handler, solve_with_symbol_c128,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // Ai
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // Aj
        .Arg<ffi::Buffer<ffi::DataType::C128>>()  // Ax
        .Arg<ffi::Buffer<ffi::DataType::C128>>()  // b
        .Arg<ffi::Buffer<ffi::DataType::U64>>()   // symbolic
        .Ret<ffi::Buffer<ffi::DataType::C128>>()  // x
);

template <typename T>
ffi::Error factor_impl(
    const ffi::Buffer<ffi::DataType::S32>& Ai,
    const ffi::Buffer<ffi::DataType::S32>& Aj,
    const ffi::AnyBuffer::Dimensions& ds_Ax,
    const ffi::Buffer<ffi::DataType::U64>& symbolic,
    const T* _Ax,
    uint64_t* _numeric) {
    if (symbolic.element_count() != 1) return ffi::Error::InvalidArgument("symbolic must be scalar");
    uint64_t sym_addr = *symbolic.typed_data();
    if (sym_addr == 0) return ffi::Error::InvalidArgument("symbolic pointer is null");
    klu_symbolic* Symbolic = reinterpret_cast<klu_symbolic*>(sym_addr);

    // We reuse validate_args but we need a dummy ds_x (b) to satisfy it.
    // We construct a dummy ds_b that matches ds_Ax's batch size.
    int n_lhs = (int)ds_Ax[0];
    int n_nz = (int)ds_Ax[1];
    // We need n_col. We can't easily get it from just Ax/Ai/Aj without passing it or inferring it.
    // However, Symbolic->n is n_col.
    int n_col = Symbolic->n;
    
    // validate_args expects ds_x to be [n_lhs, n_col, n_rhs].
    // We can just check Ai/Aj/Ax consistency manually or construct a fake vector.
    // Let's do manual consistency check for Ai/Aj/Ax to avoid constructing fake vector.
    if (Ai.dimensions().size() != 1 || Aj.dimensions().size() != 1) return ffi::Error::InvalidArgument("Ai/Aj must be 1D");
    if (Ai.dimensions()[0] != n_nz || Aj.dimensions()[0] != n_nz) return ffi::Error::InvalidArgument("Ai/Aj size mismatch with Ax");

    const int* _Ai = Ai.typed_data();
    const int* _Aj = Aj.typed_data();

    // get COO -> CSC transformation information
    auto _Bk = std::make_unique<int[]>(n_nz);
    auto _Bi = std::make_unique<int[]>(n_nz);
    auto _Bp = std::make_unique<int[]>(n_col + 1);
    auto _Bx = std::make_unique<T[]>(n_nz);

    coo_to_csc_analyze(n_col, n_nz, _Ai, _Aj, _Bi.get(), _Bp.get(), _Bk.get());

    klu_common Common;
    klu_defaults(&Common);

    for (int i = 0; i < n_lhs; i++) {
        int m = i * n_nz;
        // convert COO Ax to CSC Bx
        for (int k = 0; k < n_nz; k++) {
            _Bx[k] = _Ax[m + _Bk[k]];
        }

        klu_numeric* Numeric = KluTraits<T>::factor(_Bp.get(), _Bi.get(), _Bx.get(), Symbolic, &Common);
        if (Numeric == nullptr || Common.status < KLU_OK) {
             // Cleanup already allocated numerics in this batch?
             // For simplicity, we return error and let user handle cleanup (or leak, but this is exception path).
             // Ideally we should cleanup.
             for(int j=0; j<i; j++) {
                 klu_numeric* num = reinterpret_cast<klu_numeric*>(_numeric[j]);
                 klu_free_numeric(&num, &Common);
             }
             return ffi::Error::InvalidArgument("klu_factor/z_factor failed (singular matrix?)");
        }
        _numeric[i] = reinterpret_cast<uint64_t>(Numeric);
    }
    return ffi::Error::Success();
}

ffi::Error factor_f64(
    const ffi::Buffer<ffi::DataType::S32> Ai,
    const ffi::Buffer<ffi::DataType::S32> Aj,
    const ffi::Buffer<ffi::DataType::F64> Ax,
    const ffi::Buffer<ffi::DataType::U64> symbolic,
    ffi::Result<ffi::Buffer<ffi::DataType::U64>> numeric) {
    return factor_impl<double>(Ai, Aj, Ax.dimensions(), symbolic, Ax.typed_data(), numeric->typed_data());
}

ffi::Error factor_c128(
    const ffi::Buffer<ffi::DataType::S32> Ai,
    const ffi::Buffer<ffi::DataType::S32> Aj,
    const ffi::Buffer<ffi::DataType::C128> Ax,
    const ffi::Buffer<ffi::DataType::U64> symbolic,
    ffi::Result<ffi::Buffer<ffi::DataType::U64>> numeric) {
    return factor_impl<Complex>(Ai, Aj, Ax.dimensions(), symbolic, 
                                reinterpret_cast<const Complex*>(Ax.typed_data()), 
                                numeric->typed_data());
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    factor_f64_handler, factor_f64,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::U64>>()
        .Ret<ffi::Buffer<ffi::DataType::U64>>()
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    factor_c128_handler, factor_c128,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::C128>>()
        .Arg<ffi::Buffer<ffi::DataType::U64>>()
        .Ret<ffi::Buffer<ffi::DataType::U64>>()
);

template <typename T>
ffi::Error solve_with_numeric_impl(
    const ffi::AnyBuffer::Dimensions& ds_b,
    const ffi::AnyBuffer::Dimensions& ds_x,
    const ffi::Buffer<ffi::DataType::U64>& symbolic,
    const ffi::Buffer<ffi::DataType::U64>& numeric,
    const T* _b,
    T* _x) {
    
    if (symbolic.element_count() != 1) return ffi::Error::InvalidArgument("symbolic must be scalar");
    uint64_t sym_addr = *symbolic.typed_data();
    if (sym_addr == 0) return ffi::Error::InvalidArgument("symbolic pointer is null");
    klu_symbolic* Symbolic = reinterpret_cast<klu_symbolic*>(sym_addr);

    if (ds_b.size() != 3) return ffi::Error::InvalidArgument("b must be 3D");
    if (ds_x.size() != 3) return ffi::Error::InvalidArgument("x must be 3D");

    int n_lhs_x = (int)ds_x[0];
    int n_col = (int)ds_x[1];
    int n_rhs = (int)ds_x[2];

    int n_lhs_b = (int)ds_b[0];
    if (ds_b[1] != n_col || ds_b[2] != n_rhs) {
        return ffi::Error::InvalidArgument("b and x dimension mismatch");
    }
    
    int n_numeric = numeric.element_count();
    const uint64_t* _numeric = numeric.typed_data();

    // Determine actual output batch size
    int n_lhs = n_lhs_x;  // x determines output size
    
    bool broadcast_numeric = (n_numeric == 1);
    bool broadcast_b = (n_lhs_b == 1);
    
    // Validate broadcasting rules
    if (!broadcast_numeric && n_numeric != n_lhs) {
        return ffi::Error::InvalidArgument("numeric batch size mismatch with x: got " + 
                                          std::to_string(n_numeric) + " vs " + std::to_string(n_lhs));
    }
    if (!broadcast_b && n_lhs_b != n_lhs) {
        return ffi::Error::InvalidArgument("b batch size mismatch with x: got " + 
                                          std::to_string(n_lhs_b) + " vs " + std::to_string(n_lhs));
    }

    auto _x_temp = std::make_unique<T[]>(n_lhs * n_col * n_rhs);
    
    // Copy b to x_temp (transpose) with broadcasting
    for (int m = 0; m < n_lhs; m++) {
        int m_b = broadcast_b ? 0 : m;
        for (int n = 0; n < n_col; n++) {
            for (int p = 0; p < n_rhs; p++) {
                _x_temp[m * n_rhs * n_col + p * n_col + n] = _b[m_b * n_col * n_rhs + n * n_rhs + p];
            }
        }
    }

    klu_common Common;
    klu_defaults(&Common);

    for (int i = 0; i < n_lhs; i++) {
        int n = i * n_rhs * n_col;
        uint64_t num_addr = broadcast_numeric ? _numeric[0] : _numeric[i];
        if (num_addr == 0) return ffi::Error::InvalidArgument("numeric pointer is null");
        
        klu_numeric* Numeric = reinterpret_cast<klu_numeric*>(num_addr);
        KluTraits<T>::solve(Symbolic, Numeric, n_col, n_rhs, &_x_temp[n], &Common);
        
        if (Common.status < KLU_OK) {
            return ffi::Error::InvalidArgument("klu_solve/z_solve failed");
        }
    }

    // Copy x_temp to x (transpose)
    for (int m = 0; m < n_lhs; m++) {
        for (int n = 0; n < n_col; n++) {
            for (int p = 0; p < n_rhs; p++) {
                _x[m * n_col * n_rhs + n * n_rhs + p] = _x_temp[m * n_rhs * n_col + p * n_col + n];
            }
        }
    }
    return ffi::Error::Success();
}

ffi::Error solve_with_numeric_f64(
    const ffi::Buffer<ffi::DataType::U64> symbolic,
    const ffi::Buffer<ffi::DataType::U64> numeric,
    const ffi::Buffer<ffi::DataType::F64> b,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> x) {
    return solve_with_numeric_impl<double>(b.dimensions(), x->dimensions(), symbolic, numeric, b.typed_data(), x->typed_data());
}

ffi::Error solve_with_numeric_c128(
    const ffi::Buffer<ffi::DataType::U64> symbolic,
    const ffi::Buffer<ffi::DataType::U64> numeric,
    const ffi::Buffer<ffi::DataType::C128> b,
    ffi::Result<ffi::Buffer<ffi::DataType::C128>> x) {
    return solve_with_numeric_impl<Complex>(b.dimensions(), x->dimensions(), symbolic, numeric, 
                                            reinterpret_cast<const Complex*>(b.typed_data()), 
                                            reinterpret_cast<Complex*>(x->typed_data()));
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    solve_with_numeric_f64_handler, solve_with_numeric_f64,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::U64>>()
        .Arg<ffi::Buffer<ffi::DataType::U64>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Ret<ffi::Buffer<ffi::DataType::F64>>()
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    solve_with_numeric_c128_handler, solve_with_numeric_c128,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::U64>>()
        .Arg<ffi::Buffer<ffi::DataType::U64>>()
        .Arg<ffi::Buffer<ffi::DataType::C128>>()
        .Ret<ffi::Buffer<ffi::DataType::C128>>()
);

ffi::Error free_numeric(
    const ffi::Buffer<ffi::DataType::U64> numeric) {
    
    int n = numeric.element_count();
    const uint64_t* _numeric = numeric.typed_data();
    
    klu_common Common;
    klu_defaults(&Common);

    for(int i=0; i<n; i++) {
        uint64_t addr = _numeric[i];
        if (addr != 0) {
            klu_numeric* Numeric = reinterpret_cast<klu_numeric*>(addr);
            klu_free_numeric(&Numeric, &Common);
        }
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    free_numeric_handler, free_numeric,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::U64>>()
);

ffi::Error free_symbolic(
    const ffi::Buffer<ffi::DataType::U64> symbolic) {
    if (symbolic.element_count() != 1) {
        return ffi::Error::InvalidArgument("symbolic must be a scalar.");
    }
    uint64_t sym_addr = *symbolic.typed_data();
    if (sym_addr == 0) {
        return ffi::Error::Success();
    }

    klu_symbolic* Symbolic = reinterpret_cast<klu_symbolic*>(sym_addr);
    klu_common Common;
    klu_defaults(&Common);
    klu_free_symbolic(&Symbolic, &Common);

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    free_symbolic_handler, free_symbolic,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::U64>>()  // symbolic
);

ffi::Error analyze(
    const ffi::Buffer<ffi::DataType::S32> Ai,
    const ffi::Buffer<ffi::DataType::S32> Aj,
    const ffi::Buffer<ffi::DataType::S32> n_col_buf,
    ffi::Result<ffi::Buffer<ffi::DataType::U64>> symbolic) {
    if (n_col_buf.element_count() != 1) {
        return ffi::Error::InvalidArgument("n_col must be a scalar.");
    }
    int n_col = *n_col_buf.typed_data();

    auto ds_Ai = Ai.dimensions();
    if (ds_Ai.size() != 1) return ffi::Error::InvalidArgument("Ai must be 1D.");
    int n_nz = ds_Ai[0];

    auto ds_Aj = Aj.dimensions();
    if (ds_Aj.size() != 1) return ffi::Error::InvalidArgument("Aj must be 1D.");
    if (ds_Aj[0] != n_nz) return ffi::Error::InvalidArgument("Aj size mismatch.");

    const int* _Ai = Ai.typed_data();
    const int* _Aj = Aj.typed_data();

    // Validate indices
    for (int n = 0; n < n_nz; n++) {
        if (_Ai[n] < 0 || _Ai[n] >= n_col) return ffi::Error::InvalidArgument("Ai index out of bounds.");
        if (_Aj[n] < 0 || _Aj[n] >= n_col) return ffi::Error::InvalidArgument("Aj index out of bounds.");
    }

    auto _Bi = std::make_unique<int[]>(n_nz);
    auto _Bp = std::make_unique<int[]>(n_col + 1);
    auto _Bk = std::make_unique<int[]>(n_nz);

    coo_to_csc_analyze(n_col, n_nz, _Ai, _Aj, _Bi.get(), _Bp.get(), _Bk.get());

    klu_common Common;
    klu_defaults(&Common);
    klu_symbolic* Symbolic = klu_analyze(n_col, _Bp.get(), _Bi.get(), &Common);

    if (!Symbolic) {
        return ffi::Error::Internal("klu_analyze failed.");
    }

    *symbolic->typed_data() = reinterpret_cast<uint64_t>(Symbolic);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    analyze_handler, analyze,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // Ai
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // Aj
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // n_col
        .Ret<ffi::Buffer<ffi::DataType::U64>>()  // symbolic ptr
);

// Python wrappers
PYBIND11_MODULE(klujax_cpp, m) {
    m.def("dot_f64",
          []() { return py::capsule((void*)&dot_f64_handler); });
    m.def("dot_c128",
          []() { return py::capsule((void*)&dot_c128_handler); });
    m.def("solve_f64",
          []() { return py::capsule((void*)&solve_f64_handler); });
    m.def("solve_c128",
          []() { return py::capsule((void*)&solve_c128_handler); });
    m.def("solve_with_symbol_f64",
          []() { return py::capsule((void*)&solve_with_symbol_f64_handler); });
    m.def("solve_with_symbol_c128",
          []() { return py::capsule((void*)&solve_with_symbol_c128_handler); });
    m.def("factor_f64",
          []() { return py::capsule((void*)&factor_f64_handler); });
    m.def("factor_c128",
          []() { return py::capsule((void*)&factor_c128_handler); });
    m.def("solve_with_numeric_f64",
          []() { return py::capsule((void*)&solve_with_numeric_f64_handler); });
    m.def("solve_with_numeric_c128",
          []() { return py::capsule((void*)&solve_with_numeric_c128_handler); });
    m.def("free_numeric",
          []() { return py::capsule((void*)&free_numeric_handler); });
    m.def("analyze",
          []() { return py::capsule((void*)&analyze_handler); });
    m.def("free_symbolic",
          []() { return py::capsule((void*)&free_symbolic_handler); });
}
