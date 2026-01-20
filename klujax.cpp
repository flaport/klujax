// version: 0.4.6
// Imports

#include <cmath>
#include <complex>

#include "klu.h"
#include "pybind11/complex.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "xla/ffi/api/ffi.h"
namespace py = pybind11;
namespace ffi = xla::ffi;

#include <cstring>  // for memset
#include <memory>   // for unique_ptr
#include <mutex>
#include <unordered_map>

// =============================================================================
// Handle-based caching for KLU symbolic and numeric factorizations
// This allows reusing klu_analyze results across multiple solves
// =============================================================================

// Cache entry for symbolic factorization
struct SymbolicEntry {
    klu_symbolic* symbolic;
    klu_common common;
    int n_col;
};

// Cache entry for numeric factorization
struct NumericEntry {
    klu_numeric* numeric;
    int64_t symbolic_handle;  // reference to parent symbolic
};

// Global caches with thread-safe access
static std::mutex g_symbolic_mutex;
static std::unordered_map<int64_t, SymbolicEntry> g_symbolic_cache;
static int64_t g_next_symbolic_handle = 1;

static std::mutex g_numeric_mutex;
static std::unordered_map<int64_t, NumericEntry> g_numeric_cache;
static int64_t g_next_numeric_handle = 1;

// Helper to allocate a new symbolic handle
int64_t allocate_symbolic_handle(klu_symbolic* sym, klu_common common, int n_col) {
    std::lock_guard<std::mutex> lock(g_symbolic_mutex);
    int64_t handle = g_next_symbolic_handle++;
    g_symbolic_cache[handle] = {sym, common, n_col};
    return handle;
}

// Helper to get symbolic entry (returns nullptr if not found)
SymbolicEntry* get_symbolic_entry(int64_t handle) {
    std::lock_guard<std::mutex> lock(g_symbolic_mutex);
    auto it = g_symbolic_cache.find(handle);
    if (it == g_symbolic_cache.end()) {
        return nullptr;
    }
    return &it->second;
}

// Helper to free symbolic handle
bool free_symbolic_handle(int64_t handle) {
    std::lock_guard<std::mutex> lock(g_symbolic_mutex);
    auto it = g_symbolic_cache.find(handle);
    if (it == g_symbolic_cache.end()) {
        return false;
    }
    klu_free_symbolic(&it->second.symbolic, &it->second.common);
    g_symbolic_cache.erase(it);
    return true;
}

// Helper to allocate a new numeric handle
int64_t allocate_numeric_handle(klu_numeric* num, int64_t symbolic_handle) {
    std::lock_guard<std::mutex> lock(g_numeric_mutex);
    int64_t handle = g_next_numeric_handle++;
    g_numeric_cache[handle] = {num, symbolic_handle};
    return handle;
}

// Helper to get numeric entry (returns nullptr if not found)
NumericEntry* get_numeric_entry(int64_t handle) {
    std::lock_guard<std::mutex> lock(g_numeric_mutex);
    auto it = g_numeric_cache.find(handle);
    if (it == g_numeric_cache.end()) {
        return nullptr;
    }
    return &it->second;
}

// Helper to free numeric handle
bool free_numeric_handle(int64_t handle) {
    std::lock_guard<std::mutex> lock(g_numeric_mutex);
    auto it = g_numeric_cache.find(handle);
    if (it == g_numeric_cache.end()) {
        return false;
    }
    // Need to get the symbolic entry to free numeric properly
    SymbolicEntry* sym_entry = get_symbolic_entry(it->second.symbolic_handle);
    if (sym_entry) {
        klu_free_numeric(&it->second.numeric, &sym_entry->common);
    }
    g_numeric_cache.erase(it);
    return true;
}

ffi::Error validate_dot_f64_args(
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

ffi::Error dot_f64(
    const ffi::Buffer<ffi::DataType::S32> Ai,
    const ffi::Buffer<ffi::DataType::S32> Aj,
    const ffi::Buffer<ffi::DataType::F64> Ax,
    const ffi::Buffer<ffi::DataType::F64> x,
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
    const int* _Ai = Ai.typed_data();
    const int* _Aj = Aj.typed_data();
    const double* _Ax = Ax.typed_data();
    const double* _x = x.typed_data();
    double* _b = b->typed_data();

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
    const int* _Ai = Ai.typed_data();
    const int* _Aj = Aj.typed_data();
    const double* _Ax = (double*)Ax.typed_data();
    const double* _x = (double*)x.typed_data();
    double* _b = (double*)b->typed_data();

    // initialize empty result
    for (int i = 0; i < 2 * n_lhs * n_col * n_rhs; i++) {
        _b[i] = 0.0;
    }

    // fill result (all multi-dim arrays are row-major)
    // x_mik = A_mij × x_mjk (einsum)
    // sizes: m<n_lhs; i<n_col<--Ai; j<n_col<--Aj; k<n_rhs
    // Loop order: m (batch) outer for better cache locality on Ax
    int i;
    int j;
    for (int m = 0; m < n_lhs; m++) {
        for (int n = 0; n < n_nz; n++) {
            i = _Ai[n];
            j = _Aj[n];
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
    dot_c128_handler, dot_c128,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // Ai
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // Aj
        .Arg<ffi::Buffer<ffi::DataType::C128>>()  // Ax
        .Arg<ffi::Buffer<ffi::DataType::C128>>()  // x
        .Ret<ffi::Buffer<ffi::DataType::C128>>()  // b
);

ffi::Error solve_f64(
    ffi::Buffer<ffi::DataType::S32> Ai,
    ffi::Buffer<ffi::DataType::S32> Aj,
    ffi::Buffer<ffi::DataType::F64> Ax,
    ffi::Buffer<ffi::DataType::F64> b,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> x) {
    auto ds_b = b.dimensions();
    auto ds_Ax = Ax.dimensions();
    ffi::Error err = validate_dot_f64_args(Ai, Aj, ds_Ax, ds_b);
    if (err.failure()) {
        return err;
    }

    int n_lhs = (int)ds_b[0];
    int n_col = (int)ds_b[1];
    int n_rhs = (int)ds_b[2];
    int n_nz = (int)ds_Ax[1];
    const int* _Ai = Ai.typed_data();
    const int* _Aj = Aj.typed_data();
    const double* _Ax = Ax.typed_data();
    const double* _b = b.typed_data();
    double* _x = x->typed_data();

    // get COO -> CSC transformation information (using RAII for automatic cleanup)
    auto _Bk = std::make_unique<int[]>(n_nz);  // Ax -> Bx transformation indices
    auto _Bi = std::make_unique<int[]>(n_nz);
    auto _Bp = std::make_unique<int[]>(n_col + 1);
    auto _Bx = std::make_unique<double[]>(n_nz);

    coo_to_csc_analyze(n_col, n_nz, _Ai, _Aj, _Bi.get(), _Bp.get(), _Bk.get());

    // copy _b into _x_temp and transpose the last two dimensions since KLU expects col-major layout
    // _b itself won't be used anymore. KLU works on _x_temp in-place.
    auto _x_temp = std::make_unique<double[]>(n_lhs * n_col * n_rhs);
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
        Numeric = klu_factor(_Bp.get(), _Bi.get(), _Bx.get(), Symbolic, &Common);
        if (Numeric == nullptr || Common.status < KLU_OK) {
            klu_free_symbolic(&Symbolic, &Common);
            return ffi::Error::InvalidArgument("klu_factor failed (singular matrix?)");
        }
        klu_solve(Symbolic, Numeric, n_col, n_rhs, &_x_temp[n], &Common);
        if (Common.status < KLU_OK) {
            klu_free_numeric(&Numeric, &Common);
            klu_free_symbolic(&Symbolic, &Common);
            return ffi::Error::InvalidArgument("klu_solve failed");
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
    auto ds_x = b.dimensions();
    auto ds_Ax = Ax.dimensions();
    ffi::Error err = validate_dot_f64_args(Ai, Aj, ds_Ax, ds_x);
    if (err.failure()) {
        return err;
    }
    int n_lhs = (int)ds_x[0];
    int n_col = (int)ds_x[1];
    int n_rhs = (int)ds_x[2];
    int n_nz = (int)ds_Ax[1];
    const int* _Ai = Ai.typed_data();
    const int* _Aj = Aj.typed_data();
    const double* _Ax = (double*)Ax.typed_data();
    const double* _b = (double*)b.typed_data();
    double* _x = (double*)x->typed_data();

    // get COO -> CSC transformation information (using RAII for automatic cleanup)
    auto _Bk = std::make_unique<int[]>(n_nz);       // Ax -> Bx transformation indices
    auto _Bi = std::make_unique<int[]>(n_nz);       // CSC row indices
    auto _Bp = std::make_unique<int[]>(n_col + 1);  // CSC column pointers
    auto _Bx = std::make_unique<double[]>(2 * n_nz);
    coo_to_csc_analyze(n_col, n_nz, _Ai, _Aj, _Bi.get(), _Bp.get(), _Bk.get());

    // copy _b into _x_temp and transpose the last two dimensions since KLU expects col-major layout
    // _b itself won't be used anymore. KLU works on _x_temp in-place.
    auto _x_temp = std::make_unique<double[]>(2 * n_lhs * n_col * n_rhs);
    for (int m = 0; m < n_lhs; m++) {
        for (int n = 0; n < n_col; n++) {
            for (int p = 0; p < n_rhs; p++) {
                _x_temp[2 * (m * n_rhs * n_col + p * n_col + n)] = _b[2 * (m * n_col * n_rhs + n * n_rhs + p)];
                _x_temp[2 * (m * n_rhs * n_col + p * n_col + n) + 1] = _b[2 * (m * n_col * n_rhs + n * n_rhs + p) + 1];
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
            _Bx[2 * k] = _Ax[2 * (m + _Bk[k])];
            _Bx[2 * k + 1] = _Ax[2 * (m + _Bk[k]) + 1];
        }

        // solve using KLU
        Numeric = klu_z_factor(_Bp.get(), _Bi.get(), _Bx.get(), Symbolic, &Common);
        if (Numeric == nullptr || Common.status < KLU_OK) {
            klu_free_symbolic(&Symbolic, &Common);
            return ffi::Error::InvalidArgument("klu_z_factor failed (singular matrix?)");
        }
        klu_z_solve(Symbolic, Numeric, n_col, n_rhs, &_x_temp[2 * n], &Common);
        if (Common.status < KLU_OK) {
            klu_free_numeric(&Numeric, &Common);
            klu_free_symbolic(&Symbolic, &Common);
            return ffi::Error::InvalidArgument("klu_z_solve failed");
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
                _x[2 * (m * n_col * n_rhs + n * n_rhs + p)] = _x_temp[2 * (m * n_rhs * n_col + p * n_col + n)];
                _x[2 * (m * n_col * n_rhs + n * n_rhs + p) + 1] = _x_temp[2 * (m * n_rhs * n_col + p * n_col + n) + 1];
            }
        }
    }

    klu_free_symbolic(&Symbolic, &Common);
    return ffi::Error::Success();
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

// =============================================================================
// Low-level functions for advanced users (split-solve API)
// =============================================================================

// COO to CSC conversion: returns Bp (column pointers), Bi (row indices), Bk (value mapping)
// Input: Ai (n_nz,), Aj (n_nz,), n_col (scalar via attribute)
// Output: Bp (n_col+1,), Bi (n_nz,), Bk (n_nz,)
ffi::Error coo_to_csc(
    const ffi::Buffer<ffi::DataType::S32> Ai,
    const ffi::Buffer<ffi::DataType::S32> Aj,
    ffi::Buffer<ffi::DataType::S32> n_col_buf,
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> Bp,
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> Bi,
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> Bk) {
    auto ds_Ai = Ai.dimensions();
    auto ds_Aj = Aj.dimensions();

    if (ds_Ai.size() != 1 || ds_Aj.size() != 1) {
        return ffi::Error::InvalidArgument("Ai and Aj must be 1D");
    }

    int n_nz = (int)ds_Ai[0];
    if (n_nz != (int)ds_Aj[0]) {
        return ffi::Error::InvalidArgument("Ai and Aj must have same length");
    }

    int n_col = n_col_buf.typed_data()[0];

    const int* _Ai = Ai.typed_data();
    const int* _Aj = Aj.typed_data();
    int* _Bp = Bp->typed_data();
    int* _Bi = Bi->typed_data();
    int* _Bk = Bk->typed_data();

    // Initialize Bp to zero
    for (int i = 0; i <= n_col; i++) {
        _Bp[i] = 0;
    }

    coo_to_csc_analyze(n_col, n_nz, _Ai, _Aj, _Bi, _Bp, _Bk);

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    coo_to_csc_handler, coo_to_csc,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // Ai
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // Aj
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // n_col (as 1-element buffer)
        .Ret<ffi::Buffer<ffi::DataType::S32>>()  // Bp
        .Ret<ffi::Buffer<ffi::DataType::S32>>()  // Bi
        .Ret<ffi::Buffer<ffi::DataType::S32>>()  // Bk
);

// Solve using pre-computed CSC structure (f64)
// This skips the COO->CSC conversion step
ffi::Error solve_csc_f64(
    ffi::Buffer<ffi::DataType::S32> Bp,
    ffi::Buffer<ffi::DataType::S32> Bi,
    ffi::Buffer<ffi::DataType::S32> Bk,
    ffi::Buffer<ffi::DataType::F64> Ax,
    ffi::Buffer<ffi::DataType::F64> b,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> x) {
    auto ds_b = b.dimensions();
    auto ds_Ax = Ax.dimensions();
    auto ds_Bp = Bp.dimensions();
    auto ds_Bk = Bk.dimensions();

    if (ds_b.size() != 3) {
        return ffi::Error::InvalidArgument("b must be 3D (n_lhs, n_col, n_rhs)");
    }
    if (ds_Ax.size() != 2) {
        return ffi::Error::InvalidArgument("Ax must be 2D (n_lhs, n_nz)");
    }

    int n_lhs = (int)ds_b[0];
    int n_col = (int)ds_b[1];
    int n_rhs = (int)ds_b[2];
    int n_nz = (int)ds_Ax[1];

    if ((int)ds_Ax[0] != n_lhs) {
        return ffi::Error::InvalidArgument("Ax.shape[0] != b.shape[0]");
    }
    if ((int)ds_Bp[0] != n_col + 1) {
        return ffi::Error::InvalidArgument("Bp.shape[0] != n_col + 1");
    }
    if ((int)ds_Bk[0] != n_nz) {
        return ffi::Error::InvalidArgument("Bk.shape[0] != n_nz");
    }

    const int* _Bp = Bp.typed_data();
    const int* _Bi = Bi.typed_data();
    const int* _Bk = Bk.typed_data();
    const double* _Ax = Ax.typed_data();
    const double* _b = b.typed_data();
    double* _x = x->typed_data();

    auto _Bx = std::make_unique<double[]>(n_nz);

    // copy _b into _x_temp and transpose for KLU (col-major)
    auto _x_temp = std::make_unique<double[]>(n_lhs * n_col * n_rhs);
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
    Symbolic = klu_analyze(n_col, const_cast<int*>(_Bp), const_cast<int*>(_Bi), &Common);

    // solve for all elements in batch
    for (int i = 0; i < n_lhs; i++) {
        int m = i * n_nz;
        int n = i * n_rhs * n_col;

        // convert COO Ax to CSC Bx using pre-computed mapping
        for (int k = 0; k < n_nz; k++) {
            _Bx[k] = _Ax[m + _Bk[k]];
        }

        Numeric = klu_factor(const_cast<int*>(_Bp), const_cast<int*>(_Bi), _Bx.get(), Symbolic, &Common);
        if (Numeric == nullptr || Common.status < KLU_OK) {
            klu_free_symbolic(&Symbolic, &Common);
            return ffi::Error::InvalidArgument("klu_factor failed (singular matrix?)");
        }
        klu_solve(Symbolic, Numeric, n_col, n_rhs, &_x_temp[n], &Common);
        if (Common.status < KLU_OK) {
            klu_free_numeric(&Numeric, &Common);
            klu_free_symbolic(&Symbolic, &Common);
            return ffi::Error::InvalidArgument("klu_solve failed");
        }
        klu_free_numeric(&Numeric, &Common);
    }

    // copy _x_temp into _x and transpose back (row-major)
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

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    solve_csc_f64_handler, solve_csc_f64,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // Bp
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // Bi
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // Bk
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // Ax
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // b
        .Ret<ffi::Buffer<ffi::DataType::F64>>()  // x
);

// Solve using pre-computed CSC structure (c128)
ffi::Error solve_csc_c128(
    ffi::Buffer<ffi::DataType::S32> Bp,
    ffi::Buffer<ffi::DataType::S32> Bi,
    ffi::Buffer<ffi::DataType::S32> Bk,
    ffi::Buffer<ffi::DataType::C128> Ax,
    ffi::Buffer<ffi::DataType::C128> b,
    ffi::Result<ffi::Buffer<ffi::DataType::C128>> x) {
    auto ds_b = b.dimensions();
    auto ds_Ax = Ax.dimensions();
    auto ds_Bp = Bp.dimensions();
    auto ds_Bk = Bk.dimensions();

    if (ds_b.size() != 3) {
        return ffi::Error::InvalidArgument("b must be 3D (n_lhs, n_col, n_rhs)");
    }
    if (ds_Ax.size() != 2) {
        return ffi::Error::InvalidArgument("Ax must be 2D (n_lhs, n_nz)");
    }

    int n_lhs = (int)ds_b[0];
    int n_col = (int)ds_b[1];
    int n_rhs = (int)ds_b[2];
    int n_nz = (int)ds_Ax[1];

    if ((int)ds_Ax[0] != n_lhs) {
        return ffi::Error::InvalidArgument("Ax.shape[0] != b.shape[0]");
    }
    if ((int)ds_Bp[0] != n_col + 1) {
        return ffi::Error::InvalidArgument("Bp.shape[0] != n_col + 1");
    }
    if ((int)ds_Bk[0] != n_nz) {
        return ffi::Error::InvalidArgument("Bk.shape[0] != n_nz");
    }

    const int* _Bp = Bp.typed_data();
    const int* _Bi = Bi.typed_data();
    const int* _Bk = Bk.typed_data();
    const double* _Ax = (double*)Ax.typed_data();
    const double* _b = (double*)b.typed_data();
    double* _x = (double*)x->typed_data();

    auto _Bx = std::make_unique<double[]>(2 * n_nz);

    // copy _b into _x_temp and transpose for KLU (col-major)
    auto _x_temp = std::make_unique<double[]>(2 * n_lhs * n_col * n_rhs);
    for (int m = 0; m < n_lhs; m++) {
        for (int n = 0; n < n_col; n++) {
            for (int p = 0; p < n_rhs; p++) {
                _x_temp[2 * (m * n_rhs * n_col + p * n_col + n)] = _b[2 * (m * n_col * n_rhs + n * n_rhs + p)];
                _x_temp[2 * (m * n_rhs * n_col + p * n_col + n) + 1] = _b[2 * (m * n_col * n_rhs + n * n_rhs + p) + 1];
            }
        }
    }

    // initialize KLU for given sparsity pattern
    klu_symbolic* Symbolic;
    klu_numeric* Numeric;
    klu_common Common;
    klu_defaults(&Common);
    Symbolic = klu_analyze(n_col, const_cast<int*>(_Bp), const_cast<int*>(_Bi), &Common);

    // solve for all elements in batch
    for (int i = 0; i < n_lhs; i++) {
        int m = i * n_nz;
        int n = i * n_rhs * n_col;

        // convert COO Ax to CSC Bx using pre-computed mapping
        for (int k = 0; k < n_nz; k++) {
            _Bx[2 * k] = _Ax[2 * (m + _Bk[k])];
            _Bx[2 * k + 1] = _Ax[2 * (m + _Bk[k]) + 1];
        }

        Numeric = klu_z_factor(const_cast<int*>(_Bp), const_cast<int*>(_Bi), _Bx.get(), Symbolic, &Common);
        if (Numeric == nullptr || Common.status < KLU_OK) {
            klu_free_symbolic(&Symbolic, &Common);
            return ffi::Error::InvalidArgument("klu_z_factor failed (singular matrix?)");
        }
        klu_z_solve(Symbolic, Numeric, n_col, n_rhs, &_x_temp[2 * n], &Common);
        if (Common.status < KLU_OK) {
            klu_free_numeric(&Numeric, &Common);
            klu_free_symbolic(&Symbolic, &Common);
            return ffi::Error::InvalidArgument("klu_z_solve failed");
        }
        klu_free_numeric(&Numeric, &Common);
    }

    // copy _x_temp into _x and transpose back (row-major)
    for (int m = 0; m < n_lhs; m++) {
        for (int n = 0; n < n_col; n++) {
            for (int p = 0; p < n_rhs; p++) {
                _x[2 * (m * n_col * n_rhs + n * n_rhs + p)] = _x_temp[2 * (m * n_rhs * n_col + p * n_col + n)];
                _x[2 * (m * n_col * n_rhs + n * n_rhs + p) + 1] = _x_temp[2 * (m * n_rhs * n_col + p * n_col + n) + 1];
            }
        }
    }

    klu_free_symbolic(&Symbolic, &Common);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    solve_csc_c128_handler, solve_csc_c128,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // Bp
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // Bi
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // Bk
        .Arg<ffi::Buffer<ffi::DataType::C128>>()  // Ax
        .Arg<ffi::Buffer<ffi::DataType::C128>>()  // b
        .Ret<ffi::Buffer<ffi::DataType::C128>>()  // x
);

// =============================================================================
// True split-solve API: klu_analyze, klu_factor, klu_solve as separate functions
// These use handle-based caching to persist factorizations across calls
// =============================================================================

// klu_analyze: Perform symbolic analysis on CSC sparsity pattern
// Input: Bp (n_col+1,), Bi (n_nz,) - CSC format column pointers and row indices
// Output: handle (int64) that can be passed to klu_factor
int64_t klu_analyze_csc(
    py::array_t<int32_t> Bp,
    py::array_t<int32_t> Bi) {
    auto Bp_buf = Bp.request();
    auto Bi_buf = Bi.request();

    if (Bp_buf.ndim != 1 || Bi_buf.ndim != 1) {
        throw std::runtime_error("Bp and Bi must be 1D arrays");
    }

    int n_col = (int)Bp_buf.shape[0] - 1;
    const int* _Bp = static_cast<const int*>(Bp_buf.ptr);
    const int* _Bi = static_cast<const int*>(Bi_buf.ptr);

    klu_common Common;
    klu_defaults(&Common);
    klu_symbolic* Symbolic = klu_analyze(n_col, const_cast<int*>(_Bp), const_cast<int*>(_Bi), &Common);

    if (Symbolic == nullptr || Common.status < KLU_OK) {
        throw std::runtime_error("klu_analyze failed");
    }

    return allocate_symbolic_handle(Symbolic, Common, n_col);
}

// klu_factor_f64: Perform numeric factorization using pre-computed symbolic analysis
// Input: symbolic_handle, Bp, Bi, Bx (CSC format values)
// Output: handle (int64) that can be passed to klu_solve
int64_t klu_factor_csc_f64(
    int64_t symbolic_handle,
    py::array_t<int32_t> Bp,
    py::array_t<int32_t> Bi,
    py::array_t<double> Bx) {
    SymbolicEntry* sym_entry = get_symbolic_entry(symbolic_handle);
    if (sym_entry == nullptr) {
        throw std::runtime_error("Invalid symbolic handle");
    }

    auto Bp_buf = Bp.request();
    auto Bi_buf = Bi.request();
    auto Bx_buf = Bx.request();

    const int* _Bp = static_cast<const int*>(Bp_buf.ptr);
    const int* _Bi = static_cast<const int*>(Bi_buf.ptr);
    double* _Bx = static_cast<double*>(Bx_buf.ptr);

    klu_numeric* Numeric = klu_factor(
        const_cast<int*>(_Bp), const_cast<int*>(_Bi), _Bx,
        sym_entry->symbolic, &sym_entry->common);

    if (Numeric == nullptr || sym_entry->common.status < KLU_OK) {
        throw std::runtime_error("klu_factor failed (singular matrix?)");
    }

    return allocate_numeric_handle(Numeric, symbolic_handle);
}

// klu_factor_c128: Perform numeric factorization for complex128
int64_t klu_factor_csc_c128(
    int64_t symbolic_handle,
    py::array_t<int32_t> Bp,
    py::array_t<int32_t> Bi,
    py::array_t<std::complex<double>> Bx) {
    SymbolicEntry* sym_entry = get_symbolic_entry(symbolic_handle);
    if (sym_entry == nullptr) {
        throw std::runtime_error("Invalid symbolic handle");
    }

    auto Bp_buf = Bp.request();
    auto Bi_buf = Bi.request();
    auto Bx_buf = Bx.request();

    const int* _Bp = static_cast<const int*>(Bp_buf.ptr);
    const int* _Bi = static_cast<const int*>(Bi_buf.ptr);
    double* _Bx = static_cast<double*>(Bx_buf.ptr);

    klu_numeric* Numeric = klu_z_factor(
        const_cast<int*>(_Bp), const_cast<int*>(_Bi), _Bx,
        sym_entry->symbolic, &sym_entry->common);

    if (Numeric == nullptr || sym_entry->common.status < KLU_OK) {
        throw std::runtime_error("klu_z_factor failed (singular matrix?)");
    }

    return allocate_numeric_handle(Numeric, symbolic_handle);
}

// klu_solve_f64: Solve Ax=b using pre-computed factorizations
// Input: symbolic_handle, numeric_handle, b (col-major)
// Output: x (col-major, same shape as b)
py::array_t<double> klu_solve_handles_f64(
    int64_t symbolic_handle,
    int64_t numeric_handle,
    py::array_t<double> b) {
    SymbolicEntry* sym_entry = get_symbolic_entry(symbolic_handle);
    if (sym_entry == nullptr) {
        throw std::runtime_error("Invalid symbolic handle");
    }

    NumericEntry* num_entry = get_numeric_entry(numeric_handle);
    if (num_entry == nullptr) {
        throw std::runtime_error("Invalid numeric handle");
    }

    auto b_buf = b.request();
    if (b_buf.ndim != 2) {
        throw std::runtime_error("b must be 2D (n_col, n_rhs) in column-major format");
    }

    int n_col = (int)b_buf.shape[0];
    int n_rhs = (int)b_buf.shape[1];

    if (n_col != sym_entry->n_col) {
        throw std::runtime_error("b.shape[0] != n_col from symbolic analysis");
    }

    // Allocate output array and copy b into it (KLU solves in-place)
    py::array_t<double> x({n_col, n_rhs});
    auto x_buf = x.request();
    double* _x = static_cast<double*>(x_buf.ptr);
    const double* _b = static_cast<const double*>(b_buf.ptr);
    std::memcpy(_x, _b, n_col * n_rhs * sizeof(double));

    klu_solve(sym_entry->symbolic, num_entry->numeric, n_col, n_rhs, _x, &sym_entry->common);

    if (sym_entry->common.status < KLU_OK) {
        throw std::runtime_error("klu_solve failed");
    }

    return x;
}

// klu_solve_c128: Solve Ax=b for complex128
py::array_t<std::complex<double>> klu_solve_handles_c128(
    int64_t symbolic_handle,
    int64_t numeric_handle,
    py::array_t<std::complex<double>> b) {
    SymbolicEntry* sym_entry = get_symbolic_entry(symbolic_handle);
    if (sym_entry == nullptr) {
        throw std::runtime_error("Invalid symbolic handle");
    }

    NumericEntry* num_entry = get_numeric_entry(numeric_handle);
    if (num_entry == nullptr) {
        throw std::runtime_error("Invalid numeric handle");
    }

    auto b_buf = b.request();
    if (b_buf.ndim != 2) {
        throw std::runtime_error("b must be 2D (n_col, n_rhs) in column-major format");
    }

    int n_col = (int)b_buf.shape[0];
    int n_rhs = (int)b_buf.shape[1];

    if (n_col != sym_entry->n_col) {
        throw std::runtime_error("b.shape[0] != n_col from symbolic analysis");
    }

    // Allocate output array and copy b into it (KLU solves in-place)
    py::array_t<std::complex<double>> x({n_col, n_rhs});
    auto x_buf = x.request();
    double* _x = static_cast<double*>(x_buf.ptr);
    const double* _b = static_cast<const double*>(b_buf.ptr);
    std::memcpy(_x, _b, 2 * n_col * n_rhs * sizeof(double));

    klu_z_solve(sym_entry->symbolic, num_entry->numeric, n_col, n_rhs, _x, &sym_entry->common);

    if (sym_entry->common.status < KLU_OK) {
        throw std::runtime_error("klu_z_solve failed");
    }

    return x;
}

// Free a symbolic handle (and any associated resources)
void klu_free_symbolic_handle(int64_t handle) {
    if (!free_symbolic_handle(handle)) {
        throw std::runtime_error("Invalid symbolic handle");
    }
}

// Free a numeric handle
void klu_free_numeric_handle(int64_t handle) {
    if (!free_numeric_handle(handle)) {
        throw std::runtime_error("Invalid numeric handle");
    }
}

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
    // Low-level functions for advanced users (split-solve API via FFI)
    m.def("coo_to_csc",
          []() { return py::capsule((void*)&coo_to_csc_handler); });
    m.def("solve_csc_f64",
          []() { return py::capsule((void*)&solve_csc_f64_handler); });
    m.def("solve_csc_c128",
          []() { return py::capsule((void*)&solve_csc_c128_handler); });

    // True split-solve API: separate klu_analyze, klu_factor, klu_solve
    // These allow reusing symbolic analysis across multiple solves (for Newton-Raphson, etc.)
    m.def("klu_analyze", &klu_analyze_csc,
          "Perform symbolic analysis on CSC sparsity pattern. Returns a handle.",
          py::arg("Bp"), py::arg("Bi"));
    m.def("klu_factor_f64", &klu_factor_csc_f64,
          "Perform numeric factorization (f64). Returns a handle.",
          py::arg("symbolic_handle"), py::arg("Bp"), py::arg("Bi"), py::arg("Bx"));
    m.def("klu_factor_c128", &klu_factor_csc_c128,
          "Perform numeric factorization (c128). Returns a handle.",
          py::arg("symbolic_handle"), py::arg("Bp"), py::arg("Bi"), py::arg("Bx"));
    m.def("klu_solve_f64", &klu_solve_handles_f64,
          "Solve Ax=b using pre-computed factorizations (f64). b must be column-major (n_col, n_rhs).",
          py::arg("symbolic_handle"), py::arg("numeric_handle"), py::arg("b"));
    m.def("klu_solve_c128", &klu_solve_handles_c128,
          "Solve Ax=b using pre-computed factorizations (c128). b must be column-major (n_col, n_rhs).",
          py::arg("symbolic_handle"), py::arg("numeric_handle"), py::arg("b"));
    m.def("klu_free_symbolic", &klu_free_symbolic_handle,
          "Free a symbolic factorization handle.",
          py::arg("handle"));
    m.def("klu_free_numeric", &klu_free_numeric_handle,
          "Free a numeric factorization handle.",
          py::arg("handle"));
}
