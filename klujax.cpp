// version: 0.2.10
// author: Floris Laporte

// Imports

#include "klu.h"
#include "pybind11/pybind11.h"
#include "xla/ffi/api/ffi.h"
#include <cmath>
namespace py = pybind11;
namespace ffi = xla::ffi;

// Helper functions

void _coo_to_csc_analyze(int n_col, int n_nz, int *Ai, int *Aj, int *Bi,
                         int *Bp, int *Bk) {
  // compute number of non-zero entries per row of A
  for (int n = 0; n < n_nz; n++) {
    Bp[Aj[n]] += 1;
  }

  // cumsum the n_nz per row to get Bp
  int cumsum = 0;
  int temp = 0;
  for (int j = 0; j < n_col; j++) {
    temp = Bp[j];
    Bp[j] = cumsum;
    cumsum += temp;
  }

  // write Ai, Ax into Bi, Bk
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

// Implementations

ffi::Error _solve_f64(ffi::Buffer<ffi::DataType::S32> buf_Ai,
                      ffi::Buffer<ffi::DataType::S32> buf_Aj,
                      ffi::Buffer<ffi::DataType::F64> buf_Ax,
                      ffi::Buffer<ffi::DataType::F64> buf_b,
                      ffi::Result<ffi::Buffer<ffi::DataType::F64>> buf_x) {

  // get args
  int *Ai = buf_Ai.typed_data();
  int *Aj = buf_Aj.typed_data();
  double *Ax = buf_Ax.typed_data();
  double *b = buf_b.typed_data();
  double *x = buf_x->typed_data();
  int n_col = (int)buf_x->dimensions()[1];
  int n_lhs = (int)buf_Ax.dimensions()[0];
  int n_rhs = (int)buf_x->dimensions()[2];
  int n_nz = (int)buf_Ax.dimensions()[1];

  // copy b into result
  for (int i = 0; i < n_lhs * n_col * n_rhs; i++) {
    x[i] = b[i];
  }

  // get COO -> CSC transformation information
  int *Bk = new int[n_nz](); // Ax -> Bx transformation indices
  int *Bi = new int[n_nz]();
  int *Bp = new int[n_col + 1]();
  _coo_to_csc_analyze(n_col, n_nz, Ai, Aj, Bi, Bp, Bk);

  // initialize KLU for given sparsity pattern
  klu_symbolic *Symbolic;
  klu_numeric *Numeric;
  klu_common Common;
  klu_defaults(&Common);
  Symbolic = klu_analyze(n_col, Bp, Bi, &Common);

  // solve for other elements in batch:
  // NOTE: same sparsity pattern for each element in batch assumed
  double *Bx = new double[n_nz]();
  for (int i = 0; i < n_lhs; i++) {
    int m = i * n_nz;
    int n = i * n_rhs * n_col;

    // convert COO Ax to CSC Bx
    for (int k = 0; k < n_nz; k++) {
      Bx[k] = Ax[m + Bk[k]];
    }

    // solve using KLU
    Numeric = klu_factor(Bp, Bi, Bx, Symbolic, &Common);
    klu_solve(Symbolic, Numeric, n_col, n_rhs, &x[n], &Common);
  }

  // clean up
  klu_free_symbolic(&Symbolic, &Common);
  klu_free_numeric(&Numeric, &Common);
  delete[] Bk;
  delete[] Bi;
  delete[] Bp;
  delete[] Bx;

  return ffi::Error::Success();
}

ffi::Error
_coo_mul_vec_f64(ffi::Buffer<ffi::DataType::S32> buf_Ai,
                 ffi::Buffer<ffi::DataType::S32> buf_Aj,
                 ffi::Buffer<ffi::DataType::F64> buf_Ax,
                 ffi::Buffer<ffi::DataType::F64> buf_x,
                 ffi::Result<ffi::Buffer<ffi::DataType::F64>> buf_b) {

  // get args
  int *Ai = buf_Ai.typed_data();
  int *Aj = buf_Aj.typed_data();
  double *Ax = buf_Ax.typed_data();
  double *x = buf_x.typed_data();
  double *b = buf_b->typed_data();
  int n_col = (int)buf_b->dimensions()[1];
  int n_lhs = (int)buf_Ax.dimensions()[0];
  int n_rhs = (int)buf_b->dimensions()[2];
  int n_nz = (int)buf_Ax.dimensions()[1];

  // initialize empty result
  for (int i = 0; i < n_lhs * n_col * n_rhs; i++) {
    b[i] = 0.0;
  }

  // fill result
  for (int i = 0; i < n_lhs; i++) {
    int m = i * n_nz;
    int n = i * n_rhs * n_col;
    for (int j = 0; j < n_rhs; j++) {
      for (int k = 0; k < n_nz; k++) {
        b[n + Ai[k] + j * n_col] += Ax[m + k] * x[n + Aj[k] + j * n_col];
      }
    }
  }
  return ffi::Error::Success();
}

ffi::Error _solve_c128(ffi::Buffer<ffi::DataType::S32> buf_Ai,
                       ffi::Buffer<ffi::DataType::S32> buf_Aj,
                       ffi::Buffer<ffi::DataType::C128> buf_Ax,
                       ffi::Buffer<ffi::DataType::C128> buf_b,
                       ffi::Result<ffi::Buffer<ffi::DataType::C128>> buf_x) {

  // get args
  int *Ai = buf_Ai.typed_data();
  int *Aj = buf_Aj.typed_data();
  double *Ax = (double *)buf_Ax.typed_data();
  double *b = (double *)buf_b.typed_data();
  double *x = (double *)buf_x->typed_data();
  int n_col = (int)buf_x->dimensions()[1];
  int n_lhs = (int)buf_Ax.dimensions()[0];
  int n_rhs = (int)buf_x->dimensions()[2];
  int n_nz = (int)buf_Ax.dimensions()[1];

  // copy b into result
  for (int i = 0; i < 2 * n_lhs * n_col * n_rhs; i++) {
    x[i] = b[i];
  }

  // get COO -> CSC transformation information
  int *Bk = new int[n_nz]();      // Ax -> Bx transformation indices
  int *Bi = new int[n_nz]();      // CSC row indices
  int *Bp = new int[n_col + 1](); // CSC column pointers
  _coo_to_csc_analyze(n_col, n_nz, Ai, Aj, Bi, Bp, Bk);

  // initialize KLU for given sparsity pattern
  klu_symbolic *Symbolic;
  klu_numeric *Numeric;
  klu_common Common;
  klu_defaults(&Common);
  Symbolic = klu_analyze(n_col, Bp, Bi, &Common);

  // solve for other elements in batch:
  // NOTE: same sparsity pattern for each element in batch assumed
  double *Bx = new double[2 * n_nz]();
  for (int i = 0; i < n_lhs; i++) {
    int m = 2 * i * n_nz;
    int n = 2 * i * n_rhs * n_col;

    // convert COO Ax to CSC Bx
    for (int k = 0; k < n_nz; k++) {
      Bx[2 * k] = Ax[m + 2 * Bk[k]];
      Bx[2 * k + 1] = Ax[m + 2 * Bk[k] + 1];
    }

    // solve using KLU
    Numeric = klu_z_factor(Bp, Bi, Bx, Symbolic, &Common);
    klu_z_solve(Symbolic, Numeric, n_col, n_rhs, &x[n], &Common);
  }

  // clean up
  klu_free_symbolic(&Symbolic, &Common);
  klu_free_numeric(&Numeric, &Common);
  delete[] Bk;
  delete[] Bi;
  delete[] Bp;
  delete[] Bx;

  return ffi::Error::Success();
}

ffi::Error
_coo_mul_vec_c128(ffi::Buffer<ffi::DataType::S32> buf_Ai,
                  ffi::Buffer<ffi::DataType::S32> buf_Aj,
                  ffi::Buffer<ffi::DataType::C128> buf_Ax,
                  ffi::Buffer<ffi::DataType::C128> buf_x,
                  ffi::Result<ffi::Buffer<ffi::DataType::C128>> buf_b) {

  // get args
  int *Ai = buf_Ai.typed_data();
  int *Aj = buf_Aj.typed_data();
  double *Ax = (double *)buf_Ax.typed_data();
  double *x = (double *)buf_x.typed_data();
  double *b = (double *)buf_b->typed_data();
  int n_col = (int)buf_b->dimensions()[1];
  int n_lhs = (int)buf_Ax.dimensions()[0];
  int n_rhs = (int)buf_b->dimensions()[2];
  int n_nz = (int)buf_Ax.dimensions()[1];

  // initialize empty result
  for (int i = 0; i < 2 * n_lhs * n_col * n_rhs; i++) {
    b[i] = 0.0;
  }
  // fill result
  for (int i = 0; i < n_lhs; i++) {
    int m = 2 * i * n_nz;
    int n = 2 * i * n_rhs * n_col;
    for (int j = 0; j < n_rhs; j++) {
      for (int k = 0; k < n_nz; k++) {
        b[n + 2 * (Ai[k] + j * n_col)] +=                    // real part
            Ax[m + 2 * k] * x[n + 2 * (Aj[k] + j * n_col)] - // real * real
            Ax[m + 2 * k + 1] *
                x[n + 2 * (Aj[k] + j * n_col) + 1];              // imag * imag
        b[n + 2 * (Ai[k] + j * n_col) + 1] +=                    // imag part
            Ax[m + 2 * k] * x[n + 2 * (Aj[k] + j * n_col) + 1] + // real * imag
            Ax[m + 2 * k + 1] * x[n + 2 * (Aj[k] + j * n_col)];  // imag * real
      }
    }
  }
  return ffi::Error::Success();
}

// XLA wrappers

XLA_FFI_DEFINE_HANDLER_SYMBOL( // A x = b
    solve_f64, _solve_f64,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::S32>>() // Ai
        .Arg<ffi::Buffer<ffi::DataType::S32>>() // Aj
        .Arg<ffi::Buffer<ffi::DataType::F64>>() // Ax
        .Arg<ffi::Buffer<ffi::DataType::F64>>() // b
        .Ret<ffi::Buffer<ffi::DataType::F64>>() // x
);

XLA_FFI_DEFINE_HANDLER_SYMBOL( // b = A x
    coo_mul_vec_f64, _coo_mul_vec_f64,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::S32>>() // Ai
        .Arg<ffi::Buffer<ffi::DataType::S32>>() // Aj
        .Arg<ffi::Buffer<ffi::DataType::F64>>() // Ax
        .Arg<ffi::Buffer<ffi::DataType::F64>>() // x
        .Ret<ffi::Buffer<ffi::DataType::F64>>() // b
);

XLA_FFI_DEFINE_HANDLER_SYMBOL( // A x = b
    solve_c128, _solve_c128,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // Ai
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // Aj
        .Arg<ffi::Buffer<ffi::DataType::C128>>() // Ax
        .Arg<ffi::Buffer<ffi::DataType::C128>>() // b
        .Ret<ffi::Buffer<ffi::DataType::C128>>() // x
);

XLA_FFI_DEFINE_HANDLER_SYMBOL( // b = A x
    coo_mul_vec_c128, _coo_mul_vec_c128,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // Ai
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // Aj
        .Arg<ffi::Buffer<ffi::DataType::C128>>() // Ax
        .Arg<ffi::Buffer<ffi::DataType::C128>>() // x
        .Ret<ffi::Buffer<ffi::DataType::C128>>() // b
);

// Python wrappers

PYBIND11_MODULE(klujax_cpp, m) {
  m.def("solve_f64", []() { return py::capsule((void *)&solve_f64); });
  m.def("coo_mul_vec_f64",
        []() { return py::capsule((void *)&coo_mul_vec_f64); });
  m.def("solve_c128", []() { return py::capsule((void *)&solve_c128); });
  m.def("coo_mul_vec_c128",
        []() { return py::capsule((void *)&coo_mul_vec_c128); });
}
