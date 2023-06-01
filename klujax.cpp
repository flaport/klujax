// version: 0.1.4
// author: Floris Laporte

#include <iostream>
#include <vector>

#include <klu.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

void coo_to_csc_analyze(int n_col, int n_nz, int *Ai, int *Aj, int *Bi, int *Bp,
                        int *Bk) {

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

void solve_f64(void *out, void **in) {
  // get args
  int n_col = *reinterpret_cast<int *>(in[0]);
  int n_lhs = *reinterpret_cast<int *>(in[1]);
  int n_rhs = *reinterpret_cast<int *>(in[2]);
  int Anz = *reinterpret_cast<int *>(in[3]);
  int *Ai = reinterpret_cast<int *>(in[4]);
  int *Aj = reinterpret_cast<int *>(in[5]);
  double *Ax = reinterpret_cast<double *>(in[6]);
  double *b = reinterpret_cast<double *>(in[7]);
  double *result = reinterpret_cast<double *>(out);

  // copy b into result
  for (int i = 0; i < n_lhs * n_col * n_rhs; i++) {
    result[i] = b[i];
  }

  // get COO -> CSC transformation information
  int *Bk = new int[Anz](); // Ax -> Bx transformation indices
  int *Bi = new int[Anz]();
  int *Bp = new int[n_col + 1]();
  coo_to_csc_analyze(n_col, Anz, Ai, Aj, Bi, Bp, Bk);

  // initialize KLU for given sparsity pattern
  klu_symbolic *Symbolic;
  klu_numeric *Numeric;
  klu_common Common;
  klu_defaults(&Common);
  Symbolic = klu_analyze(n_col, Bp, Bi, &Common);

  // solve for other elements in batch:
  // NOTE: same sparsity pattern for each element in batch assumed
  double *Bx = new double[Anz]();
  for (int i = 0; i < n_lhs; i++) {
    int m = i * Anz;
    int n = i * n_rhs * n_col;

    // convert COO Ax to CSC Bx
    for (int k = 0; k < Anz; k++) {
      Bx[k] = Ax[m + Bk[k]];
    }

    // solve using KLU
    Numeric = klu_factor(Bp, Bi, Bx, Symbolic, &Common);
    klu_solve(Symbolic, Numeric, n_col, n_rhs, &result[n], &Common);
  }

  // clean up
  klu_free_symbolic(&Symbolic, &Common);
  klu_free_numeric(&Numeric, &Common);
}

void solve_c128(void *out, void **in) {
  // get args
  int n_col = *reinterpret_cast<int *>(in[0]);
  int n_lhs = *reinterpret_cast<int *>(in[1]);
  int n_rhs = *reinterpret_cast<int *>(in[2]);
  int Anz = *reinterpret_cast<int *>(in[3]);
  int *Ai = reinterpret_cast<int *>(in[4]);
  int *Aj = reinterpret_cast<int *>(in[5]);
  double *Ax = reinterpret_cast<double *>(in[6]);
  double *b = reinterpret_cast<double *>(in[7]);
  double *result = reinterpret_cast<double *>(out);

  // copy b into result
  for (int i = 0; i < 2 * n_lhs * n_col * n_rhs; i++) {
    result[i] = b[i];
  }

  // get COO -> CSC transformation information
  int *Bk = new int[Anz]();       // Ax -> Bx transformation indices
  int *Bi = new int[Anz]();       // CSC row indices
  int *Bp = new int[n_col + 1](); // CSC column pointers
  coo_to_csc_analyze(n_col, Anz, Ai, Aj, Bi, Bp, Bk);

  // initialize KLU for given sparsity pattern
  klu_symbolic *Symbolic;
  klu_numeric *Numeric;
  klu_common Common;
  klu_defaults(&Common);
  Symbolic = klu_analyze(n_col, Bp, Bi, &Common);

  // solve for other elements in batch:
  // NOTE: same sparsity pattern for each element in batch assumed
  double *Bx = new double[2 * Anz]();
  for (int i = 0; i < n_lhs; i++) {
    int m = 2 * i * Anz;
    int n = 2 * i * n_rhs * n_col;

    // convert COO Ax to CSC Bx
    for (int k = 0; k < Anz; k++) {
      Bx[2 * k] = Ax[m + 2 * Bk[k]];
      Bx[2 * k + 1] = Ax[m + 2 * Bk[k] + 1];
    }

    // solve using KLU
    Numeric = klu_z_factor(Bp, Bi, Bx, Symbolic, &Common);
    klu_z_solve(Symbolic, Numeric, n_col, n_rhs, &result[n], &Common);
  }

  // clean up
  klu_free_symbolic(&Symbolic, &Common);
  klu_free_numeric(&Numeric, &Common);
}

void coo_mul_vec_f64(void *out, void **in) {
  // get args
  int n_col = *reinterpret_cast<int *>(in[0]);
  int n_lhs = *reinterpret_cast<int *>(in[1]);
  int n_rhs = *reinterpret_cast<int *>(in[2]);
  int Anz = *reinterpret_cast<int *>(in[3]);
  int *Ai = reinterpret_cast<int *>(in[4]);
  int *Aj = reinterpret_cast<int *>(in[5]);
  double *Ax = reinterpret_cast<double *>(in[6]);
  double *b = reinterpret_cast<double *>(in[7]);
  double *result = reinterpret_cast<double *>(out);

  // initialize empty result
  for (int i = 0; i < n_lhs * n_col * n_rhs; i++) {
    result[i] = 0.0;
  }

  // fill result
  for (int i = 0; i < n_lhs; i++) {
    int m = i * Anz;
    int n = i * n_rhs * n_col;
    for (int j = 0; j < n_rhs; j++) {
      for (int k = 0; k < Anz; k++) {
        result[n + Ai[k] + j * n_col] += Ax[m + k] * b[n + Aj[k] + j * n_col];
      }
    }
  }
}

void coo_mul_vec_c128(void *out, void **in) {
  // get args
  int n_col = *reinterpret_cast<int *>(in[0]);
  int n_lhs = *reinterpret_cast<int *>(in[1]);
  int n_rhs = *reinterpret_cast<int *>(in[2]);
  int Anz = *reinterpret_cast<int *>(in[3]);
  int *Ai = reinterpret_cast<int *>(in[4]);
  int *Aj = reinterpret_cast<int *>(in[5]);
  double *Ax = reinterpret_cast<double *>(in[6]);
  double *b = reinterpret_cast<double *>(in[7]);
  double *result = reinterpret_cast<double *>(out);

  // initialize empty result
  for (int i = 0; i < 2 * n_lhs * n_col * n_rhs; i++) {
    result[i] = 0.0;
  }

  // fill result
  for (int i = 0; i < n_lhs; i++) {
    int m = 2 * i * Anz;
    int n = 2 * i * n_rhs * n_col;
    for (int j = 0; j < n_rhs; j++) {
      for (int k = 0; k < Anz; k++) {
        result[n + 2 * (Ai[k] + j * n_col)] +=               // real part
            Ax[m + 2 * k] * b[n + 2 * (Aj[k] + j * n_col)] - // real * real
            Ax[m + 2 * k + 1] *
                b[n + 2 * (Aj[k] + j * n_col) + 1];              // imag * imag
        result[n + 2 * (Ai[k] + j * n_col) + 1] +=               // imag part
            Ax[m + 2 * k] * b[n + 2 * (Aj[k] + j * n_col) + 1] + // real * imag
            Ax[m + 2 * k + 1] * b[n + 2 * (Aj[k] + j * n_col)];  // imag * real
      }
    }
  }
}

PYBIND11_MODULE(klujax_cpp, m) {
  m.def(
      "solve_f64",
      []() {
        const char *name = "xla._CUSTOM_CALL_TARGET";
        return py::capsule((void *)&solve_f64, name);
      },
      "solve a real-valued linear system of equations");
  m.def(
      "solve_c128",
      []() {
        const char *name = "xla._CUSTOM_CALL_TARGET";
        return py::capsule((void *)&solve_c128, name);
      },
      "solve a complex-valued linear system of equations");
  m.def(
      "coo_mul_vec_f64",
      []() {
        const char *name = "xla._CUSTOM_CALL_TARGET";
        return py::capsule((void *)&coo_mul_vec_f64, name);
      },
      "Multiply a real-valued COO sparse matrix with a vector.");
  m.def(
      "coo_mul_vec_c128",
      []() {
        const char *name = "xla._CUSTOM_CALL_TARGET";
        return py::capsule((void *)&coo_mul_vec_c128, name);
      },
      "Multiply a complex-valued COO sparse matrix with a vector.");
}
