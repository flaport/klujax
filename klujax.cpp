#include <iostream>
#include <vector>

#include <klu.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

void _coo_to_csc(int n_col, int n_nz, int *Ai, int *Aj, double *Ax, int *Bi,
                 int *Bp, double *Bx) {

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

  // write Ai, Ax into Bi, Bx
  int col = 0;
  int dest = 0;
  for (int n = 0; n < n_nz; n++) {
    col = Aj[n];
    dest = Bp[col];
    Bi[dest] = Ai[n];
    Bx[dest] = Ax[n];
    Bp[col] += 1;
  }

  int last = 0;
  for (int i = 0; i <= n_col; i++) {
    temp = Bp[i];
    Bp[i] = last;
    last = temp;
  }
}

void _coo_z_to_csc_z(int n_col, int n_nz, int *Ai, int *Aj, double *Ax, int *Bi,
                     int *Bp, double *Bx) {

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

  // write Ai, Ax into Bi, Bx
  int col = 0;
  int dest = 0;
  for (int n = 0; n < n_nz; n++) {
    col = Aj[n];
    dest = Bp[col];
    Bi[dest] = Ai[n];
    Bx[2 * dest] = Ax[2 * n];
    Bx[2 * dest + 1] = Ax[2 * n + 1];
    Bp[col] += 1;
  }

  int last = 0;
  for (int i = 0; i <= n_col; i++) {
    temp = Bp[i];
    Bp[i] = last;
    last = temp;
  }
}

void _klu_solve(int n_col, int n_rhs, int *Ai, int *Ap, double *Ax, double *b) {
  klu_symbolic *Symbolic;
  klu_numeric *Numeric;
  klu_common Common;
  klu_defaults(&Common);
  Symbolic = klu_analyze(n_col, Ap, Ai, &Common);
  Numeric = klu_factor(Ap, Ai, Ax, Symbolic, &Common);
  klu_solve(Symbolic, Numeric, n_col, n_rhs, b, &Common);
  klu_free_symbolic(&Symbolic, &Common);
  klu_free_numeric(&Numeric, &Common);
}

void _klu_z_solve(int n_col, int n_rhs, int *Ai, int *Ap, double *Ax,
                  double *b) {
  klu_symbolic *Symbolic;
  klu_numeric *Numeric;
  klu_common Common;
  klu_defaults(&Common);
  Symbolic = klu_analyze(n_col, Ap, Ai, &Common);
  Numeric = klu_z_factor(Ap, Ai, Ax, Symbolic, &Common);
  klu_z_solve(Symbolic, Numeric, n_col, n_rhs, b, &Common);
  klu_free_symbolic(&Symbolic, &Common);
  klu_free_numeric(&Numeric, &Common);
}

void solve_f64(void *out, void **in) {
  // get args
  int n_col = *reinterpret_cast<int *>(in[0]);
  int n_rhs = *reinterpret_cast<int *>(in[1]);
  int Anz = *reinterpret_cast<int *>(in[2]);
  int *Ai = reinterpret_cast<int *>(in[3]);
  int *Aj = reinterpret_cast<int *>(in[4]);
  double *Ax = reinterpret_cast<double *>(in[5]);
  double *b = reinterpret_cast<double *>(in[6]);
  double *result = reinterpret_cast<double *>(out);

  // copy b into result
  for (int i = 0; i < n_col * n_rhs; i++) {
    result[i] = b[i];
  }

  // convert COO Ax, Ai, Ai to CSC Bx, Bi, Bp
  double *Bx = new double[Anz]();
  int *Bi = new int[Anz]();
  int *Bp = new int[n_col + 1]();
  _coo_to_csc(n_col, Anz, Ai, Aj, Ax, Bi, Bp, Bx);

  // solve using KLU
  _klu_solve(n_col, n_rhs, Bi, Bp, Bx, /*b=*/result);
}

void solve_c128(void *out, void **in) {
  // get args
  int n_col = *reinterpret_cast<int *>(in[0]);
  int n_rhs = *reinterpret_cast<int *>(in[1]);
  int Anz = *reinterpret_cast<int *>(in[2]);
  int *Ai = reinterpret_cast<int *>(in[3]);
  int *Aj = reinterpret_cast<int *>(in[4]);
  double *Ax = reinterpret_cast<double *>(in[5]);
  double *b = reinterpret_cast<double *>(in[6]);
  double *result = reinterpret_cast<double *>(out);

  // copy b into result
  for (int i = 0; i < 2 * n_col * n_rhs; i++) {
    result[i] = b[i];
  }

  // convert COO Ax, Ai, Ai to CSC Bx, Bi, Bp
  double *Bx = new double[2 * Anz]();
  int *Bi = new int[Anz]();
  int *Bp = new int[n_col + 1]();
  _coo_z_to_csc_z(n_col, Anz, Ai, Aj, Ax, Bi, Bp, Bx);

  // solve using KLU
  _klu_z_solve(n_col, n_rhs, Bi, Bp, Bx, /*b=*/result);
}

void mul_coo_vec_f64(void *out, void **in) {
  // get args
  int n_col = *reinterpret_cast<int *>(in[0]);
  int n_rhs = *reinterpret_cast<int *>(in[1]);
  int Anz = *reinterpret_cast<int *>(in[2]);
  int *Ai = reinterpret_cast<int *>(in[3]);
  int *Aj = reinterpret_cast<int *>(in[4]);
  double *Ax = reinterpret_cast<double *>(in[5]);
  double *b = reinterpret_cast<double *>(in[6]);
  double *result = reinterpret_cast<double *>(out);

  // initialize empty result
  for (int k = 0; k < n_col * n_rhs; k++) {
    result[k] = 0.0;
  }

  // fill result
  for (int l = 0; l < n_rhs; l++) {
    for (int k = 0; k < Anz; k++) {
      result[Ai[k] + l * n_col] += Ax[k] * b[Aj[k] + l * n_col];
    }
  }
}

void mul_coo_vec_c128(void *out, void **in) {
  // get args
  int n_col = *reinterpret_cast<int *>(in[0]);
  int n_rhs = *reinterpret_cast<int *>(in[1]);
  int Anz = *reinterpret_cast<int *>(in[2]);
  int *Ai = reinterpret_cast<int *>(in[3]);
  int *Aj = reinterpret_cast<int *>(in[4]);
  double *Ax = reinterpret_cast<double *>(in[5]);
  double *b = reinterpret_cast<double *>(in[6]);
  double *result = reinterpret_cast<double *>(out);

  // initialize empty result
  for (int k = 0; k < 2 * n_col * n_rhs; k++) {
    result[k] = 0.0;
  }

  // fill result
  for (int l = 0; l < n_rhs; l++) {
    for (int k = 0; k < Anz; k++) {
      result[2 * (Ai[k] + l * n_col)] +=                  // real part
          Ax[2 * k] * b[2 * (Aj[k] + l * n_col)] -        // real * real
          Ax[2 * k + 1] * b[2 * (Aj[k] + l * n_col) + 1]; // imag * imag
      result[2 * (Ai[k] + l * n_col) + 1] +=              // imag part
          Ax[2 * k] * b[2 * (Aj[k] + l * n_col) + 1] +    // real * imag
          Ax[2 * k + 1] * b[2 * (Aj[k] + l * n_col)];     // imag * real
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
      "mul_coo_vec_f64",
      []() {
        const char *name = "xla._CUSTOM_CALL_TARGET";
        return py::capsule((void *)&mul_coo_vec_f64, name);
      },
      "matmul of real-valued sparse COO matrix with dense vector");
  m.def(
      "mul_coo_vec_c128",
      []() {
        const char *name = "xla._CUSTOM_CALL_TARGET";
        return py::capsule((void *)&mul_coo_vec_c128, name);
      },
      "matmul of real-valued sparse COO matrix with dense vector");
}
