#include <iostream>
#include <vector>

#include <klu.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

void _coo_to_csc(int n_nz, int n_col, double *Ax, int *Ai, int *Aj, double *Bx,
                 int *Bi, int *Bp) {

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

void _coo_z_to_csc_z(int n_nz, int n_col, double *Ax, int *Ai, int *Aj,
                     double *Bx, int *Bi, int *Bp) {

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

void _klu_solve(int n_col, int n_rhs, double *Ax, int *Ai, int *Ap, double *b) {
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

void _klu_z_solve(int n_col, int n_rhs, double *Ax, int *Ai, int *Ap,
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
  int n_nz = *reinterpret_cast<int *>(in[0]);
  int n_col = *reinterpret_cast<int *>(in[1]);
  int n_rhs = *reinterpret_cast<int *>(in[2]);
  double *Ax = reinterpret_cast<double *>(in[3]);
  int *Ai = reinterpret_cast<int *>(in[4]);
  int *Aj = reinterpret_cast<int *>(in[5]);
  double *b = reinterpret_cast<double *>(in[6]);
  double *result = reinterpret_cast<double *>(out);

  // copy b into result
  for (int i = 0; i < n_col * n_rhs; i++) {
    result[i] = b[i];
  }

  // convert COO Ax, Ai, Ai to CSC Bx, Bi, Bp
  double *Bx = new double[n_nz]();
  int *Bi = new int[n_nz]();
  int *Bp = new int[n_col + 1]();
  _coo_to_csc(n_nz, n_col, Ax, Ai, Aj, Bx, Bi, Bp);

  // solve using KLU
  _klu_solve(n_col, n_rhs, Bx, Bi, Bp, /*b=*/result);
}

void solve_c128(void *out, void **in) {
  // get args
  int n_nz = *reinterpret_cast<int *>(in[0]);
  int n_col = *reinterpret_cast<int *>(in[1]);
  int n_rhs = *reinterpret_cast<int *>(in[2]);
  double *Ax = reinterpret_cast<double *>(in[3]);
  int *Ai = reinterpret_cast<int *>(in[4]);
  int *Aj = reinterpret_cast<int *>(in[5]);
  double *b = reinterpret_cast<double *>(in[6]);
  double *result = reinterpret_cast<double *>(out);

  // copy b into result
  for (int i = 0; i < 2 * n_col * n_rhs; i++) {
    result[i] = b[i];
  }

  // convert COO Ax, Ai, Ai to CSC Bx, Bi, Bp
  double *Bx = new double[2 * n_nz]();
  int *Bi = new int[n_nz]();
  int *Bp = new int[n_col + 1]();
  _coo_z_to_csc_z(n_nz, n_col, Ax, Ai, Aj, Bx, Bi, Bp);

  // solve using KLU
  _klu_z_solve(n_col, n_rhs, Bx, Bi, Bp, /*b=*/result);
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
}
