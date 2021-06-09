#include <iostream>
#include <vector>

#include <klu.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

void _coo_to_csc(int n_col, int n_nz, int *Ai, int *Aj, double *Ax, int *Bi,
                 int *Bp, double *Bx) {

  // fill Bp with zeros
  for (int n = 0; n < n_col + 1; n++) {
    Bp[n] = 0.0;
  }

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

  // CSC sparse matrix initialization
  double *Bx = new double[Anz]();
  int *Bi = new int[Anz]();
  int *Bp = new int[n_col + 1]();

  // initialize KLU and solve for first element in batch:

  // convert COO Ax, Ai, Ai to CSC Bx, Bi, Bp
  _coo_to_csc(n_col, Anz, Ai, Aj, &Ax[0], Bi, Bp, Bx);

  // solve using KLU
  klu_symbolic *Symbolic;
  klu_numeric *Numeric;
  klu_common Common;
  klu_defaults(&Common);
  Symbolic = klu_analyze(n_col, Bp, Bi, &Common);
  Numeric = klu_factor(Bp, Bi, Bx, Symbolic, &Common);
  klu_solve(Symbolic, Numeric, n_col, n_rhs, &result[0], &Common);

  // solve for other elements in batch:
  // NOTE: same sparsity pattern for each element in batch assumed
  for (int i = 1; i < n_lhs; i++) {
    int m = i * Anz;
    int n = i * n_rhs * n_col;

    // convert COO Ax, Ai, Ai to CSC Bx, Bi, Bp
    _coo_to_csc(n_col, Anz, Ai, Aj, &Ax[m], Bi, Bp, Bx);

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
