#include <iostream>
#include <vector>

#include <klu.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

void _coo_to_csc(int nnz, int n_col, double *Ax, int *Ai, int *Aj, double *Bx,
                 int *Bi, int *Bp) {

  // compute number of non-zero entries per row of A
  for (int n = 0; n < nnz; n++) {
    Bp[Aj[n]] += 1;
  }

  // cumsum the nnz per row to get Bp
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
  for (int n = 0; n < nnz; n++) {
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

void _klu_solve(int n_col, double *Ax, int *Ai, int *Ap, double *b) {
  klu_symbolic *Symbolic;
  klu_numeric *Numeric;
  klu_common Common;
  klu_defaults(&Common);
  Symbolic = klu_analyze(n_col, Ap, Ai, &Common);
  Numeric = klu_factor(Ap, Ai, Ax, Symbolic, &Common);
  klu_solve(Symbolic, Numeric, n_col, /*n_rhs=*/1, b, &Common);
  klu_free_symbolic(&Symbolic, &Common);
  klu_free_numeric(&Numeric, &Common);
}

void solve_f64(void *out, void **in) {
  // get args
  int nnz = *reinterpret_cast<int *>(in[0]);
  int n_col = *reinterpret_cast<int *>(in[1]);
  double *Ax = reinterpret_cast<double *>(in[2]);
  int *Ai = reinterpret_cast<int *>(in[3]);
  int *Aj = reinterpret_cast<int *>(in[4]);
  double *b = reinterpret_cast<double *>(in[5]);
  double *result = reinterpret_cast<double *>(out);

  // copy b into result
  for (int i = 0; i < n_col; i++) {
    result[i] = b[i];
  }

  // convert COO Ax, Ai, Ai to CSC Bx, Bi, Bp
  double *Bx = new double[nnz]();
  int *Bi = new int[nnz]();
  int *Bp = new int[n_col + 1]();

  _coo_to_csc(nnz, n_col, Ax, Ai, Aj, Bx, Bi, Bp);

  // solve using KLU
  _klu_solve(n_col, Bx, Bi, Bp, /*b=*/result);
}

PYBIND11_MODULE(klujax_cpp, m) {
  m.def(
      "solve_f64",
      []() {
        const char *name = "xla._CUSTOM_CALL_TARGET";
        return py::capsule((void *)&solve_f64, name);
      },
      "solve a linear system of equations");
}
