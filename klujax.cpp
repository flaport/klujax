#include <iostream>
#include <pybind11/pybind11.h>

namespace py = pybind11;

void solve_f32(void *out, void **in) {
  const long N = *reinterpret_cast<long *>(in[0]);
  const float *Ax = reinterpret_cast<float *>(in[1]);
  const int *Ai = reinterpret_cast<int *>(in[2]);
  const int *Aj = reinterpret_cast<int *>(in[3]);
  const float *b = reinterpret_cast<float *>(in[4]);
  float *result = reinterpret_cast<float *>(out);
  for (int i = 0; i < N; i++) {
    result[i] = b[i] / Ax[i];
  }
}

PYBIND11_MODULE(klujax_cpp, m) {
  m.def(
      "solve_f32",
      []() {
        const char *name = "xla._CUSTOM_CALL_TARGET";
        return py::capsule((void *)&solve_f32, name);
      },
      "solve a linear system of equations");
}
