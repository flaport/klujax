#include <pybind11/pybind11.h>

namespace py = pybind11;

void solve_f32(void* out, void** in) {
  float* A = ((float*) in[0]);
  float* b = ((float*) in[1]);
  float* result = (float*) out;
  int length = sizeof(&b) / sizeof(b[0]);
  for (int i=0; i<length; i++){
    result[i] = b[i] / A[i];
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
