#include <pybind11/pybind11.h>

namespace py = pybind11;

float solve(float A, float b){
  return b/A;
}

PYBIND11_MODULE(klujax_cpp, m) {
    m.def("solve", &solve, "solve a linear system of equations");
}
