#include <torch/python.h>

int test(int n) {
    return n + 2;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("test", &test, "(CUDA)");
}