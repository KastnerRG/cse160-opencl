#include "opencl-functions.hpp"
#include <torch/extension.h>


torch::Tensor ocl_softmax(const torch::Tensor& a) {
    auto a_cpu = a.contiguous().to(torch::kCPU);
    float *a_ptr = a_cpu.data_ptr<float>();

    int m = a.size(0);
    int n = a.size(1);
    float *b_ptr = new float[m * n];

    //@@ Call Softmax Impl
    softmax_impl(a_ptr, b_ptr, m, n);
    torch::Tensor b = torch::empty({m, n}, torch::kFloat32);
    std::memcpy(b.data_ptr<float>(), b_ptr, m * n * sizeof(float));
    return b.to(a.device());
}

PYBIND11_MODULE(opencl_functions, m) {
    m.def("ocl_softmax", &ocl_softmax, "Custom Softmax");
    // m.def("ocl_matmul", &ocl_matmul, "Custom Matrix Multiplication");
    // m.def("ocl_conv2d", &ocl_conv2d, "Custom Convolution");
}
