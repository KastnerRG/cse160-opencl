#include "opencl-functions.hpp"
#include <torch/extension.h>


torch::Tensor ocl_matmul(const torch::Tensor& a, const torch::Tensor& b) {
    // Check if the input tensors are 2D
    TORCH_CHECK(a.dim() == 2, "Input tensor 'a' must be 2D");
    TORCH_CHECK(b.dim() == 2, "Input tensor 'b' must be 2D");
    
    // Check if the inner dimensions match
    TORCH_CHECK(a.size(1) == b.size(0), "Inner dimensions of 'a' and 'b' must match");

    auto a_cpu = a.contiguous().to(torch::kCPU); // note, doesnt always work without contiguous, also moving to target device, target based on the matmul code
    auto b_cpu = b.contiguous().to(torch::kCPU); // note, doesnt always work without contiguous, also moving to target device, target based on the matmul code

    float *a_ptr = a_cpu.data_ptr<float>();
    float *b_ptr = b_cpu.data_ptr<float>();
    int m = a.size(0);
    int n = b.size(1);
    int k = a.size(1);
    float *c_ptr = new float[m * n];
    //@@ Call matmul_impl
    // matmul_impl(a_ptr, b_ptr, c_ptr, m, n, k); //uncomment
    
    // Create a new tensor for the result
    torch::Tensor c = torch::empty({m, n}, torch::kFloat32);
    // Copy the result into the new tensor
    std::memcpy(c.data_ptr<float>(), c_ptr, m * n * sizeof(float));
    delete[] c_ptr;
    return c.to(a.device());
}


torch::Tensor ocl_conv2d(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const int strideH,
    const int strideW,
    const int dilationH,
    const int dilationW,
    const int groups
) {
    // Check if the input tensors are 2D
    TORCH_CHECK(input.dim() == 4, "Input tensor 'a' must be 4D");

    // note, doesnt always work without contiguous, also moving to target device
    auto input_cpu = input.contiguous().to(torch::kCPU); 
    auto weight_cpu = weight.contiguous().to(torch::kCPU); 
    auto bias_cpu = bias.contiguous().to(torch::kCPU); 

    float *input_ptr = input_cpu.data_ptr<float>();
    float *weight_ptr = weight_cpu.data_ptr<float>();
    float *bias_ptr = bias_cpu.data_ptr<float>();

    int B = input_cpu.size(0);
    int C_in = input_cpu.size(1);
    int H = input_cpu.size(2);
    int W = input_cpu.size(3);

    int C_out = weight_cpu.size(0);
    int k_h = weight_cpu.size(2);
    int k_w = weight_cpu.size(3);

    int H_out = (H - dilationH * (k_h - 1) - 1)/strideH + 1;
    int W_out = (W - dilationW * (k_w - 1) - 1)/strideW + 1;

    float *output_ptr = new float[B * C_out * H_out * W_out];

    conv2d_impl(
        input_ptr, weight_ptr, bias_ptr, output_ptr, 
        B, C_in, H, W, C_out, k_h, k_w,
        strideH, strideW, dilationH, dilationW
    );

    // Create a new tensor for the result
    torch::Tensor c = torch::empty({B, C_out, H_out, W_out}, torch::kFloat32);
    // Copy the result into the new tensor
    std::memcpy(c.data_ptr<float>(), output_ptr, B * C_out * H_out * W_out * sizeof(float));
    delete[] output_ptr;
    return c.to(input.device());
}

PYBIND11_MODULE(opencl_functions, m) {
    m.def("ocl_matmul", &ocl_matmul, "Custom Matrix Multiplication");
    m.def("ocl_conv2d", &ocl_conv2d, "Custom Convolution");
}