
__kernel void conv2d(
    __global const float* input,
    __global const float* weight,
    __global const float* bias,
    __global float* output,
    const int B,
    const int C_in,
    const int H,
    const int W,
    const int C_out,
    const int k_h,
    const int k_w,
    const int strideH,
    const int strideW,
    const int dilationH,
    const int dilationW
)
{
    //@@ Implemented Convolution to replace PyTorch's Conv2d
    
}