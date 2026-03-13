#pragma once

#ifndef OPENCL_FUNCTIONS_HPP
#define OPENCL_FUNCTIONS_HPP

#include "device.h"

void matmul_impl(float *a, float *b, float *c, int m, int n, int k);

void conv2d_impl(
    const float * input_ptr,
    const float * weight_ptr, 
    const float * bias_ptr, 
    float * output_ptr, 
        
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
);

#endif