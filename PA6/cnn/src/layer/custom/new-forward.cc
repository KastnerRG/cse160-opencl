#include <cmath>
#include <iostream>

#include "kernel.h"
#include "device.h"

#include "gpu-new-forward.h"

#define TILE_WIDTH 16

#define CHECK_ERR(err, msg)                           \
    if (err != CL_SUCCESS)                            \
    {                                                 \
        fprintf(stderr, "%s failed: %d.\n", msg, err); \
        exit(EXIT_FAILURE);                           \
    }
	
void GPUInterface::conv_forward_gpu_prolog(const float *host_y, const float *host_x, const float *host_k, cl_mem *device_y, cl_mem *device_x, cl_mem *device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    cl_int err;

    //@@ Allocate GPU memory here
    // Create memory buffers for input and output vectors
    *device_x = clCreateBuffer(gpu->context,
                              CL_MEM_READ_ONLY,
                              B * C * H * W * sizeof(float),
                              nullptr,
                              &err);
    CHECK_ERR(err, "clCreateBuffer device_x");

    *device_k = clCreateBuffer(gpu->context,
                              CL_MEM_READ_ONLY,
                              M * C * K * K * sizeof(float),
                              nullptr,
                              &err);
    CHECK_ERR(err, "clCreateBuffer device_k");

    *device_y = clCreateBuffer(gpu->context,
                              CL_MEM_READ_ONLY,
                              B * M * (H - K + 1) * (W - K + 1) * sizeof(float),
                              nullptr,
                              &err);
    CHECK_ERR(err, "clCreateBuffer device_y");

    //@@ Copy memory to the GPU here
    // Copy input vectors to memory buffers
    err = clEnqueueWriteBuffer(gpu->queue,
                               *device_x,
                               CL_TRUE,
                               0,
                               B * C * H * W * sizeof(float),
                               host_x,
                               0,
                               nullptr,
                               nullptr);
    CHECK_ERR(err, "clEnqueueWriteBuffer device_a");

    err = clEnqueueWriteBuffer(gpu->queue,
                               *device_k,
                               CL_TRUE,
                               0,
                               M * C * K * K * sizeof(float),
                               host_k,
                               0,
                               nullptr,
                               nullptr);
    CHECK_ERR(err, "clEnqueueWriteBuffer device_b");
}


void GPUInterface::conv_forward_gpu(cl_mem device_y, const cl_mem device_x, const cl_mem device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    cl_int err;

    //__global float *y, __constant float *x, __constant float *k,
    // const int B, const int M, const int C, const int H, const int W, const int K)
    // Set the arguments to our compute kernel
    err = clSetKernelArg(gpu->kernel, 0, sizeof(cl_mem), &device_y);
    CHECK_ERR(err, "clSetKernelArg 0");
    err = clSetKernelArg(gpu->kernel, 1, sizeof(cl_mem), &device_x);
    CHECK_ERR(err, "clSetKernelArg 1");
    err = clSetKernelArg(gpu->kernel, 2, sizeof(cl_mem), &device_k);
    CHECK_ERR(err, "clSetKernelArg 2");
    err = clSetKernelArg(gpu->kernel, 3, sizeof(int), &B);
    CHECK_ERR(err, "clSetKernelArg 3");
    err = clSetKernelArg(gpu->kernel, 4, sizeof(int), &M);
    CHECK_ERR(err, "clSetKernelArg 4");
    err = clSetKernelArg(gpu->kernel, 5, sizeof(int), &C);
    CHECK_ERR(err, "clSetKernelArg 5");
    err = clSetKernelArg(gpu->kernel, 6, sizeof(int), &H);
    CHECK_ERR(err, "clSetKernelArg 6");
    err = clSetKernelArg(gpu->kernel, 7, sizeof(int), &W);
    CHECK_ERR(err, "clSetKernelArg 7");
    err = clSetKernelArg(gpu->kernel, 8, sizeof(int), &K);
    CHECK_ERR(err, "clSetKernelArg 8");

    //@@ Set the kernel dimensions and call the kernel
    int H_out = H - K + 1;
    int W_out = W - K + 1;

    size_t W_grid = (W_out + TILE_WIDTH - 1) / TILE_WIDTH;
    size_t H_grid = (H_out + TILE_WIDTH - 1) / TILE_WIDTH;

    size_t global_item_size[3] = {(size_t)M * TILE_WIDTH, W_grid * H_grid * TILE_WIDTH, (size_t)B}; 
    size_t local_item_size[3] = {TILE_WIDTH, TILE_WIDTH, 1};

    //@@ Launch the GPU Kernel here
    // Execute the OpenCL kernel on the array
    err = clEnqueueNDRangeKernel(gpu->queue, gpu->kernel, 3, nullptr, global_item_size, local_item_size, 0, nullptr, nullptr);
    CHECK_ERR(err, "clEnqueueNDRangeKernel");
}


void GPUInterface::conv_forward_gpu_epilog(float *host_y, cl_mem device_y, cl_mem device_x, cl_mem device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    cl_int err;

    //@@ Copy the output back to host
    int H_out = H - K + 1;
    int W_out = W - K + 1;

    // Read the memory buffer output_mem_obj to the local variable result
    err = clEnqueueReadBuffer(gpu->queue, device_y, CL_TRUE, 0, B * M * H_out * W_out * sizeof(float), host_y, 0, nullptr, nullptr);
    CHECK_ERR(err, "clEnqueueReadBuffer");

    //@@ Free the GPU memory here
    // Release OpenCL resources
    clReleaseMemObject(device_x);
    clReleaseMemObject(device_y);
    clReleaseMemObject(device_k);
}
