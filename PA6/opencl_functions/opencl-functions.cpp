#include "opencl-functions.hpp"
#include <iostream>
#include <clblast.h>


#include "kernel.h"
#include "device.h"

#define TILE_WIDTH 16

#define CHECK_ERR(err, msg)                           \
    if (err != CL_SUCCESS)                            \
    {                                                 \
        fprintf(stderr, "%s failed: %d\n", msg, err); \
        exit(EXIT_FAILURE);                           \
    }

void matmul_impl(float *a, float *b, float *c, int m, int n, int k) {

    cl_int err;
    int platform_id;
    int device_index;         
    cl_device_id device_id;   
    cl_context context;       
    cl_command_queue queue;   
    cl_program program;       
    cl_kernel kernel;         

    // Find platforms and devices
    OclPlatformProp *platforms = NULL;
    cl_uint num_platforms;

    err = OclFindPlatforms((const OclPlatformProp **)&platforms, &num_platforms);
    CHECK_ERR(err, "OclFindPlatforms");

    err = OclGetDeviceInfoWithFallback(&device_id, &platform_id, &device_index, OCL_DEVICE_TYPE);
    CHECK_ERR(err, "OclGetDeviceWithFallback");

    // as there maybe many devices that get tested for this PA
    printf("Running with %s: %s\n", platforms[platform_id].name, platforms[platform_id].devices[device_index].name);

    // Create a context
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    CHECK_ERR(err, "clCreateContext");

    // Create a command queue
    queue = clCreateCommandQueueWithProperties(context, device_id, 0, &err);
    CHECK_ERR(err, "clCreateCommandQueueWithProperties");

    // Create device buffers
    cl_mem a_buf = clCreateBuffer(context, CL_MEM_READ_ONLY,  sizeof(float) * m * k, NULL, &err);
    CHECK_ERR(err, "clCreateBuffer a");
    cl_mem b_buf = clCreateBuffer(context, CL_MEM_READ_ONLY,  sizeof(float) * n * k, NULL, &err);
    CHECK_ERR(err, "clCreateBuffer b");
    cl_mem c_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * m * n, NULL, &err);
    CHECK_ERR(err, "clCreateBuffer c");

    // Copy host data to device
    err = clEnqueueWriteBuffer(queue, a_buf, CL_TRUE, 0, sizeof(float) * m * k, a, 0, NULL, NULL);
    CHECK_ERR(err, "clEnqueueWriteBuffer a");
    err = clEnqueueWriteBuffer(queue, b_buf, CL_TRUE, 0, sizeof(float) * n * k, b, 0, NULL, NULL);
    CHECK_ERR(err, "clEnqueueWriteBuffer b");

    //@@ Fill in clbast::Gemm
    //@@ start by uncommenting the following and fill in args
    // clblast::StatusCode status = clblast::Gemm<float>(
    // );
    // CHECK_ERR((cl_int)status, "gemm");

    err = clEnqueueReadBuffer(queue, c_buf, CL_TRUE, 0, sizeof(float) * m * n, c, 0, NULL, NULL);
    CHECK_ERR(err, "clEnqueueReadBuffer c");

    // Cleanup
    clReleaseMemObject(a_buf);
    clReleaseMemObject(b_buf);
    clReleaseMemObject(c_buf);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}


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
) {
    cl_mem device_in;
    cl_mem device_w;
    cl_mem device_b;
    cl_mem device_out;
    cl_int err;     

    cl_device_id device_id;    // device ID
    cl_context context;        // context
    cl_command_queue queue;    // command queue
    cl_program program;        // program
    cl_kernel kernel;          // kernel

    char *kernel_source = OclLoadKernel("opencl_functions/conv2d.cl"); // Load kernel source

    // Get the ID for the specified kind of device type.
    err = OclGetDeviceWithFallback(&device_id, OCL_DEVICE_TYPE);
    CHECK_ERR(err, "OclGetDeviceWithFallback");

    // Create a context
    context = clCreateContext(0, 1, &device_id, nullptr, nullptr, &err);
    CHECK_ERR(err, "clCreateContext");

    // Create a command queue
    queue = clCreateCommandQueueWithProperties(context, device_id, 0, &err);
    CHECK_ERR(err, "clCreateCommandQueueWithProperties");

    // Create the program from the source buffer
    program = clCreateProgramWithSource(context, 1, (const char **)&kernel_source, nullptr, &err);
    CHECK_ERR(err, "clCreateProgramWithSource");

    // Build the program executable
    err = clBuildProgram(program, 0, nullptr, nullptr, nullptr, nullptr);
    CHECK_ERR(err, "clBuildProgram");

    // Create the compute kernel in the program we wish to run
    kernel = clCreateKernel(program, "conv2d", &err);
    CHECK_ERR(err, "clCreateKernel");

    //@@ Set the size of the output of the convolution

    //@@ Allocate OpenCL memory here
    // Create memory buffers for input and output vectors

    //@@ Copy memory to the OpenCL here

    //@@ Set the kernel dimensions and call the kernel

    //@@ Launch the OpenCL Kernel here
    // Execute the OpenCL kernel on the array

    //@@ Copy the output back to host
    // Read the memory buffer output_mem_obj to the local variable result

    // Release OpenCL resources
    clReleaseMemObject(device_in);
    clReleaseMemObject(device_b);
    clReleaseMemObject(device_w);
    clReleaseMemObject(device_out);
}