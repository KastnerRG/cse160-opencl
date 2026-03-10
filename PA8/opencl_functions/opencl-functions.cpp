#include "opencl-functions.hpp"
#include <iostream>
// #include <clblast.h> //@@ Remember you are allowed to clblast if you so wish to 
#include "kernel.h"
#include "device.h"

#define CHECK_ERR(err, msg)                           \
    if (err != CL_SUCCESS)                            \
    {                                                 \
        fprintf(stderr, "%s failed: %d\n", msg, err); \
        exit(EXIT_FAILURE);                           \
    }

void softmax_impl(float *a, float *b, int m, int n) {
    cl_int err;
    int platform_id;
    int device_index;         
    cl_device_id device_id;   
    cl_context context;       
    cl_command_queue queue;   
    cl_program program;       
    cl_kernel kernel;
    cl_kernel max_kernel;
    OclPlatformProp *platforms = NULL;
    cl_uint num_platforms;

    char *kernel_source = OclLoadKernel("opencl_functions/softmax.cl");

    //@@ Implement Softmax in OpenCL, however you like
}
