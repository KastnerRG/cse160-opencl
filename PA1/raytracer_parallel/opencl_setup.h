#ifndef OPENCL_SETUP_H
#define OPENCL_SETUP_H
#include <stdlib.h>
void runKernel(cl_context context, cl_kernel kernel, cl_command_queue queue, cl_program program, 
          char* kernelSrc, int device_choice);

#endif