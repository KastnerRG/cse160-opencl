#include <stdio.h>
#ifdef __APPLE__ 
#include <OpenCL/cl.h>
#else 
#include <CL/cl.h>
#endif
#include <time.h>
#include <math.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "lib/stb_image_write.h"

const cl_int IMG_SIZE = 1024;
const size_t GLOBAL_SIZE = IMG_SIZE * IMG_SIZE;
const size_t LOCAL_SIZE = 32;
const float FOV = 3.14159265359/3.0;

void runKernel(cl_context context, cl_kernel kernel, cl_command_queue queue, 
          cl_program program, char* kernelSrc, int device_choice) {
    cl_int err;
    // Create memory buffer for output

    /* 
     * TODO: Fill in dimensions for pixel_size and pixels_h and fill in clCreateBuffer
     * pixel_size: height * width * (channels == 3)
     * pixels_h: height * width * (channels == 3)
     * Hint: Look at constant variables at the top of this file for dimensions (height = width)
     */
    size_t pixel_size = /* TODO */;
    unsigned char pixels_h[/* TODO */];

    // As we are only writing to pixels_d, we can set cl_mem_flags to CL_MEM_WRITE_ONLY
    cl_mem pixels_d = clCreateBuffer(/* TODO: Fill in this buffer*/);

    // Check if buffer is properly allocated
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error creating pixels_d\n");
    }

    // Calculate half height for raytracing
    cl_float half_height = tan(FOV * 0.5);

    // Set kernel arguments
    /* 
     * TODO: set arguments 0, 1, and 2 for kernel.cl (look at renderColor function
     * in kernel.cl for expected inputs)
     */
    err = clSetKernelArg(/* TODO */);
    err |= clSetKernelArg(/* TODO */);
    err |= clSetKernelArg(/* TODO */);

    // Check if arguments are properly filled in
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error setting kernel arguments\n");
    }

    // Execute kernel on data
    /* 
     * TODO: Fill in clEnqueueNDRangeKernel
     * GLOBAL_SIZE and LOCAL_SIZE are labeled at the top of this file
     * Hint: Look at pixels_h for expected work dimension
     */
    err = clEnqueueNDRangeKernel(/* TODO */);

    // Wait for kernel to finish
    clFinish(queue);

    // Read pixels results from device
    /* 
     * TODO: Fill in clEnqueueReadBuffer
     * We are writing to pixels_h
     * Make sure to make this function blocking by using CL_TRUE
     */
    clEnqueueReadBuffer(/* TODO */);

    // Save the result to a PNG file
    char* out_img_name;
    if (device_choice == 0) {
        out_img_name = "output_gpu.png";
    } else {
        out_img_name = "output_cpu.png";
    }
    stbi_write_png(out_img_name, IMG_SIZE, IMG_SIZE, 3, pixels_h, IMG_SIZE * 3);

    // Release OpenCL resources
    /*
     * TODO: Release buffer -> kernel -> program -> queue -> context
     */
    clReleaseMemObject(/* TODO */);

    clReleaseKernel(/* TODO */);
    clReleaseProgram(/* TODO */);
    clReleaseCommandQueue(/* TODO */);
    clReleaseContext(/* TODO */);


    // Free memory allocated to kernel source
    /*
     * TODO: Free memory used to load kernel from file
     */
    free(/* TODO */);
}