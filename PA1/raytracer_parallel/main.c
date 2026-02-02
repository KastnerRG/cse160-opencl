#include <stdio.h>
#ifdef __APPLE__ 
#include <OpenCL/cl.h>
#else 
#include <CL/cl.h>
#endif
#include <time.h>
#include <string.h>

//helper_lib 
#include "device.h"
#include "kernel.h"
#include "matrix.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define KERNEL_PATH "kernel.cl"
#include "lib/stb_image_write.h"

//@@ TODO:: Better error Function
#define CHECK_ERR(err, msg)                           \
    if (err != CL_SUCCESS)                            \
    {                                                 \
        fprintf(stderr, "%s failed: %d\n", msg, err); \
        exit(EXIT_FAILURE);                           \
    }

//@@ Hint: these might be useful
const cl_int IMG_SIZE = 1024;
const size_t GLOBAL_SIZE = IMG_SIZE * IMG_SIZE;
const size_t LOCAL_SIZE = 32;
const float FOV = 3.14159265359/3.0;

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "ERROR: Incorrect usage! Example usage: make gpu / make cpu\n");
        return 1;
    }
    // Time measurement variables
    clock_t start, end;
    double cpu_time_used;

    // Start measuring host execution time
    start = clock();

    // Load external OpenCL kernel code
    char *kernelSrc = OclLoadKernel(KERNEL_PATH); // Load kernel 
    
    cl_int err;

    cl_device_id device_id;    // device ID
    cl_context context;        // context
    cl_command_queue queue;    // command queue
    cl_program program;        // program
    cl_kernel kernel;          // kernel

    // Find platforms and devices
    OclPlatformProp *platforms = NULL;
    cl_uint num_platforms;

    err = OclFindPlatforms((const OclPlatformProp **)&platforms, &num_platforms);
    CHECK_ERR(err, "OclFindPlatforms");

    // Get the ID for the specified kind of device type.
    cl_device_type device_type;
    if (strcmp(argv[1], "gpu") == 0) {
        device_type = OCL_DEVICE_TYPE;
    } else { //default cpu
        device_type = CL_DEVICE_TYPE_CPU;
    }

    err = OclGetDeviceWithFallback(&device_id, device_type);
    CHECK_ERR(err, "OclGetDeviceWithFallback");

    // Create a context
    //context = clCreateContext(); //@@ TODO: uncomment and add args
    CHECK_ERR(err, "clCreateContext");

    // Create a command queue
    //queue = clCreateCommandQueueWithProperties(); //@@ TODO: uncomment and add args
    CHECK_ERR(err, "clCreateCommandQueueWithProperties");

    // Create the program from the source buffer
    program = clCreateProgramWithSource(context, 1, (const char **)&kernelSrc, NULL, &err);
    CHECK_ERR(err, "clCreateProgramWithSource");

    // Build the program executable
    //@@ Hint: program name is "renderColor"
    // err = clBuildProgram(); //@@ TODO: uncomment and add args
    CHECK_ERR(err, "clBuildProgram");

    // Create the compute kernel in the program we wish to run
    //@@ Hint: program name is "renderColor"
    // kernel = clCreateKernel(); //@@ TODO: uncomment and add args
    CHECK_ERR(err, "clCreateKernel");

    size_t pixel_size = IMG_SIZE * IMG_SIZE * 3 * sizeof(unsigned char);
    unsigned char pixels_h[IMG_SIZE * IMG_SIZE * 3];

    cl_float half_height = tan(FOV * 0.5);

    // cl_mem pixels_d = clCreateBuffer(); //@@ TODO: uncomment and add args
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error creating pixels_d\n");
    }

    // Set kernel arguments
    // err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &pixels_d); //@@ TODO: uncomment
    err |= clSetKernelArg(kernel, 1, sizeof(cl_int), &IMG_SIZE);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_float), &half_height);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error setting kernel arguments\n");
    }

    // Execute kernel on data
    // err = clEnqueueNDRangeKernel(); //@@ TODO: uncomment and add args

    // Wait for kernel to finish
    //clFinish(); //@@ TODO: uncomment and add args

    // Read pixels results from device
    //clEnqueueReadBuffer(); //@@ TODO: uncomment and add args

    // Save the result to a PNG file
    char out_img_name[256];
    snprintf(out_img_name, sizeof(out_img_name),  "output_%s.png", OclDeviceTypeString(device_type));

    stbi_write_png(out_img_name, IMG_SIZE, IMG_SIZE, 3, pixels_h, IMG_SIZE * 3);

    //@@ TODO: Release OpenCL resources
    //clReleaseMemObject(); //@@ TODO: uncomment and add args
    //clReleaseKernel(); //@@ TODO: uncomment and add args
    //clReleaseProgram(); //@@ TODO: uncomment and add args
    //clReleaseCommandQueue(); //@@ TODO: uncomment and add args
    //clReleaseContext(); //@@ TODO: uncomment and add args


    // Free memory allocated to kernel source
    free(kernelSrc);

    // Stop measuring host execution time
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC * 1000; // Convert to milliseconds

    printf("Total execution time (host + kernel): %.3f ms\n", cpu_time_used);
    printf("Image titled %s has been created/modified and can now be viewed!\n", out_img_name);
    printf("========================================================\n\n");
}