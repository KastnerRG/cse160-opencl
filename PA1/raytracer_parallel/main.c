#include <stdio.h>
#ifdef __APPLE__ 
#include <OpenCL/cl.h>
#else 
#include <CL/cl.h>
#endif
#include <time.h>
#include "opencl_setup.h"
#include <stdlib.h>

/* 
 * OpenCL utilization for GPU parallelization
 * Sections to fill in will be lead by a:
 * // TODO: 
 *
 * Reference guide for OpenCL functions can be found here: https://www.khronos.org/files/opencl30-reference-guide.pdf
*/
int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "ERROR: Incorrect usage! Example usage: ./raytracer_parallel <device index>\n");
        return 1;
    }
    // Time measurement variables
    clock_t start, end;
    double cpu_time_used;

    // Start measuring host execution time
    start = clock();

    // OpenCL Initialization
    cl_platform_id platform[10];
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;

    // Get platform and GPU device on platform
    cl_uint num_devices, num_platforms;
    cl_int err;

    // Run kernel on CPU or GPU depending on command line arg
    char* str_end;
    int device_choice = strtol(argv[1], &str_end, 10);
    if (*str_end != '\0') {
        fprintf(stderr, "Invalid device id. Expected an integer. Received %s.\n", argv[1]);
        return 1;
    }
    
    err = clGetPlatformIDs(2, platform, &num_platforms);
    if (device_choice >= num_platforms) {
        fprintf(stderr, "Invalid device choice %d. Only %u platforms available.\n", device_choice, num_platforms);
        return 1;
    }
    err |= clGetDeviceIDs(platform[device_choice], CL_DEVICE_TYPE_ALL, 1, &device, &num_devices);

    printf("\n========================================================\n");
    char device_name[256];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL); // Name of device
    printf("Device: %s\n", device_name);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error getting platform and device\n");
        return 1;
    }

    // Create OpenCL context
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error creating context\n");
        return 1;
    }

    // Create command queue
    cl_queue_properties properties[] = {0};
    queue = clCreateCommandQueueWithProperties(context, device, properties, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error creating queue\n");
        return 1;
    }

    // Read kernel and instantiate it
    FILE* fp;
    char* kernelSrc;
    size_t kernelSize;

    fp = fopen("kernel.cl", "rb");

    if (!fp) {
        printf("Error reading kernel");
        exit(-1);
    }

    fseek(fp, 0, SEEK_END);
    kernelSize = ftell(fp);
    rewind(fp);

    // Allocate memory to read in source of kernel
    kernelSrc = (char*)malloc(kernelSize + 1);
    fread(kernelSrc, sizeof(char), kernelSize, fp);
    kernelSrc[kernelSize] = '\0';
    fclose(fp);

    // Create program now that we have kernel source
    program = clCreateProgramWithSource(context, 1, (const char**)&kernelSrc, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error creating program\n");
    }

    // Build program
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    if (err != CL_SUCCESS) {
        char *buff_erro;
        cl_int errcode;
        size_t build_log_len;
        errcode = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &build_log_len);
        if (errcode) {
            printf("clGetProgramBuildInfo failed at line %d\n", __LINE__);
            exit(-1);
        }

        buff_erro = malloc(build_log_len);
        if (!buff_erro) {
            printf("malloc failed at line %d\n", __LINE__);
            exit(-2);
        }

        errcode = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, build_log_len, buff_erro, NULL);
        if (errcode) {
            printf("clGetProgramBuildInfo failed at line %d\n", __LINE__);
            exit(-3);
        }

        fprintf(stderr,"Build log: \n%s\n", buff_erro); //Be careful with  the fprint
        free(buff_erro);
        fprintf(stderr,"clBuildProgram failed\n");
        exit(EXIT_FAILURE);
    }

    // Build kernel
    kernel = clCreateKernel(program, "renderColor", &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error creating kernel\n");
    }


    // Implemented Function (find in opencl_setup.c)
    runKernel(context, kernel, queue, program, kernelSrc, device_choice);

    // Stop measuring host execution time
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC * 1000; // Convert to milliseconds

    // Get output
    char* out_img_name;
    if (device_choice == 0) {
        out_img_name = "output_gpu.png";
    } else {
        out_img_name = "output_cpu.png";
    }

    printf("Total execution time (host + kernel): %.3f ms\n", cpu_time_used);
    printf("Image titled %s has been created/modified and can now be viewed!\n", out_img_name);
    printf("========================================================\n\n");
}