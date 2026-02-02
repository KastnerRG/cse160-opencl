#include <stdio.h>
#include <stdlib.h>

#include "device.h"
#include "kernel.h"
#include "matrix.h"

#define CHECK_ERR(err, msg)                           \
    if (err != CL_SUCCESS)                            \
    {                                                 \
        fprintf(stderr, "%s failed: %d\n", msg, err); \
        exit(EXIT_FAILURE);                           \
    }

#define NAIVE_KERNEL_PATH "0_matmul.cl"
#define COARSENED_KERNEL_PATH "1_coarsened_matmul.cl"
#define OPTIONAL_KERNEL_PATH "2_optional_matmul.cl"

void OpenCLMatrixMultiply(Matrix *input0, Matrix *input1, Matrix *result, const char *kernel_type)
{
    // Load external OpenCL kernel code
    char *kernel_source;

    if (strcmp(kernel_type, "naive") == 0) {
        kernel_source = OclLoadKernel(NAIVE_KERNEL_PATH); // Load naive kernel source
    }
    else if (strcmp(kernel_type, "coarsened") == 0) {
        kernel_source = OclLoadKernel(COARSENED_KERNEL_PATH); // Load coarsened kernel source
    }
    else {
        kernel_source = OclLoadKernel(OPTIONAL_KERNEL_PATH); // Load optional kernel source
    }

    // Device input and output buffers
    cl_mem device_a, device_b, device_c;

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
    err = OclGetDeviceWithFallback(&device_id, OCL_DEVICE_TYPE);
    CHECK_ERR(err, "OclGetDeviceWithFallback");

    // Create a context
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    CHECK_ERR(err, "clCreateContext");

    // Create a command queue
    queue = clCreateCommandQueueWithProperties(context, device_id, 0, &err);
    CHECK_ERR(err, "clCreateCommandQueueWithProperties");

    // Create the program from the source buffer
    program = clCreateProgramWithSource(context, 1, (const char **)&kernel_source, NULL, &err);
    CHECK_ERR(err, "clCreateProgramWithSource");

    // Build the program executable
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    CHECK_ERR(err, "clBuildProgram");

    // Create the compute kernel in the program we wish to run
    kernel = clCreateKernel(program, "matrixMultiply", &err);
    CHECK_ERR(err, "clCreateKernel");

    //@@ Allocate GPU memory here
    // Create memory buffers for input and output vectors

    //@@ Copy memory to the GPU here
    // Copy input vectors to memory buffers

    // Set the arguments to our compute kernel
    // __global const int *A, __global const int *B, __global int *C,
    // const unsigned int numARows, const unsigned int numAColumns,
    // const unsigned int numBRows, const unsigned int numBColumns,
    // const unsigned int numCRows, const unsigned int numCColumns,
    // const unsigned int coarsening_factor
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &device_a);
    CHECK_ERR(err, "clSetKernelArg 0");
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &device_b);
    CHECK_ERR(err, "clSetKernelArg 1");
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &device_c);
    CHECK_ERR(err, "clSetKernelArg 2");
    err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &input0->shape[0]);
    CHECK_ERR(err, "clSetKernelArg 3");
    err |= clSetKernelArg(kernel, 4, sizeof(unsigned int), &input0->shape[1]);
    CHECK_ERR(err, "clSetKernelArg 4");
    err |= clSetKernelArg(kernel, 5, sizeof(unsigned int), &input1->shape[0]);
    CHECK_ERR(err, "clSetKernelArg 5");
    err |= clSetKernelArg(kernel, 6, sizeof(unsigned int), &input1->shape[1]);
    CHECK_ERR(err, "clSetKernelArg 6");
    err |= clSetKernelArg(kernel, 7, sizeof(unsigned int), &result->shape[0]);
    CHECK_ERR(err, "clSetKernelArg 7");
    err |= clSetKernelArg(kernel, 8, sizeof(unsigned int), &result->shape[1]);
    CHECK_ERR(err, "clSetKernelArg 8");

    // Coarsening to rows of resulting matrix per work-item
    unsigned int row_coarsening_factor = result->shape[1]; //@@ Use this for Part 2
    unsigned int optional_coarsening_factor = 13; //@@ Optional TODO: Set this

    // @@ define local and global work sizes
    if (strcmp(kernel_type, "naive") == 0) {
       //@@ Define Local and Global Size for Naive
    }
    else if (strcmp(kernel_type, "coarsened") == 0) {
        //@@ Define Local and Global Size for coarsened
        //@@ Hint Look at PA2
    }
    else {
        //@@ Define Local and Global Size for the optional coarsening task
        //@@ Hint Look at PA2
    }

    //@@ Launch the GPU Kernel here
    // Execute the OpenCL kernel on the array
    

    //@@ Copy the GPU memory back to the CPU here
    // Read the memory buffer output_mem_obj to the local variable result

    //@@ Free the GPU memory here
    
    // Release OpenCL resources
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}

int main(int argc, char *argv[])
{
    if (argc != 6 || (strcmp(argv[5], "naive") != 0 && strcmp(argv[5], "coarsened") != 0 && strcmp(argv[5], "optional") != 0))
    {
        fprintf(stderr, "Usage 1: %s <input_file_0> <input_file_1> <answer_file> <output_file> naive\n", argv[0]);
        fprintf(stderr, "Usage 2: %s <input_file_0> <input_file_1> <answer_file> <output_file> coarsened\n", argv[0]);
        fprintf(stderr, "Usage 2: %s <input_file_0> <input_file_1> <answer_file> <output_file> optional\n", argv[0]);
        return -1;
    }

    const char *input_file_a = argv[1];
    const char *input_file_b = argv[2];
    const char *input_file_c = argv[3];
    const char *input_file_d = argv[4];
    const char *kernel_type = argv[5];

    // Host input and output vectors and sizes
    Matrix host_a, host_b, host_c, answer;
    
    cl_int err;

    err = LoadMatrix(input_file_a, &host_a);
    CHECK_ERR(err, "LoadMatrix");

    err = LoadMatrix(input_file_b, &host_b);
    CHECK_ERR(err, "LoadMatrix");

    err = LoadMatrix(input_file_c, &answer);
    CHECK_ERR(err, "LoadMatrix");

    int rows, cols;
    //@@ Update these values for the output rows and cols of the output
    //@@ Do not use the results from the answer matrix

    // Allocate the memory for the target.
    host_c.shape[0] = rows;
    host_c.shape[1] = cols;
    host_c.data = (int *)malloc(sizeof(int) * host_c.shape[0] * host_c.shape[1]);

    // Call your matrix multiply.
    OpenCLMatrixMultiply(&host_a, &host_b, &host_c, kernel_type);

    // // Call to print the matrix
    // PrintMatrix(&host_c);

    // Save the matrix
    SaveMatrix(input_file_d, &host_c);

    // Check the result of the matrix multiply
    CheckMatrix(&answer, &host_c);

    // Release host memory
    free(host_a.data);
    free(host_b.data);
    free(host_c.data);
    free(answer.data);

    return 0;
}
