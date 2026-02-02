#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "device.h"
#include "kernel.h"
#include "matrix.h"

#define CHECK_ERR(err, msg)                           \
    if (err != CL_SUCCESS)                            \
    {                                                 \
        fprintf(stderr, "%s failed: %d\n", msg, err); \
        exit(EXIT_FAILURE);                           \
    }

#define VECTOR_ADD_2_KERNEL_PATH "vector_add_2.cl"
#define VECTOR_ADD_4_KERNEL_PATH "vector_add_4.cl"
#define VECTOR_ADD_2_COARSENED_KERNEL_PATH "vector_add_2_coarsened.cl"

void initializeOpenCL(cl_device_id* device_id, cl_context* context, cl_command_queue* queue) {
    cl_int err;

    // Find platforms and devices
    OclPlatformProp *platforms = NULL;
    cl_uint num_platforms;
    err = OclFindPlatforms((const OclPlatformProp **)&platforms, &num_platforms);
    CHECK_ERR(err, "OclFindPlatforms");

    // Get the ID for the specified kind of device type.
    err = OclGetDeviceWithFallback(device_id, OCL_DEVICE_TYPE);
    CHECK_ERR(err, "OclGetDeviceWithFallback");

    // Create a context
    *context = clCreateContext(0, 1, device_id, NULL, NULL, &err);
    CHECK_ERR(err, "clCreateContext");

    // Create a command queue
    *queue = clCreateCommandQueueWithProperties(*context, *device_id, 0, &err);
    CHECK_ERR(err, "clCreateCommandQueueWithProperties");
}

void callVectorAdd2Kernel(Matrix* a, Matrix* b, Matrix* out, cl_context* context, cl_command_queue* queue) {
    // OpenCL objects
    cl_program program;                 // program
    cl_kernel kernel;         // kernel

    // OpenCL setup variables
    size_t global_item_size, local_item_size;
    cl_int err;

    // Device input and output vectors
    cl_mem device_input_1, device_input_2, device_output;

    // Load external OpenCL kernel code
    char *kernel_source = OclLoadKernel(VECTOR_ADD_2_KERNEL_PATH);

    // Create the program from the source buffer
    program = clCreateProgramWithSource(*context, 1, (const char **)&kernel_source, NULL, &err);
    CHECK_ERR(err, "clCreateProgramWithSource");

    // Build the program executable
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    CHECK_ERR(err, "clBuildProgram");
    
    kernel = clCreateKernel(program, "vectorAdd", &err);
    CHECK_ERR(err, "clCreateKernel");

    //@@ Create memory buffers for input and output vectors
    // Allocate GPU memory
    
    //@@ Copy memory to the GPU here
    // Copy input and output vectors from the host device to GPU
    

    //@@ define local and global work sizes
    // Initialize the global size and local size here
    
    //@@ Set Arguments
    // Set the arguments to the kernel

    //@@ Launch the GPU Kernel here
    // Execute the OpenCL kernel to perform vector addition


    //@@ Copy the GPU memory back to the CPU here


    //@@ Free the GPU memory here
    // Free the GPU memory
    
    // Release OpenCL resources
    clReleaseProgram(program);
    clReleaseKernel(kernel);

    // Release Host Memory
    free(kernel_source);
}

void part1(Matrix* host_input_1, Matrix* host_input_2, Matrix* host_input_3, Matrix* host_input_4, Matrix* host_output, Matrix* answer, const char* output_file) {
    // Start of program one

    // OpenCL objects
    cl_device_id device_id;             // device ID
    cl_context context;                 // context
    cl_command_queue queue;             // command queue

    initializeOpenCL(&device_id, &context, &queue);

    callVectorAdd2Kernel(host_input_1, host_input_2, host_output, &context, &queue);
    callVectorAdd2Kernel(host_output, host_input_3, host_output, &context, &queue);
    callVectorAdd2Kernel(host_output, host_input_4, host_output, &context, &queue);

    // Prints the results
    // for (unsigned int i = 0; i < host_output.shape[0] * host_output.shape[1]; i++)
    // {
    //     printf("C[%u]: %d == %d\n", i, host_output.data[i], answer.data[i]);
    // }

    // Check whether the answer matches the output
    CheckMatrix(answer, host_output);
    SaveMatrix(output_file, host_output);

    // Release OpenCL resources
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    
}

void callVectorAdd4Kernel(Matrix* a, Matrix* b, Matrix* c, Matrix* d, Matrix* out, cl_context* context, cl_command_queue* queue) {
    // OpenCL objects
    cl_program program;                 // program
    cl_kernel kernel;         // kernel

    // OpenCL setup variables
    size_t global_item_size, local_item_size;
    cl_int err;

    // Device input and output vectors
    cl_mem device_input_1, device_input_2, device_input_3, device_input_4, device_output;

    // Load external OpenCL kernel code
    char *kernel_source = OclLoadKernel(VECTOR_ADD_4_KERNEL_PATH);

    // Create the program from the source buffer
    program = clCreateProgramWithSource(*context, 1, (const char **)&kernel_source, NULL, &err);
    CHECK_ERR(err, "clCreateProgramWithSource");

    // Build the program executable
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    CHECK_ERR(err, "clBuildProgram");
    
    kernel = clCreateKernel(program, "vectorAdd", &err);
    CHECK_ERR(err, "clCreateKernel");

    //@@ Create memory buffers for input and output vectors
    // Allocate GPU memory
   
    //@@ Copy memory to the GPU here
    // Copy input and output vectors from the host device to GPU

    //@@ define local and global work sizes
    // Initialize the global size and local size here
    
    //@@ Set Arguments
    // Set the arguments to the kernel

    //@@ Launch the GPU Kernel here
    // Execute the OpenCL kernel to perform vector addition
   
    //@@ Copy the GPU memory back to the CPU here

    //@@ Free the GPU memory here
    // Free the GPU memory
    
    // Release OpenCL resources
    clReleaseProgram(program);
    clReleaseKernel(kernel);

    // Release Host Memory
    free(kernel_source);
}

void part2(Matrix* host_input_1, Matrix* host_input_2, Matrix* host_input_3, Matrix* host_input_4, Matrix* host_output, Matrix* answer, const char* output_file) {
    // Start of program two

    // OpenCL objects
    cl_device_id device_id;             // device ID
    cl_context context;                 // context
    cl_command_queue queue;             // command queue

    initializeOpenCL(&device_id, &context, &queue);

    callVectorAdd4Kernel(host_input_1, host_input_2, host_input_3, host_input_4, host_output, &context, &queue);

    // Prints the results
    // for (unsigned int i = 0; i < host_output.shape[0] * host_output.shape[1]; i++)
    // {
    //     printf("C[%u]: %d == %d\n", i, host_output.data[i], answer.data[i]);
    // }

    // Check whether the answer matches the output
    CheckMatrix(answer, host_output);
    SaveMatrix(output_file, host_output);

    // Release OpenCL resources
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}

void callVectorAdd2CoarsenedKernel(Matrix* a, Matrix* b, Matrix* out, cl_context* context, cl_command_queue* queue) {
    // OpenCL objects
    cl_program program;                 // program
    cl_kernel kernel;         // kernel

    // OpenCL setup variables
    size_t global_item_size, local_item_size;
    cl_int err;

    // Device input and output vectors
    cl_mem device_input_1, device_input_2, device_output;

    // Load external OpenCL kernel code
    char *kernel_source = OclLoadKernel(VECTOR_ADD_2_COARSENED_KERNEL_PATH);

    // Create the program from the source buffer
    program = clCreateProgramWithSource(*context, 1, (const char **)&kernel_source, NULL, &err);
    CHECK_ERR(err, "clCreateProgramWithSource");

    // Build the program executable
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    CHECK_ERR(err, "clBuildProgram");
    
    kernel = clCreateKernel(program, "vectorAdd", &err);
    CHECK_ERR(err, "clCreateKernel");

    //@@ Create memory buffers for input and output vectors
    // Allocate GPU memory

    //@@ Copy memory to the GPU here
    // Copy input and output vectors from the host device to GPU
    
    // Initialize the global size and local size here
    // DO NOT CHANGE BELOW
    // If you change it for testing or experimentation
    // Make sure to set it back to these values before submission
    unsigned int coarsening_factor = 4;
    unsigned int size_a = a->shape[0] * a->shape[1];

    global_item_size = (size_a + coarsening_factor - 1) / coarsening_factor;
    local_item_size = 1;
    // DO NOT CHANGE ABOVE

    //@@ Set Arguments
    // Set the arguments to the kernel

    // DO NOT CHANGE BELOW
    // Execute the OpenCL kernel to perform vector addition
    err = clEnqueueNDRangeKernel(*queue, kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);
    CHECK_ERR(err, "clEnqueueNDRangeKernel");
    // DO NOT CHANGE ABOVE

    //@@ Copy the GPU memory back to the CPU here

    //@@ Free the GPU memory here
    // Free the GPU memory
    
    // Release OpenCL resources
    clReleaseProgram(program);
    clReleaseKernel(kernel);

    // Release Host Memory
    free(kernel_source);
}

void part3(Matrix* host_input_1, Matrix* host_input_2, Matrix* host_input_3, Matrix* host_input_4, Matrix* host_output, Matrix* answer, const char* output_file) {
    // Start of program three

    // OpenCL objects
    cl_device_id device_id;             // device ID
    cl_context context;                 // context
    cl_command_queue queue;             // command queue

    initializeOpenCL(&device_id, &context, &queue);

    callVectorAdd2CoarsenedKernel(host_input_1, host_input_2, host_output, &context, &queue);
    callVectorAdd2CoarsenedKernel(host_output, host_input_3, host_output, &context, &queue);
    callVectorAdd2CoarsenedKernel(host_output, host_input_4, host_output, &context, &queue);

    // Prints the results
    // for (unsigned int i = 0; i < host_output.shape[0] * host_output.shape[1]; i++)
    // {
    //     printf("C[%u]: %d == %d\n", i, host_output.data[i], answer.data[i]);
    // }

    // Check whether the answer matches the output
    CheckMatrix(answer, host_output);
    SaveMatrix(output_file, host_output);

    // Release OpenCL resources
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}

int main(int argc, char *argv[])
{
    if (argc != 9)
    {
        fprintf(stderr, "Usage: %s <input_file_0> <input_file_1> <input_file_2> <input_file_3> <answer_file> <output_file_program_1> <output_file_program_2> <output_file_program_3>\n", argv[0]);
        return -1;
    }

    const char *input_array_1_file = argv[1];
    const char *input_array_2_file = argv[2];
    const char *input_array_3_file = argv[3];
    const char *input_array_4_file = argv[4];
    const char *answer_file = argv[5];
    const char *program_1_output_file = argv[6];
    const char *program_2_output_file = argv[7];
    const char *program_3_output_file = argv[8];

    // Host input and output vectors
    Matrix host_input_1, host_input_2, host_input_3, host_input_4, host_output, answer;

    // OpenCL setup variables
    cl_int err;

    // Load input matrix from file and check for errors
    err = LoadMatrix(input_array_1_file, &host_input_1);
    CHECK_ERR(err, "LoadMatrix");
    err = LoadMatrix(input_array_2_file, &host_input_2);
    CHECK_ERR(err, "LoadMatrix");
    err = LoadMatrix(input_array_3_file, &host_input_3);
    CHECK_ERR(err, "LoadMatrix");
    err = LoadMatrix(input_array_4_file, &host_input_4);
    CHECK_ERR(err, "LoadMatrix");
    err = LoadMatrix(answer_file, &answer);
    CHECK_ERR(err, "LoadMatrix");

    // Allocate the memory for the output
    host_output.shape[0] = host_input_1.shape[0];
    host_output.shape[1] = host_input_1.shape[1];
    host_output.data = (int *)malloc(sizeof(int) * host_output.shape[0] * host_output.shape[1]);

    // Time measurement variables
    clock_t start, end;
    double cpu_time_used;


    

    // =================================================================
    printf("==============Starting Program 1==============\n");
    start = clock();

    part1(&host_input_1, &host_input_2, &host_input_3, &host_input_4, &host_output, &answer, program_1_output_file);
    
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC * 1000; // Convert to milliseconds

    printf("Execution time: %.2fms\n", cpu_time_used);
    printf("==============Finished Program 1==============\n");




    // =================================================================
    printf("==============Starting Program 2==============\n");
    start = clock();

    part2(&host_input_1, &host_input_2, &host_input_3, &host_input_4, &host_output, &answer, program_2_output_file);
    
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC * 1000; // Convert to milliseconds
    printf("Execution time: %.2fms\n", cpu_time_used);
    printf("==============Finished Program 2==============\n");




    // =================================================================
    printf("==============Starting Program 3==============\n");
    start = clock();

    part3(&host_input_1, &host_input_2, &host_input_3, &host_input_4, &host_output, &answer, program_3_output_file);
    
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC * 1000; // Convert to milliseconds

    printf("Execution time: %.2fms\n", cpu_time_used);
    printf("==============Finished Program 3==============\n");



    
    // Release host memory
    free(host_input_1.data);
    free(host_input_2.data);
    free(host_input_3.data);
    free(host_input_4.data);
    free(host_output.data);
    free(answer.data);

    return 0;
}
