#include "gpu_runtime.h"
#include <curand_kernel.h>

__global__ void dropout_residual_kernel(const float *input, const float *matB, float *output,
                               unsigned long long seed, const float rate,
                               size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    curandStatePhilox4_32_10_t state;
    curand_init(seed, 0, ind, &state);
    float temp = curand_uniform(&state);
    float keep_mask = (float)(temp >= rate);
    output[ind] = input[ind] * keep_mask / (1 - rate);
    output[ind] += matB[ind];
}


int DLGpuDropoutResidual(const DLArrayHandle input, const DLArrayHandle matB, const float dropout,
                 DLArrayHandle output, unsigned long long *pseed,
                 DLStreamHandle stream_handle = NULL) {
    size_t size = 1;
    assert(input->ndim == matB->ndim);
    for (index_t i = 0; i < input->ndim; i++) {
        size *= input->shape[i];
        assert(input->shape[i] == matB->shape[i]);
    }
    const float *input_data = (const float *)input->data;
    const float *matB_data = (const float *)matB->data;
    float *output_data = (float *)output->data;

    dim3 blocks;
    dim3 threads;
    if (size <= 1024) {
        threads.x = size;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (size + 1023) / 1024;
    }
    if (stream_handle) {
        dropout_residual_kernel<<<blocks, threads, 0,
                         *(cudaStream_t *)stream_handle->handle>>>(
            input_data, matB_data, output_data, *pseed, dropout, size);
    } else {
        dropout_residual_kernel<<<blocks, threads>>>(input_data, matB_data, output_data, *pseed,
                                            dropout, size);
    }
    return 0;
}