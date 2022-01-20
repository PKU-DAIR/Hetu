#include "gpu_runtime.h"
#include <curand_kernel.h>

__global__ void dropout_kernel(const float *input, float *output,
                               unsigned long long seed, const float rate,
                               size_t size);

int DLGpuSoftmaxDropout(const DLArrayHandle input, const float dropout,
                 DLArrayHandle output, unsigned long long *pseed,
                 DLStreamHandle stream_handle = NULL) {
    // softmax forward: input -> output
    size_t indim = input->ndim;
    assert(indim == output->ndim);
    int dev_id = (input->ctx).device_id;
    cudnn_init(dev_id, stream_handle);
    int n_ = 1;
    for (int i = 0; i < indim - 1; ++i) {
        n_ *= input->shape[i];
    }
    int c_ = input->shape[indim - 1];
    const float *input_data = (const float *)(input->data);
    float *output_data = (float *)(output->data);
    float alpha = 1.0f;
    float beta = 0.0f;
    cudnnTensorDescriptor_t desc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&desc));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW,
                                            CUDNN_DATA_FLOAT, n_, c_, 1, 1));
    CUDNN_CALL(cudnnSoftmaxForward(cudnn_map[dev_id], CUDNN_SOFTMAX_ACCURATE,
                                    CUDNN_SOFTMAX_MODE_INSTANCE, &alpha, desc,
                                    (const void *)input_data, &beta, desc,
                                    (void *)output_data));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(desc));

    // dropout forward: output -> output
    size_t size = 1;
    for (index_t i = 0; i < input->ndim; i++) {
        size *= input->shape[i];
    }
    input_data = (const float *)output->data;
    output_data = (float *)output->data;

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
        dropout_kernel<<<blocks, threads, 0,
                         *(cudaStream_t *)stream_handle->handle>>>(
            input_data, output_data, *pseed, dropout, size);
    } else {
        dropout_kernel<<<blocks, threads>>>(input_data, output_data, *pseed,
                                            dropout, size);
    }
    return 0;
}

int DLGpuSoftmaxDropoutGradient(DLArrayHandle grad, DLArrayHandle softmax_input, const float dropout,
                         DLArrayHandle output, unsigned long long seed,
                         DLStreamHandle stream_handle = NULL) {
    // dropout backward: grad -> grad
    size_t size = 1;
    for (index_t i = 0; i < grad->ndim; i++) {
        size *= grad->shape[i];
    }
    const float *grad_data = (const float *)grad->data;
    float *output_data = (float *)grad->data;

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
        dropout_kernel<<<blocks, threads, 0,
                         *(cudaStream_t *)stream_handle->handle>>>(
            grad_data, output_data, seed, dropout, size);
    } else {
        dropout_kernel<<<blocks, threads>>>(grad_data, output_data, seed,
                                            dropout, size);
    }

    // softmax forward: softmax_input -> output
    size_t indim = softmax_input->ndim;
    assert(indim == output->ndim);
    int dev_id = (softmax_input->ctx).device_id;
    cudnn_init(dev_id, stream_handle);
    int n_ = 1;
    for (int i = 0; i < indim - 1; ++i) {
        n_ *= softmax_input->shape[i];
    }
    int c_ = softmax_input->shape[indim - 1];
    const float *input_data = (const float *)(softmax_input->data);
    output_data = (float *)(output->data);
    float alpha = 1.0f;
    float beta = 0.0f;
    cudnnTensorDescriptor_t desc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&desc));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW,
                                            CUDNN_DATA_FLOAT, n_, c_, 1, 1));
    CUDNN_CALL(cudnnSoftmaxForward(cudnn_map[dev_id], CUDNN_SOFTMAX_ACCURATE,
                                    CUDNN_SOFTMAX_MODE_INSTANCE, &alpha, desc,
                                    (const void *)input_data, &beta, desc,
                                    (void *)output_data));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(desc));

    // softmax backward: y is output, dy is grad, dx is output

    indim = output->ndim;
    n_ = 1;
    for (int i = 0; i < indim - 1; ++i) {
        n_ *= output->shape[i];
    }
    c_ = output->shape[indim - 1];
    const float *y_data = (const float *)(output->data);
    const float *dy_data = (const float *)(grad->data);
    float *dx_data = (float *)(output->data);
    CUDNN_CALL(cudnnCreateTensorDescriptor(&desc));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT, n_, c_, 1, 1));
    CUDNN_CALL(cudnnSoftmaxBackward(
        cudnn_map[dev_id], CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
        &alpha, desc, (const void *)y_data, desc, (const void *)dy_data, &beta,
        desc, (void *)dx_data));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(desc));
    return 0;
}
