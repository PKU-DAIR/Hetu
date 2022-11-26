#include "hetu/core/ndarray.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/cuda/CUDADnn.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"

namespace hetu {
namespace impl {

template <typename spec_t>
__global__ void minus_mean_n_square_kernel1(const spec_t* in_arr,
                                            const spec_t* mean, spec_t* out_arr,
                                            int last_2dim, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  spec_t temp = in_arr[idx] - mean[idx / last_2dim];
  out_arr[idx] = temp * temp;
}

template <typename spec_t>
__global__ void std_normal_transform(const spec_t* in_arr,
                                     const spec_t* mean_arr,
                                     const spec_t* var_arr, spec_t* out_arr,
                                     int last_2dim, float eps, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  size_t mo_idx = idx / last_2dim;
  out_arr[idx] =
    (in_arr[idx] - mean_arr[mo_idx]) / sqrtf(var_arr[mo_idx] + eps);
}

void InstanceNormCuda(const NDArray& in_arr, NDArray& mean_arr,
                      NDArray& var_arr, NDArray& out_arr, float eps,
                      const Stream& stream) {
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  cudnnHandle_t handle = hetu::impl::GetCudnnHandle(cuda_stream.device_id());

  cudnnDataType_t datatype;
  cudnnIndicesType_t indicetype;
  if (in_arr->dtype() == DataType::FLOAT32) {
    datatype = CUDNN_DATA_FLOAT;
    indicetype = CUDNN_32BIT_INDICES;
  } else if (in_arr->dtype() == DataType::FLOAT64) {
    datatype = CUDNN_DATA_DOUBLE;
    indicetype = CUDNN_64BIT_INDICES;
  }

  int ndim = in_arr->ndim();
  HT_ASSERT(ndim == 4);
  int last_2dim = in_arr->shape(ndim - 1) * in_arr->shape(ndim - 2);
  size_t cpu_mem = ndim * sizeof(int);
  int* dimA = (int*) malloc(cpu_mem);
  int* strideA = (int*) malloc(cpu_mem);
  int* dimC = (int*) malloc(cpu_mem);
  int* strideC = (int*) malloc(cpu_mem);

  int temp_strideA = 1;
  int temp_strideC = 1;

  for (int i = ndim - 1; i >= 0; --i) {
    dimA[i] = (int) in_arr->shape(i);
    dimC[i] = i < in_arr->ndim() - 2 ? (int) in_arr->shape(i) : 1;
    strideA[i] = temp_strideA;
    strideC[i] = temp_strideC;
    temp_strideA *= dimA[i];
    temp_strideC *= dimC[i];
  }

  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    in_arr->dtype(), spec_t, "InstanceNormCuda", [&]() {
      size_t size = temp_strideA * sizeof(spec_t);

      float one = 1.0f;
      float zero = 0.0f;

      cudnnReduceTensorDescriptor_t rtd;
      CUDNN_CALL(cudnnCreateReduceTensorDescriptor(&rtd));
      CUDNN_CALL(cudnnSetReduceTensorDescriptor(
        rtd, CUDNN_REDUCE_TENSOR_AVG, datatype, CUDNN_PROPAGATE_NAN,
        CUDNN_REDUCE_TENSOR_NO_INDICES, indicetype));

      cudnnTensorDescriptor_t adesc;
      cudnnTensorDescriptor_t cdesc;
      CUDNN_CALL(cudnnCreateTensorDescriptor(&adesc));
      CUDNN_CALL(cudnnCreateTensorDescriptor(&cdesc));

      CUDNN_CALL(
        cudnnSetTensorNdDescriptor(adesc, datatype, ndim, dimA, strideA));
      CUDNN_CALL(
        cudnnSetTensorNdDescriptor(cdesc, datatype, ndim, dimC, strideC));

      CUDNN_CALL(cudnnReduceTensor(
        handle, rtd, NULL, 0, (void*) out_arr->data_ptr<void>(), size, &one,
        adesc, (const void*) in_arr->data_ptr<void>(), &zero, cdesc,
        (void*) mean_arr->data_ptr<void>()));
      dim3 blocks, threads;
      threads.x = MIN(temp_strideA, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
      blocks.x = DIVUP(temp_strideA, HT_DEFAULT_NUM_THREADS_PER_BLOCK);

      minus_mean_n_square_kernel1<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        in_arr->data_ptr<spec_t>(), mean_arr->data_ptr<spec_t>(),
        out_arr->data_ptr<spec_t>(), last_2dim, temp_strideA);

      CUDNN_CALL(cudnnReduceTensor(
        handle, rtd, NULL, 0, (void*) out_arr->data_ptr<void>(), size, &one,
        adesc, (const void*) out_arr->data_ptr<void>(), &zero, cdesc,
        (void*) var_arr->data_ptr<void>()));

      threads.x = MIN(temp_strideA, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
      blocks.x = DIVUP(temp_strideA, HT_DEFAULT_NUM_THREADS_PER_BLOCK);

      std_normal_transform<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        in_arr->data_ptr<spec_t>(), mean_arr->data_ptr<spec_t>(),
        var_arr->data_ptr<spec_t>(), out_arr->data_ptr<spec_t>(), last_2dim,
        eps, temp_strideA);

      CUDNN_CALL(cudnnDestroyTensorDescriptor(adesc));
      CUDNN_CALL(cudnnDestroyTensorDescriptor(cdesc));
      CUDNN_CALL(cudnnDestroyReduceTensorDescriptor(rtd));
    });
  free(dimA);
  free(dimC);
  free(strideA);
  free(strideC);
  return;
}

template <typename spec_t>
__global__ void calculate_grad_kernel(const spec_t* out_grads,
                                      const spec_t* in_arr,
                                      const spec_t* mean_arr,
                                      const spec_t* var_arr, spec_t* grad_arr,
                                      size_t last2dim, float eps, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  size_t mo_idx = idx / last2dim;
  float y = (in_arr[idx] - mean_arr[mo_idx]) / sqrtf(var_arr[mo_idx] + eps);
  grad_arr[idx] = out_grads[idx] * (1.0 - 1.0 / (float) last2dim - y * y) /
    sqrtf(var_arr[mo_idx] + eps);
}

void InstanceNormGradientCuda(const NDArray& out_grads, const NDArray& in_arr,
                              NDArray& grad_arr, const NDArray& mean_arr,
                              const NDArray& var_arr, float eps,
                              const Stream& stream) {
  /*
    already have mean and var, we directly get y = x-u / sigma
    the grad_arr = out_grad * (1 - 1/WH - y^2) / sigma
  */
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  cudnnHandle_t handle = hetu::impl::GetCudnnHandle(cuda_stream.device_id());

  int ndim = out_grads->ndim();
  HT_ASSERT(ndim == 4);
  size_t total_elements = 1;
  for (int i = 0; i < ndim; ++i)
    total_elements *= out_grads->shape(i);
  int last2dim = out_grads->shape(ndim - 1) * out_grads->shape(ndim - 2);

  size_t size = total_elements;
  if (size == 0)
    return;
  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    in_arr->dtype(), spec_t, "CauculateGradCuda", [&]() {
      calculate_grad_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        out_grads->data_ptr<spec_t>(), in_arr->data_ptr<spec_t>(),
        mean_arr->data_ptr<spec_t>(), var_arr->data_ptr<spec_t>(),
        grad_arr->data_ptr<spec_t>(), last2dim, eps, size);
    });
}

} // namespace impl
} // namespace hetu
