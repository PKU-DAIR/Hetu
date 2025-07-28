#include "hetu/core/ndarray.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/utils/cuda_math.h"

namespace hetu {
namespace impl {

template <typename spec_t>
__global__ void repeat_kernel(const spec_t* input, spec_t* output, size_t size,
                              const int64_t* stride_in,
                              const int64_t* stride_out, 
                              const int64_t* dims, int ndim) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  int index = 0;
  size_t ind = idx;
  for (int i = 0; i < ndim; i++) {
    int tmp_index = ind / stride_out[i];
    index += (tmp_index % dims[i]) * stride_in[i];
    ind -= tmp_index * stride_out[i];
  }
  output[idx] = input[index];
}

void RepeatCuda(const NDArray& input, NDArray& output, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);

  size_t size = output->numel();
  if (size == 0)
    return;

  int ndim = output->ndim();
  HTStride stride_in(ndim);
  HTShape dims(ndim);
  for (int i = 0; i < ndim; i++) {
    if (i < (ndim - input->ndim())) {
      stride_in[i] = input->stride(0);
      dims[i] = 1;
    } else {
      stride_in[i] = input->stride(i - (ndim - input->ndim()));
      dims[i] = input->shape(i - (ndim - input->ndim()));
    }
  }
  
  auto device_id = input->device().index();
  hetu::cuda::CUDADeviceGuard guard(device_id);
  CUDAStream cuda_stream(stream);
  auto stride_in_arr = hetu::cuda::to_int64_ndarray(stride_in, device_id);
  auto stride_out_arr =
    hetu::cuda::to_int64_ndarray(output->stride(), device_id);
  auto dims_arr = hetu::cuda::to_int64_ndarray(dims, device_id);
  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(input->dtype(), spec_t, "RepeatCuda", [&]() {
  // HT_DISPATCH_FLOATING_TYPES(input->dtype(), spec_t, "RepeatCuda", [&]() {
    repeat_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
      input->data_ptr<spec_t>(), output->data_ptr<spec_t>(), size, 
      stride_in_arr->data_ptr<int64_t>(), 
      stride_out_arr->data_ptr<int64_t>(), 
      dims_arr->data_ptr<int64_t>(), 
      ndim);
  });
  NDArray::MarkUsedBy({input, output, stride_in_arr, stride_out_arr, dims_arr},
                      stream);
}

template <typename spec_t>
__global__ void repeat_gradient_kernel(const spec_t* input, spec_t* output,
                                       size_t size, 
                                       const int64_t* stride_in,
                                       const int64_t* stride_out,
                                       const int64_t* dims, int ndim) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  int index = 0;
  size_t ind = idx;
  for (int i = 0; i < ndim; i++) {
    int tmp_index = ind / stride_out[i];
    index += (tmp_index % dims[i]) * stride_in[i];
    ind -= tmp_index * stride_out[i];
  }
  hetu::cuda::AtomicAdd(&output[index], input[idx]);
}

void RepeatGradientCuda(const NDArray& output, NDArray& input, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);

  size_t size = output->numel();
  if (size == 0)
    return;

  int ndim = output->ndim();
  HTStride stride_in(ndim);
  HTShape dims(ndim);
  for (int i = 0; i < ndim; i++) {
    if (i < (ndim - input->ndim())) {
      stride_in[i] = input->stride(0);
      dims[i] = 1;
    } else {
      stride_in[i] = input->stride(i - (ndim - input->ndim()));
      dims[i] = input->shape(i - (ndim - input->ndim()));
    }
  }

  auto device_id = input->device().index();
  hetu::cuda::CUDADeviceGuard guard(device_id);
  CUDAStream cuda_stream(stream);
  NDArray::zeros_(input, stream.stream_index());
  auto stride_in_arr = hetu::cuda::to_int64_ndarray(stride_in, device_id);
  auto stride_out_arr =
    hetu::cuda::to_int64_ndarray(output->stride(), device_id);
  auto dims_arr = hetu::cuda::to_int64_ndarray(dims, device_id);
  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  HT_DISPATCH_FLOATING_TYPES(
    input->dtype(), spec_t, "RepeatGradientCuda", [&]() {
      repeat_gradient_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        output->data_ptr<spec_t>(), input->data_ptr<spec_t>(), size, 
        stride_in_arr->data_ptr<int64_t>(), 
        stride_out_arr->data_ptr<int64_t>(), 
        dims_arr->data_ptr<int64_t>(), 
        ndim);
    });
  NDArray::MarkUsedBy({input, output, stride_in_arr, stride_out_arr, dims_arr},
                      stream);
}

} // namespace impl
} // namespace hetu
