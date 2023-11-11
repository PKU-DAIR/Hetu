#include "hetu/core/ndarray.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/utils/cuda_math.h"

namespace hetu {
namespace impl {

template <typename spec_t>
__global__ void tanh_kernel(const spec_t* input, size_t size, spec_t* output) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  output[idx] = hetu::cuda::cuda_tanh(input[idx]);
}

template <typename spec_t>
__global__ void tanh_gradient_kernel(const spec_t* input,
                                     const spec_t* output_grad, size_t size,
                                     spec_t* output) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  output[idx] = (1 - input[idx] * input[idx]) * output_grad[idx];
}

void TanhCuda(const NDArray& input, NDArray& output, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT_EXCHANGABLE(input, output);

  size_t size = output->numel();
  if (size == 0)
    return;
  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  HT_DISPATCH_FLOATING_TYPES(input->dtype(), spec_t, "TanhCuda", [&]() {
    tanh_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
      input->data_ptr<spec_t>(), size, output->data_ptr<spec_t>());
  });
  NDArray::MarkUsedBy({input, output}, stream);
}

void TanhGradientCuda(const NDArray& input, const NDArray& output_grad,
                      NDArray& input_grad, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output_grad);
  HT_ASSERT_SAME_DEVICE(input, input_grad);
  HT_ASSERT_EXCHANGABLE(input, output_grad);
  HT_ASSERT_EXCHANGABLE(input, input_grad);

  size_t size = input_grad->numel();
  if (size == 0)
    return;
  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "TanhGradientCuda", [&]() {
      tanh_gradient_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        input->data_ptr<spec_t>(), output_grad->data_ptr<spec_t>(), size,
        input_grad->data_ptr<spec_t>());
    });
  NDArray::MarkUsedBy({input, output_grad, input_grad}, stream);
}

} // namespace impl
} // namespace hetu
