#include "hetu/core/ndarray.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"

namespace hetu {
namespace impl {

template <typename spec_t>
__global__ void reciprocal_kernel(const spec_t* input, size_t size,
                                  spec_t* output) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  output[idx] = static_cast<spec_t>(1) / input[idx];
}

void ReciprocalCuda(const NDArray& input, NDArray& output,
                    const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_CUDA_DEVICE(output);
  HT_ASSERT_EXCHANGABLE(input, output);
  size_t size = input->numel();
  if (size == 0)
    return;
  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  HT_DISPATCH_FLOATING_TYPES(input->dtype(), spec_t, "ReciprocalCuda", [&]() {
    reciprocal_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
      input->data_ptr<spec_t>(), size, output->data_ptr<spec_t>());
  });
}

} // namespace impl
} // namespace hetu
