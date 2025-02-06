#include "hetu/core/ndarray.h"
#include "hetu/impl/cuda/CUDARand.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/random/CUDARandomState.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/utils/offset_calculator.cuh"
#include "hetu/impl/kernel/Vectorized.cuh"
#include <mutex>

namespace hetu {
namespace impl {

void DropoutCuda(const NDArray& input, double drop_rate, uint64_t seed,
                 NDArray& output, NDArray& mask, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT_SAME_DEVICE(input, mask);
  HT_ASSERT_SAME_SHAPE(input, output);
  HT_ASSERT_SAME_SHAPE(input, mask);
  size_t size = input->numel();
  if (size == 0)
    return;
  auto rand_state = GetCUDARandomState(cuda_stream.device_id(), seed, 4);
  using InType = std::tuple<spec_t>;
  using OutType = thrust::tuple<spec_t, mask_t>;
  launch_loop_kernel_with_idx<InType, OutType>({input}, {output, mask}, size, stream,
    [drop_rate, rand_state] __device__ (int idx, spec_t input) -> thrust::tuple<spec_t, mask_t> {
      curandStatePhilox4_32_10_t state;
      curand_init(rand_state.seed, idx, rand_state.offset, &state);
      float temp = curand_uniform(&state);
      mask_t keep_mask = (mask_t) (temp >= drop_rate);
      return thrust::tuple<spec_t, mask_t>(input * keep_mask / (1 - drop_rate), keep_mask);
    });
}

void DropoutGradientCuda(const NDArray& grad, const NDArray& fw_mask,
                         double drop_rate, NDArray& output,
                         const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(grad);
  HT_ASSERT_SAME_DEVICE(grad, fw_mask);
  HT_ASSERT_SAME_DEVICE(grad, output);
  HT_ASSERT_SAME_SHAPE(grad, fw_mask);
  HT_ASSERT_SAME_SHAPE(grad, output);
  size_t size = output->numel();
  if (size == 0)
    return;
  using InType = std::tuple<spec_t, mask_t>;
  using OutType = thrust::tuple<spec_t>;
  launch_loop_kernel<InType, OutType>({grad, fw_mask}, {output}, size, stream,
    [drop_rate] __device__ (spec_t grad, mask_t fw_mask) -> spec_t {
      return grad * fw_mask / (1 - drop_rate);
    });
}

} // namespace impl
} // namespace hetu
