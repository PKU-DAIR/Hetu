#include "hetu/core/ndarray.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/kernel/Vectorized.cuh"

namespace hetu {
namespace impl {

void RangeMaskCuda(const NDArray& input, int64_t min, int64_t max,
                  NDArray& output, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT_EXCHANGABLE(input, output);

  size_t size = input->numel();
  if (size == 0)
    return;
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "RangeMaskCuda", [&]() {
      using InType = std::tuple<spec_t>;
      using OutType = thrust::tuple<spec_t>;
      launch_loop_kernel<InType, OutType>({input}, {output}, size, stream,
                                         [min, max] __device__ (spec_t x) -> spec_t {
                                           spec_t zero = 0;
                                           spec_t one = 1.0f;
                                           return ((static_cast<int64_t>(x) >= min) &&
                                                   (static_cast<int64_t>(x) <= max)) ? zero : one;
                                         });
    });
  NDArray::MarkUsedBy({input, output}, stream);
}

} // namespace impl
} // namespace hetu
