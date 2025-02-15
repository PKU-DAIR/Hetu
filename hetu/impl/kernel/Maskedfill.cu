#include "hetu/core/ndarray.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/utils/offset_calculator.cuh"
#include "hetu/impl/kernel/Vectorized.cuh"

namespace hetu {
namespace impl {

void MaskedfillCuda(const NDArray& input, const NDArray& mask,
                    double val, NDArray& output, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, mask);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT_SAME_SHAPE(input, output);

  size_t size = input->numel();
  if (size == 0)
    return;
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "MaskfillCuda", [&]() {
      using InType = std::tuple<spec_t, int64_t>;
      using OutType = thrust::tuple<spec_t>;
      launch_loop_kernel<InType, OutType>({input, mask}, {output}, size, stream,
                                         [val] __device__ (spec_t in, int64_t mask_) -> spec_t {
                                           return bool(mask_) ? static_cast<spec_t>(val) : in;
                                         });
  });
  NDArray::MarkUsedBy({input, mask, output}, stream);
}

} // namespace impl
} // namespace hetu
