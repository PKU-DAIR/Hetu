#include "hetu/core/ndarray.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/utils/offset_calculator.cuh"
#include "hetu/impl/kernel/Vectorized.cuh"

namespace hetu {
namespace impl {

void ArraySetCuda(NDArray& data, double value, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(data);
  size_t size = data->numel();
  if (size == 0)
    return;
  if (data->dtype() == kFloat4 || data->dtype() == kNFloat4) {
    NDArray::MarkUsedBy({data}, stream);
    return;
  }
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    data->dtype(), spec_t, "ArraySetCuda", [&]() {
      using InType = std::tuple<>;
      using OutType = thrust::tuple<spec_t>;
      launch_loop_kernel_with_idx<InType, OutType>({}, {data}, size, stream,
                                                  [=] __device__ (int /*idx*/) -> spec_t {
                                                    return static_cast<spec_t>(value);
                                                  });
  });
  NDArray::MarkUsedBy({data}, stream);
}

} // namespace impl
} // namespace hetu
