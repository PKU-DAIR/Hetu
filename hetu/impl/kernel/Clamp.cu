#include "hetu/core/ndarray.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/utils/offset_calculator.cuh"
#include "hetu/impl/kernel/Vectorized.cuh"

namespace hetu {
namespace impl {

void ClampCuda(const NDArray& input, double min_val, double max_val, NDArray& output, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT_EXCHANGABLE(input, output);

  size_t size = input->numel();
  if (size == 0)
    return;
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "ClampCuda", [&]() {
      using InType = std::tuple<spec_t>;
      using OutType = thrust::tuple<spec_t>;
      launch_loop_kernel<InType, OutType>({input}, {output}, size, stream,
        [min_val, max_val] __device__ (spec_t in) {
          spec_t min_v = min_val, max_v = max_val;
          if (in < min_v)
            return min_v;
          else if (in > max_v)
            return max_v;
          else
            return in;
        });
    });
  NDArray::MarkUsedBy({input, output}, stream);
}

void ClampElewiseCuda(const NDArray& input, const NDArray& min_val, const NDArray& max_val, NDArray& output, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT_EXCHANGABLE(input, output);

  size_t size = input->numel();
  if (size == 0)
    return;
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "ClampCuda", [&]() {
      using InType = std::tuple<spec_t, spec_t, spec_t>;
      using OutType = thrust::tuple<spec_t>;
      launch_loop_kernel<InType, OutType>({input, min_val, max_val}, {output}, size, stream,
        [] __device__ (spec_t in, spec_t min_v, spec_t max_v) {
          if (in < min_v)
            return min_v;
          else if (in > max_v)
            return max_v;
          else
            return in;
        });
  });
  NDArray::MarkUsedBy({input, min_val, max_val, output}, stream);
}

} // namespace impl
} // namespace hetu
