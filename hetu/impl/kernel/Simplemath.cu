#include "hetu/core/ndarray.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/utils/cuda_math.h"
#include "hetu/impl/utils/offset_calculator.cuh"
#include "hetu/impl/kernel/Vectorized.cuh"

namespace hetu {
namespace impl {

void FloorCuda(const NDArray& input, NDArray& output, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT_SAME_SHAPE(input, output);

  size_t size = input->numel();
  if (size == 0)
    return;
  HT_DISPATCH_FLOATING_TYPES(
    input->dtype(), spec_t, "FloorCuda", [&]() {
      using InType = std::tuple<spec_t>;
      using OutType = thrust::tuple<spec_t>;
      launch_loop_kernel<InType, OutType>({input}, {output}, size, stream,
                                         [] __device__ (spec_t x) -> spec_t {
                                           return hetu::cuda::cuda_floor(x);
                                         });
    });
  NDArray::MarkUsedBy({input, output}, stream);
}

void CeilCuda(const NDArray& input, NDArray& output, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT_SAME_SHAPE(input, output);

  size_t size = input->numel();
  if (size == 0)
    return;
  HT_DISPATCH_FLOATING_TYPES(
    input->dtype(), spec_t, "CeilCuda", [&]() {
      using InType = std::tuple<spec_t>;
      using OutType = thrust::tuple<spec_t>;
      launch_loop_kernel<InType, OutType>({input}, {output}, size, stream,
                                         [] __device__ (spec_t x) -> spec_t {
                                           return hetu::cuda::cuda_ceil(x);
                                         });
    });
  NDArray::MarkUsedBy({input, output}, stream);
}

void RoundCuda(const NDArray& input, NDArray& output, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT_SAME_SHAPE(input, output);

  size_t size = input->numel();
  if (size == 0)
    return;
  HT_DISPATCH_FLOATING_TYPES(
    input->dtype(), spec_t, "RoundCuda", [&]() {
      using InType = std::tuple<spec_t>;
      using OutType = thrust::tuple<spec_t>;
      launch_loop_kernel<InType, OutType>({input}, {output}, size, stream,
                                         [] __device__ (spec_t x) -> spec_t {
                                           return hetu::cuda::cuda_round(x);
                                         });
    });
  NDArray::MarkUsedBy({input, output}, stream);
}

} // namespace impl
} // namespace hetu
