#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/utils/cuda_math.h"
#include "hetu/impl/utils/offset_calculator.cuh"
#include "hetu/impl/kernel/Vectorized.cuh"

namespace hetu {
namespace impl {

void AbsCuda(const NDArray& input, NDArray& output, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT_SAME_SHAPE(input, output);

  size_t size = input->numel();
  if (size == 0)
    return;
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "AbsCuda", [&]() {
      using InType = std::tuple<spec_t>;
      using OutType = thrust::tuple<spec_t>;
      launch_loop_kernel<InType, OutType>({input}, {output}, size, stream,
                                         [] __device__ (spec_t x) -> spec_t {
                                           return hetu::cuda::cuda_abs(x);
                                         });
    });
  NDArray::MarkUsedBy({input, output}, stream);
}

void AbsGradientCuda(const NDArray& input, const NDArray& output_grad, NDArray& input_grad, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output_grad);
  HT_ASSERT_SAME_DEVICE(input, input_grad);
  HT_ASSERT_SAME_SHAPE(input, output_grad);
  HT_ASSERT_SAME_SHAPE(input, input_grad);

  size_t size = input_grad->numel();
  if (size == 0)
    return;
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "AbsGradientCuda", [&]() {
      using InType = std::tuple<spec_t, spec_t>;
      using OutType = thrust::tuple<spec_t>;
      launch_loop_kernel<InType, OutType>({input, output_grad}, {input_grad}, size, stream,
                                         [] __device__ (spec_t x, spec_t y) -> spec_t {
                                                   spec_t zero = 0;
                                                   return x == zero ? zero
                                                            : x > zero ? y
                                                                       : - y;
                                                });
    });
  NDArray::MarkUsedBy({input, output_grad, input_grad}, stream);
}

} // namespace impl
} // namespace hetu
