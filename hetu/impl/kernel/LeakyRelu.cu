#include "hetu/core/ndarray.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/utils/offset_calculator.cuh"
#include "hetu/impl/kernel/Vectorized.cuh"

namespace hetu {
namespace impl {

void LeakyReluCuda(const NDArray& input, double alpha, NDArray& output,
                   const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT_SAME_SHAPE(input, output);

  size_t size = output->numel();
  if (size == 0)
    return;
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "LeakyReluCuda", [&]() {
      using InType = std::tuple<spec_t>;
      using OutType = thrust::tuple<spec_t>;
      launch_loop_kernel<InType, OutType>({input}, {output}, size, stream,
                                         [alpha] __device__ (spec_t in) -> spec_t {
                                           return static_cast<double>(in) < 0 ? in * static_cast<spec_t>(alpha)
                                                                              : in;
                                        });
  });
  NDArray::MarkUsedBy({input, output}, stream);
}

void LeakyReluGradientCuda(const NDArray& input, const NDArray& output_grad,
                           double alpha, NDArray& input_grad,
                           const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output_grad);
  HT_ASSERT_SAME_DEVICE(input, input_grad);

  size_t size = input_grad->numel();
  if (size == 0)
    return;
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "LeakyReluGradientCuda", [&]() {
      using InType = std::tuple<spec_t, spec_t>;
      using OutType = thrust::tuple<spec_t>;
      launch_loop_kernel<InType, OutType>({input, output_grad}, {input_grad}, size, stream,
                                                 [alpha] __device__ (spec_t in, spec_t out_grad) -> spec_t {
                                                   return static_cast<double>(in) < 0 ? out_grad * static_cast<spec_t>(alpha)
                                                                                      : out_grad;
                                                });
  });
  NDArray::MarkUsedBy({input, output_grad, input_grad}, stream);
}

} // namespace impl
} // namespace hetu
