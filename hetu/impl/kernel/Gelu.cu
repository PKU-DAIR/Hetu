#include "hetu/core/ndarray.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/utils/cuda_math.h"
#include "hetu/impl/utils/offset_calculator.cuh"
#include "hetu/impl/kernel/Vectorized.cuh"

#define SQRT_1_2  0.70710678118654757274f
#define pi 3.14159265358979323846f
#define e  2.71828182845904523536f

namespace hetu {
namespace impl {

void GeluCuda(const NDArray& input, NDArray& output, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT_EXCHANGABLE(input, output);

  size_t size = output->numel();
  if (size == 0)
    return;
  HT_DISPATCH_FLOATING_TYPES(
    input->dtype(), spec_t, "GeluCuda", [&]() {
      using InType = std::tuple<spec_t>;
      using OutType = thrust::tuple<spec_t>;
      launch_loop_kernel<InType, OutType>({input}, {output}, size, stream,
                                         [] __device__ (spec_t x) -> spec_t {
                                           return x * 0.5f *
                                              (1.0f + hetu::cuda::cuda_erf(x * SQRT_1_2));
                                         });
    });
  NDArray::MarkUsedBy({input, output}, stream);
}

void GeluGradientCuda(const NDArray& input, const NDArray& output_grad,
                      NDArray& input_grad, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output_grad);
  HT_ASSERT_SAME_DEVICE(input, input_grad);
  HT_ASSERT_EXCHANGABLE(input, output_grad);
  HT_ASSERT_EXCHANGABLE(input, input_grad);

  size_t size = input_grad->numel();
  if (size == 0)
    return;
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "GeluGradientCuda", [&]() {
      using InType = std::tuple<spec_t, spec_t>;
      using OutType = thrust::tuple<spec_t>;
      launch_loop_kernel<InType, OutType>({input, output_grad}, {input_grad}, size, stream,
        [] __device__ (spec_t x, spec_t y) -> spec_t {
          return y * (0.5f + 0.5f * hetu::cuda::cuda_erf(x / hetu::cuda::cuda_sqrt(2.0f)) + 
                 0.5f * x * (hetu::cuda::cuda_sqrt(2.0f) * 
                  hetu::cuda::cuda_exp(-0.5f * hetu::cuda::cuda_pow(x, spec_t(2.0f))) /
                  hetu::cuda::cuda_sqrt(pi)));
        });
  });
  NDArray::MarkUsedBy({input, output_grad, input_grad}, stream);
}

} // namespace impl
} // namespace hetu
