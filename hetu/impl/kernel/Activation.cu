#include "hetu/core/ndarray.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/utils/cuda_math.h"
#include "hetu/impl/utils/offset_calculator.cuh"
#include "hetu/impl/kernel/Vectorized.cuh"

namespace hetu {
namespace impl {

void EluCuda(const NDArray& input, double alpha, double scale, NDArray& output, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT_EXCHANGABLE(input, output);

  size_t size = output->numel();
  if (size == 0)
    return;
  HT_DISPATCH_FLOATING_TYPES(
    input->dtype(), spec_t, "EluCuda", [&]() {
      using InType = std::tuple<spec_t>;
      using OutType = thrust::tuple<spec_t>;
      launch_loop_kernel<InType, OutType>({input}, {output}, size, stream,
                                         [=] __device__ (spec_t x) -> spec_t {
                                           return x > 0 ? scale * x
                                                        : scale * alpha * (hetu::cuda::cuda_exp(x) - 1);
                                         });
    });
  NDArray::MarkUsedBy({input, output}, stream);
}

void EluGradientCuda(const NDArray& output, const NDArray& output_grad,
                     double alpha, double scale, NDArray& input_grad, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(output);
  HT_ASSERT_SAME_DEVICE(output, output_grad);
  HT_ASSERT_SAME_DEVICE(output, input_grad);
  HT_ASSERT_EXCHANGABLE(output, output_grad);
  HT_ASSERT_EXCHANGABLE(output, input_grad);

  size_t size = input_grad->numel();
  if (size == 0)
    return;
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    output->dtype(), spec_t, "EluGradientCuda", [&]() {
      using InType = std::tuple<spec_t, spec_t>;
      using OutType = thrust::tuple<spec_t>;
      launch_loop_kernel<InType, OutType>({output, output_grad}, {input_grad}, 
                                          size, stream,
        [=] __device__ (spec_t x, spec_t y) -> spec_t {
          return x <= 0 ? y * (x + alpha) * scale
                        : y * scale;
        });
  });
  NDArray::MarkUsedBy({output, output_grad, input_grad}, stream);
}

void HardshrinkCuda(const NDArray& input, double lambda, NDArray& output, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT_EXCHANGABLE(input, output);

  size_t size = output->numel();
  if (size == 0)
    return;
  HT_DISPATCH_FLOATING_TYPES(
    input->dtype(), spec_t, "HardshrinkCuda", [&]() {
      using InType = std::tuple<spec_t>;
      using OutType = thrust::tuple<spec_t>;
      launch_loop_kernel<InType, OutType>({input}, {output}, size, stream,
                                         [=] __device__ (spec_t x) -> spec_t {
                                           return (x >= -lambda && x <= lambda) ? spec_t(0)
                                                                                : x;
                                         });
    });
  NDArray::MarkUsedBy({input, output}, stream);
}

void HardshrinkGradientCuda(const NDArray& output, const NDArray& output_grad,
                            double lambda, NDArray& input_grad, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(output);
  HT_ASSERT_SAME_DEVICE(output, output_grad);
  HT_ASSERT_SAME_DEVICE(output, input_grad);
  HT_ASSERT_EXCHANGABLE(output, output_grad);
  HT_ASSERT_EXCHANGABLE(output, input_grad);

  size_t size = input_grad->numel();
  if (size == 0)
    return;
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    output->dtype(), spec_t, "HardshrinkGradientCuda", [&]() {
      using InType = std::tuple<spec_t, spec_t>;
      using OutType = thrust::tuple<spec_t>;
      launch_loop_kernel<InType, OutType>({output, output_grad}, {input_grad}, 
                                          size, stream,
        [=] __device__ (spec_t x, spec_t y) -> spec_t {
          return (x == 0) ? spec_t(0)
                          : y;
        });
  });
  NDArray::MarkUsedBy({output, output_grad, input_grad}, stream);
}

void HardsigmoidCuda(const NDArray& input, NDArray& output, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT_EXCHANGABLE(input, output);

  size_t size = output->numel();
  if (size == 0)
    return;
  HT_DISPATCH_FLOATING_TYPES(
    input->dtype(), spec_t, "HardsigmoidCuda", [&]() {
      spec_t six_percent_one = spec_t(1.0 / 6.0);
      spec_t two_percent_one = spec_t(0.5);
      using InType = std::tuple<spec_t>;
      using OutType = thrust::tuple<spec_t>;
      launch_loop_kernel<InType, OutType>({input}, {output}, size, stream,
                                         [=] __device__ (spec_t x) -> spec_t {
                                            if (x < -3) 
                                              return 0;
                                            if (x > 3)
                                              return 1;
                                            return x * six_percent_one + two_percent_one;
                                         });
    });
  NDArray::MarkUsedBy({input, output}, stream);
}

void HardsigmoidGradientCuda(const NDArray& output, const NDArray& output_grad,
                             NDArray& input_grad, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(output);
  HT_ASSERT_SAME_DEVICE(output, output_grad);
  HT_ASSERT_SAME_DEVICE(output, input_grad);
  HT_ASSERT_EXCHANGABLE(output, output_grad);
  HT_ASSERT_EXCHANGABLE(output, input_grad);

  size_t size = input_grad->numel();
  if (size == 0)
    return;
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    output->dtype(), spec_t, "HardsigmoidGradientCuda", [&]() {
      spec_t six_percent_one = spec_t(1.0 / 6.0);
      spec_t zero = spec_t(0);
      using InType = std::tuple<spec_t, spec_t>;
      using OutType = thrust::tuple<spec_t>;
      launch_loop_kernel<InType, OutType>({output, output_grad}, {input_grad}, 
                                          size, stream,
        [=] __device__ (spec_t x, spec_t y) -> spec_t {
          return (x > 0 && x < 1) ? six_percent_one * y
                                  : zero;
        });
  });
  NDArray::MarkUsedBy({output, output_grad, input_grad}, stream);
}

void HardtanhCuda(const NDArray& input, double min_val, double max_val,
                  NDArray& output, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT_EXCHANGABLE(input, output);

  size_t size = output->numel();
  if (size == 0)
    return;
  HT_DISPATCH_FLOATING_TYPES(
    input->dtype(), spec_t, "HardtanhCuda", [&]() {
      using InType = std::tuple<spec_t>;
      using OutType = thrust::tuple<spec_t>;
      launch_loop_kernel<InType, OutType>({input}, {output}, size, stream,
                                         [=] __device__ (spec_t x) -> spec_t {
                                            return x < min_val ? spec_t(min_val)
                                                               : (x > max_val ? spec_t(max_val) : x);
                                         });
    });
  NDArray::MarkUsedBy({input, output}, stream);
}

void HardtanhGradientCuda(const NDArray& output, const NDArray& output_grad,
                          double min_val, double max_val,
                          NDArray& input_grad, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(output);
  HT_ASSERT_SAME_DEVICE(output, output_grad);
  HT_ASSERT_SAME_DEVICE(output, input_grad);
  HT_ASSERT_EXCHANGABLE(output, output_grad);
  HT_ASSERT_EXCHANGABLE(output, input_grad);

  size_t size = input_grad->numel();
  if (size == 0)
    return;
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    output->dtype(), spec_t, "HardtanhGradientCuda", [&]() {
      using InType = std::tuple<spec_t, spec_t>;
      using OutType = thrust::tuple<spec_t>;
      launch_loop_kernel<InType, OutType>({output, output_grad}, {input_grad}, 
                                          size, stream,
        [=] __device__ (spec_t x, spec_t y) -> spec_t {
          return (x > min_val && x < max_val) ? y
                                              : spec_t(0);
        });
  });
  NDArray::MarkUsedBy({output, output_grad, input_grad}, stream);
}

void HardswishCuda(const NDArray& input, NDArray& output, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT_EXCHANGABLE(input, output);

  size_t size = output->numel();
  if (size == 0)
    return;
  HT_DISPATCH_FLOATING_TYPES(
    input->dtype(), spec_t, "HardswishCuda", [&]() {
      using InType = std::tuple<spec_t>;
      using OutType = thrust::tuple<spec_t>;
      launch_loop_kernel<InType, OutType>({input}, {output}, size, stream,
                                         [] __device__ (spec_t x) -> spec_t {
                                            return x < -3 ? spec_t(0)
                                                          : (x > 3 ? x : (x * (x + 3) / 6));
                                         });
    });
  NDArray::MarkUsedBy({input, output}, stream);
}

void HardswishGradientCuda(const NDArray& input, const NDArray& output_grad,
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
    input->dtype(), spec_t, "HardswishGradientCuda", [&]() {
      using InType = std::tuple<spec_t, spec_t>;
      using OutType = thrust::tuple<spec_t>;
      launch_loop_kernel<InType, OutType>({input, output_grad}, {input_grad}, 
                                          size, stream,
        [] __device__ (spec_t x, spec_t y) -> spec_t {
          return (x < -3) ? spec_t(0)
                          : (x > 3 ? y : y * (2 * x + 3) / 6);
        });
  });
  NDArray::MarkUsedBy({input, output_grad, input_grad}, stream);
}

void LogsigmoidCuda(const NDArray& input, NDArray& output, bool inverse, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT_EXCHANGABLE(input, output);

  size_t size = output->numel();
  if (size == 0)
    return;
  if (!inverse) {
    HT_DISPATCH_FLOATING_TYPES(
      input->dtype(), spec_t, "LogsigmoidCuda", [&]() {
        spec_t zero = spec_t(0);
        using InType = std::tuple<spec_t>;
        using OutType = thrust::tuple<spec_t>;
        launch_loop_kernel<InType, OutType>({input}, {output}, size, stream,
                                          [=] __device__ (spec_t x) -> spec_t {
                                              spec_t min_x_0 = hetu::cuda::cuda_min(x, zero);
                                              spec_t z = hetu::cuda::cuda_exp(-hetu::cuda::cuda_abs(x));
                                              return min_x_0 - (hetu::cuda::cuda_log(1 + z));
                                          });
    });
  }
  else {
    HT_DISPATCH_FLOATING_TYPES(
    input->dtype(), spec_t, "LogsigmoidCuda", [&]() {
      spec_t zero = spec_t(0);
      using InType = std::tuple<spec_t>;
      using OutType = thrust::tuple<spec_t>;
      launch_loop_kernel<InType, OutType>({input}, {output}, size, stream,
                                         [=] __device__ (spec_t x) -> spec_t {
                                            spec_t min_x_0 = hetu::cuda::cuda_min(-x, zero);
                                            spec_t z = hetu::cuda::cuda_exp(-hetu::cuda::cuda_abs(-x));
                                            return min_x_0 - (hetu::cuda::cuda_log(1 + z));
                                         });
    });
  }
  NDArray::MarkUsedBy({input, output}, stream);
}

void LogsigmoidGradientCuda(const NDArray& input, const NDArray& output_grad,
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
    input->dtype(), spec_t, "LogsigmoidGradientCuda", [&]() {
      using InType = std::tuple<spec_t, spec_t>;
      using OutType = thrust::tuple<spec_t>;
      launch_loop_kernel<InType, OutType>({input, output_grad}, {input_grad}, 
                                          size, stream,
        [] __device__ (spec_t x, spec_t y) -> spec_t {
          spec_t z = hetu::cuda::cuda_exp(-hetu::cuda::cuda_abs(x));
          return x < 0 ? y * (1 - (z / (1 + z)))
                       : y * (z / (1 + z));
        });
  });
  NDArray::MarkUsedBy({input, output_grad, input_grad}, stream);
}

void SiluCuda(const NDArray& input, NDArray& output, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT_EXCHANGABLE(input, output);

  size_t size = output->numel();
  if (size == 0)
    return;
  HT_DISPATCH_FLOATING_TYPES(
    input->dtype(), spec_t, "SiluCuda", [&]() {
      using InType = std::tuple<spec_t>;
      using OutType = thrust::tuple<spec_t>;
      launch_loop_kernel<InType, OutType>({input}, {output}, size, stream,
                                         [] __device__ (spec_t x) -> spec_t {
                                            return x / (1 / 1 + hetu::cuda::cuda_exp(-x));
                                         });
    });
  NDArray::MarkUsedBy({input, output}, stream);
}

void SiluGradientCuda(const NDArray& input, const NDArray& output_grad,
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
    input->dtype(), spec_t, "SiluGradientCuda", [&]() {
      using InType = std::tuple<spec_t, spec_t>;
      using OutType = thrust::tuple<spec_t>;
      launch_loop_kernel<InType, OutType>({input, output_grad}, {input_grad}, 
                                          size, stream,
        [] __device__ (spec_t x, spec_t y) -> spec_t {
          spec_t z = 1 / (1 + hetu::cuda::cuda_exp(-x));
          return y * z * (1 + x * (1 - z));
        });
  });
  NDArray::MarkUsedBy({input, output_grad, input_grad}, stream);
}

void MishCuda(const NDArray& input, NDArray& output, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT_EXCHANGABLE(input, output);

  size_t size = output->numel();
  if (size == 0)
    return;
  HT_DISPATCH_FLOATING_TYPES(
    input->dtype(), spec_t, "MishCuda", [&]() {
      using InType = std::tuple<spec_t>;
      using OutType = thrust::tuple<spec_t>;
      launch_loop_kernel<InType, OutType>({input}, {output}, size, stream,
                                         [] __device__ (spec_t x) -> spec_t {
                                            return x * hetu::cuda::cuda_tanh(
                                                       hetu::cuda::cuda_log(1 + hetu::cuda::cuda_exp(x)));
                                         });
    });
  NDArray::MarkUsedBy({input, output}, stream);
}

void MishGradientCuda(const NDArray& input, const NDArray& output_grad,
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
    input->dtype(), spec_t, "MishGradientCuda", [&]() {
      using InType = std::tuple<spec_t, spec_t>;
      using OutType = thrust::tuple<spec_t>;
      launch_loop_kernel<InType, OutType>({input, output_grad}, {input_grad}, 
                                          size, stream,
        [] __device__ (spec_t x, spec_t y) -> spec_t {
          spec_t sigmoid = 1 / (1 + hetu::cuda::cuda_exp(-x));
          spec_t z = hetu::cuda::cuda_tanh(hetu::cuda::cuda_log(1 + hetu::cuda::cuda_exp(x)));
          return y * (z + x * sigmoid * (1 - z * z));
        });
  });
  NDArray::MarkUsedBy({input, output_grad, input_grad}, stream);
}

void SoftplusCuda(const NDArray& input, double beta, double threshold,
                  NDArray& output, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT_EXCHANGABLE(input, output);

  size_t size = output->numel();
  if (size == 0)
    return;
  HT_DISPATCH_FLOATING_TYPES(
    input->dtype(), spec_t, "SoftplusCuda", [&]() {
      spec_t beta_ = spec_t(beta);
      spec_t one = spec_t(1);
      using InType = std::tuple<spec_t>;
      using OutType = thrust::tuple<spec_t>;
      launch_loop_kernel<InType, OutType>({input}, {output}, size, stream,
                                         [=] __device__ (spec_t x) -> spec_t {
                                            return x * beta > threshold ? x
                                                                        : hetu::cuda::cuda_log(
                                                                          one + hetu::cuda::cuda_exp(x * beta_)) 
                                                                          / beta_;
                                         });
    });
  NDArray::MarkUsedBy({input, output}, stream);
}

void SoftplusGradientCuda(const NDArray& input, const NDArray& output_grad,
                          double beta, double threshold, NDArray& input_grad, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output_grad);
  HT_ASSERT_SAME_DEVICE(input, input_grad);
  HT_ASSERT_EXCHANGABLE(input, output_grad);
  HT_ASSERT_EXCHANGABLE(input, input_grad);

  size_t size = input_grad->numel();
  if (size == 0)
    return;
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "SoftplusGradientCuda", [&]() {
      using InType = std::tuple<spec_t, spec_t>;
      using OutType = thrust::tuple<spec_t>;
      launch_loop_kernel<InType, OutType>({input, output_grad}, {input_grad}, 
                                          size, stream,
        [=] __device__ (spec_t x, spec_t y) -> spec_t {
          spec_t z = hetu::cuda::cuda_exp(x * beta);
          return (x * beta > threshold) ? y
                                        : y * z / (z + 1);
        });
  });
  NDArray::MarkUsedBy({input, output_grad, input_grad}, stream);
}

void SoftshrinkCuda(const NDArray& input, double lambda, NDArray& output, 
                    const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT_EXCHANGABLE(input, output);

  size_t size = output->numel();
  if (size == 0)
    return;
  HT_DISPATCH_FLOATING_TYPES(
    input->dtype(), spec_t, "SoftshrinkCuda", [&]() {
      spec_t lambda_ = spec_t(lambda);
      spec_t zero = spec_t(0);
      using InType = std::tuple<spec_t>;
      using OutType = thrust::tuple<spec_t>;
      launch_loop_kernel<InType, OutType>({input}, {output}, size, stream,
                                         [=] __device__ (spec_t x) -> spec_t {
                                            return x < -lambda_ ? x + lambda_
                                                               : (x > lambda_ ? x - lambda_ : zero);
                                         });
    });
  NDArray::MarkUsedBy({input, output}, stream);
}

void SoftshrinkGradientCuda(const NDArray& output, const NDArray& output_grad,
                            double lambda, NDArray& input_grad, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(output);
  HT_ASSERT_SAME_DEVICE(output, output_grad);
  HT_ASSERT_SAME_DEVICE(output, input_grad);
  HT_ASSERT_EXCHANGABLE(output, output_grad);
  HT_ASSERT_EXCHANGABLE(output, input_grad);

  size_t size = input_grad->numel();
  if (size == 0)
    return;
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    output->dtype(), spec_t, "SoftshrinkGradientCuda", [&]() {
      using InType = std::tuple<spec_t, spec_t>;
      using OutType = thrust::tuple<spec_t>;
      launch_loop_kernel<InType, OutType>({output, output_grad}, {input_grad}, 
                                          size, stream,
        [] __device__ (spec_t x, spec_t y) -> spec_t {
          return (x == 0) ? spec_t(0) : y;
        });
  });
  NDArray::MarkUsedBy({output, output_grad, input_grad}, stream);
}

} // namespace impl
} // namespace hetu
