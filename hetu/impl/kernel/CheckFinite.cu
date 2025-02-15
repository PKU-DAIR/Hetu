#include "hetu/core/ndarray.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/utils/offset_calculator.cuh"
#include "hetu/impl/kernel/Vectorized.cuh"

namespace hetu {
namespace impl {

template <typename spec_t>
__global__ void check_finite_kernel(const spec_t* input, size_t size, float* output,
                                    const OffsetCalculator* in_offset_calculator) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  auto in_offset = in_offset_calculator->get(idx);
  if (!isfinite(float(input[in_offset])))
    output[0] = 1.f;
  else 
    output[0] = 0.f;
}

void CheckFiniteCuda(const NDArray& input, NDArray& output, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);

  size_t size = input->numel();
  if (size == 0)
    return;
  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  NDArray in_offset_calculator_arr, out_offset_calculator_arr;
  OffsetCalculator *in_offset_calculator, *out_offset_calculator;
  std::tie(in_offset_calculator_arr, in_offset_calculator) =
    AllocOffsetCalculator(input, stream);
  std::tie(out_offset_calculator_arr, out_offset_calculator) = 
    AllocOffsetCalculator(output, stream);
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "CheckFiniteCuda", [&]() {
      check_finite_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        input->data_ptr<spec_t>(), size, output->data_ptr<float>(),
        in_offset_calculator);
    });
  NDArray::MarkUsedBy({input, output, in_offset_calculator_arr,
                      out_offset_calculator_arr}, stream);
}

template <typename spec_t>
__global__ void check_numeric_kernel(const spec_t* input, size_t size, float* output,
                                    const OffsetCalculator* in_offset_calculator) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  auto in_offset = in_offset_calculator->get(idx);
  if (isnan(input[in_offset])) {
    output[0] = 1;
  }
  if (isinf(input[in_offset])) {
    output[input[in_offset] < spec_t(0.f) ? 1 : 2] = 1;
  }
}

void CheckNumericCuda(const NDArray& input, NDArray& output, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);

  size_t size = input->numel();
  if (size == 0)
    return;
  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  NDArray in_offset_calculator_arr, out_offset_calculator_arr;
  OffsetCalculator *in_offset_calculator, *out_offset_calculator;
  std::tie(in_offset_calculator_arr, in_offset_calculator) =
    AllocOffsetCalculator(input, stream);
  std::tie(out_offset_calculator_arr, out_offset_calculator) = 
    AllocOffsetCalculator(output, stream);
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "CheckFiniteCuda", [&]() {
      using InType = std::tuple<>;
      using OutType = thrust::tuple<spec_t>;
      launch_loop_kernel_with_idx<InType, OutType>({}, {output}, size, stream,
                                                  [=] __device__ (int /*idx*/) -> spec_t {
                                                    return static_cast<spec_t>(0.f);
                                                  });
      check_numeric_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        input->data_ptr<spec_t>(), size, output->data_ptr<float>(),
        in_offset_calculator);
    });
  NDArray::MarkUsedBy({input, output, in_offset_calculator_arr,
                      out_offset_calculator_arr}, stream);
}

} // namespace impl
} // namespace hetu
