#include "hetu/core/ndarray.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/utils/cuda_math.h"
#include "hetu/impl/utils/offset_calculator.cuh"

namespace hetu {
namespace impl {
template <typename spec_t>
/*
  假设输入dim [seq_len, d * 2] -> [seq_len, d]
*/
__global__ void swiglu_kernel(const spec_t* input, size_t size, size_t d,spec_t* output,
                            const OffsetCalculator* in_offset_calculator,
                            const OffsetCalculator* out_offset_calculator) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  auto in_idx = (idx / d) * (2 * d) + (idx % d);
  auto in_x_offset = in_offset_calculator->get(in_idx);
  auto in_y_offset = in_offset_calculator->get(in_idx + d);
  auto out_offset = out_offset_calculator->get(idx);
  spec_t x = input[in_x_offset];
  spec_t y = input[in_y_offset];
  output[out_offset] = float(y) * float(x / (1.0f + hetu::cuda::cuda_exp(-x)));
}

template <typename spec_t>
__global__ void swiglu_gradient_kernel(const spec_t* input, const spec_t* output_grad,
                     size_t size, size_t d, spec_t* input_grad,
                     const OffsetCalculator* in_offset_calculator,
                     const OffsetCalculator* out_grad_offset_calculator,
                     const OffsetCalculator* in_grad_offset_calculator) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;

  auto in_idx = (idx / d) * (2 * d) + (idx % d);
  auto in_x_offset = in_offset_calculator->get(in_idx);
  auto in_y_offset = in_offset_calculator->get(in_idx + d);
  auto outg_offset = out_grad_offset_calculator->get(idx);
  auto ing_x_offset = in_grad_offset_calculator->get(in_idx);
  auto ing_y_offset = in_grad_offset_calculator->get(in_idx + d);

  spec_t x = input[in_x_offset];
  spec_t y = input[in_y_offset];

  spec_t sigmoid_x = 1 / (1.0f + hetu::cuda::cuda_exp(-x));
  input_grad[ing_y_offset] = output_grad[outg_offset] * x * sigmoid_x;
  input_grad[ing_x_offset] = output_grad[outg_offset] * y * sigmoid_x *
    (1.0f + x * (1 - sigmoid_x));
}

void SwigluCuda(const NDArray& input, NDArray& output, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);

  size_t size = output->numel();
  size_t d_size = output->shape().back();
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
    input->dtype(), spec_t, "swigluCuda", [&]() {
      swiglu_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        input->data_ptr<spec_t>(), size, d_size, output->data_ptr<spec_t>(),
        in_offset_calculator, out_offset_calculator);
    });
  NDArray::MarkUsedBy(
    {input, output, in_offset_calculator_arr, out_offset_calculator_arr},
    stream);
}

void SwigluGradientCuda(const NDArray& input, const NDArray& output_grad,
                      NDArray& input_grad, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output_grad);
  HT_ASSERT_SAME_DEVICE(input, input_grad);

  size_t d_size = output_grad->shape().back();
  size_t size = output_grad->numel();
  
  if (size == 0)
    return;
  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  NDArray in_offset_calculator_arr, out_grad_offset_calculator_arr,
    in_grad_offset_calculator_arr;
  OffsetCalculator *in_offset_calculator, *out_grad_offset_calculator,
    *in_grad_offset_calculator;
  std::tie(in_offset_calculator_arr, in_offset_calculator) =
    AllocOffsetCalculator(input, stream);
  std::tie(out_grad_offset_calculator_arr, out_grad_offset_calculator) =
    AllocOffsetCalculator(output_grad, stream);
  std::tie(in_grad_offset_calculator_arr, in_grad_offset_calculator) =
    AllocOffsetCalculator(input_grad, stream);
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "SwigluGradientCuda", [&]() {
      swiglu_gradient_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        input->data_ptr<spec_t>(), output_grad->data_ptr<spec_t>(), size, d_size,
        input_grad->data_ptr<spec_t>(), in_offset_calculator,
        out_grad_offset_calculator, in_grad_offset_calculator);
    });
  NDArray::MarkUsedBy({input, output_grad, input_grad, in_offset_calculator_arr,
                       out_grad_offset_calculator_arr,
                       in_grad_offset_calculator_arr}, stream);
}

} // namespace impl
} // namespace hetu
