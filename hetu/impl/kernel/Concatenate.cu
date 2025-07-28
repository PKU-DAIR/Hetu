#include "hetu/core/ndarray.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/utils/offset_calculator.cuh"

namespace hetu {
namespace impl {

template <typename spec_t>
__global__ void concatenate_kernel(const spec_t* input, spec_t* output,
                                   int input_width, int output_width,
                                   int offset, int concat_size, size_t size,
                                   const OffsetCalculator* in_offset_calculator,
                                   const OffsetCalculator* out_offset_calculator) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  int post_ind = idx % concat_size;
  int prev_ind = idx / concat_size;
  int mid_ind = prev_ind % input_width + offset;
  prev_ind = prev_ind / input_width;
  int out_ind = (prev_ind * output_width + mid_ind) * concat_size + post_ind;
  auto in_offset = in_offset_calculator->get(idx);
  auto out_offset = out_offset_calculator->get(out_ind);
  output[out_offset] = input[in_offset];
}

template <typename spec_t>
__global__ void concatenate_gradient_kernel(const spec_t* output_grad,
                                            spec_t* input_grad, int input_width,
                                            int output_width, int offset,
                                            int concat_size, size_t size,
                                            const OffsetCalculator* out_grad_offset_calculator,
                                            const OffsetCalculator* in_grad_offset_calculator) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  int post_ind = idx % concat_size;
  int prev_ind = idx / concat_size;
  int mid_ind = prev_ind % input_width + offset;
  prev_ind = prev_ind / input_width;
  int out_ind = (prev_ind * output_width + mid_ind) * concat_size + post_ind;
  auto out_grad_offset = out_grad_offset_calculator->get(out_ind);
  auto in_grad_offset = in_grad_offset_calculator->get(idx);
  input_grad[in_grad_offset] = output_grad[out_grad_offset];
}

void ConcatenateCuda(const NDArray& input, NDArray& output, size_t axis,
                     size_t offset, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);

  size_t size = input->numel();
  int now_ndim = output->ndim();
  HT_ASSERT(input->ndim() == now_ndim);
  int num_concats = 1;
  for (int i = 0; i < axis; ++i) {
    int cur_dim = output->shape(i);
    HT_ASSERT(input->shape(i) == cur_dim);
    num_concats *= cur_dim;
  }
  int concat_size = 1;
  for (int i = axis + 1; i < now_ndim; ++i) {
    int cur_dim = output->shape(i);
    HT_ASSERT(input->shape(i) == cur_dim);
    concat_size *= cur_dim;
  }
  int input_width = input->shape(axis);
  int output_width = output->shape(axis);
  if (size == 0 || input_width == 0 || output_width == 0)
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
    input->dtype(), spec_t, "ConcatenateCuda", [&]() {
      concatenate_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        input->data_ptr<spec_t>(), output->data_ptr<spec_t>(), input_width,
        output_width, offset, concat_size, size,
        in_offset_calculator, out_offset_calculator);
    });
  NDArray::MarkUsedBy({input, output, in_offset_calculator_arr,
                      out_offset_calculator_arr}, stream);
}

void ConcatenateGradientCuda(const NDArray& output_grad, NDArray& input_grad,
                             size_t axis, size_t offset, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(output_grad);
  HT_ASSERT_SAME_DEVICE(output_grad, input_grad);

  size_t size = input_grad->numel();
  int now_ndim = output_grad->ndim();
  HT_ASSERT(now_ndim == input_grad->ndim());
  int num_concats = 1;
  for (int i = 0; i < axis; ++i) {
    int cur_dim = output_grad->shape(i);
    HT_ASSERT(cur_dim == input_grad->shape(i));
    num_concats *= cur_dim;
  }
  int concat_size = 1;
  for (int i = axis + 1; i < now_ndim; ++i) {
    int cur_dim = output_grad->shape(i);
    HT_ASSERT(cur_dim == input_grad->shape(i));
    concat_size *= cur_dim;
  }
  int output_width = output_grad->shape(axis);
  int input_width = input_grad->shape(axis);
  if (size == 0 || input_width == 0 || output_width == 0)
    return;
  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  NDArray out_grad_offset_calculator_arr, in_grad_offset_calculator_arr;
  OffsetCalculator *out_grad_offset_calculator, *in_grad_offset_calculator;
  std::tie(out_grad_offset_calculator_arr, out_grad_offset_calculator) =
    AllocOffsetCalculator(output_grad, stream);
  std::tie(in_grad_offset_calculator_arr, in_grad_offset_calculator) = 
    AllocOffsetCalculator(input_grad, stream);
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input_grad->dtype(), spec_t, "ConcatenateGradientCuda", [&]() {
      concatenate_gradient_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        output_grad->data_ptr<spec_t>(), input_grad->data_ptr<spec_t>(),
        input_width, output_width, offset, concat_size, size,
        out_grad_offset_calculator, in_grad_offset_calculator);
    });
  NDArray::MarkUsedBy({output_grad, input_grad, out_grad_offset_calculator_arr,
                      in_grad_offset_calculator_arr}, stream);
}

} // namespace impl
} // namespace hetu
