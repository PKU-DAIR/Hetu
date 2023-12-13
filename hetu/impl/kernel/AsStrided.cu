#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/utils/cuda_math.h"
#include "hetu/impl/kernel/Binary.cuh"
#include "hetu/impl/utils/offset_calculator.cuh"

#include <vector>
#include <numeric>

namespace hetu {
namespace impl {

namespace {

inline size_t numel(const HTShape& shape) {
  size_t num = 1;
  for (auto& s : shape) {
    num *= s;
  }
  return num;
}

inline HTStride Shape2Stride(const HTShape& shape) {
  auto size = shape.size();
  HTStride stride(size);
  if (size > 0) {
    stride[size - 1] = 1;
    for (auto d = size - 1; d > 0; d--) {
      stride[d - 1] = stride[d] * shape[d];
    }
  }
  return stride;
}

inline bool maybe_overlapping_memory(const HTShape& shape, const HTStride& stride) {
  if (!shape.empty()) {
    std::vector<size_t> argsort(shape.size());
    std::iota(argsort.begin(), argsort.end(), 0);
    std::sort(
        argsort.begin(), argsort.end(), [&](size_t i, size_t j) {
          return stride[i] < stride[j];
        });

    auto max_index_in_slice = 0;
    for (auto& i : argsort) {
      const auto& stride_i = stride[i];
      if (stride_i <= max_index_in_slice) {
        return true;
      }
      max_index_in_slice += stride_i * (shape[i] - 1);
    }
  }
  return false;
}

} // namespace

// Out-of-place version of as_strided and its gradient
/* It is replaced with in-place version. */
template <typename spec_t>
__global__ void asstrided_kernel(const spec_t* input, spec_t* output,
                                 size_t size, const int64_t* stride_in,
                                 const int64_t* stride_out, int ndim) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  int index = 0;
  size_t ind = idx;
  for (int i = 0; i < ndim; i++) {
    int tmp_index = ind / stride_out[i];
    index += tmp_index * stride_in[i];
    ind = ind % stride_out[i];
  }
  output[idx] = input[index];
}

void AsStridedCuda(const NDArray& input, NDArray& output,
                   const HTStride& stride, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  size_t size = output->numel();
  int ndim = output->ndim();
  if (size == 0)
    return;

  auto device_id = input->device().index();
  hetu::cuda::CUDADeviceGuard guard(device_id);
  CUDAStream cuda_stream(stream);
  auto stride_in_arr = hetu::cuda::to_int64_ndarray(stride, device_id);
  auto stride_out_arr =
    hetu::cuda::to_int64_ndarray(output->stride(), device_id);
  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "AsStridedCuda", [&]() {
      asstrided_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        input->data_ptr<spec_t>(), output->data_ptr<spec_t>(), size,
        stride_in_arr->data_ptr<int64_t>(), 
        stride_out_arr->data_ptr<int64_t>(), 
        ndim);
    });
  NDArray::MarkUsedBy({input, output, stride_in_arr, stride_out_arr}, stream);
}

template <typename spec_t>
__global__ void asstrided_gradient_kernel(const spec_t* input, spec_t* output,
                                          size_t size, const int64_t* stride_in,
                                          const int64_t* stride_out, int ndim) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  int index = 0;
  size_t ind = idx;
  for (int i = 0; i < ndim; i++) {
    int tmp_index = ind / stride_out[i];
    index += tmp_index * stride_in[i];
    ind = ind % stride_out[i];
  }
  hetu::cuda::AtomicAdd(&output[index], input[idx]);
}

void AsStridedGradientCuda(const NDArray& output, NDArray& input,
                           const HTStride& stride, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  size_t size = output->numel();
  int ndim = output->ndim();
  if (size == 0)
    return;

  auto device_id = input->device().index();
  hetu::cuda::CUDADeviceGuard guard(input->device().index());
  CUDAStream cuda_stream(stream);
  auto stride_in_arr =
    hetu::cuda::to_int64_ndarray(stride, device_id);
  auto stride_out_arr =
    hetu::cuda::to_int64_ndarray(output->stride(), device_id);
  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  HT_DISPATCH_FLOATING_TYPES(
    input->dtype(), spec_t, "AsStridedGradientCuda", [&]() {
      spec_t* in_ptr = input->data_ptr<spec_t>();
      CudaMemsetAsync(in_ptr, 0, input->numel() * sizeof(spec_t), cuda_stream);
      asstrided_gradient_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        output->data_ptr<spec_t>(), in_ptr, size,
        stride_in_arr->data_ptr<int64_t>(), 
        stride_out_arr->data_ptr<int64_t>(), 
        ndim);
    });
  NDArray::MarkUsedBy({input, output, stride_in_arr, stride_out_arr}, stream);
}

// In-place version of as_strided gradient
template <typename spec_a_t, typename spec_b_t, typename Operator>
extern __global__ void binary_elewise_kernel(const spec_a_t* inputA, const spec_b_t* inputB,
                                             size_t size, Operator op, spec_a_t* output,
                                             const OffsetCalculator* A_offset_calculator,
                                             const OffsetCalculator* B_offset_calculator,
                                             const OffsetCalculator* out_offset_calculator);

template <typename spec_t>
__global__ void view_asstrided_gradient_kernel(const spec_t *input, spec_t *output, size_t size,
                                               const int64_t *stride_in, const int64_t *stride_out, 
                                               int ndim, int64_t storage_offset,
                                               const OffsetCalculator* in_offset_calculator) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  int index = storage_offset;
  size_t ind = idx;
  for (int i = 0; i < ndim; i++) {
    int tmp_index = ind / stride_in[i];
    index += tmp_index * stride_out[i];
    ind -= tmp_index * stride_in[i];
  }
  auto in_offset = in_offset_calculator->get(idx);
  hetu::cuda::AtomicAdd(&output[index], input[in_offset]);
}

template <typename spec_t>
__global__ void count_kernel(spec_t *output, size_t size,
                             const int64_t *stride_in, const int64_t *stride_out,
                             int ndim, int64_t storage_offset) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  int index = storage_offset;
  size_t ind = idx;
  spec_t ident = 1;
  for (int i = 0; i < ndim; i++) {
    int tmp_index = ind / stride_in[i];
    index += tmp_index * stride_out[i];
    ind = ind % stride_in[i];
  }
  hetu::cuda::AtomicAdd(&output[index], ident);
}

void AsStridedGradientCuda(const NDArray& grad_output, NDArray& grad_input,
                           const HTShape& out_shape, const HTStride& out_stride,
                           const HTShape& in_shape, const HTStride& in_stride,
                           int64_t in_storage_offset, int64_t out_storage_offset,
                           const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(grad_input);
  HT_ASSERT_SAME_DEVICE(grad_input, grad_output);

  size_t out_size = numel(out_shape);
  if (out_size == 0)
    return;

  auto device_id = grad_input->device().index();
  hetu::cuda::CUDADeviceGuard guard(device_id);
  CUDAStream cuda_stream(stream);
  dim3 blocks, threads;

  size_t odim = out_shape.size();
  HTShape out_stride_contig = Shape2Stride(out_shape);
  auto out_stride_contig_arr = 
    hetu::cuda::to_int64_ndarray(out_stride_contig, device_id);
  auto out_stride_arr =
    hetu::cuda::to_int64_ndarray(out_stride, device_id);
  threads.x = MIN(out_size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(out_size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  NDArray grad_out_offset_calculator_arr;
  OffsetCalculator *grad_out_offset_calculator;
  std::tie(grad_out_offset_calculator_arr, grad_out_offset_calculator) =
    AllocOffsetCalculator(grad_output, stream);
  HT_DISPATCH_FLOATING_TYPES(
    grad_input->dtype(), spec_t, "AsStridedGradientCuda", [&]() {
      view_asstrided_gradient_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
        grad_output->data_ptr<spec_t>(), grad_input->data_ptr<spec_t>(), out_size,
        out_stride_contig_arr->data_ptr<int64_t>(),
        out_stride_arr->data_ptr<int64_t>(),
        odim, out_storage_offset, grad_out_offset_calculator);
      });

  auto in_maybe_overlap = maybe_overlapping_memory(in_shape, in_stride);
  if (in_maybe_overlap) {
    size_t storage_size = grad_input->storage_size();
    NDArray count = NDArray::zeros({static_cast<int64_t>(storage_size)}, grad_input->device(), grad_input->dtype(),
                                   stream.stream_index());
    
    size_t in_size = numel(in_shape);
    HTShape in_stride_contig = Shape2Stride(in_shape);
    size_t idim = in_shape.size();
    auto in_stride_contig_arr =
      hetu::cuda::to_int64_ndarray(in_stride_contig, device_id);
    auto in_stride_arr = 
      hetu::cuda::to_int64_ndarray(in_stride, device_id);
    threads.x = MIN(in_size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
    blocks.x = DIVUP(in_size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
    HT_DISPATCH_FLOATING_TYPES(
      grad_input->dtype(), spec_t, "CountCuda", [&]() {
        count_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
          count->data_ptr<spec_t>(), in_size,
          in_stride_contig_arr->data_ptr<int64_t>(),
          in_stride_arr->data_ptr<int64_t>(),
          idim, in_storage_offset);
        });
    NDArray grad_in_offset_calculator_arr, cnt_offset_calculator_arr;
    OffsetCalculator *grad_in_offset_calculator, *cnt_offset_calculator;
    std::tie(grad_in_offset_calculator_arr, grad_in_offset_calculator) =
      AllocOffsetCalculator(grad_input, stream);
    std::tie(cnt_offset_calculator_arr, cnt_offset_calculator) =
      AllocOffsetCalculator(count, stream);
    threads.x = MIN(storage_size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
    blocks.x = DIVUP(storage_size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
    HT_DISPATCH_FLOATING_TYPES(
      grad_input->dtype(), spec_t, "BinaryElewiseCuda", [&]() {
        binary_elewise_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
          grad_input->data_ptr<spec_t>(), count->data_ptr<spec_t>(), storage_size,
          kdivides<spec_t, spec_t>(), grad_input->data_ptr<spec_t>(),
          grad_in_offset_calculator, cnt_offset_calculator, grad_in_offset_calculator);
        });

    NDArray::MarkUsedBy({count, in_stride_contig_arr, in_stride_arr,
                        grad_in_offset_calculator_arr, cnt_offset_calculator_arr}, stream);
  }

  auto output_meta = NDArrayMeta().set_dtype(grad_input->dtype())
                                  .set_shape(in_shape)
                                  .set_stride(in_stride)
                                  .set_device(grad_input->device());
  grad_input = NDArray(output_meta, grad_input->storage(), in_storage_offset);

  NDArray::MarkUsedBy({grad_input, grad_output, out_stride_contig_arr,
                      out_stride_arr, grad_out_offset_calculator_arr}, stream);
}

} // namespace impl
} // namespace hetu
