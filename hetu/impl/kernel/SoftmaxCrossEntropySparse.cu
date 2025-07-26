#include "hetu/core/ndarray.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/utils/cuda_math.h"
#include "hetu/impl/utils/offset_calculator.cuh"

namespace hetu {
namespace impl {

template <typename spec_t>
__global__ void softmax_cross_entropy_sparse_kernel(const spec_t* pred, const int64_t* label, 
                                                    size_t n_rows, size_t n_cols,
                                                    const int64_t ignored_index, spec_t* loss,
                                                    const OffsetCalculator* pred_offset_calculator,
                                                    const OffsetCalculator* label_offset_calculator,
                                                    const OffsetCalculator* loss_offset_calculator) {
  auto idx = blockIdx.x;
  size_t start_idx = idx * n_cols + threadIdx.x;
  size_t end_idx = (idx + 1) * n_cols;
  size_t stride = blockDim.x;
  if (start_idx >= end_idx)
    return;
  if (idx >= n_rows)
    return;
  auto label_offset = label_offset_calculator->get(idx);
  auto loss_offset = loss_offset_calculator->get(idx);
  if(int64_t(label[label_offset]) == ignored_index || int64_t(label[label_offset]) < 0) {
    loss[loss_offset] = 0;
    return;
  }
  __shared__ spec_t buffer[32];
  __shared__ float buffer_f[32];
  __shared__ spec_t wrap_max[1];
  size_t ptr = start_idx;
  auto pred_offset = pred_offset_calculator->get(ptr);
  spec_t maxval = pred[pred_offset];
  for (size_t ptr = start_idx; ptr < end_idx; ptr += stride) {
    pred_offset = pred_offset_calculator->get(ptr);
    maxval = hetu::cuda::cuda_max(pred[pred_offset], maxval);
  }
  hetu::cuda::BlockReduceArgmax(maxval, buffer, wrap_max);
  maxval = wrap_max[0];
  float sum = 0;
  for (size_t ptr = start_idx; ptr < end_idx; ptr += stride) {
    pred_offset = pred_offset_calculator->get(ptr);
    sum += hetu::cuda::cuda_exp(float(pred[pred_offset] - maxval));
  }
  hetu::cuda::BlockReduceSum(sum, buffer_f);
  if (threadIdx.x == 0) {
    size_t curid = idx * n_cols + int64_t(label[label_offset]);
    pred_offset = pred_offset_calculator->get(curid);
    loss[loss_offset] = - pred[pred_offset] + maxval + spec_t(hetu::cuda::cuda_log(sum));
  }
}

template <typename spec_t>
__global__ void softmax_cross_entropy_sparse_kernel2(const spec_t* pred, const int64_t* label, 
                                                     size_t n_rows, size_t n_cols,
                                                     const int64_t ignored_index, spec_t* loss,
                                                     const OffsetCalculator* pred_offset_calculator,
                                                     const OffsetCalculator* label_offset_calculator,
                                                     const OffsetCalculator* loss_offset_calculator) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_rows)
    return;
  auto label_offset = label_offset_calculator->get(idx);
  auto loss_offset = loss_offset_calculator->get(idx);
  if(int64_t(label[label_offset]) == ignored_index || int64_t(label[label_offset]) < 0) {
    loss[loss_offset] = 0;
    return;
  }
  auto base_idx = idx * n_cols;
  auto pred_offset = pred_offset_calculator->get(base_idx);
  spec_t maxval = pred[pred_offset];
  for (size_t i = 1; i < n_cols; ++i) {
    pred_offset = pred_offset_calculator->get(base_idx + i);
    maxval = hetu::cuda::cuda_max(maxval, pred[pred_offset]);
  }

  float sum = 0;
  for (int i = 0; i < n_cols; ++i) {
    pred_offset = pred_offset_calculator->get(base_idx + i);
    sum += hetu::cuda::cuda_exp(float(pred[pred_offset] - maxval));
  }

  size_t curid = base_idx + int64_t(label[label_offset]);
  pred_offset = pred_offset_calculator->get(curid);
  // loss[idx] = -(pred[curid] - maxval) + hetu::cuda::cuda_log(sum);
  loss[loss_offset] = - pred[pred_offset] + maxval + spec_t(hetu::cuda::cuda_log(sum));
}

template <typename spec_t>
__global__ void
softmax_cross_entropy_sparse_gradient_kernel(const spec_t* pred, const int64_t* label,
                                             const spec_t* grad_loss, size_t n_rows, size_t n_cols,
                                             const int64_t ignored_index, spec_t* output,
                                             const OffsetCalculator* pred_offset_calculator,
                                             const OffsetCalculator* label_offset_calculator,
                                             const OffsetCalculator* grad_loss_offset_calculator,
                                             const OffsetCalculator* out_offset_calculator) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_rows)
    return;

  auto label_offset = label_offset_calculator->get(idx);
  auto base_idx = idx * n_cols;
  if(int64_t(label[label_offset]) == ignored_index || int64_t(label[label_offset]) < 0) {
    for (size_t i = 0; i < n_cols; ++i) {
        auto out_offset = out_offset_calculator->get(base_idx + i);
        output[out_offset] = 0;
    }
    return;        
  }
  
  auto pred_offset = pred_offset_calculator->get(base_idx);
  spec_t maxval = pred[pred_offset];

  for (size_t i = 1; i < n_cols; ++i) {
      pred_offset = pred_offset_calculator->get(base_idx + i);
      maxval = MAX(maxval, pred[pred_offset]);
  }

  float sum = 0;
  for (size_t i = 0; i < n_cols; ++i) {
      pred_offset = pred_offset_calculator->get(base_idx + i);
      sum += hetu::cuda::cuda_exp(float(pred[pred_offset] - maxval));
  }
  auto grad_loss_offset = grad_loss_offset_calculator->get(idx);
  for (size_t i = 0; i < n_cols; ++i) {
      auto out_offset = out_offset_calculator->get(base_idx + i);
      pred_offset = pred_offset_calculator->get(base_idx + i);
      if(i == int64_t(label[label_offset]))
        output[out_offset] = (hetu::cuda::cuda_exp(pred[pred_offset] - maxval) / spec_t(sum) - 1.0)
                           * grad_loss[grad_loss_offset];
      else
        output[out_offset] = (hetu::cuda::cuda_exp(pred[pred_offset] - maxval) / spec_t(sum))
                           * grad_loss[grad_loss_offset];
  }
}

template <typename spec_t>
__forceinline__ __device__ spec_t WarpReduceSum(spec_t val) {
  unsigned int mask = __ballot_sync(0xFFFFFFFF, true);
  for (unsigned int k = (warpSize >> 1); k > 0; k >>= 1)
    val += hetu::cuda::shfl_down_sync(mask, val, k, warpSize);
  return val;
}

template <>
__forceinline__ __device__ bfloat16 WarpReduceSum<bfloat16>(bfloat16 val) {
  unsigned int mask = __ballot_sync(0xFFFFFFFF, true);
  #if(__CUDA_ARCH__ >= 800)
  for (unsigned int k = (warpSize >> 1); k > 0; k >>= 1)
    val += hetu::cuda::shfl_down_sync(mask, val, k, warpSize);
  #else
  float val_f = float(val);
  for (unsigned int k = (warpSize >> 1); k > 0; k >>= 1)
    val_f += hetu::cuda::shfl_down_sync(mask, val_f, k, warpSize); 
  val = bfloat16(val_f); 
  #endif
  return val;
}

template <typename spec_t>
__forceinline__ __device__ void BlockReduceSum(spec_t& val, spec_t* shared, spec_t* warp_sum) {
  int tid = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  val = WarpReduceSum(val);

  __syncthreads();
  if (tid == 0)
    shared[wid] = val;

  __syncthreads();
  val = (threadIdx.x < blockDim.x / warpSize) ? shared[tid] : 0;

  if (wid == 0) {
    val = WarpReduceSum(val);
    if (threadIdx.x == 0)
      warp_sum[0] = val;
  }
  __syncthreads();
}

template <typename spec_t>
__global__ void
softmax_cross_entropy_sparse_gradient_kernel2(const spec_t* pred, const int64_t* label,
                                              const spec_t* grad_loss, size_t n_rows, size_t n_cols,
                                              const int64_t ignored_index, spec_t* output,
                                              const OffsetCalculator* pred_offset_calculator,
                                              const OffsetCalculator* label_offset_calculator,
                                              const OffsetCalculator* grad_loss_offset_calculator,
                                              const OffsetCalculator* out_offset_calculator) {
  auto idx = blockIdx.x;
  if (idx >= n_rows)
    return;
  size_t start_idx = idx * n_cols + threadIdx.x;
  size_t end_idx = (idx + 1) * n_cols;
  size_t stride = blockDim.x;
  if (start_idx >= end_idx)
    return;

  __shared__ spec_t buffer[32];
  __shared__ float buffer_f[32];
  __shared__ spec_t wrap_max[1];
  __shared__ float wrap_sum[1];
  auto pred_offset = pred_offset_calculator->get(start_idx);
  spec_t maxval = pred[pred_offset];
  for (size_t ptr = start_idx; ptr < end_idx; ptr += stride) {
    pred_offset = pred_offset_calculator->get(ptr);
    maxval = hetu::cuda::cuda_max(pred[pred_offset], maxval);
  }
  hetu::cuda::BlockReduceArgmax(maxval, buffer, wrap_max);
  maxval = wrap_max[0];
  float sum = 0;
  for (size_t ptr = start_idx; ptr < end_idx; ptr += stride) {
    pred_offset = pred_offset_calculator->get(ptr);
    sum += hetu::cuda::cuda_exp(float(pred[pred_offset] - maxval));
  }
  BlockReduceSum(sum, buffer_f, wrap_sum);
  sum = wrap_sum[0];
  auto label_offset = label_offset_calculator->get(idx);
  auto grad_loss_offset = grad_loss_offset_calculator->get(idx);
  for (size_t curid = start_idx; curid < end_idx; curid += stride) {
      size_t i = curid - idx * n_cols;
      auto out_offset = out_offset_calculator->get(curid);
      pred_offset = pred_offset_calculator->get(curid);
      if(i == int64_t(label[label_offset]))
        output[out_offset] = (hetu::cuda::cuda_exp(pred[pred_offset] - maxval) / spec_t(sum) - 1.0)
                           * grad_loss[grad_loss_offset];
      else
        output[out_offset] = (hetu::cuda::cuda_exp(pred[pred_offset] - maxval) / spec_t(sum))
                           * grad_loss[grad_loss_offset];
  }
}

void SoftmaxCrossEntropySparseCuda(const NDArray& pred, const NDArray& label,
                                   NDArray& loss, const int64_t ignored_index, 
                                   const Stream& stream) {
  size_t n_rows = 1;
  for (size_t i = 0; i < pred->ndim() - 1; i++)
    n_rows *= pred->shape(i);
  size_t n_cols = pred->shape(pred->ndim() - 1);
  if (n_rows == 0)
    return;
  dim3 blocks, threads;
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  NDArray pred_offset_calculator_arr, label_offset_calculator_arr,
          loss_offset_calculator_arr;
  OffsetCalculator *pred_offset_calculator, *label_offset_calculator,
                   *loss_offset_calculator;
  std::tie(pred_offset_calculator_arr, pred_offset_calculator) =
    AllocOffsetCalculator(pred, stream);
  std::tie(label_offset_calculator_arr, label_offset_calculator) = 
    AllocOffsetCalculator(label, stream);
  std::tie(loss_offset_calculator_arr, loss_offset_calculator) = 
    AllocOffsetCalculator(loss, stream);
  threads.x = MIN(MAX(32, n_cols), HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = n_rows;
  HT_DISPATCH_FLOATING_TYPES(
    pred->dtype(), spec_t, "SoftmaxCrossEntropySparseCuda", [&]() {
      softmax_cross_entropy_sparse_kernel<<<blocks, threads, 0, cuda_stream>>>(
        pred->data_ptr<spec_t>(), label->data_ptr<int64_t>(), n_rows, n_cols,
        ignored_index, loss->data_ptr<spec_t>(), pred_offset_calculator,
        label_offset_calculator, loss_offset_calculator);
    });
  NDArray::MarkUsedBy({pred, label, loss, pred_offset_calculator_arr,
                      label_offset_calculator_arr, loss_offset_calculator_arr}, stream);
}

void SoftmaxCrossEntropySparseGradientCuda(const NDArray& pred, const NDArray& label,
                                           const NDArray& grad_loss, NDArray& output,
                                           const int64_t ignored_index,
                                           const Stream& stream) {

  size_t n_rows = 1;
  for (size_t i = 0; i < pred->ndim() - 1; i++)
    n_rows *= pred->shape(i);
  size_t n_cols = pred->shape(pred->ndim() - 1);
  if (n_rows == 0)
    return;
  dim3 blocks, threads;
  threads.x = MIN(n_rows, 1024);
  blocks.x = DIVUP(n_rows, 1024);
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  NDArray pred_offset_calculator_arr, label_offset_calculator_arr,
          grad_loss_offset_calculator_arr, out_offset_calculator_arr;
  OffsetCalculator *pred_offset_calculator, *label_offset_calculator,
                   *grad_loss_offset_calculator, *out_offset_calculator;
  std::tie(pred_offset_calculator_arr, pred_offset_calculator) =
    AllocOffsetCalculator(pred, stream);
  std::tie(label_offset_calculator_arr, label_offset_calculator) = 
    AllocOffsetCalculator(label, stream);
  std::tie(grad_loss_offset_calculator_arr, grad_loss_offset_calculator) = 
    AllocOffsetCalculator(grad_loss, stream);
  std::tie(out_offset_calculator_arr, out_offset_calculator) = 
    AllocOffsetCalculator(output, stream);
  threads.x = MIN(MAX(32, n_cols), HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = n_rows;
  HT_DISPATCH_FLOATING_TYPES(
    pred->dtype(), spec_t, "SoftmaxCrossEntropySparseGradientCuda", [&]() {
      softmax_cross_entropy_sparse_gradient_kernel2<<<blocks, threads, 0, cuda_stream>>>(
        pred->data_ptr<spec_t>(), label->data_ptr<int64_t>(),
        grad_loss->data_ptr<spec_t>(), n_rows, n_cols,
        ignored_index, output->data_ptr<spec_t>(),
        pred_offset_calculator, label_offset_calculator,
        grad_loss_offset_calculator, out_offset_calculator);
    });
  NDArray::MarkUsedBy({pred, label, grad_loss, output, pred_offset_calculator_arr,
                      label_offset_calculator_arr, grad_loss_offset_calculator_arr,
                      out_offset_calculator_arr}, stream);  
}

} // namespace impl
} // namespace hetu
