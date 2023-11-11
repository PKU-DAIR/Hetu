#include "hetu/core/ndarray.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/utils/cuda_math.h"

namespace hetu {
namespace impl {

template <typename spec_t>
__global__ void mseloss_kernel(const spec_t* pred,
                               const spec_t* label, size_t n_rows,
                               spec_t* loss) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_rows)
    return;
  loss[idx] = (pred[idx] - label[idx]) * (pred[idx] - label[idx]);
}

template <typename spec_t>
__global__ void
mseloss_gradient_kernel(const spec_t* pred, const spec_t* label,
                        const spec_t* grad_loss, size_t n_rows,
                        spec_t* output) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_rows)
    return;
  output[idx] = 2 * grad_loss[idx] * (pred[idx] - label[idx]);
}

void MSELossCuda(const NDArray& pred, const NDArray& label,
                            NDArray& loss, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(pred);
  HT_ASSERT_SAME_DEVICE(pred, label);
  HT_ASSERT_SAME_DEVICE(pred, loss);
  HT_ASSERT_SAME_NDIM(pred, label);
  HT_ASSERT_SAME_NDIM(pred, loss);

  size_t n_rows = 1;
  for (size_t i = 0; i < pred->ndim(); i++)
    n_rows *= pred->shape(i);
  if (n_rows == 0)
    return;
  dim3 blocks, threads;
  threads.x = MIN(n_rows, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(n_rows, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  HT_DISPATCH_FLOATING_TYPES(
    pred->dtype(), spec_t, "MSELossCuda", [&]() {
      mseloss_kernel<<<blocks, threads, 0, cuda_stream>>>(
        pred->data_ptr<spec_t>(), label->data_ptr<spec_t>(), n_rows,
        loss->data_ptr<spec_t>());
    });
  NDArray::MarkUsedBy({pred, label, loss}, stream);
}

void MSELossGradientCuda(const NDArray& pred, const NDArray& label,
                                    const NDArray& grad_loss, NDArray& output,
                                    const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(pred);
  HT_ASSERT_SAME_DEVICE(pred, label);
  HT_ASSERT_SAME_DEVICE(pred, grad_loss);
  HT_ASSERT_SAME_DEVICE(pred, output);
  HT_ASSERT_SAME_NDIM(pred, label);
  HT_ASSERT_SAME_NDIM(pred, grad_loss);
  HT_ASSERT_SAME_NDIM(pred, output);

  size_t n_rows = 1;
  for (size_t i = 0; i < pred->ndim(); i++)
    n_rows *= pred->shape(i);
  if (n_rows == 0)
    return;
  dim3 blocks, threads;
  threads.x = MIN(n_rows, 1024);
  blocks.x = DIVUP(n_rows, 1024);
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  HT_DISPATCH_FLOATING_TYPES(
    pred->dtype(), spec_t, "MSELossGradientCuda", [&]() {
      mseloss_gradient_kernel<<<blocks, threads, 0, cuda_stream>>>(
        pred->data_ptr<spec_t>(), label->data_ptr<spec_t>(),
        grad_loss->data_ptr<spec_t>(), n_rows, output->data_ptr<spec_t>());
    });
  NDArray::MarkUsedBy({pred, label, grad_loss, output}, stream);
}

} // namespace impl
} // namespace hetu
