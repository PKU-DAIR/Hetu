#include "hetu/core/ndarray.h"
#include "hetu/impl/cuda/CUDARand.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/utils/offset_calculator.cuh"
#include "hetu/impl/kernel/Vectorized.cuh"

namespace hetu {
namespace impl {

template <typename spec_t, typename mask_t>
struct Dropout2dOp {
  const mask_t* mask_ptr;
  int C, H, W, HW, CHW;
  float scale;

  __host__ __device__ __forceinline__
  spec_t operator()(int idx, spec_t in_val) const {
    // Compute channel index from linear idx
    int n = idx / CHW;
    int c = (idx - n * CHW) / HW;
    int mask_idx = n * C + c;
    mask_t keep_mask = mask_ptr[mask_idx];
    return in_val * (keep_mask ? scale : 0);
  }
};

void Dropout2dCuda(const NDArray& input, double drop_rate,
                   uint64_t seed, NDArray& output,
                   NDArray& mask, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT_SAME_DEVICE(input, mask);
  HT_ASSERT_SAME_SHAPE(input, output);
  HT_ASSERT(input->ndim() == 4);
  HT_ASSERT(mask->ndim() == 2);
  size_t size = output->numel();
  if (size == 0)
    return;

  const int N = input->shape()[0];
  const int C = input->shape()[1];
  const int H = input->shape()[2];
  const int W = input->shape()[3];
  const int HW = H * W;
  const int CHW = C * H * W;
  const size_t num_channels = N * C;
  const float scale = 1.0f / (1.0f - drop_rate);
  
  // Initialize random values
  NDArray rand_vals = NDArray::empty({static_cast<int64_t>(num_channels)}, input->device(),
                                     input->dtype(), stream.stream_index());
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  curandGenerator_t gen;
  CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_PHILOX4_32_10));
  CURAND_CALL(curandSetStream(gen, cuda_stream));
  CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, seed));

  HT_DISPATCH_FLOATING_TYPES(input->dtype(), spec_t, "Dropout2dMaskGenerator", [&]() {
    curand_gen_uniform<spec_t>(gen, rand_vals->data_ptr<spec_t>(), num_channels);
    using InType = std::tuple<spec_t>;
    using MaskType = thrust::tuple<bool>;
    launch_loop_kernel<InType, MaskType>({rand_vals}, {mask}, num_channels, stream,
      [drop_rate] __device__ (spec_t rand_val) -> bool {
        return rand_val >= drop_rate;
      });
    
    Dropout2dOp<spec_t, bool> op;
    op.mask_ptr = mask->data_ptr<bool>();
    op.scale = scale;
    op.C = C;
    op.H = H;
    op.W = W;
    op.HW = HW;
    op.CHW = CHW;
    using OutType = thrust::tuple<spec_t>;
    launch_loop_kernel_with_idx<InType, OutType>({input}, {output}, input->numel(), stream, op);
  });
  CURAND_CALL(curandDestroyGenerator(gen));
  NDArray::MarkUsedBy({rand_vals, mask, input, output}, stream);
}

void Dropout2dGradientCuda(const NDArray& grad, const NDArray& fw_mask,
                           double drop_rate, NDArray& output,
                           const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(grad);
  HT_ASSERT_SAME_DEVICE(grad, output);
  HT_ASSERT_SAME_DEVICE(grad, fw_mask);
  HT_ASSERT_SAME_SHAPE(grad, output);
  HT_ASSERT(grad->ndim() == 4);
  HT_ASSERT(fw_mask->ndim() == 2);
  size_t size = grad->numel();
  if (size == 0)
    return;
  
  const int N = grad->shape()[0];
  const int C = grad->shape()[1];
  const int H = grad->shape()[2];
  const int W = grad->shape()[3];
  const int HW = H * W;
  const int CHW = C * H * W;
  const float scale = 1.0f / (1.0f - static_cast<float>(drop_rate));
  
  HT_DISPATCH_FLOATING_TYPES(grad->dtype(), spec_t, "Dropout2dGradient", [&]() {
    Dropout2dOp<spec_t, bool> op;
    op.mask_ptr = fw_mask->data_ptr<bool>();
    op.scale = scale;
    op.C = C;
    op.H = H;
    op.W = W;
    op.HW = HW;
    op.CHW = CHW;

    using InType = std::tuple<spec_t>;
    using OutType = thrust::tuple<spec_t>;
    launch_loop_kernel_with_idx<InType, OutType>({grad}, {output}, grad->numel(), stream, op);
  });
  
  NDArray::MarkUsedBy({fw_mask, grad, output}, stream);
}

} // namespace impl
} // namespace hetu
