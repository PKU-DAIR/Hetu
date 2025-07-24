#include "hetu/core/ndarray.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/utils/offset_calculator.cuh"
#include "hetu/impl/kernel/Vectorized.cuh"
#include "hetu/impl/cuda/CUB.h"



namespace hetu{
namespace impl{


void MaskedScatterCuda(const NDArray& input, const NDArray& mask, const NDArray& source,
                NDArray& output, const Stream& stream) {
  hetu::cuda::CUDADeviceGuard guard(stream.device_index());
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, mask);
  HT_ASSERT_SAME_DEVICE(input, source);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT_SAME_SHAPE(input, mask);
  HT_ASSERT_SAME_SHAPE(input, output);
  auto maskPrefixSum = NDArray::empty(input->shape(), input->device(), kInt64, stream.stream_index());
  auto maskPrefixSum_data = maskPrefixSum->data_ptr<int64_t>();
  auto mask_data = mask->data_ptr<int64_t>();
  exclusive_scan(mask_data, maskPrefixSum_data, cub::Sum(), (int64_t)0, input->numel(), stream); 
  size_t size = input->numel();
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "MaskedScatterCuda", [&]() {
      auto source_ptr = source->data_ptr<spec_t>();
      launch_loop_kernel<spec_t, int64_t, int64_t, spec_t>(input, mask, maskPrefixSum, output, size, stream,
                                                  [source_ptr, size] __device__ (spec_t in, int64_t mask, int64_t maskPrefixSum) -> spec_t {
                                                    if(mask){
                                                      if(maskPrefixSum >= size || maskPrefixSum < 0){
                                                        printf("errrrror %lld %lld]\n", (int64_t)size, (int64_t)maskPrefixSum);
                                                      }
                                                      return static_cast<spec_t>(source_ptr[maskPrefixSum]);
                                                    }
                                                    else return in;
                                                 });
  });

  NDArray::MarkUsedBy({input, mask, source, output, maskPrefixSum}, stream);
}

}
}