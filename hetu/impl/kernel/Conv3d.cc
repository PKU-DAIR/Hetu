#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/omp_utils.h"
#include "hetu/impl/utils/dnnl_utils.h"
#include "hetu/impl/stream/CPUStream.h"

namespace hetu {
namespace impl {
void Conv3dCpu(const NDArray& input_x, const NDArray& input_f, NDArray& output,
               const int padding_d, const int padding_h, const int padding_w, const int stride_d, 
               const int stride_h, const int stride_w, const Stream& stream) {
  HT_NOT_IMPLEMENTED << "Conv2dCpu is not implemented";
}

void Conv3dGradientofFilterCpu(const NDArray& input_x,
                               const NDArray& gradient_y, NDArray& gradient_f,
                               const int padding_d, const int padding_h, const int padding_w,
                               const int stride_d, const int stride_h, const int stride_w,
                               const Stream& stream) {
  HT_NOT_IMPLEMENTED << "Conv2dGradientofFilterCpu is not implemented";
}

void Conv3dGradientofDataCpu(const NDArray& input_f, const NDArray& gradient_y,
                             NDArray& gradient_x, const int padding_d, const int padding_h,
                             const int padding_w, const int stride_d, const int stride_h,
                             const int stride_w, const Stream& stream) {
  HT_NOT_IMPLEMENTED << "Conv2dGradientofDataCpu is not implemented";
}


} // namespace impl
} // namespace hetu
