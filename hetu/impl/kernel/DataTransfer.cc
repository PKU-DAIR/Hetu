#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/ndarray_utils.h"
#include "hetu/impl/stream/CPUStream.h"

namespace hetu {
namespace impl {

template <typename spec_a_t, typename spec_b_t>
void data_transfer_cpu(const spec_a_t* from, size_t size, spec_b_t* to) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; idx++)
    to[idx] = static_cast<spec_b_t>(from[idx]);
}

template <typename spec_a_t, typename spec_b_t>
void data_transfer_cpu(const spec_a_t* from, size_t size, spec_b_t* to,
                       int64_t ndims, const int64_t* stride, const int64_t* stride_out,
                       const int64_t* c_shape) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; idx++) {
    int64_t i_idx = hetu::impl::get_index(idx, ndims, stride, c_shape);
    int64_t o_idx = hetu::impl::get_index(idx, ndims, stride_out, c_shape);
    to[o_idx] = static_cast<spec_b_t>(from[i_idx]);
  }
}

void DataTransferCpu(const NDArray& from, NDArray& to, const Stream& stream) {
  HT_ASSERT_COPIABLE(from, to);
  size_t numel = from->numel();
  if (numel == 0)
    return;
  void* to_ptr = to->raw_data_ptr();
  void* from_ptr = from->raw_data_ptr();
  if (to_ptr == from_ptr) {
    HT_ASSERT(from->dtype() == to->dtype())
      << "NDArrays with " << from->dtype() << " and " << to->dtype()
      << " types are sharing the same storage, which is not allowed.";
    return;
  }
  CPUStream cpu_stream(stream);
  auto _future = cpu_stream.EnqueueTask(
  [from, to, to_ptr, from_ptr, numel]() {
    if (!(from->is_contiguous() && to->is_contiguous())) {
      HT_DISPATCH_PAIRED_SIGNED_INTEGER_AND_FLOATING_TYPES(
        from->dtype(), to->dtype(), spec_a_t, spec_b_t, "DataTransferCpu", [&]() {
          auto* typed_from_ptr = reinterpret_cast<spec_a_t*>(from_ptr);
          auto* typed_to_ptr = reinterpret_cast<spec_b_t*>(to_ptr);
          data_transfer_cpu<spec_a_t, spec_b_t>(
            typed_from_ptr, numel, typed_to_ptr, from->ndim(), from->stride().data(),
            to->stride().data(), from->shape().data());
        });
    } else {
      if (from->dtype() == to->dtype()) {
        memcpy(to_ptr, from_ptr, (from->dtype() == kFloat4 || from->dtype() == kNFloat4)
                                 ? ((numel + 1) / 2) * DataType2Size(from->dtype())
                                 : numel * DataType2Size(from->dtype()));
      } else {
        HT_DISPATCH_PAIRED_SIGNED_INTEGER_AND_FLOATING_TYPES(
          from->dtype(), to->dtype(), spec_a_t, spec_b_t, "DataTransferCpu", [&]() {
            auto* typed_from_ptr = reinterpret_cast<spec_a_t*>(from_ptr);
            auto* typed_to_ptr = reinterpret_cast<spec_b_t*>(to_ptr);
            data_transfer_cpu<spec_a_t, spec_b_t>(typed_from_ptr, numel, typed_to_ptr);
          });
      }
    }
  },
  "DataTransfer");
  NDArray::MarkUsedBy({from, to}, stream);
}

} // namespace impl
} // namespace hetu
