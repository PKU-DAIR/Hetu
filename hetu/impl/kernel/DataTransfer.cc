#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/ndarray_utils.h"
#include "hetu/impl/stream/CPUStream.h"

namespace hetu {
namespace impl {

class StridedOffsetCalculator{
public:
  StridedOffsetCalculator(int dims, HTShape shape, HTShape stride)
    : _dims(dims) {
    HT_ASSERT(dims <= HT_MAX_NDIM)
      << "Currently we only support shape up to " << HT_MAX_NDIM
      << " dimensions. Got" << dims << ".";

    for (int i = 0; i < dims; i++) {
      _shape[i] = shape[i];
      _stride[i] = stride[i];
    }
  }
  
  inline size_t get(size_t linear_idx) const {
    size_t offset = 0;
#pragma unroll
    for (int i = _dims - 1; i >= 1; i--) {
      int64_t shape_i = _shape[i];
      auto div_idx = linear_idx / shape_i;
      auto mod_idx = linear_idx - div_idx * shape_i;
      offset += mod_idx * _stride[i];
      linear_idx = div_idx;
    }
    return offset + linear_idx * _stride[0];
  }

private:
  int _dims;
  int64_t _shape[HT_MAX_NDIM];
  int64_t _stride[HT_MAX_NDIM];
};

template <typename spec_a_t, typename spec_b_t>
void copy_non_contiguous(spec_a_t* from_ptr, spec_b_t* to_ptr, size_t numel,
                         StridedOffsetCalculator& from_offset_calculator,
                         StridedOffsetCalculator& to_offset_calculator) {
#pragma unroll
  for(int i = 0; i < numel; ++i) {
    size_t from_offset = from_offset_calculator.get(i);
    size_t to_offset = to_offset_calculator.get(i);
    to_ptr[to_offset] = from_ptr[from_offset];
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
    if(!(from->is_contiguous() && to->is_contiguous())) {
      StridedOffsetCalculator from_offset_calculator(from->ndim(), from->shape(), from->stride());
      StridedOffsetCalculator to_offset_calculator(to->ndim(), to->shape(), to->stride());
      HT_DISPATCH_PAIRED_SIGNED_INTEGER_AND_FLOATING_TYPES(
        from->dtype(), to->dtype(), spec_a_t, spec_b_t, "DataTransferCpu", [&]() {
          copy_non_contiguous<spec_a_t, spec_b_t>(
            reinterpret_cast<spec_a_t*>(from_ptr), 
            reinterpret_cast<spec_b_t*>(to_ptr),
            numel, from_offset_calculator, to_offset_calculator);
        }
      );
    }
    else {
      if (from->dtype() == to->dtype()) {
        memcpy(to_ptr, from_ptr, (from->dtype() == kFloat4 || from->dtype() == kNFloat4)
                                ? ((numel + 1) / 2) * DataType2Size(from->dtype())
                                : numel * DataType2Size(from->dtype()));
      } else {
        HT_DISPATCH_PAIRED_SIGNED_INTEGER_AND_FLOATING_TYPES(
          from->dtype(), to->dtype(), spec_a_t, spec_b_t, "DataTransferCpu", [&]() {
            auto* typed_from_ptr = reinterpret_cast<spec_a_t*>(from_ptr);
            auto* typed_to_ptr = reinterpret_cast<spec_b_t*>(to_ptr);
            std::copy(typed_from_ptr, typed_from_ptr + numel, typed_to_ptr);
          });
      }
    }
  },
  "DataTransfer");
  NDArray::MarkUsedBy({from, to}, stream);
}

} // namespace impl
} // namespace hetu
