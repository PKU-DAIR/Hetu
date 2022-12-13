#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/omp_utils.h"

namespace hetu {
namespace impl {

template <typename spec_t>
void where_cpu(const bool* cond, const spec_t* arr1, const spec_t* arr2,
               spec_t* output, size_t size) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; ++idx) {
    output[idx] = cond[idx] ? arr1[idx] : arr2[idx];
  }
}

void WhereCpu(const NDArray& cond, const NDArray& inputA, const NDArray& inputB,
              NDArray& output, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(cond);
  HT_ASSERT_SAME_DEVICE(cond, inputA);
  HT_ASSERT_SAME_DEVICE(cond, inputB);
  HT_ASSERT_SAME_DEVICE(cond, output);
  HT_ASSERT_EXCHANGABLE(inputA, inputB);

  size_t size = cond->numel();
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    inputA->dtype(), spec_t, "WhereCpu", [&]() {
      where_cpu<spec_t>(cond->data_ptr<bool>(), inputA->data_ptr<spec_t>(),
                        inputB->data_ptr<spec_t>(), output->data_ptr<spec_t>(),
                        size);
    });
}

} // namespace impl
} // namespace hetu
