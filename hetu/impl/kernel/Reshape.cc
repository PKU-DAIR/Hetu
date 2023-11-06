#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/omp_utils.h"
#include "hetu/impl/stream/CPUStream.h"

namespace hetu {
namespace impl {

template <typename spec_t>
void memory_copy_cpu(const spec_t* input, spec_t* output, size_t size) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; ++idx) {
    output[idx] = input[idx];
  }
}

void ReshapeCpu(const NDArray& input, NDArray& output, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);

  CPUStream cpu_stream(stream);

  size_t input_size = input->numel();
  size_t size = output->numel();
  HT_ASSERT(input_size == size) << "input size and output size are different. ";
  if (input_size == 0)
    return;
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "ReshapeCpu", [&]() {
      auto _future = cpu_stream.EnqueueTask(
        [input, output, size]() {
        memory_copy_cpu<spec_t>(input->data_ptr<spec_t>(),
                                output->data_ptr<spec_t>(), size);
        },"Reshape");
    });
}

} // namespace impl
} // namespace hetu
