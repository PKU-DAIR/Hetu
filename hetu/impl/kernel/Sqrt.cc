#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/omp_utils.h"
#include "hetu/impl/stream/CPUStream.h"
#include "cmath"

namespace hetu {
namespace impl {

template <typename spec_t>
void sqrt_cpu(const spec_t* input, size_t size, spec_t* output) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; ++idx) {
    output[idx] = 0;
  }
}

template <typename spec_t>
void reciprocal_sqrt_cpu(const spec_t* output_grad, size_t size,
                         spec_t* output) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; ++idx) {
    output[idx] = static_cast<spec_t>(1) / std::sqrt(output_grad[idx]);
  }
}

void SqrtCpu(const NDArray& input, NDArray& output, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT_EXCHANGABLE(input, output);

  size_t size = output->numel();
  if (size == 0)
    return;
  CPUStream cpu_stream(stream);
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "SqrtCpu", [&]() {
      auto _future = cpu_stream.EnqueueTask(
        [stream, input, output, size]() {
          dnnl::engine eng(dnnl::engine::kind::cpu, stream.stream_index());
          dnnl::memory::data_type mtype;
          if (input->dtype() == DataType::FLOAT32)
            mtype = dnnl::memory::data_type::f32;
          else 
            mtype = dnnl::memory::data_type::f64;
          auto mat_md = dnnl::memory::desc(input->shape(), mtype, input->stride());
          auto src_mem = dnnl::memory(mat_md, eng, input->data_ptr<spec_t>());
          auto dst_mem = dnnl::memory(mat_md, eng, output->data_ptr<spec_t>());

          auto Sqrt_pd = dnnl::eltwise_forward::primitive_desc(eng, dnnl::prop_kind::forward_training,
                              dnnl::algorithm::eltwise_sqrt, mat_md, mat_md);
          auto Sqrt = dnnl::eltwise_forward(Sqrt_pd);

          std::unordered_map<int, dnnl::memory> sqrt_args;
          sqrt_args.insert({DNNL_ARG_SRC, src_mem});
          sqrt_args.insert({DNNL_ARG_DST, dst_mem});      

          dnnl::stream engine_stream(eng);
          Sqrt.execute(engine_stream, sqrt_args);
          engine_stream.wait();
        },"Sqrt");
      //cpu_stream.Sync();
    });
}

void ReciprocalSqrtCpu(const NDArray& output_grad, NDArray& input_grad,
                       const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(output_grad);
  HT_ASSERT_SAME_DEVICE(output_grad, input_grad);
  HT_ASSERT_EXCHANGABLE(output_grad, input_grad);

  size_t size = input_grad->numel();
  if (size == 0)
    return;
  CPUStream cpu_stream(stream);
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input_grad->dtype(), spec_t, "ReciprocalSqrtCpu", [&]() {
      auto _future = cpu_stream.EnqueueTask(
        [stream, input_grad, output_grad, size]() {
          dnnl::engine eng(dnnl::engine::kind::cpu, stream.stream_index());
          auto mat_md = dnnl::memory::desc(input_grad->shape(), dnnl::memory::data_type::f32, input_grad->stride());
          auto src_mem = dnnl::memory(mat_md, eng, output_grad->data_ptr<spec_t>());
          auto dst_mem = dnnl::memory(mat_md, eng, input_grad->data_ptr<spec_t>());

          auto Reciprocal_pd = dnnl::eltwise_forward::primitive_desc(eng, dnnl::prop_kind::forward_training,
                              dnnl::algorithm::eltwise_pow, mat_md, mat_md, float(1.0), float(-0.5));
          auto Reciprocal = dnnl::eltwise_forward(Reciprocal_pd);

          std::unordered_map<int, dnnl::memory> sqrt_args;
          sqrt_args.insert({DNNL_ARG_SRC, src_mem});
          sqrt_args.insert({DNNL_ARG_DST, dst_mem});      

          dnnl::stream engine_stream(eng);
          Reciprocal.execute(engine_stream, sqrt_args);
          engine_stream.wait();
        },"ReciprocalSqrt");
      //cpu_stream.Sync();
    });
}

} // namespace impl
} // namespace hetu
