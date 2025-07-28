#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/omp_utils.h"
#include "hetu/impl/utils/dnnl_utils.h"
#include "hetu/impl/stream/CPUStream.h"

namespace hetu {
namespace impl {

template <typename spec_t>
void concatenate_cpu(const spec_t* input, spec_t* output, int input_width,
                     int output_width, int offset, int concat_size,
                     size_t size) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; ++idx) {
    int post_ind = idx % concat_size;
    int prev_ind = idx / concat_size;
    int mid_ind = prev_ind % input_width + offset;
    prev_ind = prev_ind / input_width;
    int out_ind = (prev_ind * output_width + mid_ind) * concat_size + post_ind;
    output[out_ind] = input[idx];
  }
}

template <typename spec_t>
void concatenate_gradient_cpu(const spec_t* output_grad, spec_t* input_grad,
                              int input_width, int output_width, int offset,
                              int concat_size, size_t size) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; ++idx) {
    int post_ind = idx % concat_size;
    int prev_ind = idx / concat_size;
    int mid_ind = prev_ind % input_width + offset;
    prev_ind = prev_ind / input_width;
    int out_ind = (prev_ind * output_width + mid_ind) * concat_size + post_ind;
    input_grad[idx] = output_grad[out_ind];
  }
}

void ConcatenateCpu(const NDArrayList& inputs, NDArray& output, size_t axis,
                    const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(output);

  CPUStream cpu_stream(stream);
  dnnl::engine eng(dnnl::engine::kind::cpu, 0);

  for (size_t i = 0; i < inputs.size(); ++i)
    HT_ASSERT_SAME_DEVICE(inputs[i], output);

  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    inputs[0]->dtype(), spec_t, "ConcatenateCpu", [&]() {
      std::vector<dnnl::memory::desc> src_mds;
      std::vector<dnnl::memory> src_mems;
      auto dnnltype = hetu::cpu::dtype_to_dnnltype(inputs[0]->dtype());

      for (size_t i = 0; i < inputs.size(); ++i) {
          auto md = dnnl::memory::desc(inputs[i]->shape(), dnnltype, inputs[i]->stride());
          auto mem = dnnl::memory(md, eng, inputs[i]->data_ptr<spec_t>());

          src_mds.push_back(md);
          src_mems.push_back(mem);
      }

      // Create primitive descriptor.
      auto concat_pd = dnnl::concat::primitive_desc(eng, axis, src_mds);

      // Create destination (dst) memory object using the memory descriptor
      // created by the primitive.
      auto dst_mem = dnnl::memory(concat_pd.dst_desc(), eng, output->data_ptr<spec_t>());

      // Create the primitive.
      auto concat_prim = dnnl::concat(concat_pd);

      // Primitive arguments.
      std::unordered_map<int, dnnl::memory> concat_args;
      for (size_t i = 0; i < inputs.size(); ++i)
          concat_args.insert({DNNL_ARG_MULTIPLE_SRC + i, src_mems[i]});
      concat_args.insert({DNNL_ARG_DST, dst_mem});
      auto _future = cpu_stream.EnqueueTask(
      [concat_prim, concat_args, eng]() {
        dnnl::stream engine_stream(eng);
        concat_prim.execute(engine_stream, concat_args);
        engine_stream.wait();
      },
      "Concatenate");
    });
  NDArray::MarkUsedBy(inputs, stream);
  NDArray::MarkUsedBy({output}, stream);
}

void ConcatenateGradientCpu(const NDArray& output_grad, NDArray& input_grad,
                            size_t axis, size_t offset, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(output_grad);
  HT_ASSERT_SAME_DEVICE(output_grad, input_grad);

  CPUStream cpu_stream(stream);
  dnnl::engine eng(dnnl::engine::kind::cpu, 0);

  size_t size = input_grad->numel();
  size_t now_ndim = output_grad->ndim();
  HT_ASSERT(now_ndim == input_grad->ndim());
  int num_concats = 1;
  for (size_t i = 0; i < axis; ++i) {
    int cur_dim = output_grad->shape(i);
    HT_ASSERT(cur_dim == input_grad->shape(i));
    num_concats *= cur_dim;
  }
  int concat_size = 1;
  for (size_t i = axis + 1; i < now_ndim; ++i) {
    int cur_dim = output_grad->shape(i);
    HT_ASSERT(cur_dim == input_grad->shape(i));
    concat_size *= cur_dim;
  }
  int output_width = output_grad->shape(axis);
  int input_width = input_grad->shape(axis);
  if (size == 0 || input_width == 0 || output_width == 0)
    return;
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input_grad->dtype(), spec_t, "ConcatenateGradientCpu", [&]() {
      auto _future = cpu_stream.EnqueueTask(
      [output_grad, input_grad, input_width, output_width, offset, concat_size, size]() {
      concatenate_gradient_cpu<spec_t>(
        output_grad->data_ptr<spec_t>(), input_grad->data_ptr<spec_t>(),
        input_width, output_width, offset, concat_size, size);
      },
      "ConcatGradient");  
    });
  NDArray::MarkUsedBy({output_grad, input_grad}, stream);
}

} // namespace impl
} // namespace hetu
