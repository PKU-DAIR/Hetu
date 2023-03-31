#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/omp_utils.h"
#include "hetu/impl/stream/CPUStream.h"

namespace hetu {
namespace impl {
void Conv2dCpu(const NDArray& input_x, const NDArray& input_f, NDArray& output,
               const int padding_h, const int padding_w, const int stride_h,
               const int stride_w, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input_x);
  HT_ASSERT_SAME_DEVICE(input_x, input_f);
  HT_ASSERT_SAME_DEVICE(input_x, output);

  CPUStream cpu_stream(stream);
  dnnl::engine eng(dnnl::engine::kind::cpu, cpu_stream.stream_id());

  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input_x->dtype(), spec_t, "Conv2dCpu", [&]() {
      auto _future = cpu_stream.EnqueueTask(
      [input_x, input_f, output, eng, 
      padding_h, padding_w, stride_h, stride_w]() {
        auto conv_src_md = dnnl::memory::desc(input_x->shape(), dnnl::memory::data_type::f32, 
                                              dnnl::memory::format_tag::nchw);
        auto conv_weights_md = dnnl::memory::desc(input_f->shape(), dnnl::memory::data_type::f32, 
                                                  dnnl::memory::format_tag::oihw);
        auto conv_dst_md = dnnl::memory::desc(output->shape(), dnnl::memory::data_type::f32,
                                              dnnl::memory::format_tag::nchw);

        auto conv_src_mem = dnnl::memory(conv_src_md, eng, input_x->data_ptr<spec_t>());
        auto conv_weights_mem = dnnl::memory(conv_weights_md, eng, input_f->data_ptr<spec_t>());
        auto conv_dst_mem = dnnl::memory(conv_dst_md, eng, output->data_ptr<spec_t>());


        dnnl::memory::dims strides_dims = {int(stride_h), int(stride_w)};
        dnnl::memory::dims padding_dims_l = {int(padding_h), int(padding_w)};
        dnnl::memory::dims padding_dims_r = {int(padding_h), int(padding_w)};

        // Create primitive descriptor.
        auto conv_pd = dnnl::convolution_forward::primitive_desc(eng,
                dnnl::prop_kind::forward_training, dnnl::algorithm::convolution_direct,
                conv_src_md, conv_weights_md, conv_dst_md,
                strides_dims, padding_dims_l, padding_dims_r);

        // Create the primitive.
        auto conv_prim = dnnl::convolution_forward(conv_pd);

        // Primitive arguments.
        std::unordered_map<int, dnnl::memory> conv_args;
        conv_args.insert({DNNL_ARG_SRC, conv_src_mem});
        conv_args.insert({DNNL_ARG_WEIGHTS, conv_weights_mem});
        conv_args.insert({DNNL_ARG_DST, conv_dst_mem});

        dnnl::stream engine_stream(eng);
        conv_prim.execute(engine_stream, conv_args);
      },
      "Conv2d");
      //cpu_stream.Sync();
    });
  return;
}

void Conv2dGradientofFilterCpu(const NDArray& input_x,
                               const NDArray& gradient_y, NDArray& gradient_f,
                               const int padding_h, const int padding_w,
                               const int stride_h, const int stride_w,
                               const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input_x);
  HT_ASSERT_SAME_DEVICE(input_x, gradient_y);
  HT_ASSERT_SAME_DEVICE(input_x, gradient_f);

  CPUStream cpu_stream(stream);
  dnnl::engine eng(dnnl::engine::kind::cpu, cpu_stream.stream_id()); 
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input_x->dtype(), spec_t, "Conv2dGradientofFilterCpu", [&]() {
      auto _future = cpu_stream.EnqueueTask(
      [input_x, gradient_y, gradient_f, eng,
      padding_h, padding_w, stride_h, stride_w]() {
      auto conv_src_md = dnnl::memory::desc(input_x->shape(), dnnl::memory::data_type::f32, 
                                            dnnl::memory::format_tag::nchw);
      auto conv_weights_md = dnnl::memory::desc(gradient_f->shape(), dnnl::memory::data_type::f32, 
                                                dnnl::memory::format_tag::oihw);
      auto conv_dst_md = dnnl::memory::desc(gradient_y->shape(), dnnl::memory::data_type::f32,
                                            dnnl::memory::format_tag::nchw);

      auto conv_src_mem = dnnl::memory(conv_src_md, eng, input_x->data_ptr<spec_t>());
      auto conv_weights_mem = dnnl::memory(conv_weights_md, eng, gradient_f->data_ptr<spec_t>());
      auto conv_dst_mem = dnnl::memory(conv_dst_md, eng, gradient_y->data_ptr<spec_t>());

      dnnl::memory::dims strides_dims = {int(stride_h), int(stride_w)};
      dnnl::memory::dims padding_dims_l = {int(padding_h), int(padding_w)};
      dnnl::memory::dims padding_dims_r = {int(padding_h), int(padding_w)};

      // Create primitive descriptor.
      auto conv_pd = dnnl::convolution_forward::primitive_desc(eng,
              dnnl::prop_kind::forward_training, dnnl::algorithm::convolution_direct,
              conv_src_md, conv_weights_md, conv_dst_md,
              strides_dims, padding_dims_l, padding_dims_r);
      
      auto conv_bwd_pd = dnnl::convolution_backward_weights::primitive_desc(eng,
              dnnl::algorithm::convolution_direct,
              conv_src_md, conv_dst_md, conv_weights_md,
              strides_dims, padding_dims_l, padding_dims_r, conv_pd);

      // Create the primitive.
      auto conv_prim = dnnl::convolution_backward_weights(conv_bwd_pd);

      // Primitive arguments.
      std::unordered_map<int, dnnl::memory> conv_args;
      conv_args.insert({DNNL_ARG_SRC, conv_src_mem});
      conv_args.insert({DNNL_ARG_DIFF_WEIGHTS, conv_weights_mem});
      conv_args.insert({DNNL_ARG_DIFF_DST, conv_dst_mem});

      dnnl::stream engine_stream(eng);
      conv_prim.execute(engine_stream, conv_args);         
      },
      "Conv2dFilter");
      //cpu_stream.Sync();
    });
  return;
}

void Conv2dGradientofDataCpu(const NDArray& input_f, const NDArray& gradient_y,
                             NDArray& gradient_x, const int padding_h,
                             const int padding_w, const int stride_h,
                             const int stride_w, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input_f);
  HT_ASSERT_SAME_DEVICE(input_f, gradient_y);
  HT_ASSERT_SAME_DEVICE(input_f, gradient_x);

  CPUStream cpu_stream(stream);
  dnnl::engine eng(dnnl::engine::kind::cpu, cpu_stream.stream_id());
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input_f->dtype(), spec_t, "Conv2dGradientofDataCpu", [&]() {
    auto _future = cpu_stream.EnqueueTask(
      [input_f, gradient_y, gradient_x, eng,
      padding_h, padding_w, stride_h, stride_w]() {
      auto conv_src_md = dnnl::memory::desc(gradient_x->shape(), dnnl::memory::data_type::f32, 
                                            dnnl::memory::format_tag::nchw);
      auto conv_weights_md = dnnl::memory::desc(input_f->shape(), dnnl::memory::data_type::f32, 
                                                dnnl::memory::format_tag::oihw);
      auto conv_dst_md = dnnl::memory::desc(gradient_y->shape(), dnnl::memory::data_type::f32,
                                            dnnl::memory::format_tag::nchw);

      auto conv_src_mem = dnnl::memory(conv_src_md, eng, gradient_x->data_ptr<spec_t>());
      auto conv_weights_mem = dnnl::memory(conv_weights_md, eng, input_f->data_ptr<spec_t>());
      auto conv_dst_mem = dnnl::memory(conv_dst_md, eng, gradient_y->data_ptr<spec_t>());

      dnnl::memory::dims strides_dims = {int(stride_h), int(stride_w)};
      dnnl::memory::dims padding_dims_l = {int(padding_h), int(padding_w)};
      dnnl::memory::dims padding_dims_r = {int(padding_h), int(padding_w)};

      // Create primitive descriptor.
      auto conv_pd = dnnl::convolution_forward::primitive_desc(eng,
              dnnl::prop_kind::forward_training, dnnl::algorithm::convolution_direct,
              conv_src_md, conv_weights_md, conv_dst_md,
              strides_dims, padding_dims_l, padding_dims_r);
      
      auto conv_bwd_pd = dnnl::convolution_backward_data::primitive_desc(eng,
              dnnl::algorithm::convolution_direct,
              conv_src_md, conv_weights_md, conv_dst_md,
              strides_dims, padding_dims_l, padding_dims_r, conv_pd);

      // Create the primitive.
      auto conv_prim = dnnl::convolution_backward_data(conv_bwd_pd);

      // Primitive arguments.
      std::unordered_map<int, dnnl::memory> conv_args;
      conv_args.insert({DNNL_ARG_DIFF_SRC, conv_src_mem});
      conv_args.insert({DNNL_ARG_WEIGHTS, conv_weights_mem});
      conv_args.insert({DNNL_ARG_DIFF_DST, conv_dst_mem});

      dnnl::stream engine_stream(eng);
      conv_prim.execute(engine_stream, conv_args);         
      },
      "Conv2dData");
      //cpu_stream.Sync();
    });
  return;
}

void Conv2dAddBiasCpu(const NDArray& input_x, const NDArray& input_f,
                      const NDArray& bias, NDArray& output,
                      const int padding_h, const int padding_w,
                      const int stride_h, const int stride_w,
                      const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input_x);
  HT_ASSERT_SAME_DEVICE(input_x, input_f);
  HT_ASSERT_SAME_DEVICE(input_x, bias);
  HT_ASSERT_SAME_DEVICE(input_x, output);

  CPUStream cpu_stream(stream);
  dnnl::engine eng(dnnl::engine::kind::cpu, cpu_stream.stream_id());
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input_x->dtype(), spec_t, "Conv2dAddBiasCpu", [&]() {
    auto _future = cpu_stream.EnqueueTask(
      [input_x, input_f, output, bias, eng, 
      padding_h, padding_w, stride_h, stride_w]() {
      auto conv_src_md = dnnl::memory::desc(input_x->shape(), dnnl::memory::data_type::f32, 
                                            dnnl::memory::format_tag::nchw);
      auto conv_weights_md = dnnl::memory::desc(input_f->shape(), dnnl::memory::data_type::f32, 
                                                dnnl::memory::format_tag::oihw);
      auto conv_dst_md = dnnl::memory::desc(output->shape(), dnnl::memory::data_type::f32,
                                            dnnl::memory::format_tag::nchw);
      auto conv_bias_md = dnnl::memory::desc(bias->shape(), dnnl::memory::data_type::f32, 
                                             dnnl::memory::format_tag::a);

      auto conv_src_mem = dnnl::memory(conv_src_md, eng, input_x->data_ptr<spec_t>());
      auto conv_weights_mem = dnnl::memory(conv_weights_md, eng, input_f->data_ptr<spec_t>());
      auto conv_dst_mem = dnnl::memory(conv_dst_md, eng, output->data_ptr<spec_t>());
      auto conv_bias_mem = dnnl::memory(conv_bias_md, eng, bias->data_ptr<spec_t>());

      dnnl::memory::dims strides_dims = {int(stride_h), int(stride_w)};
      dnnl::memory::dims padding_dims_l = {int(padding_h), int(padding_w)};
      dnnl::memory::dims padding_dims_r = {int(padding_h), int(padding_w)};

      // Create primitive descriptor.
      auto conv_pd = dnnl::convolution_forward::primitive_desc(eng,
              dnnl::prop_kind::forward_training, dnnl::algorithm::convolution_direct,
              conv_src_md, conv_weights_md, conv_bias_md, conv_dst_md,
              strides_dims, padding_dims_l, padding_dims_r);

      // Create the primitive.
      auto conv_prim = dnnl::convolution_forward(conv_pd);

      // Primitive arguments.
      std::unordered_map<int, dnnl::memory> conv_args;
      conv_args.insert({DNNL_ARG_SRC, conv_src_mem});
      conv_args.insert({DNNL_ARG_WEIGHTS, conv_weights_mem});
      conv_args.insert({DNNL_ARG_BIAS, conv_bias_mem});
      conv_args.insert({DNNL_ARG_DST, conv_dst_mem});

      dnnl::stream engine_stream(eng);
      conv_prim.execute(engine_stream, conv_args); 
      },
      "Conv2dBias");
      //cpu_stream.Sync();
    });
  return;
}
} // namespace impl
} // namespace hetu
