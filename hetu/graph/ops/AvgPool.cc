#include "hetu/graph/ops/AvgPool.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

void AvgPoolOpImpl::DoCompute(Operator& op,
                              const NDArrayList& inputs, NDArrayList& outputs,
                              RuntimeContext& ctx) const {
  // HT_DISPATCH_KERNEL_CUDA_ONLY(op->instantiation_ctx().placement.type(), type(), hetu::impl::AvgPool,
  //                              inputs.at(0), get_kernel_H(), get_kernel_W(),
  //                              outputs.at(0), get_padding(), get_stride(),
  //                              op->instantiation_ctx().stream());
  NDArray::avgpool(inputs.at(0), get_kernel_H(), get_kernel_W(), get_padding(), get_stride(), 
                   op->instantiation_ctx().stream_index, outputs.at(0));
}

TensorList AvgPoolOpImpl::DoGradient(Operator& op,
                                     const TensorList& grad_outputs) const {
  return {op->require_grad(0) ? MakeAvgPoolGradientOp(op->output(0), grad_outputs.at(0), op->input(0),
                                get_kernel_H(), get_kernel_W(), get_padding(),
                                get_stride(), op->grad_op_meta().set_name(op->grad_name()))
                              : Tensor()};
}

HTShapeList AvgPoolOpImpl::DoInferShape(Operator& op,
                                        const HTShapeList& input_shapes,
                                        RuntimeContext& ctx) const {
  int64_t N = input_shapes.at(0)[0];
  int64_t C = input_shapes.at(0)[1];
  int64_t H = input_shapes.at(0)[2];
  int64_t W = input_shapes.at(0)[3];
  int64_t p_H = (H + 2 * get_padding() - get_kernel_H()) / get_stride() + 1;
  int64_t p_W = (W + 2 * get_padding() - get_kernel_W()) / get_stride() + 1;
  return {{N, C, p_H, p_W}};
}

void AvgPoolGradientOpImpl::DoCompute(Operator& op,
                                      const NDArrayList& inputs,
                                      NDArrayList& outputs,
                                      RuntimeContext& ctx) const {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(
    op->instantiation_ctx().placement.type(), type(), hetu::impl::AvgPoolGradient, inputs.at(0),
    inputs.at(1), inputs.at(2), get_kernel_H(), get_kernel_W(), outputs.at(0),
    get_padding(), get_stride(), op->instantiation_ctx().stream());
}

HTShapeList
AvgPoolGradientOpImpl::DoInferShape(Operator& op,
                                    const HTShapeList& input_shapes,
                                    RuntimeContext& ctx) const {
  return {input_shapes.at(2)};
}

Tensor MakeAvgPoolOp(Tensor input, size_t kernel_H, size_t kernel_W, 
                     size_t padding, size_t stride,
                     OpMeta op_meta) {
  return Graph::MakeOp(
           std::make_shared<AvgPoolOpImpl>(kernel_H, kernel_W, padding, stride),
           {std::move(input)},
           std::move(op_meta))->output(0);
}

Tensor MakeAvgPoolGradientOp(Tensor output, Tensor output_grad, Tensor input,
                             size_t kernel_H, size_t kernel_W, size_t padding,
                             size_t stride, OpMeta op_meta) {
  return Graph::MakeOp(
           std::make_shared<AvgPoolGradientOpImpl>(kernel_H, kernel_W, padding, stride),
           {std::move(output), std::move(output_grad)},
           std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hetu
