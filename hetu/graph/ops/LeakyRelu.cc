#include "hetu/graph/ops/LeakyRelu.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

void LeakyReluOpImpl::DoCompute(Operator& op,
                                const NDArrayList& inputs, NDArrayList& outputs,
                                RuntimeContext& ctx) const {
  NDArray::leakyrelu(inputs.at(0), get_alpha(),
                     op->instantiation_ctx().stream_index, outputs.at(0));
}

NDArrayList LeakyReluOpImpl::DoCompute(Operator& op,
                                       const NDArrayList& inputs,
                                       RuntimeContext& ctx) const {
  NDArrayList outputs = inplace() && !ctx.has_runtime_allocation(op->output(0)->id()) ? inputs : DoAllocOutputs(op, inputs, ctx);
  DoCompute(op, inputs, outputs, ctx);
  return outputs;
}

TensorList LeakyReluOpImpl::DoGradient(Operator& op, const TensorList& grad_outputs) const {
  return {op->requires_grad(0) ? MakeLeakyReluGradientOp(op->input(0), grad_outputs.at(0), get_alpha(),
                                inplace(), op->grad_op_meta().set_name(op->grad_name(0)))
                              : Tensor()};
}

// NOTE: LeakyRelu backward calculation doesn't support in-place call with negative slope.
// The reason is that when calculating backward gradient, there is no way to know
// whether the original input is positive or not if the slope is negative.
// E.g., forward is 4, slope is -0.1, the original input could be 4 or -40. There is no
// way to get a correct backward gradient in this case.
void LeakyReluGradientOpImpl::DoCompute(Operator& op,const NDArrayList& inputs,
                                        NDArrayList& outputs,
                                        RuntimeContext& ctx) const {
  HT_ASSERT(!is_result() || get_alpha() >= 0)
  << "In-place LeakyRelu backward calculation is triggered with a negative slope which is not supported.\n"
  << "This is caused by calling in-place forward calculation with a negative slope, "
  << "please call out-of-place version instead.";

  HT_DISPATCH_KERNEL_CPU_AND_CUDA(
    op->instantiation_ctx().placement.type(), type(), hetu::impl::LeakyReluGradient, inputs.at(0),
    inputs.at(1), get_alpha(), outputs.at(0), op->instantiation_ctx().stream());
}

Tensor MakeLeakyReluOp(Tensor input, double alpha, OpMeta op_meta) {
  TensorList inputs = {std::move(input)};
  return Graph::MakeOp(
        std::make_shared<LeakyReluOpImpl>(alpha, false),
        std::move(inputs),
        std::move(op_meta))->output(0);   
}

Tensor MakeLeakyReluInplaceOp(Tensor input, double alpha, OpMeta op_meta) {
  TensorList inputs = {std::move(input)};
  DataType input_type = DataType::FLOAT16;
  AutoCast::Tensor_AutoCast(inputs, input_type);
  return Graph::MakeOp(
        std::make_shared<LeakyReluOpImpl>(alpha, true),
        std::move(inputs),
        std::move(op_meta))->output(0);   
}

Tensor MakeLeakyReluGradientOp(Tensor input, Tensor grad_output, double alpha,
                               bool is_result, OpMeta op_meta) {
  return Graph::MakeOp(
          std::make_shared<LeakyReluGradientOpImpl>(alpha, is_result),
          {std::move(input), std::move(grad_output)},
          std::move(op_meta))->output(0);   
}

} // namespace graph
} // namespace hetu
