#include "hetu/graph/ops/Abs.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {


NDArrayList AbsOpImpl::DoCompute(Operator& op,
                                 const NDArrayList& inputs,
                                 RuntimeContext& ctx) const {
  NDArrayList outputs = inplace() && !ctx.has_runtime_allocation(op->output(0)->id()) ? inputs : DoAllocOutputs(op, inputs, ctx);
  NDArray::abs(inputs.at(0), op->instantiation_ctx().stream_index, outputs.at(0));
  return outputs;
}

void AbsOpImpl::DoCompute(Operator& op, const NDArrayList& inputs,
                          NDArrayList& outputs, RuntimeContext& ctx) const {
  NDArray::abs(inputs.at(0), op->instantiation_ctx().stream_index, outputs.at(0));
}

TensorList AbsOpImpl::DoGradient(Operator& op, const TensorList& grad_outputs) const {
  HT_ASSERT(!inplace())
    << "This op doesn't support gradient for inplace.";
  return {op->requires_grad(0) ? MakeAbsGradientOp(op->input(0), grad_outputs.at(0),
                                 op->grad_op_meta().set_name(op->grad_name()))
                                : Tensor()};
}

void AbsGradientOpImpl::DoCompute(Operator& op,const NDArrayList& inputs,
                                  NDArrayList& outputs, RuntimeContext& ctx) const {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(),
                                  hetu::impl::AbsGradient, inputs.at(0),
                                  inputs.at(1), outputs.at(0), op->instantiation_ctx().stream());
}

Tensor MakeAbsOp(Tensor input, OpMeta op_meta) {
  TensorList inputs = {std::move(input)};
  return Graph::MakeOp(
        std::make_shared<AbsOpImpl>(false),
        std::move(inputs),
        std::move(op_meta))->output(0);
}

Tensor MakeAbsInplaceOp(Tensor input, OpMeta op_meta) {
  TensorList inputs = {std::move(input)};
  return Graph::MakeOp(
        std::make_shared<AbsOpImpl>(true),
        std::move(inputs),
        std::move(op_meta))->output(0);
}

Tensor MakeAbsGradientOp(Tensor input, Tensor grad_output,
                         OpMeta op_meta) {
  return Graph::MakeOp(
        std::make_shared<AbsGradientOpImpl>(),
        {std::move(input), std::move(grad_output)},
        std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hetu
