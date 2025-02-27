#include "hetu/graph/ops/Sin.h"
#include "hetu/graph/ops/Arithmetics.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

NDArrayList SinOpImpl::DoCompute(Operator& op,
                                 const NDArrayList& inputs,
                                 RuntimeContext& ctx) const {
  NDArrayList outputs = inplace() && !ctx.has_runtime_allocation(op->output(0)->id()) ? inputs : DoAllocOutputs(op, inputs, ctx);
  NDArray::sin(inputs.at(0), op->instantiation_ctx().stream_index, outputs.at(0));
  return outputs;
}

void SinOpImpl::DoCompute(Operator& op, const NDArrayList& inputs,
                          NDArrayList& outputs, RuntimeContext& ctx) const {
  NDArray::sin(inputs.at(0), op->instantiation_ctx().stream_index, outputs.at(0));
}

TensorList SinOpImpl::DoGradient(Operator& op, const TensorList& grad_outputs) const {
  HT_ASSERT(!inplace())
    << "This op doesn't support gradient for inplace.";
  return {op->requires_grad(0) ? MakeSinGradientOp(op->input(0), grad_outputs.at(0), 
                                 op->grad_op_meta().set_name(op->grad_name()))
                               : Tensor()};
}

void SinGradientOpImpl::DoCompute(Operator& op,const NDArrayList& inputs,
                                  NDArrayList& outputs, RuntimeContext& ctx) const {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(),
                                  hetu::impl::SinGradient, inputs.at(0),
                                  inputs.at(1), outputs.at(0), op->instantiation_ctx().stream());
}

NDArrayList CosOpImpl::DoCompute(Operator& op,
                                 const NDArrayList& inputs,
                                 RuntimeContext& ctx) const {
  NDArrayList outputs = inplace() && !ctx.has_runtime_allocation(op->output(0)->id()) ? inputs : DoAllocOutputs(op, inputs, ctx);
  NDArray::cos(inputs.at(0), op->instantiation_ctx().stream_index, outputs.at(0));
  return outputs;
}

void CosOpImpl::DoCompute(Operator& op, const NDArrayList& inputs,
                          NDArrayList& outputs, RuntimeContext& ctx) const {
  NDArray::cos(inputs.at(0), op->instantiation_ctx().stream_index, outputs.at(0));
}

TensorList CosOpImpl::DoGradient(Operator& op, const TensorList& grad_outputs) const {
  HT_ASSERT(!inplace())
    << "This op doesn't support gradient for inplace.";
  return {op->requires_grad(0) ? MakeCosGradientOp(op->input(0), grad_outputs.at(0), 
                                 op->grad_op_meta().set_name(op->grad_name()))
                               : Tensor()};
}

void CosGradientOpImpl::DoCompute(Operator& op,const NDArrayList& inputs,
                                  NDArrayList& outputs, RuntimeContext& ctx) const {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(),
                                  hetu::impl::CosGradient, inputs.at(0),
                                  inputs.at(1), outputs.at(0), op->instantiation_ctx().stream());
}

Tensor MakeSinOp(Tensor input, OpMeta op_meta) {
  TensorList inputs = {std::move(input)};
  return Graph::MakeOp(
    std::make_shared<SinOpImpl>(false),
    std::move(inputs),
    std::move(op_meta))->output(0);
}

Tensor MakeSinInplaceOp(Tensor input, OpMeta op_meta) {
  TensorList inputs = {std::move(input)};
  return Graph::MakeOp(
    std::make_shared<SinOpImpl>(true),
    std::move(inputs),
    std::move(op_meta))->output(0);
}

Tensor MakeSinGradientOp(Tensor input, Tensor grad_output,
                         OpMeta op_meta) {
  return Graph::MakeOp(
        std::make_shared<SinGradientOpImpl>(),
        {std::move(input), std::move(grad_output)},
        std::move(op_meta))->output(0);
}

Tensor MakeCosOp(Tensor input, bool inplace, OpMeta op_meta) {
  return Graph::MakeOp(
    std::make_shared<CosOpImpl>(inplace),
    {std::move(input)},
    std::move(op_meta))->output(0);
}

Tensor MakeCosGradientOp(Tensor input, Tensor grad_output,
                         OpMeta op_meta) {
  return Graph::MakeOp(
        std::make_shared<CosGradientOpImpl>(),
        {std::move(input), std::move(grad_output)},
        std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hetu
