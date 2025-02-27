#include "hetu/graph/ops/Relu.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

void ReluOpImpl::DoCompute(Operator& op, 
                           const NDArrayList& inputs, NDArrayList& outputs,
                           RuntimeContext& ctx) const {
  NDArray::relu(inputs.at(0), op->instantiation_ctx().stream_index, outputs.at(0));
}

NDArrayList ReluOpImpl::DoCompute(Operator& op,
                                  const NDArrayList& inputs,
                                  RuntimeContext& ctx) const {
  NDArrayList outputs = inplace() && !ctx.has_runtime_allocation(op->output(0)->id()) ? inputs : DoAllocOutputs(op, inputs, ctx);
  DoCompute(op, inputs, outputs, ctx);
  return outputs;
}

TensorList ReluOpImpl::DoGradient(Operator& op, const TensorList& grad_outputs) const {
  return {op->requires_grad(0) ? MakeReluGradientOp(op->input(0), grad_outputs.at(0),
                                op->grad_op_meta().set_name(op->grad_name()))
                              : Tensor()};
}

void ReluGradientOpImpl::DoCompute(Operator& op,const NDArrayList& inputs,
                                  NDArrayList& outputs, RuntimeContext& ctx) const {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(),
                                  hetu::impl::ReluGradient, inputs.at(0),
                                  inputs.at(1), outputs.at(0), op->instantiation_ctx().stream());
}

Tensor MakeReluOp(Tensor input, OpMeta op_meta) {
  TensorList inputs = {std::move(input)};
  return Graph::MakeOp(
        std::make_shared<ReluOpImpl>(false),
        std::move(inputs),
        std::move(op_meta))->output(0);
}

Tensor MakeReluInplaceOp(Tensor input, OpMeta op_meta) {
  TensorList inputs = {std::move(input)};
  DataType input_type = DataType::FLOAT16;
  AutoCast::Tensor_AutoCast(inputs, input_type);
  return Graph::MakeOp(
        std::make_shared<ReluOpImpl>(true),
        std::move(inputs),
        std::move(op_meta))->output(0);
}

Tensor MakeReluGradientOp(Tensor input, Tensor grad_output,
                          OpMeta op_meta) {
  return Graph::MakeOp(
        std::make_shared<ReluGradientOpImpl>(),
        {std::move(input), std::move(grad_output)},
        std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hetu
