#include "hetu/graph/ops/Sigmoid.h"
#include "hetu/graph/ops/Arithmetics.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

void SigmoidOpImpl::DoCompute(Operator& op, 
                              const NDArrayList& inputs, NDArrayList& outputs,
                              RuntimeContext& ctx) const {
  NDArray::sigmoid(inputs.at(0), op->instantiation_ctx().stream_index, outputs.at(0));
}

NDArrayList SigmoidOpImpl::DoCompute(Operator& op,
                                     const NDArrayList& inputs,
                                     RuntimeContext& ctx) const {
  NDArrayList outputs = inplace() && !ctx.has_runtime_allocation(op->output(0)->id()) ? inputs : DoAllocOutputs(op, inputs, ctx);
  DoCompute(op, inputs, outputs, ctx);
  return outputs;
}

TensorList SigmoidOpImpl::DoGradient(Operator& op, const TensorList& grad_outputs) const {
  auto g_op_meta = op->grad_op_meta();
  auto grad_input = op->requires_grad(0) ? MakeSigmoidGradientOp(grad_outputs.at(0), op->output(0), 
                                                                 g_op_meta.set_name(op->grad_name(0)))
                                         : Tensor();
  return {grad_input};
}

void SigmoidGradientOpImpl::DoCompute(Operator& op, 
                              const NDArrayList& inputs, NDArrayList& outputs,
                              RuntimeContext& ctx) const {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(),
                                  hetu::impl::SigmoidGradient, inputs.at(0), inputs.at(1),
                                  outputs.at(0), op->instantiation_ctx().stream());
}

Tensor MakeSigmoidOp(Tensor input, OpMeta op_meta) {
  TensorList inputs = {std::move(input)};
  return Graph::MakeOp(
    std::make_shared<SigmoidOpImpl>(false),
    std::move(inputs),
    std::move(op_meta))->output(0);
}

Tensor MakeSigmoidInplaceOp(Tensor input, OpMeta op_meta) {
  TensorList inputs = {std::move(input)};
  DataType input_type = DataType::FLOAT32;
  AutoCast::Tensor_AutoCast(inputs, input_type);
  return Graph::MakeOp(
    std::make_shared<SigmoidOpImpl>(true),
    std::move(inputs),
    std::move(op_meta))->output(0);
}

Tensor MakeSigmoidGradientOp(Tensor out_grad, Tensor output, OpMeta op_meta) {
  TensorList inputs = {std::move(out_grad), std::move(output)};
  return Graph::MakeOp(
    std::make_shared<SigmoidGradientOpImpl>(),
    std::move(inputs),
    std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hetu
