#include "hetu/graph/ops/Pow.h"
#include "hetu/graph/ops/Arithmetics.h"
#include "hetu/graph/ops/zeros_like.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

void PowTensorAndConstOpImpl::DoCompute(Operator& op, 
                                        const NDArrayList& inputs, NDArrayList& outputs,
                                        RuntimeContext& ctx) const {
  NDArray::pow(inputs.at(0), exponent(), op->instantiation_ctx().stream_index, outputs.at(0));
}

NDArrayList PowTensorAndConstOpImpl::DoCompute(Operator& op,
                                               const NDArrayList& inputs,
                                               RuntimeContext& ctx) const {
  NDArrayList outputs = inplace() && !ctx.has_runtime_allocation(op->output(0)->id()) ? inputs : DoAllocOutputs(op, inputs, ctx);
  DoCompute(op, inputs, outputs, ctx);
  return outputs;
}

TensorList PowTensorAndConstOpImpl::DoGradient(Operator& op, const TensorList& grad_outputs) const {
  HT_ASSERT(!inplace())
  << "In-place Pow backward calculation is triggered, which is not supported.";

  auto g_op_meta = op->grad_op_meta();
  return {op->requires_grad(0) ? MakeMulElewiseOp(MakeMulByConstOp(MakePowTensorAndConstOp(
                                          op->input(0), exponent() - 1, g_op_meta), exponent(), g_op_meta),
                                          grad_outputs.at(0), g_op_meta.set_name(op->grad_name()))
                                        : Tensor()};
}

Tensor MakePowTensorAndConstOp(Tensor input, double exponent, OpMeta op_meta) {
  TensorList inputs = {std::move(input)};
  return Graph::MakeOp(
    std::make_shared<PowTensorAndConstOpImpl>(exponent, false),
    std::move(inputs),
    std::move(op_meta))->output(0);
}

Tensor MakePowTensorAndConstInplaceOp(Tensor input, double exponent, OpMeta op_meta) {
  TensorList inputs = {std::move(input)};
  DataType input_type = DataType::FLOAT16;
  AutoCast::Tensor_AutoCast(inputs, input_type);
  return Graph::MakeOp(
    std::make_shared<PowTensorAndConstOpImpl>(exponent, true),
    std::move(inputs),
    std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hetu
