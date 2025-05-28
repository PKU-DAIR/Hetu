#include "hetu/graph/ops/Where.h"
#include "hetu/graph/ops/zeros_like.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

void WhereOpImpl::DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                           RuntimeContext& ctx) const {
  NDArray::where(inputs.at(0), inputs.at(1), inputs.at(2),
                 op->instantiation_ctx().stream_index, outputs.at(0));
}

NDArrayList WhereOpImpl::DoCompute(Operator& op,
                                   const NDArrayList& inputs,
                                   RuntimeContext& ctx) const {
  NDArrayList outputs = inplace() && !ctx.has_runtime_allocation(op->output(0)->id()) ? NDArrayList{inputs.at(1)} : DoAllocOutputs(op, inputs, ctx);
  DoCompute(op, inputs, outputs, ctx);
  return outputs;
}

TensorList WhereOpImpl::DoGradient(Operator& op, const TensorList& grad_outputs) const {
  auto g_op_meta = op->grad_op_meta();
  auto zero_grad = MakeZerosLikeOp(op->input(1), g_op_meta);
  auto grad_inputA = op->requires_grad(1) ? MakeWhereOp(op->input(0), grad_outputs.at(0), zero_grad,
                                           g_op_meta.set_name(op->grad_name(1)))
                                         : Tensor();
  auto grad_inputB = op->requires_grad(2) ? MakeWhereOp(op->input(0), zero_grad, grad_outputs.at(0),
                                           g_op_meta.set_name(op->grad_name(2)))
                                         : Tensor();
  return {Tensor(), grad_inputA, grad_inputB};
}

HTShapeList WhereOpImpl::DoInferShape(Operator& op, 
                                      const HTShapeList& input_shapes, 
                                      RuntimeContext& ctx) const {
  HT_ASSERT(input_shapes.at(0).size() == input_shapes.at(1).size() &&
            input_shapes.at(0).size() == input_shapes.at(2).size())
          << input_shapes.at(0) << " " << input_shapes.at(1) <<
          " " << input_shapes.at(2);
  return {input_shapes.at(1)};
}

HTShapeList WhereOpImpl::DoInferDynamicShape(Operator& op, 
                                      const HTShapeList& input_shapes, 
                                      RuntimeContext& ctx) const {
  // shape 1 should have the smallest dynamic shape
  HT_ASSERT(input_shapes.at(0).size() >= input_shapes.at(1).size() &&
            input_shapes.at(1).size() <= input_shapes.at(2).size())
          << input_shapes.at(0) << " " << input_shapes.at(1) <<
          " " << input_shapes.at(2);
  return {input_shapes.at(1)};
}

Tensor MakeWhereOp(Tensor cond, Tensor inputA, Tensor inputB,
                   OpMeta op_meta) {
  return Graph::MakeOp(
    std::make_shared<WhereOpImpl>(false),
    {std::move(cond), std::move(inputA), std::move(inputB)},
    std::move(op_meta))->output(0);
}

Tensor MakeWhereInplaceOp(Tensor cond, Tensor inputA, Tensor inputB,
                          OpMeta op_meta) {
  return Graph::MakeOp(
    std::make_shared<WhereOpImpl>(true),
    {std::move(cond), std::move(inputA), std::move(inputB)},
    std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hetu
