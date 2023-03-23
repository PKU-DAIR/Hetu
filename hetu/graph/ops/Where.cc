#include "hetu/graph/ops/Where.h"
#include "hetu/graph/ops/zeros_like.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

void WhereOpImpl::DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                           RuntimeContext& ctx) const {
  // HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(), hetu::impl::Where,
  //                                 inputs.at(0), inputs.at(1), inputs.at(2),
  //                                 outputs.at(0), op->instantiation_ctx().stream());
  NDArray::where(inputs.at(0), inputs.at(1), inputs.at(2),
                 op->instantiation_ctx().stream_index, outputs.at(0));
}

TensorList WhereOpImpl::DoGradient(Operator& op, const TensorList& grad_outputs) const {
  auto g_op_meta = op->grad_op_meta();
  auto zero_grad = MakeZerosLikeOp(op->input(1), g_op_meta);
  auto grad_inputA = op->require_grad(1) ? MakeWhereOp(op->input(0), grad_outputs.at(0), zero_grad,
                                           g_op_meta.set_name(op->grad_name(1)))
                                         : Tensor();
  auto grad_inputB = op->require_grad(2) ? MakeWhereOp(op->input(0), zero_grad, grad_outputs.at(0),
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

Tensor MakeWhereOp(Tensor cond, Tensor inputA, Tensor inputB,
                   const OpMeta& op_meta) {
  return Graph::MakeOp(
    std::make_shared<WhereOpImpl>(),
    {std::move(cond), std::move(inputA), std::move(inputB)},
    std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hetu
