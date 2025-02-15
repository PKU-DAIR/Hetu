#include "hetu/graph/ops/MatDot.h"
#include "hetu/graph/ops/Arithmetics.h"
#include "hetu/graph/ops/Reduce.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

void MatDotOpImpl::DoCompute(Operator& op,
                             const NDArrayList& inputs, NDArrayList& outputs,
                             RuntimeContext& ctx) const {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(),
                                  hetu::impl::MatDot, inputs.at(0),
                                  inputs.at(1), outputs.at(0), op->instantiation_ctx().stream());
}

TensorList MatDotOpImpl::DoGradient(Operator& op, const TensorList& grad_outputs) const {
  auto g_op_meta = op->grad_op_meta();
  auto grad_inputA = op->requires_grad(0) ? MakeMatDotOp(grad_outputs.at(0), op->input(1), 1,
                                           g_op_meta.set_name(op->grad_name(0)))
                                         : Tensor();
  auto grad_inputB =
    MakeReduceOp(MakeMulElewiseOp(op->input(0), grad_outputs.at(0), g_op_meta),
                 "sum", {1}, {false}, g_op_meta.set_name(op->grad_name(1)));
  return {grad_inputA, grad_inputB};
}

HTShapeList MatDotOpImpl::DoInferShape(Operator& op, 
                                       const HTShapeList& input_shapes,
                                       RuntimeContext& ctx) const {
  return {input_shapes.at(0)};
}

void MatDotOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                  const OpMeta& op_meta,
                                  const InstantiationContext& inst_ctx) const {
  const DistributedStates& ds_a = inputs.at(0)->get_distributed_states();
  const DistributedStates& ds_b = inputs.at(1)->get_distributed_states();
  HT_ASSERT(ds_a.is_valid() && ds_b.is_valid() && 
            ds_a.get_device_num() == ds_b.get_device_num()) 
    << "MatDotOpDef: distributed states for a and b must be valid!";
  HT_ASSERT(ds_a.get_dim(-2) == 1 && ds_b.get_dim(-2) == 1) 
    << "Tensor a and b shouldn't be partial";
  // TODO: support tensor parallel
  HT_ASSERT(ds_a.check_pure_duplicate() && ds_b.check_pure_duplicate())
    << "MatDot only require input tensor a and b are both pure duplicate!";
  outputs.at(0)->set_distributed_states(ds_a);
}

Tensor MakeMatDotOp(Tensor a, Tensor b, int64_t axes,
                    OpMeta op_meta) {
  return Graph::MakeOp(
          std::make_shared<MatDotOpImpl>(axes),
          {std::move(a), std::move(b)},
          std::move(op_meta))->output(0);  
}

} // namespace graph
} // namespace hetu
