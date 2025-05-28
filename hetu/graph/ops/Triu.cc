#include "hetu/graph/ops/Triu.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

void TriuTrilOpImpl::DoCompute(Operator& op, 
                               const NDArrayList& inputs, NDArrayList& outputs,
                               RuntimeContext& ctx) const {
  // HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(), hetu::impl::TriuTril,
  //                                 inputs.at(0), outputs.at(0), lower(), diagonal(), op->instantiation_ctx().stream());
  NDArray::triu(inputs.at(0), lower(), diagonal(),
                op->instantiation_ctx().stream_index, outputs.at(0));
}

TensorList TriuTrilOpImpl::DoGradient(Operator& op, const TensorList& grad_outputs) const {
  return {op->requires_grad(0) ? MakeTriuTrilOp(grad_outputs.at(0), lower(), diagonal(),
                                op->grad_op_meta().set_name(op->grad_name()))
                              : Tensor()};
}

HTShapeList TriuTrilOpImpl::DoInferShape(Operator& op, 
                                         const HTShapeList& input_shapes, 
                                         RuntimeContext& ctx) const {
  return {input_shapes.at(0)};
}

void TriuTrilOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                    const OpMeta& op_meta,
                                    const InstantiationContext& inst_ctx) const {
  const DistributedStates& ds_input = inputs.at(0)->get_distributed_states();
  HT_ASSERT(ds_input.check_pure_duplicate())
    << "Input tensor cannot be splited in any dimension!";
  outputs.at(0)->set_distributed_states(ds_input);    
}

Tensor MakeTriuTrilOp(Tensor input, bool lower, int64_t diagonal,  
                      OpMeta op_meta) {
  return Graph::MakeOp(
    std::make_shared<TriuTrilOpImpl>(lower, diagonal),
    {std::move(input)},
    std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hetu
