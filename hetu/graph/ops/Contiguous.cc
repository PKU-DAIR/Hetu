#include "hetu/graph/ops/Contiguous.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

NDArrayList ContiguousOpImpl::DoCompute(Operator& op,
                                        const NDArrayList& inputs,
                                        RuntimeContext& ctx) const {
  NDArrayList outputs = inputs.at(0)->is_contiguous() && !ctx.has_runtime_allocation(op->output(0)->id()) ? 
                        inputs : DoAllocOutputs(op, inputs, ctx);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(),
                                  hetu::impl::DataTransfer, inputs.at(0),
                                  outputs.at(0), op->instantiation_ctx().stream());
  return outputs;
}

void ContiguousOpImpl::DoCompute(Operator& op, const NDArrayList& inputs,
                                 NDArrayList& outputs, RuntimeContext& ctx) const {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(),
                                  hetu::impl::DataTransfer, inputs.at(0),
                                  outputs.at(0), op->instantiation_ctx().stream());                  
}

TensorList ContiguousOpImpl::DoGradient(Operator& op, const TensorList& grad_outputs) const {
  return {op->requires_grad(0) ? MakeContiguousGradientOp(grad_outputs.at(0), op->input(0)->stride(),
                                 op->grad_op_meta().set_name(op->grad_name()))
                               : Tensor()};
}

HTShapeList ContiguousOpImpl::DoInferShape(Operator& op, 
                                          const HTShapeList& input_shapes, 
                                          RuntimeContext& ctx) const {
  return {input_shapes.at(0)};
}

void ContiguousOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                      const OpMeta& op_meta,
                                      const InstantiationContext& inst_ctx) const {
  auto& ds_input = inputs.at(0)->get_distributed_states();
  HT_ASSERT(ds_input.is_valid()) << op_meta.name << ": input states must be valid! and " 
                                 << "input: " << inputs.at(0) << ", input_ds: " << ds_input.ds_info();
  outputs.at(0)->set_distributed_states(ds_input);
}

Tensor MakeContiguousOp(Tensor input, OpMeta op_meta) {
  auto contig_op = Graph::MakeOp(
    std::make_shared<ContiguousOpImpl>(),
    {std::move(input)},
    std::move(op_meta));
  input->set_contiguous_op_id(contig_op->id());
  return contig_op->output(0);
}

void ContiguousGradientOpImpl::DoCompute(Operator& op, 
                                         const NDArrayList& inputs, NDArrayList& outputs,
                                         RuntimeContext& ctx) const {                             
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(),
                                  hetu::impl::ContiguousGradient, inputs.at(0),
                                  outputs.at(0), op->instantiation_ctx().stream());
}

HTShapeList ContiguousGradientOpImpl::DoInferShape(Operator& op, 
                                                   const HTShapeList& input_shapes, 
                                                   RuntimeContext& ctx) const {
  return {input_shapes.at(0)};
}

void ContiguousGradientOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                              const OpMeta& op_meta,
                                              const InstantiationContext& inst_ctx) const {
  outputs.at(0)->set_distributed_states(inputs.at(0)->get_distributed_states());
}

Tensor MakeContiguousGradientOp(Tensor input, const HTStride& stride, OpMeta op_meta) {
  return Graph::MakeOp(
    std::make_shared<ContiguousGradientOpImpl>(stride),
    {std::move(input)},
    std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hetu
