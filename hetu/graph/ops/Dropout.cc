#include "hetu/graph/ops/Dropout.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"
#include "hetu/impl/random/CPURandomState.h"

namespace hetu {
namespace graph {

void DropoutOpImpl::DoCompute(Operator& op, const NDArrayList& inputs,
                              NDArrayList& outputs,
                              RuntimeContext& ctx) const {
  uint64_t seed = hetu::impl::GenNextRandomSeed();
  // record seed for recomputed dropout in original op
  if (op->op_meta().get_recompute(op->graph().COMPUTE_STRATEGY_ID, op->suggested_hetero_id())) {
    ctx.get_or_create(op->id()).put("seed", seed);
  }
  // get seed for recomputed dropout in recompute op
  if (op->op_meta().origin_op_id != -1) {
    seed = ctx.get_or_create(op->op_meta().origin_op_id).get<uint64_t>("seed");
  }
  HT_DISPATCH_KERNEL_CUDA_ONLY(op->instantiation_ctx().placement.type(), type(),
                               hetu::impl::Dropout, inputs.at(0), 1 - keep_prob(),
                               seed, outputs.at(0), outputs.at(1), op->instantiation_ctx().stream());
};

NDArrayList DropoutOpImpl::DoCompute(Operator& op,
                                     const NDArrayList& inputs,
                                     RuntimeContext& ctx) const {
  NDArrayList outputs;
  if (inplace() && !ctx.has_runtime_allocation(op->output(0)->id())) {
    outputs = inputs;
    outputs.push_back(DoAllocOutput(op, inputs, 1, ctx));
  } else {
    outputs = DoAllocOutputs(op, inputs, ctx);
  }
  DoCompute(op, inputs, outputs, ctx);
  return outputs;
}

TensorList DropoutOpImpl::DoGradient(Operator& op, const TensorList& grad_outputs) const {
  return {op->requires_grad(0) ? MakeDropoutGradientOp(grad_outputs.at(0),
                                op->output(1), keep_prob(), inplace(),
                                op->grad_op_meta().set_name(op->grad_name()))
                               : Tensor()};
}

HTShapeList DropoutOpImpl::DoInferShape(Operator& op,
                                        const HTShapeList& input_shapes,
                                        RuntimeContext& ctx) const {
  return {input_shapes[0], input_shapes[0]};
}

void DropoutOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                   const OpMeta& op_meta,
                                   const InstantiationContext& inst_ctx) const {
  const DistributedStates& ds_input = inputs.at(0)->get_distributed_states();
  HT_ASSERT(ds_input.is_valid()) 
    << "DropoutOpDef: distributed states for input must be valid!";
  outputs.at(0)->set_distributed_states(ds_input);
  outputs.at(1)->set_distributed_states(ds_input);
}

void DropoutGradientOpImpl::DoCompute(Operator& op, const NDArrayList& inputs,
                                      NDArrayList& outputs,
                                      RuntimeContext& ctx) const {
  HT_DISPATCH_KERNEL_CUDA_ONLY(
    op->instantiation_ctx().placement.type(), type(), hetu::impl::DropoutGradient, inputs.at(0),
    inputs.at(1), 1 - keep_prob(), outputs.at(0), op->instantiation_ctx().stream());
};

NDArrayList DropoutGradientOpImpl::DoCompute(Operator& op,const NDArrayList& inputs,
                                            RuntimeContext& ctx) const {
  NDArrayList outputs;
  if (fw_inplace()) {
    outputs.push_back(inputs[0]);
  } else {
    outputs.push_back(DoAllocOutput(op, inputs, 0, ctx));
  }
  DoCompute(op, inputs, outputs, ctx);
  return outputs;
}

HTShapeList DropoutGradientOpImpl::DoInferShape(Operator& op,
                                                const HTShapeList& input_shapes,
                                                RuntimeContext& ctx) const {
  return {input_shapes[0]};
}

void DropoutGradientOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                           const OpMeta& op_meta,
                                           const InstantiationContext& inst_ctx) const {
  outputs.at(0)->set_distributed_states(inputs.at(0)->get_distributed_states());
}

Tensor MakeDropoutOp(Tensor input, double keep_prob,
                     OpMeta op_meta) {
  return Graph::MakeOp(
          std::make_shared<DropoutOpImpl>(keep_prob, false),
          {std::move(input)},
          std::move(op_meta))->output(0);
}

Tensor MakeDropoutInplaceOp(Tensor input, double keep_prob,
                            OpMeta op_meta) {
  return Graph::MakeOp(
          std::make_shared<DropoutOpImpl>(keep_prob, true),
          {std::move(input)},
          std::move(op_meta))->output(0);
}

Tensor MakeDropoutGradientOp(Tensor grad_output, Tensor mask,
                             double keep_prob, bool fw_inplace,
                             OpMeta op_meta) {
  return Graph::MakeOp(
          std::make_shared<DropoutGradientOpImpl>(keep_prob, fw_inplace),
          {std::move(grad_output), std::move(mask)},
          std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hetu
