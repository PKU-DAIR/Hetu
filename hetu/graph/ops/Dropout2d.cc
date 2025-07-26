#include "hetu/graph/ops/Dropout2d.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"
#include "hetu/impl/random/CPURandomState.h"

namespace hetu {
namespace graph {

void Dropout2dOpImpl::DoCompute(Operator& op, const NDArrayList& inputs,
                                NDArrayList& outputs,
                                RuntimeContext& ctx) const {
  uint64_t seed = hetu::impl::GenNextRandomSeed();
  // record seed for recomputed dropout
  if (op->op_meta().get_recompute(op->graph().COMPUTE_STRATEGY_ID, op->suggested_hetero_id())) {
    ctx.get_or_create(op->id()).put("seed", seed);
  }
  // recomputed dropout
  if (op->op_meta().origin_op_id != -1) {
    seed = ctx.get_or_create(op->op_meta().origin_op_id).get<uint64_t>("seed");
  }
  HT_DISPATCH_KERNEL_CUDA_ONLY(op->instantiation_ctx().placement.type(), type(),
                                hetu::impl::Dropout2d, inputs.at(0), 1 - keep_prob(),
                                seed, outputs.at(0), outputs.at(1), op->instantiation_ctx().stream());
};

NDArrayList Dropout2dOpImpl::DoCompute(Operator& op,
                                       const NDArrayList& inputs,
                                       RuntimeContext& ctx) const {
  NDArrayList outputs;
  if (inplace()) {
    outputs = inputs;
    outputs.push_back(DoAllocOutput(op, inputs, 1, ctx));
  } else {
    outputs = DoAllocOutputs(op, inputs, ctx);
  }
  DoCompute(op, inputs, outputs, ctx);
  return outputs;
}

TensorList Dropout2dOpImpl::DoGradient(Operator& op, const TensorList& grad_outputs) const {
  return {op->requires_grad(0) ? MakeDropout2dGradientOp(grad_outputs.at(0),
                                  op->output(1), keep_prob(), inplace(),
                                  op->grad_op_meta().set_name(op->grad_name()))
                               : Tensor()};
}

HTShapeList Dropout2dOpImpl::DoInferShape(Operator& op,
                                          const HTShapeList& input_shapes,
                                          RuntimeContext& ctx) const {
  return {input_shapes[0], HTShape({input_shapes[0][0], input_shapes[0][1]})};
}

void Dropout2dOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                     const OpMeta& op_meta,
                                     const InstantiationContext& inst_ctx) const {
  const DistributedStates& ds_input = inputs.at(0)->get_distributed_states();
  HT_ASSERT(ds_input.is_valid()) 
    << "Dropout2dOpDef: distributed states for input must be valid!";
  HT_ASSERT(ds_input.get_dim(-2) == 1) 
    << "Tensor input shouldn't be partial";
  HT_ASSERT(ds_input.check_max_dim(2))
    << "droupout2d only support split dimensions N&C in [N, C, H, W] now!";
  outputs.at(0)->set_distributed_states(ds_input);  
}

void Dropout2dGradientOpImpl::DoCompute(Operator& op, const NDArrayList& inputs,
                                        NDArrayList& outputs,
                                        RuntimeContext& ctx) const {
  HT_DISPATCH_KERNEL_CUDA_ONLY(
    op->instantiation_ctx().placement.type(), type(), hetu::impl::Dropout2dGradient,
    inputs.at(0), inputs.at(1), 1 - keep_prob(), outputs.at(0), op->instantiation_ctx().stream());
}

NDArrayList
Dropout2dGradientOpImpl::DoCompute(Operator& op,const NDArrayList& inputs,
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

HTShapeList Dropout2dGradientOpImpl::DoInferShape(Operator& op,
                                                  const HTShapeList& input_shapes,
                                                  RuntimeContext& ctx) const {
  return {input_shapes[0]};
}

void Dropout2dGradientOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                             const OpMeta& op_meta,
                                             const InstantiationContext& inst_ctx) const {
  outputs.at(0)->set_distributed_states(inputs.at(0)->get_distributed_states());    
}

Tensor MakeDropout2dOp(Tensor input, double keep_prob,
                       OpMeta op_meta) {
  return Graph::MakeOp(
          std::make_shared<Dropout2dOpImpl>(keep_prob, false),
          {std::move(input)},
          std::move(op_meta))->output(0);
}

Tensor MakeDropout2dInplaceOp(Tensor input, double keep_prob,
                              OpMeta op_meta) {
  return Graph::MakeOp(
          std::make_shared<Dropout2dOpImpl>(keep_prob, true),
          {std::move(input)},
          std::move(op_meta))->output(0);
}

Tensor MakeDropout2dGradientOp(Tensor grad_output, Tensor mask,
                               double keep_prob, bool fw_inplace,
                               OpMeta op_meta) {
  return Graph::MakeOp(
          std::make_shared<Dropout2dGradientOpImpl>(keep_prob, fw_inplace),
          {std::move(grad_output), std::move(mask)},
          std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hetu
