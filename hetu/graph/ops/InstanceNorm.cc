#include "hetu/graph/ops/InstanceNorm.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

void InstanceNormOpImpl::DoCompute(Operator& op,
                                   const NDArrayList& inputs,
                                   NDArrayList& outputs, RuntimeContext& ctx) const {
  NDArray::instancenorm(inputs.at(0), get_eps(), op->instantiation_ctx().stream_index,
                        outputs.at(0), outputs.at(1), outputs.at(2));
}

TensorList InstanceNormOpImpl::DoGradient(Operator& op,
                                          const TensorList& grad_outputs) const {
  auto grad_input = op->requires_grad(0) ? MakeInstanceNormGradientOp(grad_outputs.at(0), op->input(0), op->output(1), op->output(2), get_eps(),
                                          op->grad_op_meta().set_name(op->grad_name()))
                                        : Tensor();
  return {grad_input};
}

HTShapeList InstanceNormOpImpl::DoInferShape(Operator&op,
                                             const HTShapeList& input_shapes,
                                             RuntimeContext& ctx) const {
  HTShape local_shape = input_shapes.at(0);
  local_shape[3] = 1;
  local_shape[2] = 1;
  return {input_shapes.at(0), local_shape, local_shape};
}

void InstanceNormOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                        const OpMeta& op_meta,
                                        const InstantiationContext& inst_ctx) const {
  const DistributedStates& ds_input = inputs.at(0)->get_distributed_states();
  HT_ASSERT(ds_input.is_valid()) << "InstanceNormOpDef: input states must be valid!";
  HT_ASSERT(ds_input.get_dim(-2) == 1) << "Input tensor shouldn't be partial!";
  HT_ASSERT(ds_input.check_max_dim(2))
    << "InstanceNormOp only support split dimensions N&C in [N, C, H, W]!";  
  outputs.at(0)->set_distributed_states(ds_input);
}

void InstanceNormGradientOpImpl::DoCompute(Operator& op,
                                           const NDArrayList& inputs,
                                           NDArrayList& outputs,
                                           RuntimeContext& ctx) const {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(
    op->instantiation_ctx().placement.type(), type(), hetu::impl::InstanceNormGradient, inputs.at(0),
    inputs.at(1), outputs.at(0), const_cast<NDArray&>(inputs.at(2)),
    const_cast<NDArray&>(inputs.at(3)), get_eps(), op->instantiation_ctx().stream());
}

HTShapeList
InstanceNormGradientOpImpl::DoInferShape(Operator& op,
                                         const HTShapeList& input_shapes,
                                         RuntimeContext& ctx) const {
  return {input_shapes.at(0)};
}

void InstanceNormGradientOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                                const OpMeta& op_meta,
                                                const InstantiationContext& inst_ctx) const {
  outputs.at(0)->set_distributed_states(inputs.at(1)->get_distributed_states());
}

TensorList MakeInstanceNormOp(Tensor input, double eps,
                              OpMeta op_meta) {
  TensorList inputs = {std::move(input)};
  auto ss = Graph::MakeOp(
          std::make_shared<InstanceNormOpImpl>(eps),
          std::move(inputs),
          std::move(op_meta));   
  return ss->outputs();                             
}

Tensor MakeInstanceNormGradientOp(Tensor output_grad, Tensor input,
                                  Tensor save_mean, Tensor save_var,
                                  double eps,
                                  OpMeta op_meta) {
  return Graph::MakeOp(
          std::make_shared<InstanceNormGradientOpImpl>(eps),
          {std::move(output_grad), std::move(input), std::move(save_mean), std::move(save_var)},
          std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hetu
