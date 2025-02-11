#include "hetu/graph/ops/BatchNorm.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

void BatchNormOpImpl::DoCompute(Operator& op, 
                                const NDArrayList& inputs, NDArrayList& outputs,
                                RuntimeContext& ctx) const {
  // TODO: Convert these states to VariableOps
  // HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(),
  //                                 hetu::impl::ArraySet, const_cast<NDArray&>(inputs.at(3)), 0,
  //                                 op->instantiation_ctx().stream());

  // HT_DISPATCH_KERNEL_CPU_AND_CUDA(
  //   op->instantiation_ctx().placement.type(), type(), hetu::impl::ArraySet, 
  //   const_cast<NDArray&>(inputs.at(4)), 1, op->instantiation_ctx().stream());
  NDArray::arrayset(const_cast<NDArray&>(inputs.at(3)), 0,
                    op->instantiation_ctx().stream_index);
  NDArray::arrayset(const_cast<NDArray&>(inputs.at(4)), 1,
                    op->instantiation_ctx().stream_index);
  // HT_DISPATCH_KERNEL_CPU_AND_CUDA(
  //   op->instantiation_ctx().placement.type(), type(), hetu::impl::BatchNorm, inputs.at(0),
  //   inputs.at(1), inputs.at(2), outputs.at(0), get_momentum(), get_eps(),
  //   const_cast<NDArray&>(inputs.at(3)), const_cast<NDArray&>(inputs.at(4)), 
  //   outputs.at(1), outputs.at(2), op->instantiation_ctx().stream());
  NDArray::batchnorm(inputs.at(0), inputs.at(1), inputs.at(2), 
                     const_cast<NDArray&>(inputs.at(3)),
                     const_cast<NDArray&>(inputs.at(4)),
                     get_momentum(), get_eps(),
                     op->instantiation_ctx().stream_index,
                     outputs.at(0), outputs.at(1), outputs.at(2));
}

TensorList BatchNormOpImpl::DoGradient(Operator& op,
                                       const TensorList& grad_outputs) const {
  auto g_op_meta = op->grad_op_meta();
  TensorList empty = {Tensor(), Tensor(), Tensor()};
  auto grad = op->requires_grad(0) ? MakeBatchNormGradientOp(grad_outputs.at(0), op->input(0), op->input(1),
                                     op->output(1), op->output(2), get_eps(), g_op_meta)
                                   : empty;                         
  return {grad.at(0), grad.at(1), grad.at(2), Tensor(), Tensor()};
}

HTShapeList BatchNormOpImpl::DoInferShape(Operator& op,
                                          const HTShapeList& input_shapes,
                                          RuntimeContext& ctx) const {
  HT_ASSERT(input_shapes.at(0).size() == 4);
  return {input_shapes.at(0), {input_shapes.at(0)[1]}, {input_shapes.at(0)[1]}};
}

// 注: input tensor shape=[N, C, H, W], 在N, H, W维上做切分均会影响到batch norm的mean和var, 
// 导致最终结果产生差异(类比于batch和mini-batch做batchnorm的区别)
void BatchNormOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                     const OpMeta& op_meta,
                                     const InstantiationContext& inst_ctx) const {
  const auto& ds_input = inputs.at(0)->get_distributed_states();
  const auto& ds_scale = inputs.at(1)->get_distributed_states();
  const auto& ds_bias = inputs.at(2)->get_distributed_states();
  HT_ASSERT(ds_input.is_valid()) << op_meta.name << ": input states must be valid!";
  HT_ASSERT(ds_input.get_dim(-2) == 1) << "Input tensor shouldn't be partial!";
  HT_ASSERT(ds_input.check_max_dim(2)) // cannot split in H,W dimension
    << "Input tensor can only support split in dimension N, C!";
  HT_ASSERT(ds_input.get_dim(1) == ds_scale.get_dim(0) && ds_input.get_dim(1) == ds_bias.get_dim(0))
    << "Split states for bn_scale and bn_bias should be equal to split states for input dimension C!";  
  outputs.at(0)->set_distributed_states(ds_input);
}

void BatchNormGradientOpImpl::DoCompute(Operator& op,
                                        const NDArrayList& inputs,
                                        NDArrayList& outputs,
                                        RuntimeContext& ctx) const {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(
    op->instantiation_ctx().placement.type(), type(), hetu::impl::BatchNormGradient, inputs.at(0),
    inputs.at(1), inputs.at(2), outputs.at(0), outputs.at(1),
    outputs.at(2), get_eps(), const_cast<NDArray&>(inputs.at(3)),
    const_cast<NDArray&>(inputs.at(4)), op->instantiation_ctx().stream());
}

HTShapeList
BatchNormGradientOpImpl::DoInferShape(Operator& op,
                                      const HTShapeList& input_shapes,
                                      RuntimeContext& ctx) const {
  int64_t channels = input_shapes.at(0)[1];
  return {input_shapes.at(1), {channels}, {channels}};
}

void BatchNormGradientOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                             const OpMeta& op_meta,
                                             const InstantiationContext& inst_ctx) const {
  outputs.at(0)->set_distributed_states(inputs.at(1)->get_distributed_states());
  outputs.at(1)->set_distributed_states(inputs.at(2)->get_distributed_states());
  outputs.at(2)->set_distributed_states(inputs.at(2)->get_distributed_states());  
}

TensorList MakeBatchNormOp(Tensor input, Tensor bn_scale, Tensor bn_bias,
                       Tensor running_mean, Tensor running_var,
                       double momentum, double eps,
                       OpMeta op_meta) {
  return Graph::MakeOp(
        std::make_shared<BatchNormOpImpl>(momentum, eps),
        {std::move(input), std::move(bn_scale), std::move(bn_bias),
         std::move(running_mean), std::move(running_var)},
        std::move(op_meta))->outputs();
}

TensorList MakeBatchNormGradientOp(Tensor output_grad, Tensor input, Tensor bn_scale,
                               Tensor save_mean, Tensor save_var, double eps,
                               OpMeta op_meta) {
  return Graph::MakeOp(
                 std::make_shared<BatchNormGradientOpImpl>(eps),
                 {std::move(output_grad), std::move(input),
                 std::move(bn_scale), std::move(save_mean),
                 std::move(save_var)},
                 std::move(op_meta))->outputs();
}

} // namespace graph
} // namespace hetu
