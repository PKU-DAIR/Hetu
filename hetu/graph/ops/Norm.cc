#include "hetu/graph/ops/Norm.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

void NormOpImpl::DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                          RuntimeContext& ctx) const {
  // HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(), hetu::impl::Norm,
  //                              inputs.at(0), outputs.at(0), dim(), getp(), op->instantiation_ctx().stream());
  NDArray::norm(inputs.at(0), getp(), dim(), keepdim(),
                op->instantiation_ctx().stream_index, outputs.at(0));
}

TensorList NormOpImpl::DoGradient(Operator& op, const TensorList& grad_outputs) const {
  return {op->requires_grad(0) ? MakeNormGradientOp(op->input(0), op->output(0), grad_outputs.at(0), getp(), dim(),
                                op->grad_op_meta().set_name(op->grad_name()))
                              : Tensor()};
}

HTShapeList NormOpImpl::DoInferShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const {
  HTShape outshape = input_shapes.at(0);
  int64_t axi = dim() >= 0 ? dim(): dim() + outshape.size();
  if (keepdim()) 
    outshape[axi] = 1;
  else 
    outshape.erase(outshape.begin() + axi);
  return {outshape};
}

void NormOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                const OpMeta& op_meta,
                                const InstantiationContext& inst_ctx) const {
  const DistributedStates& ds = inputs.at(0)->get_distributed_states();
  HT_ASSERT(ds.is_valid()) 
    << "NormOpImpl: distributed states for input tensor must be valid!";
  HT_ASSERT(ds.get_dim(-2) == 1)
    << "Input tensor shouldn't be partial!";
  HTShape outshape = inputs.at(0)->shape();    
  int64_t axi = dim() >= 0 ? dim(): dim() + outshape.size();
  HT_ASSERT(ds.get_dim(axi) == 1)
    << "The norm dim " << axi << " shouldn't be split!";
  if (keepdim()) {
    outputs.at(0)->set_distributed_states(ds);
  } else {
    std::unordered_map<int32_t, int32_t> new_states;
    std::vector<int32_t> new_order;
    for (auto& state : ds.get_states()) {
      if (state.first < axi) {
        new_states[state.first] = state.second;
      } else {
        new_states[state.first - 1] = state.second;
      }
    }
    for (auto& o : ds.get_order()) {
      if (o < axi) {
        new_order.push_back(o);
      } else {
        new_order.push_back(o - 1);
      }
    }
    outputs.at(0)->set_distributed_states({ds.get_device_num(), new_states, new_order});
  }
}

void NormGradientOpImpl::DoCompute(Operator& op,
                                   const NDArrayList& inputs, NDArrayList& outputs,
                                   RuntimeContext& ctx) const {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(),
                               hetu::impl::NormGradient, inputs.at(0),
                               inputs.at(1), inputs.at(2), outputs.at(0), dim(), getp(), 
                               op->instantiation_ctx().stream());
}

HTShapeList NormGradientOpImpl::DoInferShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const {
  return {input_shapes.at(0)};
}

void NormGradientOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                        const OpMeta& op_meta,
                                        const InstantiationContext& inst_ctx) const {
  outputs.at(0)->set_distributed_states(inputs.at(0)->get_distributed_states());
}

Tensor MakeNormOp(Tensor input, int64_t p, int64_t dim, 
                  bool keepdim, OpMeta op_meta) {
  TensorList inputs = {std::move(input)};
  return Graph::MakeOp(
        std::make_shared<NormOpImpl>(p, dim, keepdim),
        std::move(inputs),
        std::move(op_meta))->output(0);
}

Tensor MakeNormGradientOp(Tensor input, Tensor output, Tensor grad_output, int64_t p, 
                          int64_t dim, OpMeta op_meta) {
  return Graph::MakeOp(
        std::make_shared<NormGradientOpImpl>(p, dim),
        {std::move(input), std::move(output), std::move(grad_output)},
        std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hetu
