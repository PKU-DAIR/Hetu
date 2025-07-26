#include "hetu/graph/ops/binary_cross_entropy.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

TensorList
BinaryCrossEntropyOpImpl::DoGradient(Operator& op,
                                     const TensorList& grad_outputs) const {
  auto grad_probs = op->requires_grad(0)
    ? MakeBinaryCrossEntropyGradientOp(
        op->input(0), op->input(1), grad_outputs.at(0), reduction(),
        op->grad_op_meta().set_name(op->grad_name()))
    : Tensor();
  return {grad_probs, Tensor()};
}

void BinaryCrossEntropyOpImpl::DoCompute(Operator& op,
                                         const NDArrayList& inputs,
                                         NDArrayList& outputs,
                                         RuntimeContext& runtime_ctx) const {
  NDArray unreduced =
    reduction() == kNONE ? outputs.at(0) : NDArray::empty_like(inputs.at(0));
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(),
                                  type(), hetu::impl::BinaryCrossEntropy,
                                  inputs.at(0), inputs.at(1), unreduced,
                                  op->instantiation_ctx().stream());
  if (reduction() != kNONE) {
    NDArray::reduce(unreduced, reduction(), HTAxes(), false,
                    op->instantiation_ctx().stream_index, outputs.at(0));
  }
}

void BinaryCrossEntropyOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                              const OpMeta& op_meta,
                                              const InstantiationContext& inst_ctx) const {
  const DistributedStates& ds_preds = inputs.at(0)->get_distributed_states();
  const DistributedStates& ds_labels = inputs.at(1)->get_distributed_states();
  HT_ASSERT(ds_preds.is_valid() && ds_labels.is_valid()
            && ds_preds.get_device_num() == ds_labels.get_device_num())
    << "BCEOpDef: distributed states for inputs tensor must be valid!";
  HT_ASSERT(ds_preds.get_dim(-2) == 1 && ds_labels.get_dim(-2) == 1)
    << "Inputs tensor shouldn't be partial!";
  HT_ASSERT(ds_preds.check_equal(ds_labels))
    << "Distributed states among preds and labels should be equal!";
  // HT_ASSERT(ds_preds.check_max_dim(1))
  //   << "BCEOp only support data parallel!";
  outputs.at(0)->set_distributed_states(ds_preds);  
}

void BinaryCrossEntropyGradientOpImpl::DoCompute(
  Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
  RuntimeContext& runtime_ctx) const {
  NDArray broadcasted =
    reduction() == kNONE ? inputs.at(2) : NDArray::empty_like(inputs.at(0));
  if (reduction() == kMEAN) {
    HT_DISPATCH_KERNEL_CPU_AND_CUDA(
      op->instantiation_ctx().placement.type(), type(), hetu::impl::BroadcastShapeMul, inputs.at(2),
      1.0f / broadcasted->numel(), broadcasted, HTAxes(), op->instantiation_ctx().stream());
  } else if (reduction() == kSUM) {
    HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(),
                                    hetu::impl::BroadcastShape, inputs.at(2),
                                    broadcasted, HTAxes(), op->instantiation_ctx().stream());
  }
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(
    op->instantiation_ctx().placement.type(), type(), hetu::impl::BinaryCrossEntropyGradient,
    inputs.at(0), inputs.at(1), broadcasted, outputs.at(0), op->instantiation_ctx().stream());
}

Tensor MakeBinaryCrossEntropyOp(Tensor probs, Tensor labels,
                                ReductionType reduction, OpMeta op_meta) {
  return Graph::MakeOp(std::make_shared<BinaryCrossEntropyOpImpl>(reduction),
                       {std::move(probs), std::move(labels)},
                       std::move(op_meta))
    ->output(0);
}

Tensor MakeBinaryCrossEntropyGradientOp(Tensor probs, Tensor labels,
                                        Tensor grad_outputs,
                                        ReductionType reduction,
                                        OpMeta op_meta) {
  return Graph::MakeOp(
           std::make_shared<BinaryCrossEntropyGradientOpImpl>(reduction),
           {std::move(probs), std::move(labels), std::move(grad_outputs)},
           std::move(op_meta))
    ->output(0);
}

} // namespace graph
} // namespace hetu
