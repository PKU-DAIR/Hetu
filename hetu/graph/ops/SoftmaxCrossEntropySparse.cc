#include "hetu/graph/ops/SoftmaxCrossEntropySparse.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"
#include <numeric>

namespace hetu {
namespace graph {

using SCESOpImpl = SoftmaxCrossEntropySparseOpImpl;
using SCESGradOpImpl = SoftmaxCrossEntropySparseGradientOpImpl;

void SCESOpImpl::DoCompute(Operator& op, 
                           const NDArrayList& inputs, NDArrayList& outputs,
                           RuntimeContext& ctx) const {
  NDArray::sceloss(inputs.at(0), inputs.at(1), ignored_index(), reduction(),
                   op->instantiation_ctx().stream_index, outputs.at(0));
}

TensorList SCESOpImpl::DoGradient(Operator& op, const TensorList& grad_outputs) const {
  auto grad_input = op->requires_grad(0) ? MakeSoftmaxCrossEntropySparseGradientOp(
                                          op->input(0), op->input(1), grad_outputs.at(0), ignored_index(), reduction(),
                                          op->grad_op_meta().set_name(op->grad_name()))
                                        : Tensor();
  return {grad_input, Tensor()};
}

HTShapeList SCESOpImpl::DoInferShape(Operator& op, 
                                     const HTShapeList& input_shapes, 
                                     RuntimeContext& ctx) const {
  HT_ASSERT_GE(input_shapes.at(0).size(), 2)
    << "Invalid shape for " << type() << ": " << input_shapes.at(0);
  if (reduction() != kNONE)
    return {{1}};
  else {
    HTShape output_shape = {};
    for (size_t i = 0; i < input_shapes.at(0).size() - 1; ++i) {
      output_shape.emplace_back(input_shapes.at(0)[i]);
    }
    return {output_shape};
  }
}

void SCESOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                               const OpMeta& op_meta,
                               const InstantiationContext& inst_ctx) const {
  const DistributedStates& ds_preds = inputs.at(0)->get_distributed_states();
  const DistributedStates& ds_labels = inputs.at(1)->get_distributed_states();
  int ndim = inputs.at(0)->ndim();
  HT_ASSERT(ds_preds.is_valid() && ds_labels.is_valid())
    << "SoftmaxCrossEntropySparseOp: distributed states for input must be valid!";
  HT_ASSERT(ds_preds.get_dim(-2) == 1 && ds_labels.get_dim(-2) == 1)
    << "Input tensor shouldn't be partial!";
  HT_ASSERT(ds_preds.check_equal(ds_labels))
    << "Distributed states among preds and labels should be equal!";
  HT_ASSERT(ds_preds.check_max_dim(ndim - 1)) // cannot split in last dimension
    << "Input tensor can only support split in dimension < " << ndim - 1;
  outputs.at(0)->set_distributed_states(ds_preds);
}

void SCESOpImpl::DoDeduceHeterProp(const std::vector<int32_t>& inputs_hetero_dim,
                                   TensorList& outputs, const OpMeta& op_meta,
                                   const InstantiationContext& inst_ctx) const {
  outputs.at(0)->cur_ds_union().set_hetero_dim(inputs_hetero_dim.at(0));
}

void SCESGradOpImpl::DoCompute(Operator& op, 
                               const NDArrayList& inputs, NDArrayList& outputs,
                               RuntimeContext& ctx) const {
    HTShape output_shape = HTShape(inputs.at(0)->shape().begin(), inputs.at(0)->shape().end() - 1);
    NDArray broadcasted = reduction() == kNONE
      ? inputs.at(2)
      : NDArray::empty(output_shape, inputs.at(0)->device(),
                       inputs.at(0)->dtype(),
                       op->instantiation_ctx().stream_index);
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
    op->instantiation_ctx().placement.type(), type(), hetu::impl::SoftmaxCrossEntropySparseGradient,
    inputs.at(0), inputs.at(1), broadcasted, outputs.at(0), ignored_index(), op->instantiation_ctx().stream());
}

HTShapeList SCESGradOpImpl::DoInferShape(Operator& op, 
                                         const HTShapeList& input_shapes, 
                                         RuntimeContext& ctx) const {
  HT_ASSERT_GE(input_shapes.at(0).size(), 2)
    << "Invalid shape for " << type() << ": " << input_shapes.at(0);
  return {input_shapes.at(0)};
}

void SCESGradOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                    const OpMeta& op_meta,
                                    const InstantiationContext& inst_ctx) const {
  outputs.at(0)->set_distributed_states(inputs.at(0)->get_distributed_states());
}

void SCESGradOpImpl::DoDeduceHeterProp(const std::vector<int32_t>& inputs_hetero_dim,
                                       TensorList& outputs, const OpMeta& op_meta,
                                       const InstantiationContext& inst_ctx) const {
  outputs.at(0)->cur_ds_union().set_hetero_dim(inputs_hetero_dim.at(0));
}

Tensor MakeSoftmaxCrossEntropySparseOp(Tensor preds, Tensor labels, const int64_t ignored_index, 
                                       ReductionType reduction,
                                       OpMeta op_meta) {
  TensorList inputs = {preds, labels};
  return Graph::MakeOp(
    std::make_shared<SoftmaxCrossEntropySparseOpImpl>(ignored_index, reduction),
    std::move(inputs),
    std::move(op_meta))->output(0);
}

Tensor MakeSoftmaxCrossEntropySparseOp(Tensor preds, Tensor labels, const int64_t ignored_index, 
                                       const std::string& reduction,
                                       OpMeta op_meta) {
  TensorList inputs = {preds, labels};
  return Graph::MakeOp(
    std::make_shared<SoftmaxCrossEntropySparseOpImpl>(ignored_index, Str2ReductionType(reduction)),
    std::move(inputs),
    std::move(op_meta))->output(0);
}

Tensor MakeSoftmaxCrossEntropySparseGradientOp(Tensor preds, Tensor labels, Tensor grad_output,
                                               const int64_t ignored_index, 
                                               ReductionType reduction,
                                               OpMeta op_meta) {
  return Graph::MakeOp(
    std::make_shared<SoftmaxCrossEntropySparseGradientOpImpl>(ignored_index, reduction),
    {std::move(preds), std::move(labels), std::move(grad_output)},
    std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hetu
