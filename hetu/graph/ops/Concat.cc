#include "hetu/graph/ops/Concat.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

void ConcatOpImpl::DoCompute(Operator& op,
                             const NDArrayList& inputs, NDArrayList& outputs,
                             RuntimeContext& ctx) const {
  NDArray::cat(inputs, get_axis(), op->instantiation_ctx().stream_index, outputs.at(0));
}

TensorList ConcatOpImpl::DoGradient(Operator &op,
                                    const TensorList& grad_outputs) const {
  bool requires_grad = false;
  for (size_t i = 0; i < op->inputs().size(); i++) {
    if (op->requires_grad(i)) {
      requires_grad = true;
    } else if (!requires_grad) {
      HT_RUNTIME_ERROR << "Now Concat only supports all inputs requires grad case!";
    }
  }
  if (!requires_grad) {
    TensorList ret;
    for (size_t i = 0; i < inputs.size(); i++) {
      ret.push_back(Tensor());
    }
    return ret;
  }
  auto g_op_meta = op->grad_op_meta();
  auto grad_inputs = MakeConcatGradientOp(grad_outputs.at(0), get_axis(), g_op_meta.set_name(op->grad_name(0)));
  // insert input_axis_size_list to inst_ctx
  auto& inst_ctx = grad_inputs.at(0)->producer()->instantiation_ctx();
  HTShape input_axis_size_list;
  for (auto& input : op->inputs()) {
    input_axis_size_list.push_back(input->shape(get_axis()));
  }
  inst_ctx.ctx.put_int64_list("input_axis_size_list", input_axis_size_list);
  return std::move(grad_inputs);
}

HTShapeList ConcatOpImpl::DoInferShape(Operator& op,
                                       const HTShapeList& input_shapes,
                                       RuntimeContext& ctx) const { 
  HTShape shapeA = input_shapes.at(0);
  shapeA[get_axis()] = 0;
  HTShape input_axis_size_list;
  for (size_t i = 0; i < input_shapes.size(); i++) {
    shapeA[get_axis()] += input_shapes.at(i)[get_axis()];
    input_axis_size_list.push_back(input_shapes.at(i)[get_axis()]);
  }
  ctx.get_or_create(op->id()).put_int64_list("input_axis_size_list", input_axis_size_list);
  return {shapeA};
}

void ConcatOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                  const OpMeta& op_meta) const {
  const DistributedStates& ds_0 = inputs.at(0)->get_distributed_states();
  auto device_num = ds_0.get_device_num();
  for (size_t i = 0; i < inputs.size(); i++) {
    const DistributedStates& ds = inputs.at(i)->get_distributed_states();
    HT_ASSERT(ds.is_valid() && ds.get_device_num() == device_num)
      << "ConcatOpDef: distributed states for input " << i << " must be valid!";
    HT_ASSERT(ds.get_dim(-2) == 1)
      << "Tensor " << i << " shouldn't be partial";
    HT_ASSERT(ds.check_equal(ds_0))
      << "Distributed states for tensor " << i << " and tensor 0 must be equal!";
    HT_ASSERT(ds.get_dim(get_axis()) == 1)
      << "Concat was not allowed in splited dimension: " << get_axis();
  }
  outputs.at(0)->set_distributed_states(ds_0);
}

void ConcatGradientOpImpl::DoCompute(Operator& op,
                                     const NDArrayList& inputs,
                                     NDArrayList& outputs, RuntimeContext& ctx) const {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(
    op->instantiation_ctx().placement.type(), type(), hetu::impl::ConcatGradient, 
    outputs.at(0), inputs, get_axis(), op->instantiation_ctx().stream());
}

HTShapeList ConcatGradientOpImpl::DoInferShape(Operator& op,
                                               const HTShapeList& input_shapes,
                                               RuntimeContext& ctx) const {
  HTShape input_axis_size_list = ctx.get_or_create(op->fw_op_id()).get_int64_list("input_axis_size_list");
  HTShapeList ret;
  for (size_t i = 0; i < input_axis_size_list.size(); i++) {
    ret.push_back(input_shapes.at(0));
    ret.back()[get_axis()] = input_axis_size_list.at(i);
  }
  return ret;
}

void ConcatGradientOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                          const OpMeta& op_meta) const {
  for (auto& output : outputs) {
    output->set_distributed_states(inputs.at(0)->get_distributed_states());
  }
}

Tensor MakeConcatOp(TensorList inputs, size_t axis,
                    OpMeta op_meta) {
  return Graph::MakeOp(
          std::make_shared<ConcatOpImpl>(axis),
          std::move(inputs),
          std::move(op_meta))->output(0);
}

TensorList MakeConcatGradientOp(Tensor grad_output, size_t axis,
                                OpMeta op_meta) {
  return Graph::MakeOp(
          std::make_shared<ConcatGradientOpImpl>(axis),
          std::move({std::move(grad_output)}),
          std::move(op_meta));
}

} // namespace graph
} // namespace hetu
