#include "hetu/graph/ops/IndexAdd.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

NDArrayList IndexAddOpImpl::DoCompute(Operator& op,
                                      const NDArrayList& inputs,
                                      RuntimeContext& ctx) const {
  NDArrayList outputs = inputs;
  auto output_shape = inputs.at(0)->shape();
  HTShape _start_and_end_idx = inputs.size() == 3 ?
      HTShape(
        {inputs.at(0)->shape(0) * inputs.at(2)->shape(0) / inputs.at(2)->shape(2),
        inputs.at(0)->shape(0) * (inputs.at(2)->shape(0) + inputs.at(2)->shape(1)) / inputs.at(2)->shape(2)}
      )
      : start_and_end_idx();
  output_shape[0] = _start_and_end_idx[1] - _start_and_end_idx[0];
  NDArray slice_x = NDArray::slice(outputs.at(0), {_start_and_end_idx[0], 0}, output_shape, op->instantiation_ctx().stream_index);
  NDArray::add(slice_x, inputs.at(1), op->instantiation_ctx().stream_index, slice_x);
  return outputs;
}

TensorList IndexAddOpImpl::DoGradient(Operator& op,
                                      const TensorList& grad_outputs) const {
  HTShape input_start_and_end_idx = {};
  auto grad_input = op->requires_grad(0) ? MakeIndexAddGradientOp(grad_outputs.at(0), op->input(0), dim(), false, std::move(input_start_and_end_idx),
                                          op->grad_op_meta().set_name(op->grad_name()))
                                        : Tensor();
  if (op->inputs().size() == 3) {
    if (symbolic()) {
      auto grad_y = op->requires_grad(1) ? MakeIndexAddGradientOp(grad_outputs.at(0), op->input(1), op->input(2), dim(), true,
                                                                  op->grad_op_meta().set_name(op->grad_name()))
                                        : Tensor();
      return {grad_input, grad_y, Tensor()};
    } else {
      auto grad_y = op->requires_grad(1) ? MakeIndexAddGradientOp(grad_outputs.at(0), op->input(1), op->input(2), dim(), true,
                                                                  op->grad_op_meta().set_name(op->grad_name()))
                                        : Tensor();
      return {grad_input, grad_y, Tensor()};
    }
  } else {
    if (symbolic()) {
      auto grad_y = op->requires_grad(1) ? MakeIndexAddGradientOp(grad_outputs.at(0), op->input(1), dim(), true, symbolic_start_and_end_idx(),
                                                                  op->grad_op_meta().set_name(op->grad_name()))
                                        : Tensor();
      return {grad_input, grad_y};
    } else {
      auto grad_y = op->requires_grad(1) ? MakeIndexAddGradientOp(grad_outputs.at(0), op->input(1), dim(), true, start_and_end_idx(),
                                                                  op->grad_op_meta().set_name(op->grad_name()))
                                        : Tensor();
      return {grad_input, grad_y};
    }
  }
}

HTShapeList
IndexAddOpImpl::DoInferShape(Operator& op,
                             const HTShapeList& input_shapes,
                             RuntimeContext& ctx) const {
  HT_ASSERT(input_shapes[0].size() == input_shapes[1].size());
  int64_t len = input_shapes[0].size();
  for (int64_t i = 1; i < len; ++i) {
    HT_ASSERT(input_shapes[0][i] == input_shapes[1][i]);
  }
  return {input_shapes.at(0)};
}

void IndexAddOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                    const OpMeta& op_meta,
                                    const InstantiationContext& inst_ctx) const {
  const DistributedStates& ds_input = inputs.at(0)->get_distributed_states();
  const DistributedStates& ds_id = inputs.at(1)->get_distributed_states();
  HT_ASSERT(ds_input.is_valid() && ds_id.is_valid()) 
    << "GatherOpImpl: distributed states for input must be valid!";
  outputs.at(0)->set_distributed_states(ds_input);    
}

void IndexAddOpImpl::DoDeduceHeterProp(const std::vector<int32_t>& inputs_hetero_dim,
                                       TensorList& outputs, const OpMeta& op_meta,
                                       const InstantiationContext& inst_ctx) const {
  outputs.at(0)->cur_ds_union().set_hetero_dim(inputs_hetero_dim.at(0));
}

void IndexAddGradientOpImpl::DoCompute(Operator& op,
                                       const NDArrayList& inputs,
                                       NDArrayList& outputs,
                                       RuntimeContext& ctx) const {
  if (!require_slice()) {
    NDArray::copy(inputs.at(0), op->instantiation_ctx().stream_index, outputs.at(0));
    return;
  } else {
    auto grad_shape = inputs.at(0)->shape();
    auto _start_and_end_idx = start_and_end_idx();
    grad_shape[0] = _start_and_end_idx[1] - _start_and_end_idx[0];
    NDArray slice_grad = NDArray::slice(inputs.at(0), {_start_and_end_idx[0], 0}, grad_shape, op->instantiation_ctx().stream_index);
    NDArray::copy(slice_grad, op->instantiation_ctx().stream_index, outputs.at(0));
    return;
  }
}

NDArrayList IndexAddGradientOpImpl::DoCompute(Operator& op,
                                              const NDArrayList& inputs,
                                              RuntimeContext& ctx) const {
  if (!require_slice()) {
    NDArrayList outputs = {inputs[0]};
    return outputs;
  } else {
    auto grad_shape = inputs.at(0)->shape();
    auto _start_and_end_idx = inputs.size() == 3 ? HTShape({inputs.at(0)->shape(0) * inputs.at(2)->shape(0) / inputs.at(2)->shape(2),
                                                            inputs.at(0)->shape(0) * (inputs.at(2)->shape(0) + inputs.at(2)->shape(1)) / inputs.at(2)->shape(2)})
                                                 : start_and_end_idx();
    grad_shape[0] = _start_and_end_idx[1] - _start_and_end_idx[0];
    NDArray slice_grad = NDArray::slice(inputs.at(0), {_start_and_end_idx[0], 0}, grad_shape, op->instantiation_ctx().stream_index);
    NDArrayList outputs = {slice_grad};
    return outputs;
  }
}

HTShapeList
IndexAddGradientOpImpl::DoInferShape(Operator& op,
                                     const HTShapeList& input_shapes,
                                     RuntimeContext& ctx) const {
  return {input_shapes.at(1)};
}

void IndexAddGradientOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                            const OpMeta& op_meta,
                                            const InstantiationContext& inst_ctx) const {
  outputs.at(0)->set_distributed_states(inputs.at(1)->get_distributed_states());
}

void IndexAddGradientOpImpl::DoDeduceHeterProp(const std::vector<int32_t>& inputs_hetero_dim,
                                               TensorList& outputs, const OpMeta& op_meta,
                                               const InstantiationContext& inst_ctx) const {
  outputs.at(0)->cur_ds_union().set_hetero_dim(inputs_hetero_dim.at(1));
}

Tensor MakeIndexAddOp(Tensor x, Tensor y, int64_t dim, const SyShape& start_and_end_idx, OpMeta op_meta) {
  return Graph::MakeOp(
          std::make_shared<IndexAddOpImpl>(dim, start_and_end_idx),
          {std::move(x), std::move(y)},
          std::move(op_meta))->output(0);
}

Tensor MakeIndexAddOp(Tensor x, Tensor y, int64_t dim, const HTShape& start_and_end_idx, OpMeta op_meta) {
  return Graph::MakeOp(
          std::make_shared<IndexAddOpImpl>(dim, start_and_end_idx),
          {std::move(x), std::move(y)},
          std::move(op_meta))->output(0);
}

Tensor MakeIndexAddOp(Tensor x, Tensor y, Tensor task_batch_idx, int64_t dim, OpMeta op_meta) {
  return Graph::MakeOp(
          std::make_shared<IndexAddOpImpl>(dim),
          {std::move(x), std::move(y), std::move(task_batch_idx)},
          std::move(op_meta))->output(0);
}

Tensor MakeIndexAddGradientOp(Tensor grad_output, Tensor x, int64_t dim, bool require_slice, const SyShape& start_and_end_idx,
                              OpMeta op_meta) {
  return Graph::MakeOp(
          std::make_shared<IndexAddGradientOpImpl>(dim, require_slice, start_and_end_idx),
          {std::move(grad_output), std::move(x)},
          std::move(op_meta))->output(0);
}

Tensor MakeIndexAddGradientOp(Tensor grad_output, Tensor x, int64_t dim, bool require_slice, const HTShape& start_and_end_idx,
                              OpMeta op_meta) {
  return Graph::MakeOp(
          std::make_shared<IndexAddGradientOpImpl>(dim, require_slice, start_and_end_idx),
          {std::move(grad_output), std::move(x)},
          std::move(op_meta))->output(0);
}

Tensor MakeIndexAddGradientOp(Tensor grad_output, Tensor x, Tensor task_batch_idx, int64_t dim, bool require_slice,
                              OpMeta op_meta) {
  return Graph::MakeOp(
          std::make_shared<IndexAddGradientOpImpl>(dim, require_slice),
          {std::move(grad_output), std::move(x), std::move(task_batch_idx)},
          std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hetu
