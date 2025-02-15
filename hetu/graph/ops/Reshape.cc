#include "hetu/graph/ops/Reshape.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

NDArrayList ArrayReshapeOpImpl::DoCompute(Operator& op,
                                          const NDArrayList& inputs,
                                          RuntimeContext& ctx) const {
  auto output_shape = ctx.get_runtime_shape(op->output(0)->id());
  NDArray output = NDArray::reshape(inputs.at(0), output_shape, op->instantiation_ctx().stream_index);
  return {output};
}

void ArrayReshapeOpImpl::DoCompute(Operator& op, const NDArrayList& inputs,
                                   NDArrayList& outputs, RuntimeContext& ctx) const {
  auto output_shape = ctx.get_runtime_shape(op->output(0)->id());
  outputs.at(0) = NDArray::reshape(inputs.at(0), output_shape, op->instantiation_ctx().stream_index, outputs.at(0));
}

TensorList ArrayReshapeOpImpl::DoGradient(Operator& op, 
                                          const TensorList& grad_outputs) const {
  if (symbolic()) {
    // 要将输入的tensor设置成symbolic的，之后shape发生改变时，
    // 直接overwrite该tensor中的symbolic shape的value即可，
    // 后续ArrayReshapeGradientOp算子的shape均会发生改变
    if (!op->input(0)->symbolic()) {
      op->input(0)->init_symbolic_shape(); // leaf
    }
    return {op->requires_grad(0) ? MakeArrayReshapeGradientOp(grad_outputs.at(0),
                                                            op->grad_op_meta().set_name(op->grad_name()))
                                 : Tensor()};
  } else {
    return {op->requires_grad(0) ? MakeArrayReshapeGradientOp(grad_outputs.at(0),
                                                              op->grad_op_meta().set_name(op->grad_name()))
                                 : Tensor()};
  }
}

HTShapeList ArrayReshapeOpImpl::DoInferShape(Operator& op, 
                                             const HTShapeList& input_shapes, 
                                             RuntimeContext& ctx) const {
  int64_t input_size = 1;
  HTShape input_shape = input_shapes.at(0);
  int64_t input_len = input_shape.size();
  // check if there exists -1 in output_shape
  int64_t idx = -1;
  size_t cnt = 0;
  int64_t output_size = 1;
  HTShape output_shape = get_output_shape();
  if (op->input(0)->has_distributed_states()) {
    HTShape global_shape(input_shape.size());
    const auto& input_ds = op->input(0)->get_distributed_states();
    for (size_t d = 0; d < input_shape.size(); d++) {
      global_shape[d] = input_shape[d] * input_ds.get_dim(d);
    }
    output_shape = get_local_output_shape(global_shape, input_ds);
  }  
  int64_t output_len = output_shape.size();
  for (size_t i = 0; i < input_len; ++i) {
    if (input_shape[i] == -1) {
      cnt = cnt + 1;
      HT_ASSERT(cnt != 2) << "Input shape has more than one '-1' dims. ";
    }
    input_size *= input_shape[i];
  }
  cnt = 0;
  for (int64_t i = 0; i < output_len; ++i) {
    if (output_shape[i] == -1) {
      idx = i;
      cnt = cnt + 1;
      HT_ASSERT(cnt != 2) << "Output shape has more than one '-1' dims. ";
    }
    output_size *= output_shape[i];
  }
  if (idx == -1) {
    HT_ASSERT(input_size == output_size) << "Invalid output size.";
  } else {
    output_size = output_size * (-1);
    HT_ASSERT(input_size % output_size == 0) << "Invalid output size." << input_shape << "," << output_shape
                                             << input_size << "," << output_size;
    output_shape[idx] = input_size / output_size;
  }
  return {output_shape};
}

void ArrayReshapeOpImpl::DoSaveCtxForBackward(const TensorList& inputs, ContextStore& dst_ctx) const {
  dst_ctx.put("in_meta", inputs.at(0)->meta());
  dst_ctx.put("in_dstate", inputs.at(0)->get_distributed_states());
  dst_ctx.put("hetero_dim", inputs.at(0)->cur_ds_union().hetero_dim());
}

// deprecated: only used in gpt inference, before symbolic shape is realized
HTShapeList ArrayReshapeOpImpl::DoInferDynamicShape(Operator& op, 
                                                    const HTShapeList& input_shapes, 
                                                    RuntimeContext& ctx) const {
  int64_t input_size = 1;
  HTShape input_shape = input_shapes.at(0);
  int64_t input_len = input_shape.size();
  int64_t output_size = 1;
  HTShape output_shape = get_output_shape();
  if (op->input(0)->has_distributed_states()) {
    output_shape = get_local_output_shape(op->input(0)->global_shape(), 
                                          op->input(0)->get_distributed_states());
  }  
  int64_t output_len = output_shape.size();
  for (size_t i = 0; i < input_len; ++i) {
    HT_ASSERT(input_shape[i] != -1) << "The shape of input shouldn't consist of -1 when having paddings.";
    input_size *= input_shape[i];
  }
  for (int64_t i = 0; i < output_len; ++i) {
    HT_ASSERT(input_shape[i] != -1) << "The shape of output shouldn't consist of -1 when having paddings.";
    output_size *= output_shape[i];
  }
  int64_t fixed_output_size = output_size / output_shape[get_padding_axis()];
  HT_ASSERT(input_size % fixed_output_size == 0) << "The dynamic shape: " << input_shape << " can't support reshape.";
  int64_t padding_output_size = input_size / fixed_output_size;
  output_shape[get_padding_axis()] = padding_output_size;
  return {output_shape};
}

void ArrayReshapeOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                        const OpMeta& op_meta,
                                        const InstantiationContext& inst_ctx) const {
  const DistributedStates& ds_input = inputs.at(0)->get_distributed_states();
  HT_ASSERT(ds_input.is_valid()) 
    << "ArrayReshapeOpDef: distributed states for input must be valid!";
  HTShape global_output_shape = get_output_shape(inputs[0]->global_shape());
  DistributedStates ds_output = get_output_ds(inputs[0]->global_shape(), ds_input, global_output_shape);
  outputs.at(0)->set_distributed_states(ds_output);
}

void ArrayReshapeOpImpl::DoDeduceHeterProp(const std::vector<int32_t>& inputs_hetero_dim,
                                           TensorList& outputs, const OpMeta& op_meta,
                                           const InstantiationContext& inst_ctx) const {
  outputs.at(0)->cur_ds_union().set_hetero_dim(inputs_hetero_dim.at(0));
}

NDArrayList ArrayReshapeGradientOpImpl::DoCompute(Operator& op,
                                                  const NDArrayList& inputs,
                                                  RuntimeContext& ctx) const {
  NDArray output = NDArray::reshape(inputs.at(0), inputs.at(1)->shape(), op->instantiation_ctx().stream_index);
  return {output};
}

HTShapeList
ArrayReshapeGradientOpImpl::DoInferShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const {
  return {ctx.get_or_create(op->id()).get<NDArrayMeta>("in_meta").shape};
}

void ArrayReshapeGradientOpImpl::DoLoadCtxForBackward(ContextStore& src_ctx, ContextStore& dst_ctx) const {
  dst_ctx.migrate_from<NDArrayMeta>(src_ctx, "in_meta");
  dst_ctx.migrate_from<DistributedStates>(src_ctx, "in_dstate");
  dst_ctx.migrate_from<int32_t>(src_ctx, "hetero_dim");
}

void ArrayReshapeGradientOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                                const OpMeta& op_meta,
                                                const InstantiationContext& inst_ctx) const {
  outputs.at(0)->set_distributed_states(inst_ctx.get<DistributedStates>("in_dstate"));    
}

void ArrayReshapeGradientOpImpl::DoDeduceHeterProp(const std::vector<int32_t>& inputs_hetero_dim,
                                                   TensorList& outputs, const OpMeta& op_meta,
                                                   const InstantiationContext& inst_ctx) const {
  outputs.at(0)->cur_ds_union().set_hetero_dim(inst_ctx.get<int32_t>("hetero_dim"));
}

// fixed shape
Tensor MakeArrayReshapeOp(Tensor input, const HTShape& output_shape,
                          OpMeta op_meta) {
  return Graph::MakeOp(
      std::make_shared<ArrayReshapeOpImpl>(output_shape),
      {std::move(input)},
      std::move(op_meta))->output(0);
}

// symbolic shape
Tensor MakeArrayReshapeOp(Tensor input, const SyShape& output_shape,
                          OpMeta op_meta) {
  return Graph::MakeOp(
      std::make_shared<ArrayReshapeOpImpl>(output_shape),
      {std::move(input)},
      std::move(op_meta))->output(0);
}

// deprecated: only used in gpt inference, before symbolic shape is realized
Tensor MakeArrayReshapeOp(Tensor input, const HTShape& output_shape,
                          int64_t padding_axis, OpMeta op_meta) {
  padding_axis = NDArrayMeta::ParseAxis(padding_axis, output_shape.size());
  for (auto& x : output_shape) {
    HT_ASSERT(x != -1) << "The shape of output shouldn't consist of -1 when having paddings.";
  }
  return Graph::MakeOp(
      std::make_shared<ArrayReshapeOpImpl>(output_shape, padding_axis),
      {std::move(input)},
      std::move(op_meta))->output(0);
}

Tensor MakeArrayReshapeGradientOp(Tensor grad_output, OpMeta op_meta) {
  return Graph::MakeOp(
      std::make_shared<ArrayReshapeGradientOpImpl>(),
      {std::move(grad_output)},
      std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hetu
