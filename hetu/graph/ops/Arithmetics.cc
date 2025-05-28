#include "hetu/graph/ops/Arithmetics.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"
#include "hetu/graph/ops/Reduce.h"

namespace hetu {
namespace graph {

std::pair<HTAxes, HTKeepDims> GradInfer(const HTShapeList& input_shapes) {
  HTShape output_shape = input_shapes[1];
  HTShape input_shape = input_shapes[0];
  size_t ndim = output_shape.size();
  HT_ASSERT(input_shape.size() <= ndim);
  size_t diff = ndim - input_shape.size();
  HTAxes add_axes(diff);
  HTKeepDims keep_dims(diff);
  size_t len = diff + input_shape.size();
  HTShape n_input_shape(len);
  for (size_t i = 0; i < diff; ++i) {
    add_axes[i] = i;
    keep_dims[i] = false;
    n_input_shape[i] = 1;
  }
  for (size_t i = diff; i < len; ++i) {
    n_input_shape[i] = input_shape[i - diff];
  }
  for (size_t i = 0; i < ndim; ++i) {
    if (output_shape[i] == -1) {
      output_shape[i] = n_input_shape[i];
    }
    // HT_ASSERT(output_shape[i] > 0);
    HT_ASSERT(n_input_shape[i] == 1 || n_input_shape[i] == output_shape[i]);
    if (i >= diff && input_shape[i] == 1 && output_shape[i] > 1) {
      add_axes.emplace_back(i);
      keep_dims.emplace_back(true);
    }
  }
  return std::make_pair(add_axes, keep_dims);
}

DistributedStates ElewiseDeduceStates(Tensor a, Tensor b) {
  const DistributedStates& ds_a = a->get_distributed_states();
  const DistributedStates& ds_b = b->get_distributed_states();
  HT_ASSERT(ds_a.is_valid() && ds_b.is_valid() && ds_a.get_device_num() == ds_b.get_device_num()) 
    << "AddElewiseOpDef: distributed states for input a and input b must be valid!";
  // allow sum/sub input tensor states to be partial
  HT_ASSERT(ds_a.get_dim(-2) == ds_b.get_dim(-2)) << "Tensors a and b should have the same partial dimension (-2)";
  HTShape shape_a = a->global_shape();
  HTShape shape_b = b->global_shape();
  int size_a = shape_a.size();
  int size_b = shape_b.size();
  int min_size = std::min(size_a, size_b);
  // now only allow dim extention(1) in one of the two input tensor
  int last_ext_distributed = -1; // -1: init, 0: tensor_a, 1: tensor_b
  for (int i = 0; i < min_size; i++) {
    int dim_a = size_a - 1 - i;
    int dim_b = size_b - 1 - i;
    if (shape_a[dim_a] == shape_b[dim_b]) {
      HT_ASSERT(ds_a.get_dim(dim_a) == ds_b.get_dim(dim_b))
        << "Split states in " << dim_a << " for tensor a should be equal to split states in " << dim_b << " for tensor b!";
    } else if (shape_a[dim_a] == 1) {
      HT_ASSERT(last_ext_distributed != 1)
        << "Only allow shape [1] appear in one of the two input tensor!";
      if (last_ext_distributed == -1) {
        last_ext_distributed = 0;
      }
    } else if (shape_b[dim_b] == 1) {
      HT_ASSERT(last_ext_distributed != 0)
        << "Only allow shape [1] appear in one of the two input tensor!";
      if (last_ext_distributed == -1) {
        last_ext_distributed = 1;
      }    
    } else {
      HT_RUNTIME_ERROR << "This case shouldn't be happened!"; 
    }
  }
  if (size_a > size_b) { // [1] is only in tensor b
    HT_ASSERT(last_ext_distributed != 0)
      << "Only allow shape [1] appear in one of the two input tensor!";    
    return ds_a;
  } else if (size_a < size_b) { // [1] is only in tensor a
    HT_ASSERT(last_ext_distributed != 1)
      << "Only allow shape [1] appear in one of the two input tensor!";    
    return ds_b;
  } else {
    if (last_ext_distributed == 0)
      return ds_b;
    else if (last_ext_distributed == 1)
      return ds_a;
    else
      return ds_a;
  }  
}

void AddElewiseOpImpl::DoCompute(Operator& op,
                                 const NDArrayList& inputs, NDArrayList& outputs,
                                 RuntimeContext& ctx) const {
  NDArray::add(inputs.at(0), inputs.at(1), 
               op->instantiation_ctx().stream_index, outputs.at(0));
}

NDArrayList AddElewiseOpImpl::DoCompute(Operator& op,
                                        const NDArrayList& inputs,
                                        RuntimeContext& ctx) const {
  NDArrayList outputs = inplace() && !ctx.has_runtime_allocation(op->output(0)->id()) ? inputs : DoAllocOutputs(op, inputs, ctx);
  DoCompute(op, inputs, outputs, ctx);
  return outputs;
}

TensorList AddElewiseOpImpl::DoGradient(Operator& op,
                                        const TensorList& grad_outputs) const {
  auto g_op_meta = op->grad_op_meta();
  auto grad_a = op->requires_grad(0) ? MakeAddElewiseGradientOp(grad_outputs.at(0), op->input(0),
                                      op->output(0), 0,
                                      g_op_meta.set_name(op->grad_name(0)))
                                    : Tensor();
  auto grad_b = op->requires_grad(1) ? MakeAddElewiseGradientOp(grad_outputs.at(0), op->input(1),
                                      op->output(0), 1,
                                      g_op_meta.set_name(op->grad_name(1)))
                                    : Tensor();
  return {grad_a, grad_b};
}

HTShapeList AddElewiseOpImpl::DoInferShape(Operator& op, const HTShapeList& input_shapes,
                                           RuntimeContext& runtime_ctx) const {
  HTShape output_shape =
    NDArrayMeta::Broadcast(input_shapes.at(0), input_shapes.at(1));
  return {output_shape};
}

void AddElewiseOpImpl::DoSaveCtxForBackward(const TensorList& inputs, ContextStore& dst_ctx) const {
  dst_ctx.put("in_meta_0", inputs.at(0)->meta());
  dst_ctx.put("in_tensor_0", inputs.at(0));
  dst_ctx.put("in_meta_1", inputs.at(1)->meta());
  dst_ctx.put("in_tensor_1", inputs.at(1));
}

void AddElewiseOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                      const OpMeta& op_meta,
                                      const InstantiationContext& inst_ctx) const {
  outputs.at(0)->set_distributed_states(ElewiseDeduceStates(inputs.at(0), inputs.at(1)));
}

void AddByConstOpImpl::DoCompute(Operator& op,
                                 const NDArrayList& inputs, NDArrayList& outputs,
                                 RuntimeContext& ctx) const {
  NDArray::add(inputs.at(0), const_value(), 
               op->instantiation_ctx().stream_index, outputs.at(0));
}

TensorList AddByConstOpImpl::DoGradient(Operator& op,
                                        const TensorList& grad_outputs) const {
  return {op->requires_grad(0) ? grad_outputs.at(0) : Tensor()};
}

HTShapeList AddByConstOpImpl::DoInferShape(Operator& op,
                                           const HTShapeList& input_shapes,
                                           RuntimeContext& ctx) const {
  return {input_shapes.at(0)};
}

void SubElewiseOpImpl::DoCompute(Operator& op,
                                 const NDArrayList& inputs, NDArrayList& outputs,
                                 RuntimeContext& ctx) const {
  NDArray::sub(inputs.at(0), inputs.at(1), 
               op->instantiation_ctx().stream_index, outputs.at(0));
}

TensorList SubElewiseOpImpl::DoGradient(Operator& op,
                                        const TensorList& grad_outputs) const {
  auto g_op_meta = op->grad_op_meta();
  auto grad_a = op->requires_grad(0) ? MakeSubElewiseGradientOp(grad_outputs.at(0), op->input(0),
                                      op->output(0), 0,
                                      g_op_meta.set_name(op->grad_name(0)))
                                    : Tensor();
  auto grad_b = op->requires_grad(1) ? MakeSubElewiseGradientOp(grad_outputs.at(0), op->input(1),
                                      op->output(0), 1,
                                      g_op_meta.set_name(op->grad_name(1)))
                                    : Tensor();
  return {grad_a, grad_b};
}

HTShapeList SubElewiseOpImpl::DoInferShape(Operator& op, const HTShapeList& input_shapes,
                                           RuntimeContext& runtime_ctx) const {
  HTShape output_shape =
    NDArrayMeta::Broadcast(input_shapes.at(0), input_shapes.at(1));
  return {output_shape};
}

void SubElewiseOpImpl::DoSaveCtxForBackward(const TensorList& inputs, ContextStore& dst_ctx) const {
  dst_ctx.put("in_meta_0", inputs.at(0)->meta());
  dst_ctx.put("in_tensor_0", inputs.at(0));
  dst_ctx.put("in_meta_1", inputs.at(1)->meta());
  dst_ctx.put("in_tensor_1", inputs.at(1));
}

void SubElewiseOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                      const OpMeta& op_meta,
                                      const InstantiationContext& inst_ctx) const {
  outputs.at(0)->set_distributed_states(ElewiseDeduceStates(inputs.at(0), inputs.at(1)));
}

void SubByConstOpImpl::DoCompute(Operator& op,
                                 const NDArrayList& inputs, NDArrayList& outputs,
                                 RuntimeContext& ctx) const {
  NDArray::sub(inputs.at(0), const_value(), 
               op->instantiation_ctx().stream_index, outputs.at(0));
}

TensorList SubByConstOpImpl::DoGradient(Operator& op,
                                        const TensorList& grad_outputs) const {
  return {op->requires_grad(0) ? grad_outputs.at(0) : Tensor()};
}

HTShapeList SubByConstOpImpl::DoInferShape(Operator& op,
                                           const HTShapeList& input_shapes,
                                           RuntimeContext& ctx) const {
  return {input_shapes.at(0)};
}

void SubFromConstOpImpl::DoCompute(Operator& op,
                                   const NDArrayList& inputs, NDArrayList& outputs,
                                   RuntimeContext& ctx) const {
  NDArray::sub(const_value(), inputs.at(0),
               op->instantiation_ctx().stream_index, outputs.at(0));
}

TensorList SubFromConstOpImpl::DoGradient(Operator& op,
                                          const TensorList& grad_outputs) const {
  auto grad_input =  op->requires_grad(0) ? MakeNegateOp(grad_outputs.at(0),
                                            op->grad_op_meta().set_name(op->grad_name()))
                                         : Tensor();
  return {grad_input};
}

HTShapeList SubFromConstOpImpl::DoInferShape(Operator& op,
                                             const HTShapeList& input_shapes,
                                             RuntimeContext& ctx) const {
  return {input_shapes.at(0)};
}

NDArrayList NegateOpImpl::DoCompute(Operator& op,
                                    const NDArrayList& inputs,
                                    RuntimeContext& ctx) const {
  NDArrayList outputs = inplace() && !ctx.has_runtime_allocation(op->output(0)->id()) ? inputs : DoAllocOutputs(op, inputs, ctx);
  DoCompute(op, inputs, outputs, ctx);
  return outputs;
}

void NegateOpImpl::DoCompute(Operator& op,
                             const NDArrayList& inputs, NDArrayList& outputs,
                             RuntimeContext& ctx) const {
  NDArray::neg(inputs.at(0), op->instantiation_ctx().stream_index, outputs.at(0));
}

TensorList NegateOpImpl::DoGradient(Operator& op,
                                    const TensorList& grad_outputs) const {
  auto grad_input = op->requires_grad(0) ? MakeNegateOp(grad_outputs.at(0),
                                          op->grad_op_meta().set_name(op->grad_name()))
                                        : Tensor();
  return {grad_input};
}

void MulElewiseOpImpl::DoCompute(Operator& op,
                                 const NDArrayList& inputs, NDArrayList& outputs,
                                 RuntimeContext& ctx) const {
  NDArray::mul(inputs.at(0), inputs.at(1), 
               op->instantiation_ctx().stream_index, outputs.at(0));
}

TensorList MulElewiseOpImpl::DoGradient(Operator& op,
                                        const TensorList& grad_outputs) const {
  auto g_op_meta = op->grad_op_meta();
  HT_ASSERT(!inplace() || !op->requires_grad(1))
    << "This op doesn't support gradient for inplace.";
  auto grad_a = op->requires_grad(0) ? MakeMulElewiseGradientOp(grad_outputs.at(0), op->input(1), op->input(0),
                                      op->output(0), 0,
                                      g_op_meta.set_name(op->grad_name(0)))
                                    : Tensor();
  auto grad_b = op->requires_grad(1) ? MakeMulElewiseGradientOp(grad_outputs.at(0), op->input(0), op->input(1),
                                      op->output(0), 1,
                                      g_op_meta.set_name(op->grad_name(1)))
                                    : Tensor();
  return {grad_a, grad_b};
}

HTShapeList MulElewiseOpImpl::DoInferShape(Operator& op, const HTShapeList& input_shapes,
                                           RuntimeContext& runtime_ctx) const {
  HTShape output_shape =
    NDArrayMeta::Broadcast(input_shapes.at(0), input_shapes.at(1));
  return {output_shape};
}

void MulElewiseOpImpl::DoSaveCtxForBackward(const TensorList& inputs, ContextStore& dst_ctx) const {
  dst_ctx.put("in_meta_0", inputs.at(0)->meta());
  dst_ctx.put("in_tensor_0", inputs.at(0));
  dst_ctx.put("in_meta_1", inputs.at(1)->meta());
  dst_ctx.put("in_tensor_1", inputs.at(1));
}

void MulElewiseOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                      const OpMeta& op_meta,
                                      const InstantiationContext& inst_ctx) const {
  outputs.at(0)->set_distributed_states(ElewiseDeduceStates(inputs.at(0), inputs.at(1)));
}

void MulByConstOpImpl::DoCompute(Operator& op,
                                 const NDArrayList& inputs, NDArrayList& outputs,
                                 RuntimeContext& ctx) const {
  NDArray::mul(inputs.at(0), const_value(), 
               op->instantiation_ctx().stream_index, outputs.at(0));
}

TensorList MulByConstOpImpl::DoGradient(Operator& op,
                                        const TensorList& grad_outputs) const {
  return {op->requires_grad(0) ? MakeMulByConstOp(grad_outputs.at(0), const_value_symbol(),
                                op->grad_op_meta().set_name(op->grad_name()))
                              : Tensor()};
}

HTShapeList MulByConstOpImpl::DoInferShape(Operator& op,
                                           const HTShapeList& input_shapes,
                                           RuntimeContext& ctx) const {
  return {input_shapes.at(0)};
}

void DivElewiseOpImpl::DoCompute(Operator& op,
                                 const NDArrayList& inputs, NDArrayList& outputs,
                                 RuntimeContext& ctx) const {
  NDArray::div(inputs.at(0), inputs.at(1), 
               op->instantiation_ctx().stream_index, outputs.at(0));
}

TensorList DivElewiseOpImpl::DoGradient(Operator& op,
                                        const TensorList& grad_outputs) const {
  auto g_op_meta = op->grad_op_meta();
  auto grad_a = op->requires_grad(0) ? MakeDivElewiseGradientOp(grad_outputs.at(0), op->input(1), op->input(0),
                                      op->output(0), 0,
                                      g_op_meta.set_name(op->grad_name(0)))
                                    : Tensor();
  auto grad_b = op->requires_grad(1) ? MakeDivElewiseGradientOp(grad_outputs.at(0), op->input(0), op->input(1),
                                      op->output(0), 1,
                                      g_op_meta.set_name(op->grad_name(1)))
                                    : Tensor();
  return {grad_a, grad_b};
}

HTShapeList DivElewiseOpImpl::DoInferShape(Operator& op, const HTShapeList& input_shapes,
                                           RuntimeContext& runtime_ctx) const {
  HTShape output_shape =
    NDArrayMeta::Broadcast(input_shapes.at(0), input_shapes.at(1));
  return {output_shape};
}

void DivElewiseOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                      const OpMeta& op_meta,
                                      const InstantiationContext& inst_ctx) const {
  outputs.at(0)->set_distributed_states(ElewiseDeduceStates(inputs.at(0), inputs.at(1)));
}

void DivByConstOpImpl::DoCompute(Operator& op,
                                 const NDArrayList& inputs, NDArrayList& outputs,
                                 RuntimeContext& ctx) const {
  NDArray::div(inputs.at(0), const_value(), 
               op->instantiation_ctx().stream_index, outputs.at(0));
}

TensorList DivByConstOpImpl::DoGradient(Operator& op,
                                        const TensorList& grad_outputs) const {
  return {op->requires_grad(0) ? MakeDivByConstOp(grad_outputs.at(0), const_value_symbol(),
                                op->grad_op_meta().set_name(op->grad_name()))
                              : Tensor()};
}

HTShapeList DivByConstOpImpl::DoInferShape(Operator& op,
                                           const HTShapeList& input_shapes,
                                           RuntimeContext& ctx) const {
  return {input_shapes.at(0)};
}

void DivFromConstOpImpl::DoCompute(Operator& op,
                                   const NDArrayList& inputs, NDArrayList& outputs,
                                   RuntimeContext& ctx) const {
  NDArray::div(const_value(), inputs.at(0),
               op->instantiation_ctx().stream_index, outputs.at(0));
}

TensorList DivFromConstOpImpl::DoGradient(Operator& op,
                                          const TensorList& grad_outputs) const {
  auto g_op_meta = op->grad_op_meta();
  DoubleSymbol grad_const_value_symbol = DoubleSymbol{0} - const_value_symbol();
  auto grad_input = op->requires_grad(0) ? MakeMulElewiseOp(MakeDivByConstOp(MakeMulElewiseOp(
                                           op->output(0), op->output(0), g_op_meta), grad_const_value_symbol,
                                           g_op_meta), grad_outputs.at(0),
                                           g_op_meta.set_name(op->grad_name(1)))
                                         : Tensor();
  return {grad_input};
}

HTShapeList DivFromConstOpImpl::DoInferShape(Operator& op,
                                             const HTShapeList& input_shapes,
                                             RuntimeContext& ctx) const {
  return {input_shapes.at(0)};
}

NDArrayList ReciprocalOpImpl::DoCompute(Operator& op,
                                        const NDArrayList& inputs,
                                        RuntimeContext& ctx) const {
  NDArrayList outputs = inplace() && !ctx.has_runtime_allocation(op->output(0)->id()) ? inputs : DoAllocOutputs(op, inputs, ctx);
  DoCompute(op, inputs, outputs, ctx);
  return outputs;
}

void ReciprocalOpImpl::DoCompute(Operator& op,
                                 const NDArrayList& inputs, NDArrayList& outputs,
                                 RuntimeContext& ctx) const {
  NDArray::reciprocal(inputs.at(0), op->instantiation_ctx().stream_index, outputs.at(0));
}

TensorList ReciprocalOpImpl::DoGradient(Operator& op,
                                        const TensorList& grad_outputs) const {
  auto g_op_meta = op->grad_op_meta();
  // 1 / (x^2) = (1 / x) * (1 / x)
  if (!op->requires_grad(0))
    return {Tensor()};
  auto ret = MakeMulElewiseOp(op->output(0), op->output(0), g_op_meta);
  ret = MakeNegateOp(ret, g_op_meta);
  ret = MakeMulElewiseOp(ret, grad_outputs.at(0), g_op_meta.set_name(op->grad_name()));
  return {ret};
}

void AddElewiseGradientOpImpl::DoCompute(Operator& op,
                                         const NDArrayList& inputs, NDArrayList& outputs,
                                         RuntimeContext& ctx) const {
  if (axes().size() == 0) {
    NDArray::copy(inputs.at(0), op->instantiation_ctx().stream_index, outputs.at(0));
    return;
  }
  NDArray::reduce(inputs.at(0), ReductionType::SUM, axes(), false, op->instantiation_ctx().stream_index,
                  outputs.at(0));
}

HTShapeList AddElewiseGradientOpImpl::DoInferShape(Operator& op,
                                                   const HTShapeList& input_shapes,
                                                   RuntimeContext& ctx) const {
  return {ctx.get_or_create(op->id()).get<Tensor>("in_tensor")->temp_shape()};
}

void AddElewiseGradientOpImpl::DoLoadCtxForBackward(ContextStore& src_ctx, ContextStore& dst_ctx) const {
  auto meta_key = std::string("in_meta_") + std::to_string(index());
  auto dstate_tensor_key = std::string("in_tensor_") + std::to_string(index());
  dst_ctx.migrate_from<NDArrayMeta>(src_ctx, meta_key, "in_meta");
  dst_ctx.migrate_from<Tensor>(src_ctx, dstate_tensor_key, "in_tensor");
}

void AddElewiseGradientOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                              const OpMeta& op_meta,
                                              const InstantiationContext& inst_ctx) const {
  DistributedStates ds_output = inst_ctx.get<Tensor>("in_tensor")->get_distributed_states();
  if (axes().size() > 0) 
    ds_output = ReduceOpImpl::StatesForDistributedReduce(inputs.at(0), axes(), keep_dims());
  outputs.at(0)->set_distributed_states(ds_output);
}

void SubElewiseGradientOpImpl::DoCompute(Operator& op,
                                         const NDArrayList& inputs, NDArrayList& outputs,
                                         RuntimeContext& ctx) const {
  if (axes().size() == 0) {
    if (index() == 0) {
      NDArray::copy(inputs.at(0), op->instantiation_ctx().stream_index, outputs.at(0));
    } else {
      NDArray::neg(inputs.at(0), op->instantiation_ctx().stream_index, outputs.at(0));
    }
  } else {
    if (index() == 0) {
      NDArray::reduce(inputs.at(0), ReductionType::SUM, axes(), false, op->instantiation_ctx().stream_index,
                      outputs.at(0));
    } else {
      NDArray::neg(NDArray::reduce(inputs.at(0), ReductionType::SUM, axes(), false,
                   op->instantiation_ctx().stream_index, outputs.at(0)),
                   op->instantiation_ctx().stream_index, outputs.at(0));
    }
  }
}

HTShapeList SubElewiseGradientOpImpl::DoInferShape(Operator& op,
                                                   const HTShapeList& input_shapes,
                                                   RuntimeContext& ctx) const {
  return {ctx.get_or_create(op->id()).get<Tensor>("in_tensor")->temp_shape()};
}

void SubElewiseGradientOpImpl::DoLoadCtxForBackward(ContextStore& src_ctx, ContextStore& dst_ctx) const {
  auto meta_key = std::string("in_meta_") + std::to_string(index());
  auto dstate_tensor_key = std::string("in_tensor_") + std::to_string(index());
  dst_ctx.migrate_from<NDArrayMeta>(src_ctx, meta_key, "in_meta");
  dst_ctx.migrate_from<Tensor>(src_ctx, dstate_tensor_key, "in_tensor");
}

void SubElewiseGradientOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                              const OpMeta& op_meta,
                                              const InstantiationContext& inst_ctx) const {
  DistributedStates ds_output = inst_ctx.get<Tensor>("in_tensor")->get_distributed_states();
  if (axes().size() > 0) 
    ds_output = ReduceOpImpl::StatesForDistributedReduce(inputs.at(0), axes(), keep_dims());
  outputs.at(0)->set_distributed_states(ds_output);
}

void MulElewiseGradientOpImpl::DoCompute(Operator& op,
                                         const NDArrayList& inputs, NDArrayList& outputs,
                                         RuntimeContext& ctx) const {
  NDArray unreduced = axes().size() == 0 ? outputs.at(0)
                      : NDArray::empty_like(inputs.at(0));
  NDArray::mul(inputs.at(0), inputs.at(1), op->instantiation_ctx().stream_index, unreduced);
  if (axes().size() > 0)
    NDArray::reduce(unreduced, ReductionType::SUM, axes(), false, op->instantiation_ctx().stream_index,
                    outputs.at(0));
}

HTShapeList MulElewiseGradientOpImpl::DoInferShape(Operator& op,
                                                   const HTShapeList& input_shapes,
                                                   RuntimeContext& ctx) const {
  return {ctx.get_or_create(op->id()).get<Tensor>("in_tensor")->temp_shape()};
}

void MulElewiseGradientOpImpl::DoLoadCtxForBackward(ContextStore& src_ctx, ContextStore& dst_ctx) const {
  auto meta_key = std::string("in_meta_") + std::to_string(index());
  auto dstate_tensor_key = std::string("in_tensor_") + std::to_string(index());
  dst_ctx.migrate_from<NDArrayMeta>(src_ctx, meta_key, "in_meta");
  dst_ctx.migrate_from<Tensor>(src_ctx, dstate_tensor_key, "in_tensor");
}

void MulElewiseGradientOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                              const OpMeta& op_meta,
                                              const InstantiationContext& inst_ctx) const {
  DistributedStates ds_output = inst_ctx.get<Tensor>("in_tensor")->get_distributed_states();
  if (axes().size() > 0) 
    ds_output = ReduceOpImpl::StatesForDistributedReduce(inputs.at(0), axes(), keep_dims());
  outputs.at(0)->set_distributed_states(ds_output);
}

void DivElewiseGradientOpImpl::DoCompute(Operator& op,
                                         const NDArrayList& inputs, NDArrayList& outputs,
                                         RuntimeContext& ctx) const {
  NDArray unreduced = axes().size() == 0 ? outputs.at(0)
                      : NDArray::empty_like(inputs.at(0));
  if (index() == 0) {
    NDArray::div(inputs.at(0), inputs.at(1), op->instantiation_ctx().stream_index, unreduced);
  }
  else {
    NDArray::mul(inputs.at(0), inputs.at(3), op->instantiation_ctx().stream_index, unreduced);
    NDArray::div(unreduced, inputs.at(2), op->instantiation_ctx().stream_index, unreduced);
    NDArray::neg(unreduced, op->instantiation_ctx().stream_index, unreduced); 
  }
  if (axes().size() > 0)
    NDArray::reduce(unreduced, ReductionType::SUM, axes(), false, op->instantiation_ctx().stream_index,
                    outputs.at(0));
}

HTShapeList DivElewiseGradientOpImpl::DoInferShape(Operator& op,
                                                   const HTShapeList& input_shapes,
                                                   RuntimeContext& ctx) const {
  return {input_shapes.at(2)};
}

void DivElewiseGradientOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                              const OpMeta& op_meta,
                                              const InstantiationContext& inst_ctx) const {
  DistributedStates ds_output = inputs.at(2)->get_distributed_states();
  if (axes().size() > 0) 
    ds_output = ReduceOpImpl::StatesForDistributedReduce(inputs.at(0), axes(), keep_dims());
  outputs.at(0)->set_distributed_states(ds_output);
}

Tensor MakeAddElewiseOp(Tensor a, Tensor b, OpMeta op_meta) {

  TensorList inputs = {std::move(a), std::move(b)};
  return Graph::MakeOp(
           std::make_shared<AddElewiseOpImpl>(false),
           std::move(inputs),
           std::move(op_meta))->output(0);
}

Tensor MakeSubElewiseOp(Tensor a, Tensor b, OpMeta op_meta) {
  TensorList inputs = {std::move(a), std::move(b)};
  return Graph::MakeOp(
           std::make_shared<SubElewiseOpImpl>(false),
           std::move(inputs),
           std::move(op_meta))->output(0);
}

Tensor MakeMulElewiseOp(Tensor a, Tensor b, OpMeta op_meta) {
  TensorList inputs = {std::move(a), std::move(b)};
  return Graph::MakeOp(
           std::make_shared<MulElewiseOpImpl>(false),
           std::move(inputs),
           std::move(op_meta))->output(0);
}

Tensor MakeDivElewiseOp(Tensor a, Tensor b, OpMeta op_meta) {
  TensorList inputs = {std::move(a), std::move(b)};
  return Graph::MakeOp(
           std::make_shared<DivElewiseOpImpl>(false),
           std::move(inputs),
           std::move(op_meta))->output(0);
}

Tensor MakeAddByConstOp(Tensor input, double value, OpMeta op_meta) {
  return Graph::MakeOp(
           std::make_shared<AddByConstOpImpl>(DoubleSymbol{value}, false),
           {std::move(input)},
           std::move(op_meta))->output(0);
}

Tensor MakeAddByConstOp(Tensor input, DoubleSymbol value, OpMeta op_meta) {
  return Graph::MakeOp(
           std::make_shared<AddByConstOpImpl>(value, false),
           {std::move(input)},
           std::move(op_meta))->output(0);
}

Tensor MakeAddByConstOp(double value, Tensor input, OpMeta op_meta) {
  return Graph::MakeOp(
           std::make_shared<AddByConstOpImpl>(DoubleSymbol{value}, false),
           {std::move(input)},
           std::move(op_meta))->output(0);
}

Tensor MakeAddByConstOp(DoubleSymbol value, Tensor input, OpMeta op_meta) {
  return Graph::MakeOp(
           std::make_shared<AddByConstOpImpl>(value, false),
           {std::move(input)},
           std::move(op_meta))->output(0);
}

Tensor MakeSubByConstOp(Tensor input, double value, OpMeta op_meta) {
  return Graph::MakeOp(
           std::make_shared<SubByConstOpImpl>(DoubleSymbol{value}, false),
           {std::move(input)},
           std::move(op_meta))->output(0);
}

Tensor MakeSubByConstOp(Tensor input, DoubleSymbol value, OpMeta op_meta) {
  return Graph::MakeOp(
           std::make_shared<SubByConstOpImpl>(value, false),
           {std::move(input)},
           std::move(op_meta))->output(0);
}

Tensor MakeSubFromConstOp(double value, Tensor input, OpMeta op_meta) {
  return Graph::MakeOp(
           std::make_shared<SubFromConstOpImpl>(DoubleSymbol{value}, false),
           {std::move(input)},
           std::move(op_meta))->output(0);
}

Tensor MakeSubFromConstOp(DoubleSymbol value, Tensor input, OpMeta op_meta) {
  return Graph::MakeOp(
           std::make_shared<SubFromConstOpImpl>(value, false),
           {std::move(input)},
           std::move(op_meta))->output(0);
}

Tensor MakeMulByConstOp(Tensor input, double value, OpMeta op_meta) {
  return Graph::MakeOp(
           std::make_shared<MulByConstOpImpl>(DoubleSymbol{value}, false),
           {std::move(input)},
           std::move(op_meta))->output(0);
}

Tensor MakeMulByConstOp(Tensor input, DoubleSymbol value, OpMeta op_meta) {
  return Graph::MakeOp(
           std::make_shared<MulByConstOpImpl>(value, false),
           {std::move(input)},
           std::move(op_meta))->output(0);
}

Tensor MakeMulByConstOp(double value, Tensor input, OpMeta op_meta) {
  return Graph::MakeOp(
           std::make_shared<MulByConstOpImpl>(DoubleSymbol{value}, false),
           {std::move(input)},
           std::move(op_meta))->output(0);
}

Tensor MakeMulByConstOp(DoubleSymbol value, Tensor input, OpMeta op_meta) {
  return Graph::MakeOp(
           std::make_shared<MulByConstOpImpl>(value, false),
           {std::move(input)},
           std::move(op_meta))->output(0);
}

Tensor MakeDivByConstOp(Tensor input, double value, OpMeta op_meta) {
  return Graph::MakeOp(
           std::make_shared<DivByConstOpImpl>(DoubleSymbol{value}, false),
           {std::move(input)},
           std::move(op_meta))->output(0);
}

Tensor MakeDivByConstOp(Tensor input, DoubleSymbol value, OpMeta op_meta) {
  return Graph::MakeOp(
           std::make_shared<DivByConstOpImpl>(value, false),
           {std::move(input)},
           std::move(op_meta))->output(0);
}

Tensor MakeDivFromConstOp(double value, Tensor input, OpMeta op_meta) {
  return Graph::MakeOp(
           std::make_shared<DivFromConstOpImpl>(DoubleSymbol{value}, false),
           {std::move(input)},
           std::move(op_meta))->output(0);
}

Tensor MakeDivFromConstOp(DoubleSymbol value, Tensor input, OpMeta op_meta) {
  return Graph::MakeOp(
           std::make_shared<DivFromConstOpImpl>(value, false),
           {std::move(input)},
           std::move(op_meta))->output(0);
}

Tensor MakeAddElewiseInplaceOp(Tensor a, Tensor b, OpMeta op_meta) {
  TensorList inputs = {std::move(a), std::move(b)};
  return Graph::MakeOp(
           std::make_shared<AddElewiseOpImpl>(true),
           std::move(inputs),
           std::move(op_meta))->output(0);
}

Tensor MakeSubElewiseInplaceOp(Tensor a, Tensor b, OpMeta op_meta) {
  TensorList inputs = {std::move(a), std::move(b)};
  return Graph::MakeOp(
           std::make_shared<SubElewiseOpImpl>(true),
           std::move(inputs),
           std::move(op_meta))->output(0);
}

Tensor MakeMulElewiseInplaceOp(Tensor a, Tensor b, OpMeta op_meta) {
  TensorList inputs = {std::move(a), std::move(b)};
  return Graph::MakeOp(
           std::make_shared<MulElewiseOpImpl>(true),
           std::move(inputs),
           std::move(op_meta))->output(0);
}

Tensor MakeDivElewiseInplaceOp(Tensor a, Tensor b, OpMeta op_meta) {
  TensorList inputs = {std::move(a), std::move(b)};
  return Graph::MakeOp(
           std::make_shared<DivElewiseOpImpl>(true),
           std::move(inputs),
           std::move(op_meta))->output(0);
}

Tensor MakeAddByConstInplaceOp(Tensor input, double value, OpMeta op_meta) {
  return Graph::MakeOp(
           std::make_shared<AddByConstOpImpl>(DoubleSymbol{value}, true),
           {std::move(input)},
           std::move(op_meta))->output(0);
}

Tensor MakeAddByConstInplaceOp(Tensor input, DoubleSymbol value, OpMeta op_meta) {
  return Graph::MakeOp(
           std::make_shared<AddByConstOpImpl>(value, true),
           {std::move(input)},
           std::move(op_meta))->output(0);
}

Tensor MakeAddByConstInplaceOp(double value, Tensor input, OpMeta op_meta) {
  return Graph::MakeOp(
           std::make_shared<AddByConstOpImpl>(DoubleSymbol{value}, true),
           {std::move(input)},
           std::move(op_meta))->output(0);
}

Tensor MakeAddByConstInplaceOp(DoubleSymbol value, Tensor input, OpMeta op_meta) {
  return Graph::MakeOp(
           std::make_shared<AddByConstOpImpl>(value, true),
           {std::move(input)},
           std::move(op_meta))->output(0);
}

Tensor MakeSubByConstInplaceOp(Tensor input, double value, OpMeta op_meta) {
  return Graph::MakeOp(
           std::make_shared<SubByConstOpImpl>(DoubleSymbol{value}, true),
           {std::move(input)},
           std::move(op_meta))->output(0);
}

Tensor MakeSubByConstInplaceOp(Tensor input, DoubleSymbol value, OpMeta op_meta) {
  return Graph::MakeOp(
           std::make_shared<SubByConstOpImpl>(value, true),
           {std::move(input)},
           std::move(op_meta))->output(0);
}

Tensor MakeSubFromConstInplaceOp(double value, Tensor input, OpMeta op_meta) {
  return Graph::MakeOp(
           std::make_shared<SubFromConstOpImpl>(DoubleSymbol{value}, true),
           {std::move(input)},
           std::move(op_meta))->output(0);
}

Tensor MakeSubFromConstInplaceOp(DoubleSymbol value, Tensor input, OpMeta op_meta) {
  return Graph::MakeOp(
           std::make_shared<SubFromConstOpImpl>(value, true),
           {std::move(input)},
           std::move(op_meta))->output(0);
}

Tensor MakeMulByConstInplaceOp(Tensor input, double value, OpMeta op_meta) {
  return Graph::MakeOp(
           std::make_shared<MulByConstOpImpl>(DoubleSymbol{value}, true),
           {std::move(input)},
           std::move(op_meta))->output(0);
}

Tensor MakeMulByConstInplaceOp(Tensor input, DoubleSymbol value, OpMeta op_meta) {
  return Graph::MakeOp(
           std::make_shared<MulByConstOpImpl>(value, true),
           {std::move(input)},
           std::move(op_meta))->output(0);
}

Tensor MakeMulByConstInplaceOp(double value, Tensor input, OpMeta op_meta) {
  return Graph::MakeOp(
           std::make_shared<MulByConstOpImpl>(DoubleSymbol{value}, true),
           {std::move(input)},
           std::move(op_meta))->output(0);
}

Tensor MakeMulByConstInplaceOp(DoubleSymbol value, Tensor input, OpMeta op_meta) {
  return Graph::MakeOp(
           std::make_shared<MulByConstOpImpl>(value, true),
           {std::move(input)},
           std::move(op_meta))->output(0);
}

Tensor MakeDivByConstInplaceOp(Tensor input, double value, OpMeta op_meta) {
  return Graph::MakeOp(
           std::make_shared<DivByConstOpImpl>(DoubleSymbol{value}, true),
           {std::move(input)},
           std::move(op_meta))->output(0);
}

Tensor MakeDivByConstInplaceOp(Tensor input, DoubleSymbol value, OpMeta op_meta) {
  return Graph::MakeOp(
           std::make_shared<DivByConstOpImpl>(value, true),
           {std::move(input)},
           std::move(op_meta))->output(0);
}

Tensor MakeDivFromConstInplaceOp(double value, Tensor input, OpMeta op_meta) {
  return Graph::MakeOp(
           std::make_shared<DivFromConstOpImpl>(DoubleSymbol{value}, true),
           {std::move(input)},
           std::move(op_meta))->output(0);
}

Tensor MakeDivFromConstInplaceOp(DoubleSymbol value, Tensor input, OpMeta op_meta) {
  return Graph::MakeOp(
           std::make_shared<DivFromConstOpImpl>(value, true),
           {std::move(input)},
           std::move(op_meta))->output(0);
}

Tensor MakeAddElewiseGradientOp(Tensor a, Tensor input, Tensor output, int index, 
                                OpMeta op_meta) {
  auto grad_pair = GradInfer({input->shape(), output->shape()});
  return Graph::MakeOp(
           std::make_shared<AddElewiseGradientOpImpl>(grad_pair.first, grad_pair.second, index),
           {std::move(a)},
           std::move(op_meta))->output(0);
}

Tensor MakeSubElewiseGradientOp(Tensor a, Tensor input, Tensor output, int index, 
                                OpMeta op_meta) {
  auto grad_pair = GradInfer({input->shape(), output->shape()});
  return Graph::MakeOp(
           std::make_shared<SubElewiseGradientOpImpl>(grad_pair.first, grad_pair.second, index),
           {std::move(a), std::move(output)},
           std::move(op_meta))->output(0);
}

Tensor MakeMulElewiseGradientOp(Tensor a, Tensor b, Tensor input, Tensor output, int index, 
                                OpMeta op_meta) {
  auto grad_pair = GradInfer({input->shape(), output->shape()});
  return Graph::MakeOp(
           std::make_shared<MulElewiseGradientOpImpl>(grad_pair.first, grad_pair.second, index),
           {std::move(a), std::move(b)},
           std::move(op_meta))->output(0);
}

Tensor MakeDivElewiseGradientOp(Tensor a, Tensor b, Tensor input, Tensor output, int index, 
                                OpMeta op_meta) {
  auto grad_pair = GradInfer({input->shape(), output->shape()});
  return Graph::MakeOp(
           std::make_shared<DivElewiseGradientOpImpl>(grad_pair.first, grad_pair.second, index),
           {std::move(a), std::move(b), std::move(input), std::move(output)},
           std::move(op_meta))->output(0);
}

Tensor MakeNegateOp(Tensor input, OpMeta op_meta) {
  return Graph::MakeOp(
           std::make_shared<NegateOpImpl>(false),
           {std::move(input)},
           std::move(op_meta))->output(0);
}

Tensor MakeNegateInplaceOp(Tensor input, OpMeta op_meta) {
  return Graph::MakeOp(
           std::make_shared<NegateOpImpl>(true),
           {std::move(input)},
           std::move(op_meta))->output(0);
}

Tensor MakeReciprocalOp(Tensor input, OpMeta op_meta) {
  return Graph::MakeOp(
           std::make_shared<ReciprocalOpImpl>(false),
           {std::move(input)},
           std::move(op_meta))->output(0);
}

Tensor MakeReciprocalInplaceOp(Tensor input, OpMeta op_meta) {
  return Graph::MakeOp(
           std::make_shared<ReciprocalOpImpl>(true),
           {std::move(input)},
           std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hetu
