#include "hetu/graph/ops/Slice.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

NDArrayList SliceOpImpl::DoCompute(Operator& op,
                                   const NDArrayList& inputs,
                                   RuntimeContext& ctx) const {
  if (ctx.has_runtime_allocation(op->output(0)->id())) {
    auto output = ctx.get_runtime_allocation(op->output(0)->id());
    NDArray::contiguous(NDArray::slice(inputs.at(0), get_begin_pos(), get_output_shape(), op->instantiation_ctx().stream_index), 
                        op->instantiation_ctx().stream_index, output);
    return {output};
  }
  return {NDArray::slice(inputs.at(0), get_begin_pos(), get_output_shape(), op->instantiation_ctx().stream_index)};
}

// caution: if the op is symbolic, then the corresponding gradient op should also be symbolic!
TensorList SliceOpImpl::DoGradient(Operator& op, const TensorList& grad_outputs) const {
  if (symbolic())
    return {op->requires_grad(0) ? MakeSliceGradientOp(grad_outputs.at(0), get_symbolic_begin_pos(),
                                    get_symbolic_output_shape(),
                                    op->grad_op_meta().set_name(op->grad_name()))
                                 : Tensor()};
  else
    return {op->requires_grad(0) ? MakeSliceGradientOp(grad_outputs.at(0), get_begin_pos(),
                                    get_output_shape(),
                                    op->grad_op_meta().set_name(op->grad_name()))
                                 : Tensor()};
}

HTShapeList SliceOpImpl::DoInferShape(Operator& op, 
                                      const HTShapeList& input_shapes, 
                                      RuntimeContext& ctx) const {
  HTShape output_shape = get_output_shape();
  return {output_shape};
}

void SliceOpImpl::DoSaveCtxForBackward(const TensorList& inputs, ContextStore& dst_ctx) const {
  dst_ctx.put("in_meta", inputs.at(0)->meta());
  dst_ctx.put("hetero_dim", inputs.at(0)->cur_ds_union().hetero_dim());
}

// deprecated: only used in gpt inference, before symbolic shape is realized
HTShapeList SliceOpImpl::DoInferDynamicShape(Operator& op, 
                                             const HTShapeList& input_shapes, 
                                             RuntimeContext& ctx) const {
  HTShape output_shape = get_output_shape();
  int64_t ndim = output_shape.size();
  // TODO: a more scalable approach to infer the dynamic shape
  // HTShape begin_pos = get_begin_pos();
  // int64_t padding_axis = get_padding_axis();
  for (int64_t i; i < ndim; i++)
    output_shape[i] = output_shape[i] < input_shapes[0][i] ? output_shape[i] : input_shapes[0][i];
  HT_LOG_TRACE << "SliceOpImpl::DoInferDynamicShape, input_shape: " << input_shapes[0]
   << " output_shape: " << output_shape;
  return {output_shape};
}

void SliceOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                 const OpMeta& op_meta,
                                 const InstantiationContext& inst_ctx) const {
  const DistributedStates& ds_input = inputs.at(0)->get_distributed_states();
  HT_ASSERT(ds_input.is_valid()) 
    << "SliceOpDef: distributed states for input must be valid!";
  HT_ASSERT(ds_input.get_dim(-2) == 1)
    << "Input tensor shouldn't be partial!";
  // HT_ASSERT(ds_input.check_pure_duplicate())
  //   << "Input tensor cannot be splited in any dimension!";
  HTShape ori_shape = inputs.at(0)->shape();
  int ndim = ori_shape.size();
  const HTShape output_shape = get_output_shape();
  const HTShape begin_pos = get_begin_pos();
  for (int i = 0; i < ndim; i++) {
    if (!(begin_pos[i] == 0 && begin_pos[i] + output_shape[i] == ori_shape[i])) {
      HT_ASSERT(ds_input.get_dim(i) == 1)
        << "Slice dimension " << i << " shouldn't be splited!"; 
    }
  }
  outputs.at(0)->set_distributed_states(ds_input); 
  HT_LOG_DEBUG << hetu::impl::comm::GetLocalDevice() << " slice op DoDeduceStates() finished";     
}

void SliceOpImpl::DoDeduceHeterProp(const std::vector<int32_t>& inputs_hetero_dim,
                                    TensorList& outputs, const OpMeta& op_meta,
                                    const InstantiationContext& inst_ctx) const {
  outputs.at(0)->cur_ds_union().set_hetero_dim(inputs_hetero_dim.at(0));
}

void SliceGradientOpImpl::DoCompute(Operator& op,const NDArrayList& inputs,
                                    NDArrayList& outputs, RuntimeContext& ctx) const {
  auto stream_idx = op->instantiation_ctx().stream_index;
  if (outputs.at(0)->numel() == 0) {
    return;
  }
  NDArray::zeros_(outputs.at(0), stream_idx);
  auto slice_grad_input = NDArray::slice(outputs.at(0), get_begin_pos(),
                                         get_output_shape(), stream_idx);
  NDArray::copy(inputs.at(0), stream_idx, slice_grad_input);
}

HTShapeList SliceGradientOpImpl::DoInferShape(Operator& op, 
                                              const HTShapeList& input_shapes, 
                                              RuntimeContext& ctx) const {
  return {ctx.get_or_create(op->id()).get<NDArrayMeta>("in_meta").shape};
}

void SliceGradientOpImpl::DoLoadCtxForBackward(ContextStore& src_ctx, ContextStore& dst_ctx) const {
  dst_ctx.migrate_from<NDArrayMeta>(src_ctx, "in_meta");
  dst_ctx.migrate_from<int32_t>(src_ctx, "hetero_dim");
}

void SliceGradientOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                         const OpMeta& op_meta,
                                         const InstantiationContext& inst_ctx) const {
  outputs.at(0)->set_distributed_states(inputs.at(0)->get_distributed_states());  
}

void SliceGradientOpImpl::DoDeduceHeterProp(const std::vector<int32_t>& inputs_hetero_dim,
                                            TensorList& outputs, const OpMeta& op_meta,
                                            const InstantiationContext& inst_ctx) const {
  outputs.at(0)->cur_ds_union().set_hetero_dim(inst_ctx.get<int32_t>("hetero_dim"));
}

// fixed shape
Tensor MakeSliceOp(Tensor input, const HTShape& begin_pos, const HTShape& output_shape,
                   OpMeta op_meta) {
  return Graph::MakeOp(
    std::make_shared<SliceOpImpl>(begin_pos, output_shape),
    {std::move(input)},
    std::move(op_meta))->output(0);
}

// symbolic shape
Tensor MakeSliceOp(Tensor input, const SyShape& begin_pos, const SyShape& output_shape,
                   OpMeta op_meta) {
  return Graph::MakeOp(
    std::make_shared<SliceOpImpl>(begin_pos, output_shape),
    {std::move(input)},
    std::move(op_meta))->output(0);
}

Tensor MakeSliceGradientOp(Tensor grad_output,
                           const HTShape& begin_pos, const HTShape& output_shape,
                           OpMeta op_meta) {
  return Graph::MakeOp(
    std::make_shared<SliceGradientOpImpl>(begin_pos, output_shape),
    {std::move(grad_output)},
    std::move(op_meta))->output(0);
}

// symbolic shape
Tensor MakeSliceGradientOp(Tensor grad_output,
                           const SyShape& begin_pos, const SyShape& output_shape,
                           OpMeta op_meta) {
  return Graph::MakeOp(
    std::make_shared<SliceGradientOpImpl>(begin_pos, output_shape),
    {std::move(grad_output)},
    std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hetu
