#include "hetu/graph/ops/Diagonal.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

NDArrayList DiagonalOpImpl::DoCompute(Operator& op,
                                      const NDArrayList& inputs,
                                      RuntimeContext& ctx) const {
  NDArray output = NDArray::diagonal(inputs.at(0), dim1(), dim2(), offset(),
                                     op->instantiation_ctx().stream_index);
  return {output};
}

TensorList DiagonalOpImpl::DoGradient(Operator& op, const TensorList& grad_outputs) const {
  return {op->requires_grad(0) ? MakeDiagonalGradientOp(grad_outputs.at(0), offset(), dim1(), dim2(),
                                op->grad_op_meta().set_name(op->grad_name()))
                              : Tensor()};
}

HTShapeList DiagonalOpImpl::DoInferShape(Operator& op, 
                                         const HTShapeList& input_shapes, 
                                         RuntimeContext& ctx) const {
  HTShape ori_shape = input_shapes.at(0);
  int64_t ndim = ori_shape.size();
  int64_t dim1_ = NDArrayMeta::ParseAxis(dim1(), ndim);
  int64_t dim2_ = NDArrayMeta::ParseAxis(dim2(), ndim);
  HTShape res_shape(ori_shape.begin(), ori_shape.end());
  res_shape.erase(res_shape.begin() + std::max(dim1_, dim2_));
  res_shape.erase(res_shape.begin() + std::min(dim1_, dim2_));
  if (offset() >= 0) {
    res_shape.emplace_back(
        std::min(ori_shape[dim1_], ori_shape[dim2_] - offset()));
  } else {
    res_shape.emplace_back(
        std::min(ori_shape[dim1_] + offset(), ori_shape[dim2_]));
  }
  return {res_shape};
}

void DiagonalOpImpl::DoSaveCtxForBackward(const TensorList& inputs, ContextStore& dst_ctx) const {
  dst_ctx.put("in_meta", inputs.at(0)->meta());
  dst_ctx.put("in_dstate", inputs.at(0)->get_distributed_states());
}

void DiagonalOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                    const OpMeta& op_meta,
                                    const InstantiationContext& inst_ctx) const {
  const DistributedStates& ds_input = inputs.at(0)->get_distributed_states();
  HT_ASSERT(ds_input.is_valid()) 
    << "DiagonalOpImpl: distributed states for input must be valid!";
  HT_ASSERT(ds_input.get_dim(-2) == 1)
    << "Input tensor shouldn't be partial!";
  HT_ASSERT(ds_input.check_pure_duplicate())
    << "Input tensor cannot be splited in any dimension!";
  outputs.at(0)->set_distributed_states(ds_input);
}

void DiagonalGradientOpImpl::DoCompute(Operator& op,
                                       const NDArrayList& inputs, NDArrayList& outputs,
                                       RuntimeContext& ctx) const {
  auto stream_idx = op->instantiation_ctx().stream_index;
  NDArray::zeros_(outputs.at(0), stream_idx);
  auto diag = NDArray::diagonal(outputs.at(0), dim1(), dim2(), offset(), stream_idx);
  NDArray::copy(inputs.at(0), stream_idx, diag);
}

HTShapeList DiagonalGradientOpImpl::DoInferShape(Operator& op,
                                                 const HTShapeList& input_shapes,
                                                 RuntimeContext& ctx) const {
  return {ctx.get_or_create(op->id()).get<NDArrayMeta>("in_meta").shape};
}

void DiagonalGradientOpImpl::DoLoadCtxForBackward(ContextStore& src_ctx, ContextStore& dst_ctx) const {
  dst_ctx.migrate_from<NDArrayMeta>(src_ctx, "in_meta");
  dst_ctx.migrate_from<DistributedStates>(src_ctx, "in_dstate");
}

void DiagonalGradientOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs,
                                            const OpMeta& op_meta,
                                            const InstantiationContext& inst_ctx) const {
  const DistributedStates& ds_input = inst_ctx.get<DistributedStates>("in_dstate");
  outputs.at(0)->set_distributed_states(ds_input);
}

Tensor MakeDiagonalOp(Tensor input, int64_t offset, int64_t dim1, int64_t dim2, OpMeta op_meta) {
  return Graph::MakeOp(
    std::make_shared<DiagonalOpImpl>(offset, dim1, dim2),
    {std::move(input)},
    std::move(op_meta))->output(0);
}

Tensor MakeDiagonalGradientOp(Tensor grad_output, int64_t offset,
                              int64_t dim1, int64_t dim2, OpMeta op_meta) {
  return Graph::MakeOp(
    std::make_shared<DiagonalGradientOpImpl>(offset, dim1, dim2),
    {std::move(grad_output)},
    std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hetu
