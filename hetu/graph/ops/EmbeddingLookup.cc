#include "hetu/graph/ops/EmbeddingLookup.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

void EmbeddingLookupOpImpl::DoCompute(Operator& op,
                                      const NDArrayList& inputs,
                                      NDArrayList& outputs,
                                      RuntimeContext& ctx) const {
  // do offset for vocab parallel embedding
  NDArray id_offset = inputs.at(1);
  if (offset(op) != 0) 
    id_offset = NDArray::sub(inputs.at(1), offset(op), op->instantiation_ctx().stream_index);
                      
  NDArray::embedding(inputs.at(0), id_offset, 
                     op->instantiation_ctx().stream_index, outputs.at(0));
}

TensorList EmbeddingLookupOpImpl::DoGradient(Operator& op,
                                             const TensorList& grad_outputs) const {
  auto grad_input = op->requires_grad(0) ? MakeEmbeddingLookupGradientOp(grad_outputs.at(0), op->input(1), op->output(0),
                                           _multi_offset, op->grad_op_meta().set_name(op->grad_name()))
                                        : Tensor();
  return {grad_input, Tensor()};
}

HTShapeList
EmbeddingLookupOpImpl::DoInferShape(Operator& op,
                                    const HTShapeList& input_shapes,
                                    RuntimeContext& ctx) const {
  HTShape output_shape = input_shapes[1];
  output_shape.emplace_back(input_shapes[0][1]);
  return {output_shape};
}

void EmbeddingLookupOpImpl::DoSaveCtxForBackward(const TensorList& inputs, ContextStore& dst_ctx) const {
  dst_ctx.put("in_meta", inputs.at(0)->meta());
  dst_ctx.put("in_tensor", inputs.at(0));
}

void EmbeddingLookupOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                           const OpMeta& op_meta,
                                           const InstantiationContext& inst_ctx) const {
  const DistributedStates& ds_input = inputs.at(0)->get_distributed_states();
  const DistributedStates& ds_id = inputs.at(1)->get_distributed_states();
  HT_ASSERT(ds_input.is_valid() && ds_id.is_valid() && 
            ds_input.get_device_num() == ds_id.get_device_num()) 
    << "EmbeddingLookupOpDef: distributed states for input and id must be valid!";
  HT_ASSERT(ds_input.get_dim(-2) == 1 && ds_id.get_dim(-2) == 1) 
    << "Tensor input and id shouldn't be partial";
  HT_ASSERT(ds_input.get_dim(1) == 1)
    << "Tensor input(embedding table) should not be splited in dimension 1!";
  DistributedStates ds_output(ds_id);
  // embedding table is pure duplicate
  if (ds_input.get_dim(0) == 1) {
    ; // ds_output = ds_id
  }
  else if (ds_input.get_dim(0) > 1) {
    HT_ASSERT(ds_input.get_dim(0) == ds_id.get_dim(-1)
              && ds_input.get_dim(-1) == ds_id.get_dim(0))
      << "Embedding: now only support id split in dimension 0 & table split in dimension 0";
    std::pair<std::vector<int32_t>, int32_t> src2dst({{-1}, -2});
    auto res_states = ds_output.combine_states(src2dst);
    auto res_order = ds_output.combine_order(src2dst);
    auto device_num = ds_output.get_device_num();
    ds_output.set_distributed_states({device_num, res_states, res_order});
  }
  outputs.at(0)->set_distributed_states(ds_output);
}

void EmbeddingLookupOpImpl::DoDeduceHeterProp(const std::vector<int32_t>& inputs_hetero_dim,
                                              TensorList& outputs, const OpMeta& op_meta,
                                              const InstantiationContext& inst_ctx) const {
  HT_ASSERT(inputs_hetero_dim.at(1) >= 0)
    << "Only support hetero dim is a split dim";
  outputs.at(0)->cur_ds_union().set_hetero_dim(inputs_hetero_dim.at(1));
}

void EmbeddingLookupGradientOpImpl::DoCompute(Operator& op,
                                              const NDArrayList& inputs,
                                              NDArrayList& outputs,
                                              RuntimeContext& ctx) const {
  // do offset for vocab parallel emebedding
  NDArray id_offset = inputs.at(1);
  if (offset(op) != 0) 
    id_offset = NDArray::sub(inputs.at(1), offset(op), op->instantiation_ctx().stream_index);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(
    op->instantiation_ctx().placement.type(), type(), hetu::impl::EmbeddingLookupGradient,
    inputs.at(0), id_offset, outputs.at(0), op->instantiation_ctx().stream());
}

HTShapeList
EmbeddingLookupGradientOpImpl::DoInferShape(Operator& op,
                                            const HTShapeList& input_shapes,
                                            RuntimeContext &ctx) const {
  return {ctx.get_or_create(op->id()).get<Tensor>("in_tensor")->temp_shape()};
}

void EmbeddingLookupGradientOpImpl::DoLoadCtxForBackward(ContextStore& src_ctx, ContextStore& dst_ctx) const {
  dst_ctx.migrate_from<NDArrayMeta>(src_ctx, "in_meta");
  dst_ctx.migrate_from<Tensor>(src_ctx, "in_tensor");
}

void EmbeddingLookupGradientOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                                   const OpMeta& op_meta,
                                                   const InstantiationContext& inst_ctx) const {
  const DistributedStates& ds_grad_output = inputs.at(0)->get_distributed_states();
  const DistributedStates& ds_id = inputs.at(1)->get_distributed_states();
  const DistributedStates& ds_tb = inst_ctx.get<Tensor>("in_tensor")->get_distributed_states();

  DistributedStates ds_tb_grad(ds_grad_output);
  if (ds_tb.check_pure_duplicate()) {
    std::pair<std::vector<int32_t>, int32_t> src2dst({{0}, -2});
    auto res_states = ds_tb_grad.combine_states(src2dst);
    auto res_order = ds_tb_grad.combine_order(src2dst);
    auto device_num = ds_tb_grad.get_device_num();
    ds_tb_grad.set_distributed_states({device_num, res_states, res_order});    
  } else {
    std::pair<std::vector<int32_t>, int32_t> src2dst1({{0}, -2});
    std::pair<std::vector<int32_t>, int32_t> src2dst2({{-1}, 0});
    auto res_states = DistributedStates::combine_states(src2dst1, ds_tb_grad.get_states());
    res_states = DistributedStates::combine_states(src2dst2, res_states);
    auto res_order = DistributedStates::combine_order(src2dst1, ds_tb_grad.get_order());
    res_order = DistributedStates::combine_order(src2dst2, res_order);
    auto device_num = ds_tb_grad.get_device_num();
    ds_tb_grad.set_distributed_states({device_num, res_states, res_order});
  }
  outputs.at(0)->set_distributed_states(ds_tb_grad);
}

void EmbeddingLookupGradientOpImpl::DoDeduceHeterProp(const std::vector<int32_t>& inputs_hetero_dim,
                                                      TensorList& outputs, const OpMeta& op_meta,
                                                      const InstantiationContext& inst_ctx) const {
  HT_ASSERT(inputs_hetero_dim.at(0) == 0)
    << "Only support hetero dim is 0";
  outputs.at(0)->cur_ds_union().set_hetero_dim(-2);
}

Tensor MakeEmbeddingLookupOp(Tensor input, Tensor id, std::vector<int64_t> multi_offset, OpMeta op_meta) {
  return Graph::MakeOp(
          std::make_shared<EmbeddingLookupOpImpl>(std::move(multi_offset)),
          {std::move(input), std::move(id)},
          std::move(op_meta))->output(0);
}

Tensor MakeEmbeddingLookupGradientOp(Tensor grad_output, Tensor id, Tensor ori_input,
                                     std::vector<int64_t> multi_offset, OpMeta op_meta) {
  return Graph::MakeOp(
          std::make_shared<EmbeddingLookupGradientOpImpl>(multi_offset),
          {std::move(grad_output), std::move(id), std::move(ori_input)},
          std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hetu
