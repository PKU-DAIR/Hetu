#include "hetu/graph/ops/Attention.h"
#include "hetu/graph/ops/Reshape.h"
#include "hetu/graph/ops/Slice.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

void AttentionOpImpl::DoCompute(Operator& op, 
                                const NDArrayList& inputs, NDArrayList& outputs,
                                RuntimeContext& ctx) const {
  
  double softmax_scale_ = softmax_scale() >= 0 ? softmax_scale() : std::pow(inputs.at(0)->shape(3), -0.5);
  HT_DISPATCH_KERNEL_CUDA_ONLY(op->instantiation_ctx().placement.type(), type(), hetu::impl::FlashAttn,
                               inputs.at(0), inputs.at(1), inputs.at(2), outputs.at(0), outputs.at(1),
                               outputs.at(2), outputs.at(3), outputs.at(4), outputs.at(5),
                               outputs.at(6), outputs.at(7), p_dropout(), softmax_scale_,
                               is_causal(), return_softmax(), op->instantiation_ctx().stream());
}

void AttentionOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                     const OpMeta& op_meta,
                                     const InstantiationContext& inst_ctx) const {
  const DistributedStates& ds_input = inputs.at(0)->get_distributed_states();
  HT_ASSERT(ds_input.is_valid()) 
    << "ParallelAttentionOpImpl: distributed states for input must be valid!";
  HT_ASSERT(ds_input.get_dim(-2) == 1)
    << "Input tensor shouldn't be partial!";
  outputs.at(0)->set_distributed_states(ds_input);    
}

TensorList AttentionOpImpl::DoGradient(Operator& op, const TensorList& grad_outputs) const {
  TensorList empty = {Tensor(), Tensor(), Tensor()};
  return op->requires_grad(0) ? MakeAttentionGradientOp(grad_outputs.at(0), op->input(0), op->input(1), 
                                                        op->input(2), op->output(0), op->output(5),
                                                        op->output(7), p_dropout(), softmax_scale(),
                                                        is_causal(), op->grad_op_meta().set_name(op->grad_name()))
                              : empty;
}

HTShapeList AttentionOpImpl::DoInferShape(Operator& op, 
                                          const HTShapeList& input_shapes, 
                                          RuntimeContext& ctx) const {
  HTShapeList out_shapes = {input_shapes.at(0)};
  const int batch_size = input_shapes.at(0)[0];
  const int seqlen_q = input_shapes.at(0)[1];
  const int num_heads = input_shapes.at(0)[2];
  const int head_size_og = input_shapes.at(0)[3];
  const int seqlen_k = input_shapes.at(1)[1];
  const int num_heads_k = input_shapes.at(1)[2];
  HT_ASSERT(batch_size > 0)
  << "batch size must be postive";
  HT_ASSERT(head_size_og <= 256)
  << "FlashAttention forward only supports head dimension at most 256";
  HT_ASSERT(num_heads % num_heads_k == 0)
  << "Number of heads in key/value must divide number of heads in query";
  const int pad_len = head_size_og % 8 == 0 ? 0 : 8 - head_size_og % 8;
  HTShape padded_shape;
  for (int i = 0; i < 3; ++i) {
    padded_shape = input_shapes.at(i);
    padded_shape[3] += pad_len;
    out_shapes.emplace_back(padded_shape); // q_padded, k_padded, v_padded.
  }
  padded_shape = input_shapes.at(0);
  padded_shape[3] += pad_len;
  out_shapes.emplace_back(padded_shape); // out_padded
  HTShape lse_shape = {batch_size, num_heads, seqlen_q},
          p_shape = {batch_size, num_heads, seqlen_q + pad_len, seqlen_k + pad_len},
          rng_shape = {2};
  out_shapes.emplace_back(lse_shape); // softmax_lse
  out_shapes.emplace_back(p_shape); // p
  out_shapes.emplace_back(rng_shape); // rng_state
  return out_shapes;
}

void AttentionGradientOpImpl::DoCompute(Operator& op, const NDArrayList& inputs,
                                        NDArrayList& outputs, RuntimeContext& ctx) const {
  double softmax_scale_ = softmax_scale() >= 0 ? softmax_scale() : std::pow(inputs.at(1)->shape(3), -0.5);
  HT_DISPATCH_KERNEL_CUDA_ONLY(op->instantiation_ctx().placement.type(), type(),
                               hetu::impl::FlashAttnGradient, inputs.at(0),
                               inputs.at(1), inputs.at(2), inputs.at(3), const_cast<NDArray&>(inputs.at(4)),
                               const_cast<NDArray&>(inputs.at(5)), const_cast<NDArray&>(inputs.at(6)), 
                               outputs.at(0), outputs.at(1), outputs.at(2), p_dropout(), softmax_scale_,
                               is_causal(), op->instantiation_ctx().stream());
}

void AttentionGradientOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                             const OpMeta& op_meta,
                                             const InstantiationContext& inst_ctx) const {
  const DistributedStates& ds_input = inputs.at(0)->get_distributed_states();
  HT_ASSERT(ds_input.is_valid()) 
    << "ParallelAttentionGradientOpImpl: distributed states for input must be valid!";
  HT_ASSERT(ds_input.get_dim(-2) == 1)
    << "Input tensor shouldn't be partial!";
  outputs.at(0)->set_distributed_states(ds_input);    
}

HTShapeList AttentionGradientOpImpl::DoInferShape(Operator& op, 
                                             const HTShapeList& input_shapes, 
                                             RuntimeContext& ctx) const {
  return {input_shapes.at(1), input_shapes.at(2), input_shapes.at(3)};
}

void AttentionVarlenOpImpl::DoCompute(Operator& op, 
                                      const NDArrayList& inputs, NDArrayList& outputs,
                                      RuntimeContext& ctx) const {
  
  double softmax_scale_ = softmax_scale() >= 0 ? softmax_scale() : std::pow(inputs.at(0)->shape(3), -0.5);
  HT_DISPATCH_KERNEL_CUDA_ONLY(op->instantiation_ctx().placement.type(), type(), hetu::impl::FlashAttnVarlen,
                               inputs.at(0), inputs.at(1), inputs.at(2), inputs.at(3), inputs.at(4),
                               outputs.at(0), outputs.at(1),
                               outputs.at(2), outputs.at(3), outputs.at(4), outputs.at(5),
                               outputs.at(6), outputs.at(7), max_seqlen_q(), max_seqlen_k(),
                               p_dropout(), softmax_scale_, zero_tensors(),
                               is_causal(), return_softmax(), op->instantiation_ctx().stream());
}

TensorList AttentionVarlenOpImpl::DoGradient(Operator& op, const TensorList& grad_outputs) const {
  TensorList empty = {Tensor(), Tensor(), Tensor()};
  return op->requires_grad(0) ? MakeAttentionVarlenGradientOp(grad_outputs.at(0), op->input(0), op->input(1), 
                                                              op->input(2), op->input(3), op->input(4), 
                                                              op->output(0), op->output(5), op->output(7), 
                                                              max_seqlen_q(), max_seqlen_k(), p_dropout(), softmax_scale(),
                                                              zero_tensors(), is_causal(), op->grad_op_meta().set_name(op->grad_name()))
                              : empty;
}

HTShapeList AttentionVarlenOpImpl::DoInferShape(Operator& op, 
                                          const HTShapeList& input_shapes, 
                                          RuntimeContext& ctx) const {
  HTShapeList out_shapes = {input_shapes.at(0)};
  const int total_q = input_shapes.at(0)[0];
  int batch_size = 1;
  for (auto& val: input_shapes.at(3))
    batch_size *= val;
  batch_size = batch_size - 1;
  const int num_heads = input_shapes.at(0)[1];
  const int head_size_og = input_shapes.at(0)[2];
  const int total_k = input_shapes.at(1)[0];
  const int num_heads_k = input_shapes.at(1)[1];
  HT_ASSERT(batch_size > 0)
  << "batch size must be postive";
  HT_ASSERT(head_size_og <= 256)
  << "FlashAttentionVarlen forward only supports head dimension at most 256";
  HT_ASSERT(num_heads % num_heads_k == 0)
  << "Number of heads in key/value must divide number of heads in query";
  const int pad_len = head_size_og % 8 == 0 ? 0 : 8 - head_size_og % 8;
  HTShape padded_shape;
  for (int i = 0; i < 3; ++i) {
    padded_shape = input_shapes.at(i);
    padded_shape[2] += pad_len;
    out_shapes.emplace_back(padded_shape); //q_padded, k_padded, v_padded.
  }
  padded_shape = input_shapes.at(0);
  padded_shape[2] += pad_len;
  out_shapes.emplace_back(padded_shape); //out_padded
  HTShape lse_shape = {batch_size, num_heads, _max_seqlen_q},
          p_shape = {batch_size, num_heads, _max_seqlen_q + pad_len, _max_seqlen_k + pad_len},
          rng_shape = {2};
  out_shapes.emplace_back(lse_shape); //softmax_lse
  out_shapes.emplace_back(p_shape); //p
  out_shapes.emplace_back(rng_shape); //rng_state
  return out_shapes;
}

void AttentionVarlenGradientOpImpl::DoCompute(Operator& op,const NDArrayList& inputs,
                                        NDArrayList& outputs, RuntimeContext& ctx) const {
  double softmax_scale_ = softmax_scale() >= 0 ? softmax_scale() : std::pow(inputs.at(1)->shape(3), -0.5);
  HT_DISPATCH_KERNEL_CUDA_ONLY(op->instantiation_ctx().placement.type(), type(),
                               hetu::impl::FlashAttnVarlenGradient, inputs.at(0),
                               inputs.at(1), inputs.at(2), inputs.at(3), inputs.at(4), 
                               inputs.at(5), const_cast<NDArray&>(inputs.at(6)),
                               const_cast<NDArray&>(inputs.at(7)), const_cast<NDArray&>(inputs.at(8)), 
                               outputs.at(0), outputs.at(1), outputs.at(2), max_seqlen_q(), max_seqlen_k(), 
                               p_dropout(), softmax_scale_, zero_tensors(),
                               is_causal(), op->instantiation_ctx().stream());
}

HTShapeList AttentionVarlenGradientOpImpl::DoInferShape(Operator& op, 
                                             const HTShapeList& input_shapes, 
                                             RuntimeContext& ctx) const {
  return {input_shapes.at(1), input_shapes.at(2), input_shapes.at(3)};
}

TensorList MakeAttentionOp(Tensor q, Tensor k, Tensor v, double p_dropout, double softmax_scale, 
                           bool is_causal, bool return_softmax, OpMeta op_meta) {
  TensorList inputs = {std::move(q), std::move(k), std::move(v)};
  return Graph::MakeOp(
        std::make_shared<AttentionOpImpl>(p_dropout, softmax_scale, is_causal, return_softmax),
        std::move(inputs),
        std::move(op_meta))->outputs();
}

TensorList MakeAttentionGradientOp(Tensor grad_out, Tensor q, Tensor k, Tensor v,
                                   Tensor out, Tensor softmax_lse, Tensor rng_state,
                                   double p_dropout, double softmax_scale,
                                   bool is_causal, OpMeta op_meta) {
  return Graph::MakeOp(
        std::make_shared<AttentionGradientOpImpl>(p_dropout, softmax_scale, is_causal),
        {std::move(grad_out), std::move(q), std::move(k), std::move(v),
         std::move(out), std::move(softmax_lse), std::move(rng_state)},
         std::move(op_meta))->outputs();
}

TensorList MakeAttentionVarlenOp(Tensor q, Tensor k, Tensor v, Tensor cu_seqlens_q, Tensor cu_seqlens_k,
                                 int max_seqlen_q, int max_seqlen_k, double p_dropout, double softmax_scale, 
                                 bool zero_tensors, bool is_causal, bool return_softmax, OpMeta op_meta) {
  TensorList inputs = {std::move(q), std::move(k), std::move(v), std::move(cu_seqlens_q), std::move(cu_seqlens_k)};
  return Graph::MakeOp(
        std::make_shared<AttentionVarlenOpImpl>(max_seqlen_q, max_seqlen_k, p_dropout, softmax_scale, 
                                                zero_tensors, is_causal, return_softmax),
        std::move(inputs),
        std::move(op_meta))->outputs();
}

TensorList MakeAttentionVarlenGradientOp(Tensor grad_out, Tensor q, Tensor k, Tensor v,
                                         Tensor cu_seqlens_q, Tensor cu_seqlens_k,
                                         Tensor out, Tensor softmax_lse, Tensor rng_state,
                                         int max_seqlen_q, int max_seqlen_k, 
                                         double p_dropout, double softmax_scale, 
                                         bool zero_tensors, bool is_causal, OpMeta op_meta) {
  return Graph::MakeOp(
        std::make_shared<AttentionVarlenGradientOpImpl>(max_seqlen_q, max_seqlen_k, p_dropout, 
                                                        softmax_scale, zero_tensors, is_causal),
        {std::move(grad_out), std::move(q), std::move(k), std::move(v),
         std::move(cu_seqlens_q), std::move(cu_seqlens_k),
         std::move(out), std::move(softmax_lse), std::move(rng_state)},
         std::move(op_meta))->outputs();
}

TensorList MakeAttentionPackedOp(Tensor qkv, double p_dropout, double softmax_scale, 
                                 bool is_causal, bool return_softmax, OpMeta op_meta) {
  HTShape out_shape = qkv->shape();
  HT_ASSERT(out_shape.size() == 5 && out_shape[2] == 3);
  out_shape[2] = 1;
  HTShape qkvshape = {out_shape[0], out_shape[1], out_shape[3], out_shape[4]};
  Tensor q = MakeArrayReshapeOp(MakeSliceOp(qkv, {0,0,0,0,0}, out_shape), qkvshape);
  Tensor k = MakeArrayReshapeOp(MakeSliceOp(qkv, {0,0,1,0,0}, out_shape), qkvshape);
  Tensor v = MakeArrayReshapeOp(MakeSliceOp(qkv, {0,0,2,0,0}, out_shape), qkvshape);
  TensorList inputs = {std::move(q), std::move(k), std::move(v)};
  return Graph::MakeOp(
         std::make_shared<AttentionOpImpl>(p_dropout, softmax_scale, is_causal, return_softmax),
         std::move(inputs),
         std::move(op_meta))->outputs();
}

TensorList MakeAttentionVarlenPackedOp(Tensor qkv, Tensor cu_seqlens_q, Tensor cu_seqlens_k,
                                       int max_seqlen_q, int max_seqlen_k, double p_dropout, double softmax_scale, 
                                       bool zero_tensors, bool is_causal, bool return_softmax, OpMeta op_meta) {
  HTShape out_shape = qkv->shape();
  HT_ASSERT(out_shape.size() == 4 && out_shape[1] == 3);
  out_shape[1] = 1;
  HTShape qkvshape = {out_shape[0], out_shape[2], out_shape[3]};
  Tensor q = MakeArrayReshapeOp(MakeSliceOp(qkv, {0,0,0,0}, out_shape), qkvshape);
  Tensor k = MakeArrayReshapeOp(MakeSliceOp(qkv, {0,1,0,0}, out_shape), qkvshape);
  Tensor v = MakeArrayReshapeOp(MakeSliceOp(qkv, {0,2,0,0}, out_shape), qkvshape);
  TensorList inputs = {std::move(q), std::move(k), std::move(v), std::move(cu_seqlens_q), std::move(cu_seqlens_k)};
  return Graph::MakeOp(
        std::make_shared<AttentionVarlenOpImpl>(max_seqlen_q, max_seqlen_k, p_dropout, softmax_scale, 
                                                zero_tensors, is_causal, return_softmax),
        std::move(inputs),
        std::move(op_meta))->outputs();
}

} // namespace graph
} // namespace hetu
