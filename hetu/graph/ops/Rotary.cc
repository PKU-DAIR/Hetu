#include "hetu/graph/ops/Rotary.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"
#include <numeric>

namespace hetu {
namespace graph {

NDArrayList RotaryOpImpl::DoCompute(Operator& op,
                                    const NDArrayList& inputs,
                                    RuntimeContext& ctx) const {
  NDArrayList outputs = inplace() ? inputs : DoAllocOutputs(op, inputs, ctx);
  DoCompute(op, inputs, outputs, ctx);
  return outputs;
}

void RotaryOpImpl::DoCompute(Operator& op, 
                             const NDArrayList& inputs, NDArrayList& outputs,
                             RuntimeContext& ctx) const {
  auto x = inputs.at(0);
  auto cos = inputs.at(1);
  auto sin = inputs.at(2);
  auto out = outputs.at(0);
  int64_t rotary_dim = cos->shape(1);
  HTShape rotary_size = x->shape();
  rotary_size[3] = rotary_dim;
  NDArray x1, x2, o1, o2;
  if (interleaved())
    HT_NOT_IMPLEMENTED << "GPT-J style Not Implemented.";
  else {
    HTShape begin_pos_x1 = {0, 0, 0, 0};
    HTShape begin_pos_x2 = {0, 0, 0, rotary_dim};
    x1 = NDArray::slice(x, begin_pos_x1, rotary_size, op->instantiation_ctx().stream_index);
    x2 = NDArray::slice(x, begin_pos_x2, rotary_size, op->instantiation_ctx().stream_index);
    o1 = NDArray::slice(out, begin_pos_x1, rotary_size, op->instantiation_ctx().stream_index);
    o2 = NDArray::slice(out, begin_pos_x2, rotary_size, op->instantiation_ctx().stream_index);
  }
  NDArray cos_ = NDArray::unsqueeze(NDArray::slice(cos, HTShape{0, 0}, HTShape{x->shape(1), cos->shape(2)}), 1);
  NDArray sin_ = NDArray::unsqueeze(NDArray::slice(sin, HTShape{0, 0}, HTShape{x->shape(1), sin->shape(2)}), 1);
  HT_DISPATCH_KERNEL_CUDA_ONLY(op->instantiation_ctx().placement.type(), type(),
                               hetu::impl::Rotary, x1, x2, cos_, sin_, o1, o2,
                               false, op->instantiation_ctx().stream());
  if (!inplace()) {
    HTShape begin_pos_remain = {0, 0, 0, 2 * rotary_dim};
    HTShape remain_size = x->shape();
    remain_size[3] -= (2 * rotary_dim);
    NDArray x_remain = NDArray::slice(x, begin_pos_remain, remain_size, op->instantiation_ctx().stream_index);
    NDArray out_remain = NDArray::slice(out, begin_pos_remain, remain_size, op->instantiation_ctx().stream_index);
    NDArray::copy(x_remain, op->instantiation_ctx().stream_index, out_remain);
  }
}

TensorList RotaryOpImpl::DoGradient(Operator& op, const TensorList& grad_outputs) const {
  auto grad_input = op->requires_grad(0) ? MakeRotaryGradientOp(
                                           grad_outputs.at(0), op->input(1), op->input(2),
                                           interleaved(), inplace(),
                                           op->grad_op_meta().set_name(op->grad_name()))
                                         : Tensor();
  return {grad_input, Tensor(), Tensor()};
}

HTShapeList RotaryOpImpl::DoInferShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const {
  return {input_shapes.at(0)};
}

void RotaryOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                  const OpMeta& op_meta,
                                  const InstantiationContext& inst_ctx) const {
  return outputs.at(0)->set_distributed_states(inputs.at(0)->get_distributed_states());
}

void RotaryOpImpl::DoDeduceHeterProp(const std::vector<int32_t>& inputs_hetero_dim,
                                     TensorList& outputs, const OpMeta& op_meta,
                                     const InstantiationContext& inst_ctx) const {
  outputs.at(0)->cur_ds_union().set_hetero_dim(inputs_hetero_dim.at(0));
}

NDArrayList RotaryGradientOpImpl::DoCompute(Operator& op,
                                            const NDArrayList& inputs,
                                            RuntimeContext& ctx) const {
  NDArrayList outputs = inplace() ? inputs : DoAllocOutputs(op, inputs, ctx);
  DoCompute(op, inputs, outputs, ctx);
  return outputs;
}

void RotaryGradientOpImpl::DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                             RuntimeContext& ctx) const {
  auto dout = inputs.at(0);
  auto cos = inputs.at(1);
  auto sin = inputs.at(2);
  auto dx = outputs.at(0);
  int64_t rotary_dim = cos->shape(1);
  HTShape rotary_size = dout->shape();
  rotary_size[3] = rotary_dim;
  NDArray dout1, dout2, dx1, dx2;
  if (interleaved())
    HT_NOT_IMPLEMENTED << "GPT-J style Not Implemented.";
  else {
    HTShape begin_pos_x1 = {0, 0, 0, 0};
    HTShape begin_pos_x2 = {0, 0, 0, rotary_dim};
    dout1 = NDArray::slice(dout, begin_pos_x1, rotary_size, op->instantiation_ctx().stream_index);
    dout2 = NDArray::slice(dout, begin_pos_x2, rotary_size, op->instantiation_ctx().stream_index);
    dx1 = NDArray::slice(dx, begin_pos_x1, rotary_size, op->instantiation_ctx().stream_index);
    dx2 = NDArray::slice(dx, begin_pos_x2, rotary_size, op->instantiation_ctx().stream_index);
  }
  NDArray cos_ = NDArray::unsqueeze(NDArray::slice(cos, HTShape{0, 0}, HTShape{dout->shape(1), cos->shape(2)}), 1);
  NDArray sin_ = NDArray::unsqueeze(NDArray::slice(sin, HTShape{0, 0}, HTShape{dout->shape(1), sin->shape(2)}), 1);
  HT_DISPATCH_KERNEL_CUDA_ONLY(op->instantiation_ctx().placement.type(), type(),
                               hetu::impl::Rotary, dout1, dout2, cos_, sin_, dx1, dx2,
                               true, op->instantiation_ctx().stream());
  if (!inplace()) {
    HTShape begin_pos_remain = {0, 0, 0, 2 * rotary_dim};
    HTShape remain_size = dout->shape();
    remain_size[3] -= (2 * rotary_dim);
    NDArray dout_remain = NDArray::slice(dout, begin_pos_remain, remain_size, op->instantiation_ctx().stream_index);
    NDArray dx_remain = NDArray::slice(dx, begin_pos_remain, remain_size, op->instantiation_ctx().stream_index);
    NDArray::copy(dout_remain, op->instantiation_ctx().stream_index, dx_remain);
  }
}

HTShapeList RotaryGradientOpImpl::DoInferShape(Operator& op, 
                                           const HTShapeList& input_shapes, 
                                           RuntimeContext& ctx) const {
  return {input_shapes.at(0)};
}

void RotaryGradientOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                          const OpMeta& op_meta,
                                          const InstantiationContext& inst_ctx) const {
  return outputs.at(0)->set_distributed_states(inputs.at(0)->get_distributed_states());
}

void RotaryGradientOpImpl::DoDeduceHeterProp(const std::vector<int32_t>& inputs_hetero_dim,
                                             TensorList& outputs, const OpMeta& op_meta,
                                             const InstantiationContext& inst_ctx) const {
  outputs.at(0)->cur_ds_union().set_hetero_dim(inputs_hetero_dim.at(0));
}

Tensor MakeRotaryOp(Tensor x, Tensor cos, Tensor sin,
                    bool interleaved, bool inplace,
                    OpMeta op_meta) {
  TensorList inputs = {x, cos, sin};
  return Graph::MakeOp(
           std::make_shared<RotaryOpImpl>(interleaved, inplace),
           std::move(inputs),
           std::move(op_meta))->output(0);
}

Tensor MakeRotaryGradientOp(Tensor dout, Tensor cos, Tensor sin,
                            bool interleaved, bool inplace,
                            OpMeta op_meta) {
  TensorList inputs = {dout, cos, sin};
  return Graph::MakeOp(
           std::make_shared<RotaryGradientOpImpl>(interleaved, inplace),
           std::move(inputs),
           std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hetu
