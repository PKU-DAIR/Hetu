#include "hetu/graph/ops/Sqrt.h"
#include "hetu/graph/ops/Arithmetics.h"
#include "hetu/graph/ops/Pow.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

void SqrtOpImpl::DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                          RuntimeContext& ctx) const {
  NDArray::sqrt(inputs.at(0), op->instantiation_ctx().stream_index, outputs.at(0));
}

NDArrayList SqrtOpImpl::DoCompute(Operator& op,
                                  const NDArrayList& inputs,
                                  RuntimeContext& ctx) const {
  NDArrayList outputs = inplace() && !ctx.has_runtime_allocation(op->output(0)->id()) ? inputs : DoAllocOutputs(op, inputs, ctx);
  DoCompute(op, inputs, outputs, ctx);
  return outputs;
}

TensorList SqrtOpImpl::DoGradient(Operator& op, const TensorList& grad_outputs) const {
  auto g_op_meta = op->grad_op_meta();
  auto grad_input = op->requires_grad(0) ? MakeMulElewiseOp(MakeReciprocalOp(MakeMulByConstOp(
                                          op->output(0), 2, g_op_meta), g_op_meta),
                                          grad_outputs.at(0), g_op_meta.set_name(op->grad_name()))
                                        : Tensor();
  return {grad_input};
}

void ReciprocalSqrtOpImpl::DoCompute(Operator& op,const NDArrayList& inputs,
                                     NDArrayList& outputs, 
                                     RuntimeContext& ctx) const {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(),
                                  hetu::impl::ReciprocalSqrt, inputs.at(0),
                                  outputs.at(0), op->instantiation_ctx().stream());
}

NDArrayList ReciprocalSqrtOpImpl::DoCompute(Operator& op,
                                            const NDArrayList& inputs,
                                            RuntimeContext& ctx) const {
  NDArrayList outputs = inplace() && !ctx.has_runtime_allocation(op->output(0)->id()) ? inputs : DoAllocOutputs(op, inputs, ctx);
  DoCompute(op, inputs, outputs, ctx);
  return outputs;
}

TensorList ReciprocalSqrtOpImpl::DoGradient(Operator& op, const TensorList& grad_outputs) const {
  auto g_op_meta = op->grad_op_meta();
  auto grad_input = op->requires_grad(0) ? MakeMulByConstOp(MakeMulElewiseOp(MakePowTensorAndConstOp(
                                          op->output(0), 3, g_op_meta), grad_outputs.at(0), g_op_meta),
                                          -0.5, g_op_meta.set_name(op->grad_name()))
                                        : Tensor();
  return {grad_input};
}

Tensor MakeSqrtOp(Tensor input, OpMeta op_meta) {
  TensorList inputs = {std::move(input)};
  return Graph::MakeOp(
    std::make_shared<SqrtOpImpl>(false),
    std::move(inputs),
    std::move(op_meta))->output(0);
}

Tensor MakeSqrtInplaceOp(Tensor input, OpMeta op_meta) {
  TensorList inputs = {std::move(input)};
  return Graph::MakeOp(
    std::make_shared<SqrtOpImpl>(true),
    std::move(inputs),
    std::move(op_meta))->output(0);
}

Tensor MakeReciprocalSqrtOp(Tensor input, OpMeta op_meta) {
  TensorList inputs = {std::move(input)};
  DataType input_type = DataType::FLOAT32;
  AutoCast::Tensor_AutoCast(inputs, input_type);
  return Graph::MakeOp(
    std::make_shared<ReciprocalSqrtOpImpl>(false),
    std::move(inputs),
    std::move(op_meta))->output(0);
}

Tensor MakeReciprocalSqrtInplaceOp(Tensor input, OpMeta op_meta) {
  TensorList inputs = {std::move(input)};
  DataType input_type = DataType::FLOAT32;
  AutoCast::Tensor_AutoCast(inputs, input_type);
  return Graph::MakeOp(
    std::make_shared<ReciprocalSqrtOpImpl>(true),
    std::move(inputs),
    std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hetu
