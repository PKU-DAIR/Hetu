#include "hetu/graph/ops/AvgPool.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

void AvgPoolOpImpl::DoCompute(Operator& op,
                              const NDArrayList& inputs, NDArrayList& outputs,
                              RuntimeContext& ctx) const {
  NDArray::avgpool(inputs.at(0), get_kernel_H(), get_kernel_W(), get_padding(), get_stride(), 
                   op->instantiation_ctx().stream_index, outputs.at(0));
}

TensorList AvgPoolOpImpl::DoGradient(Operator& op,
                                     const TensorList& grad_outputs) const {
  return {op->requires_grad(0) ? MakeAvgPoolGradientOp(op->output(0), grad_outputs.at(0), op->input(0),
                                get_kernel_H(), get_kernel_W(), get_padding(),
                                get_stride(), op->grad_op_meta().set_name(op->grad_name()))
                              : Tensor()};
}

void AvgPoolOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                   const OpMeta& op_meta,
                                   const InstantiationContext& inst_ctx) const {
  const DistributedStates& ds = inputs.at(0)->get_distributed_states();
  HT_ASSERT(ds.is_valid()) 
    << "AvgPoolOpDef: distributed states for input tensor must be valid!";
  HT_ASSERT(ds.get_dim(-2) == 1)
    << "Input tensor shouldn't be partial!";
  HT_ASSERT(ds.get_dim(2) == 1 && ds.get_dim(3) == 1)
    << "H & W dimension shouldn't be splited, H: "
    << ds.get_dim(2) << ", W: " << ds.get_dim(3);
  outputs.at(0)->set_distributed_states(ds);
}

void AvgPoolGradientOpImpl::DoCompute(Operator& op,
                                      const NDArrayList& inputs,
                                      NDArrayList& outputs,
                                      RuntimeContext& ctx) const {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(
    op->instantiation_ctx().placement.type(), type(), hetu::impl::AvgPoolGradient, inputs.at(0),
    inputs.at(1), inputs.at(2), get_kernel_H(), get_kernel_W(), outputs.at(0),
    get_padding(), get_stride(), op->instantiation_ctx().stream());
}

void AvgPoolGradientOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                           const OpMeta& op_meta,
                                           const InstantiationContext& inst_ctx) const {
  outputs.at(0)->set_distributed_states(inputs.at(2)->get_distributed_states());
}

Tensor MakeAvgPoolOp(Tensor input, size_t kernel_H, size_t kernel_W, 
                     size_t padding, size_t stride,
                     OpMeta op_meta) {
  return Graph::MakeOp(
           std::make_shared<AvgPoolOpImpl>(kernel_H, kernel_W, padding, stride),
           {std::move(input)},
           std::move(op_meta))->output(0);
}

Tensor MakeAvgPoolGradientOp(Tensor output, Tensor output_grad, Tensor input,
                             size_t kernel_H, size_t kernel_W, size_t padding,
                             size_t stride, OpMeta op_meta) {
  return Graph::MakeOp(
           std::make_shared<AvgPoolGradientOpImpl>(kernel_H, kernel_W, padding, stride),
           {std::move(output), std::move(output_grad), std::move(input)},
           std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hetu
