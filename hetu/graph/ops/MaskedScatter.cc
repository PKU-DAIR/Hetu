#include "hetu/graph/ops/Maskedfill.h"
#include "hetu/graph/ops/MaskedScatter.h"
#include "hetu/graph/ops/zeros_like.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"


namespace hetu {
namespace graph {



void MaskedScatterOpImpl::DoCompute(Operator& op, 
                                 const NDArrayList& inputs, NDArrayList& outputs,
                                 RuntimeContext& ctx) const {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(), hetu::impl::MaskedScatter,
                                  inputs.at(0), inputs.at(1), inputs.at(2),
                                  outputs.at(0), op->instantiation_ctx().stream());
}

NDArrayList MaskedScatterOpImpl::DoCompute(Operator& op,
                                               const NDArrayList& inputs,
                                               RuntimeContext& ctx) const {
                                          
  NDArrayList outputs;
  if (inplace()) {
      outputs.push_back(inputs.at(0));
  } else {
      outputs = DoAllocOutputs(op, inputs, ctx);
  }
  DoCompute(op, inputs, outputs, ctx);
  return outputs;
}

TensorList MaskedScatterOpImpl::DoGradient(Operator& op, 
                                        const TensorList& grad_outputs) const {
                                        
  auto g_op_meta = op->grad_op_meta();
  auto grad_input0 = op->requires_grad(0) ? MakeMaskedfillOp(grad_outputs.at(0), op->input(1), 0.0,
                                                           g_op_meta.set_name(op->grad_name(0)))
                                        : Tensor();

  auto grad_input2 = op->requires_grad(2) ? MaskedScatterGradientOp(grad_outputs.at(0), op->input(1), op->input(2),
                                                           g_op_meta.set_name(op->grad_name(2)))
                                        : Tensor();

  return {grad_input0, Tensor(), grad_input2};
}

void MaskedScatterGradientOpImpl::DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                RuntimeContext& ctx) const{
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(), hetu::impl::MaskedSelect,
                                  inputs.at(0), inputs.at(1),
                                  outputs.at(0), op->instantiation_ctx().stream());
}

NDArrayList MaskedScatterGradientOpImpl::DoCompute(Operator& op,
                                               const NDArrayList& inputs,
                                               RuntimeContext& ctx) const {
  NDArrayList outputs = inplace() ? inputs : DoAllocOutputs(op, inputs, ctx);
  DoCompute(op, inputs, outputs, ctx);
  return outputs;
}

Tensor MakeMaskedScatterOp(Tensor input, Tensor mask, Tensor source,
                        OpMeta op_meta) {
  return Graph::MakeOp(
          std::make_shared<MaskedScatterOpImpl>(),
          {std::move(input), std::move(mask), std::move(source)},
          std::move(op_meta))->output(0);  
}

Tensor MaskedScatterGradientOp(Tensor grad_output, Tensor mask, Tensor source,
                        OpMeta op_meta) {
  return Graph::MakeOp(
          std::make_shared<MaskedScatterGradientOpImpl>(),
          {std::move(grad_output), std::move(mask), std::move(source)},
          std::move(op_meta))->output(0);  
}


}
}