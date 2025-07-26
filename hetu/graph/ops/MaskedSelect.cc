// #include "hetu/graph/ops/MaskedSelect.h"
// #include "hetu/graph/ops/MaskedScatter.h"
// #include "hetu/graph/ops/zeros_like.h"
// #include "hetu/graph/headers.h"
// #include "hetu/graph/ops/kernel_links.h"


// namespace hetu {
// namespace graph {



// void MaskedSelectOpImpl::DoCompute(Operator& op, 
//                                  const NDArrayList& inputs, NDArrayList& outputs,
//                                  RuntimeContext& ctx) const {
//   HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(), hetu::impl::MaskedSelect,
//                                   inputs.at(0), inputs.at(1), inputs.at(2),
//                                   outputs.at(0), op->instantiation_ctx().stream());
// }

// NDArrayList MaskedSelectOpImpl::DoCompute(Operator& op,
//                                                const NDArrayList& inputs,
//                                                RuntimeContext& ctx) const {
//   NDArrayList outputs = inplace() ? inputs : DoAllocOutputs(op, inputs, ctx);
//   DoCompute(op, inputs, outputs, ctx);
//   return outputs;
// }

// TensorList MaskedSelectOpImpl::DoGradient(Operator& op, 
//                                         const TensorList& grad_outputs) const {

//   auto g_op_meta = op->grad_op_meta();
//   auto grad_input0 = op->requires_grad(0) ? 
//                                         : Tensor();
//   auto grad_input1 = op->requires_grad(1) ? MakeMaskedfillOp(grad_outputs.at(0), op->input(1), 0.0,
//                                                            g_op_meta.set_name(op->grad_name(0)))
//                                         : Tensor();

//   return {grad_input0, grad_input1, Tensor()};
// }

// Tensor MakeMaskedSelectOp(Tensor input, Tensor mask,
//                         OpMeta op_meta) {
//   return Graph::MakeOp(
//           std::make_shared<MaskedSelectOpImpl>(),
//           {std::move(input), std::move(mask)},
//           std::move(op_meta))->output(0);  
// }


// }
// }