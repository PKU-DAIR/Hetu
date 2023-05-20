#include "hetu/graph/ops/Slice.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

void SliceOpImpl::DoCompute(Operator& op, 
                            const NDArrayList& inputs, NDArrayList& outputs,
                            RuntimeContext& ctx) const {
  // HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(), hetu::impl::Slice,
  //                                 inputs.at(0), outputs.at(0),
  //                                 get_begin_pos().data(), op->instantiation_ctx().stream());
  NDArray::slice(inputs.at(0), get_begin_pos(), outputs.at(0)->shape(),
                 op->instantiation_ctx().stream_index, outputs.at(0));
}

TensorList SliceOpImpl::DoGradient(Operator& op, const TensorList& grad_outputs) const {
  return {op->requires_grad(0) ? MakeSliceGradientOp(grad_outputs.at(0), op->output(0), op->input(0), get_begin_pos(),
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

void SliceOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                 const OpMeta& op_meta) const {
  const DistributedStates& ds_input = inputs.at(0)->get_distributed_states();
  HT_ASSERT(ds_input.is_valid()) 
    << "SliceOpDef: distributed states for input must be valid!";
  HT_ASSERT(ds_input.get_dim(-2) == 1)
    << "Input tensor shouldn't be partial!";
  // HT_ASSERT(ds_input.check_pure_duplicate())
  //   << "Input tensor cannot be splited in any dimension!";
  HTShape ori_shape = inputs.at(0)->shape();
  int ndim = ori_shape.size();
  HTShape output_shape = get_output_shape();
  HTShape begin_pos = get_begin_pos();
  for (int i = 0; i < ndim; i++) {
    if (!(begin_pos[i] == 0 && begin_pos[i] + output_shape[i] == ori_shape[i])) {
      HT_ASSERT(ds_input.get_dim(i) == 1)
        << "Slice dimension " << i << " shouldn't be splited!"; 
    }
  }
  outputs.at(0)->set_distributed_states(ds_input);      
}

void SliceGradientOpImpl::DoCompute(Operator& op,const NDArrayList& inputs,
                                   NDArrayList& outputs, RuntimeContext& ctx) const {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(
    op->instantiation_ctx().placement.type(), type(), hetu::impl::SliceGradient, inputs.at(0),
    outputs.at(0), get_begin_pos().data(), op->instantiation_ctx().stream());
}


HTShapeList SliceGradientOpImpl::DoInferShape(Operator& op, 
                                              const HTShapeList& input_shapes, 
                                              RuntimeContext& ctx) const {
  return {input_shapes.at(2)};
}

void SliceGradientOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                         const OpMeta& op_meta) const {
  outputs.at(0)->set_distributed_states(inputs.at(2)->get_distributed_states());  
}

Tensor MakeSliceOp(Tensor input, const HTShape& begin_pos, const HTShape& output_shape,
                   OpMeta op_meta) {
  return Graph::MakeOp(
    std::make_shared<SliceOpImpl>(begin_pos, output_shape),
    {std::move(input)},
    std::move(op_meta))->output(0);
}

Tensor MakeSliceGradientOp(Tensor grad_output, Tensor ori_output, Tensor ori_input,
                           const HTShape& begin_pos, const HTShape& output_shape,
                           OpMeta op_meta) {
  return Graph::MakeOp(
    std::make_shared<SliceGradientOpImpl>(begin_pos, output_shape),
    {std::move(grad_output), std::move(ori_output), std::move(ori_input)},
    std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hetu
