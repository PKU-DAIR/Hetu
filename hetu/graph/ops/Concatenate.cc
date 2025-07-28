#include "hetu/graph/ops/Concatenate.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

NDArrayList ConcatenateOpImpl::DoCompute(Operator& op,
                                         const NDArrayList& inputs,
                                         RuntimeContext& runtime_ctx) const {
  // inplace
  if (inputs.size() == 1 && !runtime_ctx.has_runtime_allocation(op->output(0)->id())) {
    return inputs;
  }
  auto outputs = DoAllocOutputs(op, inputs, runtime_ctx);
  DoCompute(op, inputs, outputs, runtime_ctx);
  return outputs;
}

void ConcatenateOpImpl::DoCompute(Operator& op,
                                  const NDArrayList& inputs,
                                  NDArrayList& outputs, RuntimeContext& ctx) const {
  if (op->instantiation_ctx().placement.is_cpu()) {
    HT_DISPATCH_KERNEL_CPU_ONLY(op->instantiation_ctx().placement.type(), type(),
                                hetu::impl::Concatenate, inputs,
                                outputs.at(0), get_axis(), op->instantiation_ctx().stream());
  }
  else {
    int num = op->num_inputs();
    size_t offset = 0;
    size_t axis = get_axis();
    if (axis == 0) {
      for (int i = 0; i < num; ++i) {
        auto output = NDArray(inputs.at(i)->meta(), 
                              outputs.at(0)->storage(),
                              outputs.at(0)->storage_offset() + offset);
        if (!inputs.at(i)->is_equal(output)) {
          HT_DISPATCH_KERNEL_CUDA_ONLY(op->instantiation_ctx().placement.type(), type(),
                                       hetu::impl::DataTransfer, inputs.at(i), output,
                                       op->instantiation_ctx().stream());
          // HT_LOG_WARN << op << " data transfer may be avoided (unless you are hot switching)";
        }
        offset += inputs.at(i)->numel();
      }
    } 
    else {
      for (int i = 0; i < num; ++i) {
        HT_DISPATCH_KERNEL_CUDA_ONLY(op->instantiation_ctx().placement.type(), type(),
                                      hetu::impl::Concatenate, inputs.at(i),
                                      outputs.at(0), axis, offset, op->instantiation_ctx().stream());
        offset += inputs.at(i)->shape(axis);
      }
    }
  }
}

TensorList ConcatenateOpImpl::DoGradient(Operator& op,
                                         const TensorList& grad_outputs) const {
  TensorList grads = {};
  int num = op->num_inputs();
  int outdim = 0; 
  size_t axis = get_axis();
  auto g_op_meta = op->grad_op_meta();
  for (int i = 0; i < num; ++i) {
    HT_ASSERT(op->input(i)->shape(axis) >= 0);
    auto grad_input = op->requires_grad(i) ? MakeConcatenateGradientOp(
                                            op->input(i), op->output(0), grad_outputs.at(0), 
                                            axis, outdim, g_op_meta.set_name(op->grad_name(i)))
                                          : Tensor();
    outdim += op->input(i)->shape(axis);
    grads.emplace_back(grad_input);
  }
  return grads;
}

HTShapeList ConcatenateOpImpl::DoInferShape(Operator& op,
                                            const HTShapeList& input_shapes,
                                            RuntimeContext& ctx) const {
  int len = input_shapes.size();
  HTShape out_shape = input_shapes.at(0);
  int n_dim = out_shape.size();
  int out_dim = out_shape[get_axis()];
  int ind = 0;
  ind += 1;
  for (int i = 1; i < len; ++i) {
    HTShape shape = input_shapes.at(i);
    HT_ASSERT(shape.size() == out_shape.size());
    for (int j = 0; j < n_dim; ++j) {
      if (j != (int) get_axis()) {
        HT_ASSERT(shape[j] == out_shape[j]);
      } else {
        ind += 1;
        out_dim += shape[j];
      }
    }
  }
  out_shape[get_axis()] = out_dim;
  return {out_shape};
}

void ConcatenateOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                       const OpMeta& op_meta, const InstantiationContext& inst_ctx) const {
  for (const auto& input : inputs) {
    const DistributedStates& ds_input = input->get_distributed_states();
    HT_ASSERT(ds_input.get_dim(get_axis()) == 1)
      << "Concat was not allowed in splited dimension: " << get_axis();
  }
  // 直接调用默认的states copy函数做检查和赋值
  OpInterface::DoDeduceStates(inputs, outputs, op_meta, inst_ctx);
}

void ConcatenateGradientOpImpl::DoCompute(Operator& op,
                                          const NDArrayList& inputs,
                                          NDArrayList& outputs,
                                          RuntimeContext& ctx) const {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(
    op->instantiation_ctx().placement.type(), type(), hetu::impl::ConcatenateGradient, inputs.at(2),
    outputs.at(0), get_axis(), get_offset(), op->instantiation_ctx().stream());
}

HTShapeList
ConcatenateGradientOpImpl::DoInferShape(Operator& op,
                                        const HTShapeList& input_shapes,
                                        RuntimeContext& ctx) const {
  return {input_shapes.at(0)};
}

void ConcatenateGradientOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                               const OpMeta& op_meta, const InstantiationContext& inst_ctx) const {
  outputs.at(0)->set_distributed_states(inputs.at(0)->get_distributed_states());
}

Tensor MakeConcatenateOp(TensorList inputs, size_t axis,
                         OpMeta op_meta) {
  TensorList inputs_ = inputs;
  DataType input_type = DataType::UNDETERMINED;
  AutoCast::Tensor_AutoCast(inputs_, input_type);
  return Graph::MakeOp(
          std::make_shared<ConcatenateOpImpl>(axis),
          std::move(inputs_),
          std::move(op_meta))->output(0);
}

Tensor MakeConcatenateGradientOp(Tensor input, Tensor output, Tensor grad_output, size_t axis, size_t offset,
                                 OpMeta op_meta) {
  return Graph::MakeOp(
          std::make_shared<ConcatenateGradientOpImpl>(axis, offset),
          {std::move(input), std::move(output), std::move(grad_output)},
          std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hetu
