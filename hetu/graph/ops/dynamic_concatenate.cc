#include "hetu/graph/ops/dynamic_concatenate.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

void DynamicConcatenateOpImpl::DoCompute(Operator& op,
                                  const NDArrayList& inputs,
                                  NDArrayList& outputs, RuntimeContext& ctx) const {
  if (op->instantiation_ctx().placement.is_cpu()) {
    HT_RUNTIME_ERROR << "DynamicConcatenateOpImpl can't run on cpu.";
    __builtin_unreachable();
    // TODO: implement DynamicConcatenate on cpu
  }
  else {
    int num = op->num_inputs();
    int n_dims = outputs.at(0)->ndim();
    size_t axis = get_axis();
    size_t offset = 0;
    size_t max_offset = outputs.at(0)->shape(axis);
    // only support DynamicConcatenate on the single padding axis
    for (int i = 0; i < n_dims; ++i) {
      if (i != axis) {
        for (int j = 0; j < num; j++)
        HT_ASSERT(inputs.at(j)->shape(i) == inputs.at(j)->dynamic_shape(i))
          << "There shouldn't be any paddings besides the axis to concat.";
      }
    }
    for (int i = 0; i < num; ++i) {
      HT_ASSERT(offset + inputs.at(i)->dynamic_shape(axis) <= max_offset)
        << "The concat axis is out of range, please provide more paddings when building the graph.";
      HT_DISPATCH_KERNEL_CUDA_ONLY(op->instantiation_ctx().placement.type(), type(),
                                  hetu::impl::DynamicConcatenate, inputs.at(i),
                                  outputs.at(0), axis, offset, op->instantiation_ctx().stream());
      offset += inputs.at(i)->dynamic_shape(axis);
    }
  }
}

HTShapeList DynamicConcatenateOpImpl::DoInferShape(Operator& op,
                                            const HTShapeList& input_shapes,
                                            RuntimeContext& ctx) const {
  int len = input_shapes.size();
  HTShape out_shape = input_shapes.at(0);
  int n_dim = out_shape.size();
  int out_dim = out_shape[get_axis()];
  for (int i = 1; i < len; ++i) {
    HTShape shape = input_shapes.at(i);
    HT_ASSERT(shape.size() == out_shape.size());
    for (int j = 0; j < n_dim; ++j) {
      if (j != (int) get_axis()) {
        HT_ASSERT(shape[j] == out_shape[j]);
      } else {
        out_dim = out_dim > shape[j] ? out_dim : shape[j];
      }
    }
  }
  out_shape[get_axis()] = out_dim;
  return {out_shape};
}

HTShapeList DynamicConcatenateOpImpl::DoInferDynamicShape(Operator& op,
                                            const HTShapeList& input_shapes,
                                            RuntimeContext& ctx) const {
  int len = input_shapes.size();
  HTShape out_shape = input_shapes.at(0);
  int n_dim = out_shape.size();
  int out_dim = out_shape[get_axis()];
  for (int i = 1; i < len; ++i) {
    HTShape shape = input_shapes.at(i);
    HT_ASSERT(shape.size() == out_shape.size());
    for (int j = 0; j < n_dim; ++j) {
      if (j != (int) get_axis()) {
        HT_ASSERT(shape[j] == out_shape[j]);
      } else {
        out_dim += shape[j];
      }
    }
  }
  out_shape[get_axis()] = out_dim;
  return {out_shape};
}

void DynamicConcatenateOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                       const OpMeta& op_meta,
                                       const InstantiationContext& inst_ctx) const {
  for (const auto& input : inputs) {
    const DistributedStates& ds_input = input->get_distributed_states();
    HT_ASSERT(ds_input.get_dim(get_axis()) == 1)
      << "Concat was not allowed in splited dimension: " << get_axis();
  }
  // 直接调用默认的states copy函数做检查和赋值
  OpInterface::DoDeduceStates(inputs, outputs, op_meta, inst_ctx);
}

Tensor MakeDynamicConcatenateOp(TensorList inputs, int64_t axis,
                         OpMeta op_meta) {
  axis = NDArrayMeta::ParseAxis(axis, inputs[0]->ndim());
  TensorList inputs_ = inputs;
  DataType input_type = DataType::UNDETERMINED;
  AutoCast::Tensor_AutoCast(inputs_, input_type);
  return Graph::MakeOp(
          std::make_shared<DynamicConcatenateOpImpl>(axis),
          std::move(inputs_),
          std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hetu
