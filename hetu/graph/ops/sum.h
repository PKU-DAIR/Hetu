#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"

namespace hetu {
namespace graph {

class SumOpImpl final : public OpInterface {
 public:
  SumOpImpl() : OpInterface(quote(SumOp)) {}

  uint64_t op_indicator() const noexcept override {
    return SUM_OP;
  }

 protected:
  std::vector<NDArrayMeta>
  DoInferMeta(const TensorList& inputs) const override {
    HT_VALUE_ERROR_IF(inputs.empty()) << "No inputs are provided";
    // TODO: support broadcast
    int len = inputs.size();
    HTShape output_shape = inputs[0]->shape();
    for (int i = 1; i < len; ++i) {
      if (inputs[i]->has_shape())
        output_shape = Broadcast(output_shape, inputs[i]->shape());
    }
    auto output_meta = NDArrayMeta(output_shape, inputs[0]->dtype(), inputs[0]->device());
    return {output_meta};
  }

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                        const OpMeta& op_meta) const override {
    // TODO: care about broadcase case
    // allow partial sum
    DistributedStates default_ds;
    for (auto& input : inputs) {
      const auto& input_ds = input->get_distributed_states(); 
      HT_ASSERT(input_ds.is_valid()) << op_meta.name << ": input states must be valid!";    
      if (!default_ds.is_valid()) {
        default_ds.set_distributed_states(input_ds);
      } else {
        HT_ASSERT(default_ds.check_equal(input_ds))
          << op_meta.name << ": in SumOp DoDeduceStates: distributed states of all input tensor must be same!"
          << ", " << default_ds.ds_info() << " vs " << input_ds.ds_info();
      }
    }
    outputs.at(0)->set_distributed_states(default_ds);
  }

  TensorList DoGradient(Operator& op,
                        const TensorList& grad_outputs) const override {
    TensorList grad_inputs;
    grad_inputs.reserve(op->num_inputs());
    for (size_t i = 0; i < op->num_inputs(); i++)
      grad_inputs.push_back(op->requires_grad(i) ? grad_outputs.front() : Tensor());
    return grad_inputs;
  }

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override {
    // TODO: support broadcast
    int len = input_shapes.size();
    HTShape output_shape = input_shapes[0];
    for (int i = 1; i < len; ++i) {
      output_shape = Broadcast(output_shape, input_shapes[i]);
    }
    return {output_shape};
  }

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override {
    auto stream_id = op->instantiation_ctx().stream_index;
    auto& output = outputs.front();
    auto input_num = inputs.size();
    if (input_num == 1) {
      NDArray::copy(inputs[0], stream_id, output);
    } else {
      NDArray::add(inputs[0], inputs[1], stream_id, output);
      for (int i = 2; i < input_num; i++) {
        NDArray::add(inputs[i], output, stream_id, output);
      }
    }
  }

 public:
  inline bool require_contig_inputs() const override {
    return false;
  }
};

Tensor MakeSumOp(TensorList inputs, OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu
