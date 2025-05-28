#pragma once

#include "hetu/graph/operator.h"

namespace hetu {
namespace graph {

class ScalarsLikeOpImpl : public OpInterface {
 public:
  ScalarsLikeOpImpl(double scalar_value)
  : ScalarsLikeOpImpl(quote(ScalarsLikeOp), scalar_value) {}

 protected:
  ScalarsLikeOpImpl(OpType&& type, double scalar_value)
  : OpInterface(std::move(type)), _scalar_value(scalar_value) {}

  std::vector<NDArrayMeta>
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    return {inputs.front()->meta()};
  }

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta,
                      const InstantiationContext& inst_ctx) const override;
  
  TensorList DoGradient(Operator&, const TensorList&) const override {
    return {Tensor()};
  }

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override {
    return input_shapes;
  }

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override {
    NDArray::full_(outputs.front(), scalar_value(),
                   op->instantiation_ctx().stream_index);
  }

 public:
  inline bool require_contig_inputs() const override {
    return false;
  }

  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const ScalarsLikeOpImpl&>(rhs);
      return scalar_value() == rhs_.scalar_value();
    }
    return false;
  }

  double scalar_value() const {
    return _scalar_value;
  }

 protected:
  double _scalar_value;
};

Tensor MakeScalarsLikeOp(Tensor input, double scalar_value,
                         OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu
