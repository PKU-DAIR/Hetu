#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"

namespace hetu {
namespace graph {

class RollOpImpl;
class RollOp;

class RollOpImpl final : public OpInterface {
 public:
  RollOpImpl(HTShape shifts,
             HTAxes dims)
  : OpInterface(quote(RollOp)),
  _shifts(shifts),
  _dims(dims) {
  }

  inline HTShape shifts() const{
    return _shifts;
  }

  inline HTAxes dims() const{
    return _dims;
  }

 protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    return {inputs[0]->meta()};
  };

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta,
                      const InstantiationContext& inst_ctx) const override;
                      
  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

  TensorList DoGradient(Operator& op, const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const override;

  HTShape _shifts;

  HTAxes _dims;
  
 public:
  inline bool require_contig_inputs() const override {
    return false;
  }

  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const RollOpImpl&>(rhs);
      return (shifts() == rhs_.shifts()
              && dims() == rhs_.dims());
    }
    return false;
  }
};

Tensor MakeRollOp(Tensor input, HTShape shifts,
                  HTAxes dims, OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu
