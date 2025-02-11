#pragma once

#include "hetu/graph/operator.h"

namespace hetu {
namespace graph {

class UpdateScaleOpImpl;
class UpdateScaleOp;

class UpdateScaleOpImpl final : public OpInterface {
 private:
  friend class UpdateScaleOp;
  struct constructor_access_key {};

 public:
  UpdateScaleOpImpl(double growth_factor,
                    double backoff_factor, int growth_interval)
  : OpInterface(quote(UpdateScaleOp)),
   _growth_factor(growth_factor),
  _backoff_factor(backoff_factor), _growth_interval(growth_interval) {
  }


  double growth_factor() const {
    return _growth_factor;
  }  

  double backoff_factor() const {
    return _backoff_factor;
  }    

  int growth_interval() const {
    return _growth_interval;
  }  
 protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    return {inputs[0]->meta(), inputs[1]->meta()};
  };

  NDArrayList DoAllocOutputs(Operator& op, const NDArrayList& inputs,
                             RuntimeContext& runtime_ctx) const override {
    return {inputs[0], inputs[1]};
  }

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

  TensorList DoGradient(Operator& op, const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const override;

  double _growth_factor;
  double _backoff_factor;
  int _growth_interval;

 public:
  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const UpdateScaleOpImpl&>(rhs);
      return (growth_factor() == rhs_.growth_factor()
              && backoff_factor() == rhs_.backoff_factor()
              && growth_interval() == rhs_.growth_interval());
    }
    return false;
  }
};

TensorList MakeUpdateScaleOp(Tensor scale, Tensor growth_tracker, Tensor found_inf, double growth_factor,
                             double backoff_factor, int growth_interval, OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu