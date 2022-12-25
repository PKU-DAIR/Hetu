#pragma once

#include "hetu/autograd/operator.h"

namespace hetu {
namespace autograd {

class SigmoidOpDef;
class SigmoidOp;

class SigmoidOpDef : public OperatorDef {
 private:
  friend class SigmoidOp;
  struct constrcutor_access_key {};

 public:
  SigmoidOpDef(const constrcutor_access_key&, Tensor input,
               const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(SigmoidOp), {input}, op_meta) {
    AddOutput(input->meta());
  }

 protected:
  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  TensorList DoGradient(const TensorList& grad_outputs) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;
};

class SigmoidOp final : public OpWrapper<SigmoidOpDef> {
 public:
  SigmoidOp(Tensor input, const OpMeta& op_meta = OpMeta())
  : OpWrapper<SigmoidOpDef>(make_ptr<SigmoidOpDef>(
      SigmoidOpDef::constrcutor_access_key(), input, op_meta)) {}
};
} // namespace autograd
} // namespace hetu