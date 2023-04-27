#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"

namespace hetu {
namespace graph {

class SqrtOpImpl;
class SqrtOp;
class ReciprocalSqrtOpImpl;
class ReciprocalSqrtOp;

class SqrtOpImpl : public OpInterface {

 public:
  SqrtOpImpl()
  : OpInterface(quote(SqrtOp)) {
  }

 protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs) const override {
    return {inputs[0]->meta()};
  };

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

  TensorList DoGradient(Operator& op, const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const override;

 public:
  bool operator==(const OpInterface& rhs) const override {
    return OpInterface::operator==(rhs);
  }
};

Tensor MakeSqrtOp(Tensor input, OpMeta op_meta = OpMeta());

class ReciprocalSqrtOpImpl : public OpInterface {

 public:
  ReciprocalSqrtOpImpl()
  : OpInterface(quote(ReciprocalSqrtOp)) {
  }

 protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs) const override {
    return {inputs[0]->meta()};
  };

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const override;

 public:
  bool operator==(const OpInterface& rhs) const override {
    return OpInterface::operator==(rhs);
  }
};

Tensor MakeReciprocalSqrtOp(Tensor grad_output, OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu
