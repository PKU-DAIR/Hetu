#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"
#include "hetu/graph/ops/Unary.h"

namespace hetu {
namespace graph {

class SwiGLUOpImpl;
class SwiGLUOp;
class SwiGLUGradientOpImpl;
class SwiGLUGradientOp;

class SwiGLUOpImpl final : public UnaryOpImpl {
 private:
  friend class SwiGLUOp;
  struct constructor_access_key {};

 public:
  SwiGLUOpImpl()
  : UnaryOpImpl(quote(SwiGLUOp)) {
  }

 protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    HTShape output_shape(inputs[0]->shape());
    output_shape.back() /= 2;
    NDArrayMeta output_meta = NDArrayMeta().set_dtype(inputs[0]->dtype())
                                           .set_shape(output_shape)
                                           .set_device(inputs[0]->device());    
    return {output_meta};
  }
  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

  TensorList DoGradient(Operator& op, const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const override;

 public:
  bool operator==(const OpInterface& rhs) const override {
    return UnaryOpImpl::operator==(rhs); 
  }
};

Tensor MakeSwiGLUOp(Tensor input, OpMeta op_meta = OpMeta());

class SwiGLUGradientOpImpl final : public UnaryGradientOpImpl {

 public:
  SwiGLUGradientOpImpl()
  : UnaryGradientOpImpl(quote(SwiGLUGradientOp)) {
  }

 protected:
  std::vector<NDArrayMeta>
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    NDArrayMeta output_meta = NDArrayMeta().set_dtype(inputs[0]->dtype())
                                           .set_shape(inputs[0]->shape())
                                           .set_device(inputs[0]->device());
    return {output_meta};
  }


  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const override;
 public:
  bool operator==(const OpInterface& rhs) const override {
    return UnaryGradientOpImpl::operator==(rhs); 
  }
};

Tensor MakeSwiGLUGradientOp(Tensor input, Tensor grad_output,
                          OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu
