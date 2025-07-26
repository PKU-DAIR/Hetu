#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"
#include "hetu/graph/ops/Unary.h"

namespace hetu {
namespace graph {


class MaskedScatterOpImpl;
class MaskedScatterOp;
class MaskedScatterGradientOpImpl;
class MaskedScatterGradientOp;

class MaskedScatterOpImpl : public OpInterface {

 private:
  friend class MaskedScatterOp;
  struct constructor_access_key {};

 public:
  MaskedScatterOpImpl(bool inplace = true)
  : OpInterface(quote(MaskedScatterOp)),
    _inplace(inplace) {}

  inline bool require_contig_inputs() const override {
    return false;
  }

  inline bool inplace() const {
    return _inplace;
  }

  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    HTShape shape = inputs[0]->shape();
    NDArrayMeta output_meta = NDArrayMeta().set_dtype(inputs[0]->dtype())
                                           .set_shape(shape)
                                           .set_device(inputs[0]->device());
    return {output_meta};
  }

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override {
    return {input_shapes[0]};
  }

  TensorList DoGradient(Operator& op,
                        const TensorList& grad_outputs) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

  NDArrayList DoCompute(Operator& op,
                        const NDArrayList& inputs,
                        RuntimeContext& ctx) const override;

  bool operator==(const OpInterface& rhs) const override {
    return OpInterface::operator==(rhs);
  }
  protected:
    bool _inplace;
};



class MaskedScatterGradientOpImpl : public OpInterface {

 private:
  friend class MaskedScatterGradientOp;
  struct constructor_access_key {};

 public:
  MaskedScatterGradientOpImpl()
  : OpInterface(quote(MaskedScatterGradientOpImpl)) {
  }

  inline bool require_contig_inputs() const override {
    return false;
  }

  inline bool inplace() const {
    return _inplace;
  }

  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    HTShape shape = inputs[2]->shape();
    NDArrayMeta output_meta = NDArrayMeta().set_dtype(inputs[2]->dtype())
                                           .set_shape(shape)
                                           .set_device(inputs[2]->device());
    return {output_meta};
  }

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override {
    return {input_shapes[2]};
  }

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

  NDArrayList DoCompute(Operator& op,
                        const NDArrayList& inputs,
                        RuntimeContext& ctx) const override;

  bool operator==(const OpInterface& rhs) const override {
    return OpInterface::operator==(rhs);
  }
  protected:
    bool _inplace;
};


Tensor MakeMaskedScatterOp(Tensor input, Tensor mask, Tensor source,
                        OpMeta op_meta = OpMeta());


Tensor MaskedScatterGradientOp(Tensor grad_output, Tensor mask, Tensor source,
                        OpMeta op_meta = OpMeta());

}
}