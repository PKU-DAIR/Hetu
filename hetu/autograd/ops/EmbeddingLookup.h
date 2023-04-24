#pragma once

#include "hetu/autograd/operator.h"

namespace hetu {
namespace autograd {

class EmbeddingLookupOpDef;
class EmbeddingLookupOp;
class EmbeddingLookupGradientOpDef;
class EmbeddingLookupGradientOp;

class EmbeddingLookupGradientOpDef : public OperatorDef {
 private:
  friend class EmbeddingLookupGradientOp;
  struct constrcutor_access_key {};

 public:
  EmbeddingLookupGradientOpDef(const constrcutor_access_key&,
                               Tensor grad_output, Tensor id, Tensor ori_input,
                               const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(EmbeddingLookupGradientOp), {grad_output, id, ori_input}, // ori_output ?
                op_meta) {
    AddOutput(NDArrayMeta().set_dtype(_inputs[0]->dtype()));
    DeduceStates();
  }

  void DeduceStates() override;

  HTShape get_embed_shape() {
    return _embed_shape;
  }

  void set_embed_shape(HTShape shape) {
    _embed_shape = shape;
  }

 protected:
  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  HTShape _embed_shape;
};

class EmbeddingLookupGradientOp final
: public OpWrapper<EmbeddingLookupGradientOpDef> {
 public:
  EmbeddingLookupGradientOp(Tensor grad_output, Tensor id, Tensor ori_input,
                            const OpMeta& op_meta = OpMeta())
  : OpWrapper<EmbeddingLookupGradientOpDef>(
      make_ptr<EmbeddingLookupGradientOpDef>(
        EmbeddingLookupGradientOpDef::constrcutor_access_key(), grad_output, id,
        ori_input, op_meta)) {}
};

class EmbeddingLookupOpDef : public OperatorDef {
 private:
  friend class EmbeddingLookupOp;
  struct constrcutor_access_key {};

 public:
  EmbeddingLookupOpDef(const constrcutor_access_key&, Tensor input, Tensor id,
                       const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(EmbeddingLookupOp), {input, id}, op_meta) {
    HTShape shape;
    if (input->has_shape() && id->has_shape()) {
      shape = id->shape();
      shape.emplace_back(input->shape(1));
    }
    AddOutput(NDArrayMeta().set_dtype(_inputs[0]->dtype()).set_shape(shape));
    DeduceStates();
  }

  void DeduceStates() override;

  HTShape get_grad_embed() const {
    return _grad_embed_shape;
  }

  void set_grad_embed(HTShape shape) {
    _grad_embed_shape = shape;
  }

 protected:
  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  TensorList DoGradient(const TensorList& grad_outputs) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  HTShape _grad_embed_shape;
};

class EmbeddingLookupOp final : public OpWrapper<EmbeddingLookupOpDef> {
 public:
  EmbeddingLookupOp(Tensor input, Tensor id, const OpMeta& op_meta = OpMeta())
  : OpWrapper<EmbeddingLookupOpDef>(make_ptr<EmbeddingLookupOpDef>(
      EmbeddingLookupOpDef::constrcutor_access_key(), input, id, op_meta)) {}
};

} // namespace autograd
} // namespace hetu
