#pragma once

#include "hetu/autograd/operator.h"
#include "hetu/autograd/utils/tensor_utils.h"

namespace hetu {
namespace autograd {

class LinearOpDef;
class LinearOp;

class LinearOpDef : public OperatorDef {
 private:
  friend class LinearOp;
  struct constrcutor_access_key {};

 public:
  LinearOpDef(const constrcutor_access_key&, Tensor a, Tensor b, Tensor bias,
              bool trans_a = false, bool trans_b = true,
              const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(LinearOp), {a, b, bias}, op_meta),
    _trans_a(trans_a),
    _trans_b(trans_b) {
    if (a->has_shape() && b->has_shape()) {
      HT_ASSERT(a->ndim() == 2 && b->ndim() == 2)
        << "Failed to construct the \"" << type() << "\" operation "
        << "(with name \"" << name() << "\"): "
        << "Dimensions must be 2. "
        << "Got " << a->ndim() << ", " << b->ndim() << ".";
      int64_t dim_a = a->shape(trans_a ? 0 : 1);
      int64_t dim_b = b->shape(trans_b ? 1 : 0);
      HT_ASSERT(dim_a == -1 || dim_b == -1 || dim_a == dim_b)
        << "Failed to construct the \"" << type() << "\" operation "
        << "(with name \"" << name() << "\"): "
        << "Dimensions must be compatible. "
        << "Got " << dim_a << " vs. " << dim_b << ". "
        << "Input shapes: " << a->shape() << " vs. " << b->shape() << ".";
    }
    HTShape shape = {-1, -1};
    if (a->has_shape())
      shape[0] = a->shape(trans_a ? 1 : 0);
    if (b->has_shape())
      shape[1] = b->shape(trans_b ? 0 : 1);
    HT_ASSERT_TENSORS_SAME_DTYPE(_inputs);
    AddOutput(NDArrayMeta().set_dtype(_inputs[0]->dtype()).set_shape(shape));
    DeduceStates();
  }

  void DeduceStates() override;

  inline bool trans_a() const {
    return _trans_a;
  }

  inline bool trans_b() const {
    return _trans_b;
  }

 protected:
  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  TensorList DoGradient(const TensorList& grad_outputs) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  bool _trans_a;
  bool _trans_b;
};

class LinearOp final : public OpWrapper<LinearOpDef> {
 public:
  LinearOp(Tensor a, Tensor b, Tensor bias, bool trans_a = false,
           bool trans_b = true, const OpMeta& op_meta = OpMeta())
  : OpWrapper<LinearOpDef>(
      make_ptr<LinearOpDef>(LinearOpDef::constrcutor_access_key(), a, b, bias,
                            trans_a, trans_b, op_meta)) {}
};

} // namespace autograd
} // namespace hetu
