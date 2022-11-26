#pragma once

#include "hetu/autograd/operator.h"

namespace hetu {
namespace autograd {

class ReduceSumOpDef;
class ReduceSumOp;

class ReduceSumOpDef : public OperatorDef {
 private:
  friend class ReduceSumOp;
  struct constrcutor_access_key {};

 public:
  ReduceSumOpDef(const constrcutor_access_key&, Tensor input,
                 const HTAxes& axes, const HTKeepDims& keepdims = {false},
                 const OpMeta& op_meta = OpMeta())
  : OperatorDef(quote(ReduceSumOp), {input}, op_meta),
    _axes(axes),
    _keepdims(keepdims) {
    HT_ASSERT(keepdims.size() == axes.size() || keepdims.size() == 1);
    if (keepdims.size() == 1) {
      int len = axes.size();
      bool keepdim = keepdims[0];
      for (int i = 1; i < len; ++i) {
        _keepdims.emplace_back(keepdim);
      }
    }
    int ndim = input->ndim();
    HTShape tmp_axes = axes;
    HTShape input_shape = input->shape();
    int len = axes.size();
    for (int i = 0; i < len; ++i) {
      if (tmp_axes[i] < 0) {
        tmp_axes[i] += ndim;
      }
      HT_ASSERT(tmp_axes[i] >= 0 && tmp_axes[i] < ndim);
      if (keepdims[i] == true)
        input_shape[tmp_axes[i]] = 1;
      else
        input_shape[tmp_axes[i]] = 0;
    }
    HTShape output_shape(0);
    for (int i = 0; i < ndim; ++i) {
      if (input_shape[i] > 0)
        output_shape.emplace_back(input_shape[i]);
    }
    if (output_shape.size() == 0)
      output_shape.emplace_back(1);
    AddOutput(
      NDArrayMeta().set_dtype(_inputs[0]->dtype()).set_shape(output_shape));
  }

  const HTShape& get_axes() const {
    return _axes;
  }

  const HTKeepDims& get_keepdims() const {
    return _keepdims;
  }

  void set_axes(const HTShape& axes) {
    _axes = axes;
  }

  void set_keepdims(const HTKeepDims& keepdims) {
    _keepdims = keepdims;
  }

  const HTShape& get_grad_axes() const {
    return _grad_add_axes;
  }

  const HTShape& get_grad_shape() const {
    return _grad_shape;
  }

  void set_grad_axes(const HTShape& axes) {
    _grad_add_axes = axes;
  }

  void set_grad_shape(const HTShape& shape) {
    _grad_shape = shape;
  }

 protected:
  void DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) override;

  TensorList DoGradient(const TensorList& grad_outputs) override;

  HTShapeList DoInferShape(const HTShapeList& input_shapes) override;

  HTShape _axes;

  HTKeepDims _keepdims;

  HTShape _grad_add_axes;

  HTShape _grad_shape;
};

class ReduceSumOp final : public OpWrapper<ReduceSumOpDef> {
 public:
  // ReduceSumOp() : OpWrapper<ReduceSumOpDef>() {}
  ReduceSumOp(Tensor input, const HTAxes& axes,
              const HTKeepDims& keepdims = {false},
              const OpMeta& op_meta = OpMeta())
  : OpWrapper<ReduceSumOpDef>(
      make_ptr<ReduceSumOpDef>(ReduceSumOpDef::constrcutor_access_key(), input,
                               axes, keepdims, op_meta)) {}
};

} // namespace autograd
} // namespace hetu
