#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"

namespace hetu {
namespace graph {

class InterpolateOpImpl;
class InterpolateOp;
class InterpolateGradientOpImpl;
class InterpolateGradientOp;

class InterpolateOpImpl final : public OpInterface {
 public:
  InterpolateOpImpl(const HTShape& outshape,
                    bool align_corners = false, double scale_factor = 0)
  : OpInterface(quote(InterpolateOp)),
  _out_shape(outshape), 
  _align_corners(align_corners),
  _scale_factor(scale_factor) {

  }

  inline HTShape out_shape() const{
    return _out_shape;
  }

  inline bool align_corners() const{
    return _align_corners;
  }

  inline double scale_factor() const{
    return _scale_factor;
  }

protected:
  std::vector<NDArrayMeta>
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    HTShape output = {};
    if (inputs[0]->has_shape()) {
      HT_ASSERT_HAS_DIMS(inputs[0], 4);
      output = inputs[0]->shape();
      if (out_shape().size() == 2) {
        output[2] = out_shape()[0];
        output[3] = out_shape()[1];
      }
      else {
        HT_ASSERT(scale_factor() > 0);
        output[2] = output[2] * scale_factor();
        output[3] = output[3] * scale_factor();
      }
    }
    NDArrayMeta output_meta = NDArrayMeta().set_dtype(inputs[0]->dtype())
                                           .set_shape(output)
                                           .set_device(inputs[0]->device());
    return {output_meta};
  }

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta,
                      const InstantiationContext& inst_ctx) const override;  

  TensorList DoGradient(Operator& op,
                        const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;

  void DoSaveCtxForBackward(const TensorList& inputs, ContextStore& dst_ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;

  HTShape _out_shape;

  bool _align_corners;

  double _scale_factor;
 public:
  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const InterpolateOpImpl&>(rhs);
      return (out_shape() == rhs_.out_shape()
              && align_corners() == rhs_.align_corners()
              && scale_factor() == rhs_.scale_factor());
    }
    return false;
  }
};

Tensor MakeInterpolateOp(Tensor input, const HTShape& outshape,
                         bool align_corners = false, double scale_factor = 0,
                         OpMeta op_meta = OpMeta());

class InterpolateGradientOpImpl final : public OpInterface {
 public:
  InterpolateGradientOpImpl(bool align_corners = false, double scale_factor = 0)
  : OpInterface(quote(InterpolateGradientOp)),
  _align_corners(align_corners),
  _scale_factor(scale_factor) {
  }

  inline bool align_corners() const{
    return _align_corners;
  }

  inline double scale_factor() const{
    return _scale_factor;
  }

protected:
  std::vector<NDArrayMeta>
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    return {inst_ctx.get<NDArrayMeta>("in_meta")};
  }

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta,
                      const InstantiationContext& inst_ctx) const override;  

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;

  void DoLoadCtxForBackward(ContextStore& src_ctx, ContextStore& dst_ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;

  bool _align_corners;

  double _scale_factor;
 public:
  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const InterpolateGradientOpImpl&>(rhs);
      return (align_corners() == rhs_.align_corners()
              && scale_factor() == rhs_.scale_factor());
    }
    return false;
  }
};

Tensor MakeInterpolateGradientOp(Tensor grad_output,
                                 bool align_corners = false, double scale_factor = 0,
                                 OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu
