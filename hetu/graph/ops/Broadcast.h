#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"

namespace hetu {
namespace graph {

class BroadcastOpImpl;
class BroadcastOp;

class BroadcastOpImpl final : public OpInterface {
 private:
  friend class BroadcastOp;
  struct constructor_access_key {};

 public:
  BroadcastOpImpl(OpMeta op_meta = OpMeta())
  : OpInterface(quote(BroadcastOp)), _mode(0) {
  }

  BroadcastOpImpl(const HTShape& shape, const HTShape& add_axes = HTShape(),
                  OpMeta op_meta = OpMeta())
  : OpInterface(quote(BroadcastOp)),
    _mode(1),
    _shape(shape),
    _add_axes(add_axes) {
  }

  const HTShape& get_shape() const {
    return _shape;
  }

  const HTAxes& get_add_axes() const {
    return _add_axes;
  }

  void set_shape(const HTShape& shape) {
    _shape = shape;
  }

  void set_add_axes(const HTAxes& shape) {
    _add_axes = shape;
  }

  int mode() const {
    return _mode;
  }


 protected:
  std::vector<NDArrayMeta>
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    NDArrayMeta output_meta;
    if (mode() == 0) {
      output_meta = inputs[1]->meta();
    }
    else {
        // HT_ASSERT_CHECK_AXES(get_add_axes(), inputs[0]->ndim()); 
      output_meta = NDArrayMeta().set_dtype(inputs[0]->dtype()).set_shape(get_shape()).set_device(inputs[0]->device());
    }
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

  int _mode; // 0 = Broadcast, 1 = BroadcastShape

  HTShape _shape;

  HTShape _add_axes;


 public:
  inline bool require_contig_inputs() const override {
    return false;
  }

  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const BroadcastOpImpl&>(rhs);
      if (mode() == rhs_.mode()){
        if (mode() == 0)
          return true;
        else 
          return (get_shape() == rhs_.get_shape()
                  && get_add_axes() == rhs_.get_add_axes());
      }
    }
    return false;
  }
};

Tensor MakeBroadcastOp(Tensor input, Tensor output, OpMeta op_meta = OpMeta());

Tensor MakeBroadcastOp(Tensor input, const HTShape& shape,
                       const HTShape& add_axes = HTShape(),
                       OpMeta op_meta = OpMeta());

class BroadcastGradientOpImpl final : public OpInterface {
 public:
  BroadcastGradientOpImpl(const HTShape& axes,
                          const HTKeepDims& keepdims,
                          OpMeta op_meta = OpMeta())
  : OpInterface(quote(BroadcastGradientOp)),
    _axes(axes),
    _keepdims(keepdims) {
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

  HTShape _axes;

  HTKeepDims _keepdims;

 public:
  inline bool require_contig_inputs() const override {
    return false;
  }

  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const BroadcastGradientOpImpl&>(rhs);
      return (get_axes() == rhs_.get_axes()
              && get_keepdims() == rhs_.get_keepdims());
    }
    return false;
  }
};

Tensor MakeBroadcastGradientOp(Tensor input, Tensor ori_input, 
                               Tensor ori_output, const HTAxes& axes,
                               OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu
