#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"

namespace hetu {
namespace graph {

class Conv2dOpImpl;
class Conv2dOp;
class Conv2dGradientofFilterOpImpl;
class Conv2dGradientofFilterOp;
class Conv2dGradientofDataOpImpl;
class Conv2dGradientofDataOp;
class Conv2dAddBiasOpImpl;
class Conv2dAddBiasOp;

class Conv2dOpImpl final : public OpInterface {
 public:
  Conv2dOpImpl(int64_t padding, int64_t stride, OpMeta op_meta = OpMeta())
  : OpInterface(quote(Conv2dOp)) {
    HT_ASSERT(padding >= 0)
    << "padding < 0, padding = " << padding;
    HT_ASSERT(stride >= 1)
    << "stride < 1, stride = " << padding;
    _padding = {padding, padding};
    _stride = {stride, stride};
  }

  HTShape get_padding() const {
    return _padding;
  }

  HTShape get_stride() const {
    return _stride;
  }

protected:
  std::vector<NDArrayMeta>
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    HT_ASSERT_TENSORS_SAME_DTYPE(inputs);
    HTShape shape = {-1, -1, -1, -1};
    if (inputs[0]->has_shape() && inputs[1]->has_shape()) {
      HT_ASSERT_HAS_DIMS(inputs[0], 4);
      HT_ASSERT_HAS_DIMS(inputs[1], 4);
      HT_ASSERT(inputs[0]->shape(1) == inputs[1]->shape(1))
      << "input and filter has different C, while input has " 
      << inputs[0]->shape(1) << " and filter has "
      << inputs[1]->shape(1);
      int64_t N = inputs[0]->shape(0);
      int64_t H = inputs[0]->shape(2);
      int64_t W = inputs[0]->shape(3);
      int64_t f_O = inputs[1]->shape(0);
      int64_t f_H = inputs[1]->shape(2);
      int64_t f_W = inputs[1]->shape(3);
      int64_t out_H = (H + 2 * get_padding()[0] - f_H) / get_stride()[0] + 1;
      int64_t out_W = (W + 2 * get_padding()[1] - f_W) / get_stride()[1] + 1;
      HT_ASSERT(out_H > 0 && out_W > 0)
      << "invalid output shape, outW = " << out_W 
      << ", outH = " << out_H;
      shape = {N, f_O, out_H, out_W};
    }
    NDArrayMeta output_meta = NDArrayMeta().set_dtype(inputs[0]->dtype())
                                           .set_shape(shape)
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

  HTShape _padding;

  HTShape _stride;


 public:
  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const Conv2dOpImpl&>(rhs);
      return (get_padding() == rhs_.get_padding()
              && get_stride() == rhs_.get_stride());
    }
    return false;
  }
};

Tensor MakeConv2dOp(Tensor input, Tensor filter, int64_t padding, int64_t stride,
                    OpMeta op_meta = OpMeta());

/*——————————————————————Conv2dGradientofFilter————————————————————————*/

class Conv2dGradientofFilterOpImpl final : public OpInterface {
 public:
  Conv2dGradientofFilterOpImpl(const HTShape& padding, const HTStride& stride,
                               OpMeta op_meta = OpMeta())
  : OpInterface(quote(Conv2dGradientofFilterOp)),
    _padding(padding),
    _stride(stride) {
  }

  HTShape get_padding() const {
    return _padding;
  }

  HTShape get_stride() const {
    return _stride;
  }

protected:
  std::vector<NDArrayMeta>
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    HT_ASSERT_TENSORS_SAME_DTYPE(inputs);
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

  HTShape _padding;

  HTShape _stride;


 public:
  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const Conv2dOpImpl&>(rhs);
      return (get_padding() == rhs_.get_padding()
              && get_stride() == rhs_.get_stride());
    }
    return false;
  }
};

Tensor MakeConv2dGradientofFilterOp(Tensor input, Tensor grad_output,
                                    const HTShape& padding, const HTStride& stride,
                                    OpMeta op_meta = OpMeta());

/*——————————————————————Conv2dGradientofData————————————————————————*/

class Conv2dGradientofDataOpImpl final : public OpInterface {
 public:
  Conv2dGradientofDataOpImpl(const HTShape& padding, const HTStride& stride,
                            OpMeta op_meta = OpMeta())
  : OpInterface(quote(Conv2dGradientofDataOp)),
    _padding(padding),
    _stride(stride) {
  }

  HTShape get_padding() const {
    return _padding;
  }

  HTShape get_stride() const {
    return _stride;
  }

protected:
  std::vector<NDArrayMeta>
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    HT_ASSERT_TENSORS_SAME_DTYPE(inputs);
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

  HTShape _padding;

  HTShape _stride;


 public:
  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const Conv2dOpImpl&>(rhs);
      return (get_padding() == rhs_.get_padding()
              && get_stride() == rhs_.get_stride());
    }
    return false;
  }
};

Tensor MakeConv2dGradientofDataOp(Tensor filter, Tensor grad_output,
                                  const HTShape& padding, const HTStride& stride,
                                  OpMeta op_meta = OpMeta());

/*——————————————————————Conv2dAddBias————————————————————————*/

class Conv2dAddBiasOpImpl final : public OpInterface {
 public:
  Conv2dAddBiasOpImpl(int64_t padding, int64_t stride,
                      OpMeta op_meta = OpMeta())
  : OpInterface(quote(Conv2dAddBiasOp)) {
    _padding = {padding, padding};
    _stride = {stride, stride};
  }

  HTShape get_padding() const {
    return _padding;
  }

  HTShape get_stride() const {
    return _stride;
  }

protected:
  std::vector<NDArrayMeta>
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    HT_ASSERT_TENSORS_SAME_DTYPE(inputs);
    HTShape shape = {-1, -1, -1, -1};
    if (inputs[0]->has_shape() && inputs[1]->has_shape()) {
      int64_t N = inputs[0]->shape(0);
      int64_t H = inputs[0]->shape(2);
      int64_t W = inputs[0]->shape(3);
      int64_t f_O = inputs[1]->shape(0);
      int64_t f_H = inputs[1]->shape(2);
      int64_t f_W = inputs[1]->shape(3);
      int64_t out_H = (H + 2 * get_padding()[0] - f_H) / get_stride()[0] + 1;
      int64_t out_W = (W + 2 * get_padding()[1] - f_W) / get_stride()[1] + 1;
      shape = {N, f_O, out_H, out_W};
    }
    NDArrayMeta output_meta = NDArrayMeta().set_dtype(inputs[0]->dtype())
                                           .set_shape(shape)
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

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;

  HTShape _padding;

  HTShape _stride;


 public:
  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const Conv2dOpImpl&>(rhs);
      return (get_padding() == rhs_.get_padding()
              && get_stride() == rhs_.get_stride());
    }
    return false;
  }
};

Tensor MakeConv2dAddBiasOp(Tensor input, Tensor filter, Tensor bias, int64_t padding,
                           int64_t stride, OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu
