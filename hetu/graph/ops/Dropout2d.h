#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"
#include "hetu/graph/ops/Unary.h"

namespace hetu {
namespace graph {

class Dropout2dOpImpl;
class Dropout2dOp;
class Dropout2dGradientOpImpl;
class Dropout2dGradientOp;

class Dropout2dOpImpl final : public UnaryOpImpl {
 public:
  Dropout2dOpImpl(double keep_prob, bool inplace = false)
  : UnaryOpImpl(quote(Dropout2dOp), inplace),
    _keep_prob(keep_prob) {}

  double keep_prob() const {
    return _keep_prob;
  };

protected:
  std::vector<NDArrayMeta>
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    HT_ASSERT_TENSORS_SAME_DTYPE(inputs);
    NDArrayMeta output_meta = inputs[0]->meta();
    NDArrayMeta mask_meta = inputs[0]->meta();
    mask_meta.set_shape(HTShape({inputs[0]->shape()[0], inputs[0]->shape()[1]}));
    mask_meta.set_dtype(DataType::BOOL);
    return {output_meta, mask_meta};
  }

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta,
                      const InstantiationContext& inst_ctx) const override;

  TensorList DoGradient(Operator& op,
                        const TensorList& grad_outputs) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs,
                 NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;

  NDArrayList DoCompute(Operator& op, const NDArrayList& inputs,
                        RuntimeContext& runtime_ctx) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const override;

  double _keep_prob;

 public:
  bool operator==(const OpInterface& rhs) const override {
    if (UnaryOpImpl::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const Dropout2dOpImpl&>(rhs);
      return (keep_prob() == rhs_.keep_prob() &&
              inplace() == rhs_.inplace());
    }
    return false;
  }
};

Tensor MakeDropout2dOp(Tensor input, double keep_prob, OpMeta op_meta = OpMeta());

Tensor MakeDropout2dInplaceOp(Tensor input, double keep_prob, OpMeta op_meta = OpMeta());

class Dropout2dGradientOpImpl final : public UnaryGradientOpImpl {
 public:
  Dropout2dGradientOpImpl(double keep_prob,
                          bool fw_inplace,
                          OpMeta op_meta = OpMeta())
  : UnaryGradientOpImpl(quote(Dropout2dGradientOp)),
    _keep_prob(keep_prob),
    _fw_inplace(fw_inplace) {
  }

  double keep_prob() const {
    return _keep_prob;
  }

  bool fw_inplace() const {
    return _fw_inplace;
  }

protected:
  std::vector<NDArrayMeta>
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    NDArrayMeta output_meta = inputs[0]->meta();
    return {output_meta};
  }

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta,
                      const InstantiationContext& inst_ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs,
                 NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;

  NDArrayList DoCompute(Operator& op, const NDArrayList& inputs,
                        RuntimeContext& runtime_ctx) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const override;

  double _keep_prob;
  bool _fw_inplace;

 public:
  bool operator==(const OpInterface& rhs) const override {
    if (UnaryGradientOpImpl::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const Dropout2dGradientOpImpl&>(rhs);
      return (keep_prob() == rhs_.keep_prob() &&
              fw_inplace() == rhs_.fw_inplace());
    }
    return false;
  }
};

Tensor MakeDropout2dGradientOp(Tensor grad_output, Tensor mask,
                               double keep_prob, bool fw_inplace,
                               OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu
