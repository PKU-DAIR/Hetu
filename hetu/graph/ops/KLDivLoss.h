#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"
#include "hetu/graph/ops/Loss.h"

namespace hetu {
namespace graph {

class KLDivLossOpImpl;
class KLDivLossOp;
class KLDivLossGradientOpImpl;
class KLDivLossGradientOp;

class KLDivLossOpImpl final : public LossOpImpl {

 public:
  KLDivLossOpImpl(ReductionType reduction = kMEAN)
  : LossOpImpl(quote(KLDivLossOp), reduction) {}

protected:
  std::vector<NDArrayMeta>
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    HT_ASSERT_TENSORS_SAME_SHAPE(inputs[0], inputs[1]);
    NDArrayMeta output_meta;
    if (_reduction != kNONE)
      output_meta = NDArrayMeta().set_dtype(inputs[0]->dtype()).set_shape({1}).set_device(inputs[0]->device());
    else
      output_meta = inputs[0]->meta();
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

 public:
  bool operator==(const OpInterface& rhs) const override {
    return LossOpImpl::operator==(rhs);
  }
};

Tensor MakeKLDivLossOp(Tensor preds, Tensor labels,
                       ReductionType reduction = kMEAN,
                       OpMeta op_meta = OpMeta());

Tensor MakeKLDivLossOp(Tensor preds, Tensor labels,
                       const std::string& reduction = "mean",
                       OpMeta op_meta = OpMeta());

class KLDivLossGradientOpImpl final : public LossGradientOpImpl {

 public:
  KLDivLossGradientOpImpl(ReductionType reduction = kMEAN)
  : LossGradientOpImpl(quote(KLDivLossGradientOp), reduction) {}

 protected:
  std::vector<NDArrayMeta>
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    return {inputs[0]->meta()};
  }

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta,
                      const InstantiationContext& inst_ctx) const override;  

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;

 public:
  bool operator==(const OpInterface& rhs) const override {
    return LossGradientOpImpl::operator==(rhs);
  }
};

Tensor MakeKLDivLossGradientOp(Tensor preds, Tensor labels, Tensor grad_output,
                               ReductionType reduction = kMEAN,
                               OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu
