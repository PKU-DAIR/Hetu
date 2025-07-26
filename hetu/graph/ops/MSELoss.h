#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"
#include "hetu/graph/ops/Loss.h"

namespace hetu {
namespace graph {

class MSELossOpImpl;
class MSELossOp;
class MSELossGradientOpImpl;
class MSELossGradientOp;

class MSELossOpImpl final : public LossOpImpl {
 public:
  MSELossOpImpl(ReductionType reduction = kMEAN)
  : LossOpImpl(quote(MSELossOp), reduction) {}

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
  };

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

  TensorList DoGradient(Operator& op, const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const override;

 public:
  bool operator==(const OpInterface& rhs) const override {
    return LossOpImpl::operator==(rhs);
  }
};

Tensor MakeMSELossOp(Tensor preds, Tensor labels,
                     ReductionType reduction = kMEAN,
                     OpMeta op_meta = OpMeta());

Tensor MakeMSELossOp(Tensor preds, Tensor labels,
                     const std::string& reduction = "mean",
                     OpMeta op_meta = OpMeta());

class MSELossGradientOpImpl final : public LossGradientOpImpl {
 public:
  MSELossGradientOpImpl(ReductionType reduction = kMEAN)
  : LossGradientOpImpl(quote(MSELossGradientOp), reduction) {}

 protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    HT_ASSERT(_reduction == kSUM || _reduction == kMEAN || _reduction == kNONE)
    << "Unsupported reduction type \'" << _reduction << "\' for " << type()
    << " operators. Expected: [\'mean\', \'sum\', \'none\']";
    return {inputs[0]->meta()};
  };

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const override;

 public:
  bool operator==(const OpInterface& rhs) const override {
    return LossGradientOpImpl::operator==(rhs);
  }
};

Tensor MakeMSELossGradientOp(Tensor preds, Tensor labels, Tensor grad_output,
                             ReductionType reduction = kMEAN,
                             OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu
