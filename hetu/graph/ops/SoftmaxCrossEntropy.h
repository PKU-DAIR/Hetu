#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"
#include "hetu/graph/ops/Loss.h"

namespace hetu {
namespace graph {

class SoftmaxCrossEntropyOpImpl;
class SoftmaxCrossEntropyOp;
class SoftmaxCrossEntropyGradientOpImpl;
class SoftmaxCrossEntropyGradientOp;

class SoftmaxCrossEntropyOpImpl final : public LossOpImpl {

 public:
  SoftmaxCrossEntropyOpImpl(ReductionType reduction = kMEAN)
  : LossOpImpl(quote(SoftmaxCrossEntropyOp), reduction) {}

 protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    HTShape output_shape = {};
    for (size_t i = 0; i < inputs[0]->ndim() - 1; ++i) {
      output_shape.emplace_back(inputs[0]->shape(i));
    }
    NDArrayMeta out_meta = inputs[0]->meta();
    if (_reduction != kNONE)
      out_meta.set_shape({1});
    else
      out_meta.set_shape(output_shape);
    return {out_meta};
  };

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta,
                      const InstantiationContext& inst_ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

  TensorList DoGradient(Operator& op, const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const override;

 public:
  inline bool require_contig_inputs() const override {
    return false;
  }

  bool operator==(const OpInterface& rhs) const override {
    return LossOpImpl::operator==(rhs);
  }
};

Tensor MakeSoftmaxCrossEntropyOp(Tensor preds, Tensor labels,
                                 ReductionType reduction = kMEAN,
                                 OpMeta op_meta = OpMeta());

Tensor MakeSoftmaxCrossEntropyOp(Tensor preds, Tensor labels,
                                 const std::string& reduction = "mean",
                                 OpMeta op_meta = OpMeta());

class SoftmaxCrossEntropyGradientOpImpl final : public LossGradientOpImpl {
 public:
  SoftmaxCrossEntropyGradientOpImpl(ReductionType reduction = kMEAN)
  : LossGradientOpImpl(quote(SoftmaxCrossEntropyGradientOp), reduction) {}

 protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    HT_ASSERT(_reduction == kSUM || _reduction == kMEAN || _reduction == kNONE)
      << "Unsupported reduction type \'" << _reduction << "\' for " << type()
      << " operators. Expected: [\'mean\', \'sum\', \'none\']";
    return {inputs[0]->meta()};
  };

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta,
                      const InstantiationContext& inst_ctx) const override;
  
  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const override;

 public:
  inline bool require_contig_inputs() const override {
    return false;
  }

  bool operator==(const OpInterface& rhs) const override {
    return LossGradientOpImpl::operator==(rhs);
  }
};

Tensor MakeSoftmaxCrossEntropyGradientOp(Tensor preds, Tensor labels, Tensor grad_output,
                                         ReductionType reduction = kMEAN,
                                         OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu
