#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"
#include "hetu/graph/ops/Loss.h"

namespace hetu {
namespace graph {

class SoftmaxCrossEntropySparseOpImpl;
class SoftmaxCrossEntropySparseOp;
class SoftmaxCrossEntropySparseGradientOpImpl;
class SoftmaxCrossEntropySparseGradientOp;

class SoftmaxCrossEntropySparseOpImpl final : public LossOpImpl {
 public:
  SoftmaxCrossEntropySparseOpImpl(const int64_t ignored_index = -1, 
                          ReductionType reduction = kMEAN)
  : LossOpImpl(quote(SoftmaxCrossEntropySparseOp), reduction),
    _ignored_index(ignored_index) {}

  int64_t ignored_index() const {
    return _ignored_index;
  }

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
    out_meta.set_device(inputs[0]->device());
    return {out_meta};
  };

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta,
                      const InstantiationContext& inst_ctx) const override;

  void DoDeduceHeterProp(const std::vector<int32_t>& inputs_hetero_dim,
                         TensorList& outputs, const OpMeta& op_meta,
                         const InstantiationContext& inst_ctx) const override;  

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

  TensorList DoGradient(Operator& op, const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const override;

  int64_t _ignored_index;

 public:
  inline bool require_contig_inputs() const override {
    return false;
  }

  bool operator==(const OpInterface& rhs) const override {
    if (LossOpImpl::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const SoftmaxCrossEntropySparseOpImpl&>(rhs);
      return ignored_index() == rhs_.ignored_index();
    }
    return false;
  }
};

Tensor MakeSoftmaxCrossEntropySparseOp(Tensor preds, Tensor labels, const int64_t ignored_index = -1, 
                                       ReductionType reduction = kMEAN,
                                       OpMeta op_meta = OpMeta());

Tensor MakeSoftmaxCrossEntropySparseOp(Tensor preds, Tensor labels, const int64_t ignored_index = -1, 
                                       const std::string& reduction = "mean",
                                       OpMeta op_meta = OpMeta());

class SoftmaxCrossEntropySparseGradientOpImpl final : public LossGradientOpImpl {

 public:
  SoftmaxCrossEntropySparseGradientOpImpl(const int64_t ignored_index = -1, 
                                          ReductionType reduction = kMEAN)
  : LossGradientOpImpl(quote(SoftmaxCrossEntropySparseGradientOp), reduction),
    _ignored_index(ignored_index) {}

  ino64_t ignored_index() const {
    return _ignored_index;
  }

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

  void DoDeduceHeterProp(const std::vector<int32_t>& inputs_hetero_dim,
                         TensorList& outputs, const OpMeta& op_meta,
                         const InstantiationContext& inst_ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const override;

  int64_t _ignored_index; 

 public:
  inline bool require_contig_inputs() const override {
    return false;
  }

  bool operator==(const OpInterface& rhs) const override {
    if (LossGradientOpImpl::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const SoftmaxCrossEntropySparseGradientOpImpl&>(rhs);
      return ignored_index() == rhs_.ignored_index();
    }
    return false;
  }
};

Tensor MakeSoftmaxCrossEntropySparseGradientOp(Tensor preds, Tensor labels, Tensor grad_output,
                                               const int64_t ignored_index = -1, 
                                               ReductionType reduction = kMEAN,
                                               OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu
