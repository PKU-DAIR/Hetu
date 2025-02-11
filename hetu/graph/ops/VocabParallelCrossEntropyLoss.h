#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"
#include "hetu/graph/ops/Loss.h"

namespace hetu {
namespace graph {

class VocabParallelCrossEntropyOpImpl;
class VocabParallelCrossEntropyOp;
class VocabParallelCrossEntropyGradientOpImpl;
class VocabParallelCrossEntropyGradientOp;

// input: [batch_size * seq_len, vocab_size], splited by tp in vocab_size dimension
// target: [batch_size, seq_len], duplicate
class VocabParallelCrossEntropyOpImpl : public LossOpImpl {
 public:
  VocabParallelCrossEntropyOpImpl(const int64_t ignored_index = -1, 
                                  ReductionType reduction = kMEAN)
  : LossOpImpl(quote(VocabParallelCrossEntropyOp), reduction),
    _ignored_index(ignored_index) {}

  int64_t ignored_index() const {
    return _ignored_index;
  }

  static DeviceGroup get_devices_by_dim(const Tensor& input, int32_t dim);
  
 protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    HTShape output_shape = {inputs.at(0)->shape(0), 1};
    NDArrayMeta out_meta = inputs[0]->meta();
    if (_reduction != kNONE)
      out_meta.set_shape({1});
    else
      out_meta.set_shape(output_shape);
    out_meta.set_device(inputs[0]->device());
    return {out_meta};
  };

  bool DoInstantiate(Operator& op, const Device& placement,
                     StreamIndex stream_index) const override;

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
  // DeviceGroup _comm_group;
};

Tensor MakeVocabParallelCrossEntropyOp(Tensor preds, Tensor labels,
                                       const int64_t ignored_index = -1, 
                                       ReductionType reduction = kMEAN,
                                       OpMeta op_meta = OpMeta());

Tensor MakeVocabParallelCrossEntropyOp(Tensor preds, Tensor labels,
                                       const int64_t ignored_index = -1, 
                                       const std::string& reduction = "mean",
                                       OpMeta op_meta = OpMeta());

class VocabParallelCrossEntropyGradientOpImpl : public LossGradientOpImpl {

 public:
  VocabParallelCrossEntropyGradientOpImpl(const int64_t ignored_index = -1, 
                                          ReductionType reduction = kMEAN)
  : LossGradientOpImpl(quote(VocabParallelCrossEntropyGradientOp), reduction),
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
};

Tensor MakeVocabParallelCrossEntropyGradientOp(Tensor preds, Tensor labels, Tensor grad_output,
                                               const int64_t ignored_index = -1, 
                                               ReductionType reduction = kMEAN,
                                               OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu
