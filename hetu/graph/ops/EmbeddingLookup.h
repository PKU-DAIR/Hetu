#pragma once

#include "hetu/graph/graph.h"
#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"

namespace hetu {
namespace graph {

class EmbeddingLookupOpImpl;
class EmbeddingLookupOp;
class EmbeddingLookupGradientOpImpl;
class EmbeddingLookupGradientOp;

class EmbeddingLookupOpImpl : public OpInterface {
 public:
  EmbeddingLookupOpImpl(std::vector<int64_t> multi_offset = {0})
  : OpInterface(quote(EmbeddingLookupOp)), _multi_offset(multi_offset) {
  }

protected:
  std::vector<NDArrayMeta>
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    HTShape shape;
    if (inputs[0]->has_shape() && inputs[1]->has_shape()) {
      HT_ASSERT_HAS_DIMS(inputs[0], 2);
      shape = inputs[1]->shape();
      shape.emplace_back(inputs[0]->shape(1));
    }
    NDArrayMeta output_meta = NDArrayMeta().set_dtype(inputs[0]->dtype())
                                           .set_shape(shape)
                                           .set_device(inputs[0]->device());
    return {output_meta};
  }

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta,
                      const InstantiationContext& inst_ctx) const override;

  void DoDeduceHeterProp(const std::vector<int32_t>& inputs_hetero_dim,
                         TensorList& outputs, const OpMeta& op_meta,
                         const InstantiationContext& inst_ctx) const override;  

  TensorList DoGradient(Operator& op,
                        const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;

  void DoSaveCtxForBackward(const TensorList& inputs, ContextStore& dst_ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;
 public:
  inline bool require_contig_inputs() const override {
    return false;
  }

  bool operator==(const OpInterface& rhs) const override {
    return OpInterface::operator==(rhs);
  }

  // for multi ds
  int offset(Operator& op) const {
    auto& graph = op->graph();
    HT_ASSERT(_multi_offset.size() == 1 || _multi_offset.size() == graph.NUM_STRATEGY)
      << "EmbeddingLookupOp get offset error!";
    if (_multi_offset.size() == 1) {
      return _multi_offset[0];
    } else {
      return _multi_offset[graph.COMPUTE_STRATEGY_ID];
    }
  }

 protected:
  std::vector<int64_t> _multi_offset;
};

Tensor MakeEmbeddingLookupOp(Tensor input, Tensor id, std::vector<int64_t> multi_offset={0}, OpMeta op_meta = OpMeta());

class EmbeddingLookupGradientOpImpl : public OpInterface {
 public:
  EmbeddingLookupGradientOpImpl(std::vector<int64_t> multi_offset, OpMeta op_meta = OpMeta())
  : OpInterface(quote(EmbeddingLookupGradientOp)), _multi_offset(multi_offset) {
  }

protected:
  std::vector<NDArrayMeta>
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    return {inst_ctx.get<NDArrayMeta>("in_meta")};
  }

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta,
                      const InstantiationContext& inst_ctx) const override;
  
  void DoDeduceHeterProp(const std::vector<int32_t>& inputs_hetero_dim,
                         TensorList& outputs, const OpMeta& op_meta,
                         const InstantiationContext& inst_ctx) const override;  

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;

  void DoLoadCtxForBackward(ContextStore& src_ctx, ContextStore& dst_ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;

 public:
  inline bool require_contig_inputs() const override {
    return false;
  }

  bool operator==(const OpInterface& rhs) const override {
    return OpInterface::operator==(rhs);
  }

  // for multi ds
  int offset(Operator& op) const {
    auto& graph = op->graph();
    HT_ASSERT(_multi_offset.size() == 1 || _multi_offset.size() == graph.NUM_STRATEGY)
      << "EmbeddingLookupOp get offset error!";
    if (_multi_offset.size() == 1) {
      return _multi_offset[0];
    } else {
      return _multi_offset[graph.COMPUTE_STRATEGY_ID];
    }
  }

 protected:
  std::vector<int64_t> _multi_offset;
};

Tensor MakeEmbeddingLookupGradientOp(Tensor grad_output, Tensor id, Tensor ori_input,
                                     std::vector<int64_t> multi_offset, OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu
