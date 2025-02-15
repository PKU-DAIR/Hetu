#pragma once

#include "hetu/graph/operator.h"

namespace hetu {
namespace graph {

class PlaceholderOpImpl final : public OpInterface {
 public:
  PlaceholderOpImpl(const NDArrayMeta& data_meta)
  : OpInterface(quote(PlaceholderOp)), _data_meta(std::move(data_meta)) {}

  uint64_t op_indicator() const noexcept override {
    return PLACEHOLDER_OP;
  }

 protected:
  std::vector<NDArrayMeta>
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    return {_data_meta};
  }

  TensorList DoGradient(Operator& op,
                        const TensorList& grad_outputs) const override {
    return {Tensor()};
  }

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override {
    HT_RUNTIME_ERROR << "Should not call InferShape fn of " << type();
    __builtin_unreachable();
  }

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override {
    HT_RUNTIME_ERROR << "Should not call Compute fn of " << type();
    __builtin_unreachable();
  }

 public:
  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const PlaceholderOpImpl&>(rhs);
      return rhs_.data_meta() == data_meta();
    }
    return false;
  }

  const NDArrayMeta data_meta() const {
    return _data_meta;
  }

 protected:
  NDArrayMeta _data_meta;
};

Tensor MakePlaceholderOp(NDArrayMeta data_meta, const DistributedStatesHierarchy& ds_hierarchy = DistributedStatesHierarchy(), OpMeta op_meta = OpMeta());

Tensor MakeParallelPlaceholderOp(NDArrayMeta data_meta, const DistributedStatesHierarchy& ds_hierarchy, OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu
