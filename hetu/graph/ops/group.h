#pragma once

#include "hetu/graph/operator.h"

namespace hetu {
namespace graph {

class GroupOpImpl final : public OpInterface {
 public:
  GroupOpImpl() : OpInterface(quote(GroupOp)) {}

  uint64_t op_indicator() const noexcept {
    return GROUP_OP;
  }

 protected:
  std::vector<NDArrayMeta> DoInferMeta(const TensorList&, const InstantiationContext&) const override {
    return {};
  }

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta,
                      const InstantiationContext& inst_ctx) const {}  

  void DoDeduceHeterProp(const std::vector<int32_t>& inputs_hetero_dim,
                         TensorList& outputs, const OpMeta& op_meta,
                         const InstantiationContext& inst_ctx) const {}  

  void DoCompute(Operator&, const NDArrayList&, NDArrayList&,
                 RuntimeContext&) const {}
};

Tensor MakeGroupOp(OpMeta op_meta = OpMeta());

Tensor MakeGroupOp(TensorList deps, OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu
