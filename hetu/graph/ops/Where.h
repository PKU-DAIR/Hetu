#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"

namespace hetu {
namespace graph {

class WhereOpImpl;
class WhereOp;

class WhereOpImpl final : public OpInterface {
 private:
  friend class WhereOp;
  struct constructor_access_key {};

 public:
  WhereOpImpl(bool inplace)
  : OpInterface(quote(WhereOp)), _inplace(inplace) {
  }

  inline bool inplace() const {
    return _inplace;
  }

  inline uint64_t inplace_pos() const {
    return 1;
  }

  inline bool inplace_at(size_t input_position) const override {
    return inplace() && input_position == inplace_pos();
  }

  inline uint64_t op_indicator() const noexcept override {
    return _inplace ? INPLACE_OP : 0;
  }

 protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    return {inputs[1]->meta()};
  };

  NDArrayList DoCompute(Operator& op,
                        const NDArrayList& inputs,
                        RuntimeContext& ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

  TensorList DoGradient(Operator& op, const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const override;

  HTShapeList DoInferDynamicShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const override;

  bool _inplace;
 public:
  inline bool require_contig_inputs() const override {
    return false;
  }

  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const WhereOpImpl&>(rhs);
      return (inplace() == rhs_.inplace());
    }
    return false;
  }
};

Tensor MakeWhereOp(Tensor cond, Tensor inputA, Tensor inputB,
                   OpMeta op_meta = OpMeta());

Tensor MakeWhereInplaceOp(Tensor cond, Tensor inputA, Tensor inputB,
                          OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu
