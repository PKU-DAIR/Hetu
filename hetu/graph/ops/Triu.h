#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"

namespace hetu {
namespace graph {

class TriuTrilOpImpl;
class TriuTrilOp;

class TriuTrilOpImpl final : public OpInterface {
 private:
  friend class TriuTrilOp;
  struct constructor_access_key {};

 public:
  TriuTrilOpImpl(bool lower = false, int64_t diagonal = 0)
  : OpInterface(quote(TriuTrilOp)),
  _lower(lower),
  _diagonal(diagonal) {
  }

  inline bool lower() const {
    return _lower;
  }

  inline int64_t diagonal() const {
    return _diagonal;
  }

 protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    return {inputs[0]->meta()};
  };

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta,
                      const InstantiationContext& inst_ctx) const override;  

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

  TensorList DoGradient(Operator& op, const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const override;

  bool _lower;

  int64_t _diagonal;

 public:
  inline bool require_contig_inputs() const override {
    return false;
  }

  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const TriuTrilOpImpl&>(rhs);
      return (lower() == rhs_.lower()
              && diagonal() == rhs_.diagonal());
    }
    return false;
  }
};

Tensor MakeTriuTrilOp(Tensor input,  bool lower = false, int64_t diagonal = 0,  
                      OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu
