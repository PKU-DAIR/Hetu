#pragma once

#include "hetu/graph/operator.h"

namespace hetu {
namespace graph {

class UnaryOpImpl;
class UnaryGradientOpImpl;

class UnaryOpImpl : public OpInterface {
 protected:
  UnaryOpImpl(OpType&& op_type, bool inplace = false)
  : OpInterface(std::move(op_type)), _inplace(inplace) {}
 
 public:
  inline bool require_contig_inputs() const override {
    return false;
  }

  inline bool inplace() const {
    return _inplace;
  }

  inline uint64_t inplace_pos() const {
    return 0;
  }

  inline bool inplace_at(size_t input_position) const override {
    return inplace() && input_position == inplace_pos();
  }

  inline uint64_t op_indicator() const noexcept override {
    return _inplace ? INPLACE_OP : 0;
  }

  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const UnaryOpImpl&>(rhs);
      return inplace() == rhs_.inplace();
    }
    return false;
  }

 protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    return {inputs[0]->meta()};
  }

  HTShapeList DoInferShape(Operator& op, 
                           const HTShapeList& input_shapes, 
                           RuntimeContext& ctx) const override {
    return {input_shapes.at(0)};
  }

  bool _inplace;
};

class UnaryGradientOpImpl : public OpInterface {
 protected:
  UnaryGradientOpImpl(OpType&& op_type)
  : OpInterface(std::move(op_type)) {}
 
 public:
  inline bool require_contig_inputs() const override {
    return false;
  }

 protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    return {inputs[0]->meta()};
  }

  HTShapeList DoInferShape(Operator& op, 
                           const HTShapeList& input_shapes, 
                           RuntimeContext& ctx) const override {
    return {input_shapes.at(0)};
  }
};

} // namespace graph
} // namespace hetu