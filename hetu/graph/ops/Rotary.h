#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"
#include "hetu/graph/ops/Loss.h"

namespace hetu {
namespace graph {

class RotaryOpImpl;
class RotaryOp;
class RotaryGradientOpImpl;
class RotaryGradientOp;

class RotaryOpImpl final : public OpInterface {
 public:
  RotaryOpImpl(bool interleaved, bool inplace)
  : OpInterface(quote(RotaryOp)), 
  _interleaved(interleaved), _inplace(inplace) {}

  inline bool interleaved() const {
    return _interleaved;
  }

  inline bool inplace() const {
    return _inplace;
  }

 protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    HT_ASSERT(inputs.at(0)->ndim() == 4);
    HT_ASSERT(inputs.at(1)->ndim() == 2);
    HT_ASSERT(inputs.at(2)->ndim() == 2);
    return {inputs.at(0)->meta()};
  };

  NDArrayList DoCompute(Operator& op,
                        const NDArrayList& inputs,
                        RuntimeContext& ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

  TensorList DoGradient(Operator& op, const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const override;

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta,
                      const InstantiationContext& inst_ctx) const override;

  void DoDeduceHeterProp(const std::vector<int32_t>& inputs_hetero_dim,
                         TensorList& outputs, const OpMeta& op_meta,
                         const InstantiationContext& inst_ctx) const override;  

 public:
  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const RotaryOpImpl&>(rhs);
      return interleaved() == rhs_.interleaved()
             && inplace() == rhs_.inplace();
    } else
      return false;
  }

  bool _interleaved;
  bool _inplace;
};

Tensor MakeRotaryOp(Tensor x, Tensor cos, Tensor sin,
                    bool interleaved = false, bool inplace = false,
                    OpMeta op_meta = OpMeta());

class RotaryGradientOpImpl final : public OpInterface {
 public:
  RotaryGradientOpImpl(bool interleaved, bool inplace)
  : OpInterface(quote(RotaryGradientOp)),
  _interleaved(interleaved), _inplace(inplace) {}

  inline bool interleaved() const {
    return _interleaved;
  }

  inline bool inplace() const {
    return _inplace;
  }

 protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    return {inputs[0]->meta()};
  };

  NDArrayList DoCompute(Operator& op,
                        const NDArrayList& inputs,
                        RuntimeContext& ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const override;

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta,
                      const InstantiationContext& inst_ctx) const override;

  void DoDeduceHeterProp(const std::vector<int32_t>& inputs_hetero_dim,
                         TensorList& outputs, const OpMeta& op_meta,
                         const InstantiationContext& inst_ctx) const override;  

 public:
  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const RotaryGradientOpImpl&>(rhs);
      return interleaved() == rhs_.interleaved()
             && inplace() == rhs_.inplace();
    } else
      return false;
  }

  bool _interleaved;
  bool _inplace;
};

Tensor MakeRotaryGradientOp(Tensor dout, Tensor cos, Tensor sin,
                            bool interleaved = false, bool inplace = false,
                            OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu
