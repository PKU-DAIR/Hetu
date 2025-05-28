#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"

namespace hetu {
namespace graph {

class PadOpImpl;
class PadOp;
class PadGradientOpImpl;
class PadGradientOp;

class PadOpImpl final : public OpInterface {
 public:
  PadOpImpl(const HTShape& paddings,
            const std::string& mode, double constant)
  : OpInterface(quote(PadOp)),
    _mode(mode),
    _paddings(paddings),
    _constant(constant) {
    HT_ASSERT(_mode == "constant")
      << "Now we only support padding mode \'constant\'";
    for (size_t i = 0; i < paddings.size(); ++i) {
      HT_ASSERT(paddings[i] >= 0)
      << "padding in dim " << i << " < 0,"
      << "which has value" << paddings[i];
    }
  }

  const std::string& get_mode() const {
    return _mode;
  }

  HTShape get_paddings() const {
    return _paddings;
  }

  double get_constant() const {
    return _constant;
  }

 protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    HTShape shape;
    if (inputs[0]->has_shape()) {
      shape = inputs[0]->shape();
      size_t len = _paddings.size();
      for (size_t i = 0; i < 4; ++i) {
        if (i >= (4 - len / 2)) {
          shape[i] = shape[i] + _paddings[(i - (4 - len / 2)) * 2] +
            _paddings[(i - (4 - len / 2)) * 2 + 1];
        }
      }
    }
    NDArrayMeta output_meta = NDArrayMeta().set_dtype(inputs[0]->dtype())
                                           .set_shape(shape)
                                           .set_device(inputs[0]->device());
    return {output_meta};
  };

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta,
                      const InstantiationContext& inst_ctx) const override;
  
  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

  TensorList DoGradient(Operator& op, const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const override;

  std::string _mode;

  HTShape _paddings;

  double _constant;

 public:
  inline bool require_contig_inputs() const override {
    return false;
  }

  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const PadOpImpl&>(rhs);
      return (get_mode() == rhs_.get_mode()
              && get_constant() == rhs_.get_constant()
              && get_paddings() == rhs_.get_paddings());
    }
    return false;
  }
};

Tensor MakePadOp(Tensor input, const HTShape& paddings, std::string mode, double constant,
                 OpMeta op_meta = OpMeta());

class PadGradientOpImpl final : public OpInterface {

 public:
  PadGradientOpImpl(const HTShape& paddings, const std::string& mode)
  : OpInterface(quote(PadGradientOp)),
    _mode(mode),
    _paddings(paddings) {
  }

  const std::string& get_mode() const {
    return _mode;
  }

  HTShape get_paddings() const {
    return _paddings;
  }

 protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    HTShape shape = inputs[0]->shape();
    size_t len = _paddings.size();
    for (size_t i = 0; i < 4; ++i) {
      if (i >= (4 - len / 2)) {
        shape[i] = shape[i] - _paddings[(i - (4 - len / 2)) * 2] -
          _paddings[(i - (4 - len / 2)) * 2 + 1];
      }
    }
    NDArrayMeta output_meta = NDArrayMeta().set_dtype(inputs[0]->dtype())
                                           .set_shape(shape)
                                           .set_device(inputs[0]->device());
    return {output_meta};
  };

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const override;

  std::string _mode;

  HTShape _paddings;

 public:
  inline bool require_contig_inputs() const override {
    return false;
  }

  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const PadGradientOpImpl&>(rhs);
      return (get_mode() == rhs_.get_mode()
              && get_paddings() == rhs_.get_paddings());
    }
    return false;
  }
};

Tensor MakePadGradientOp(Tensor grad_output, const HTShape& paddings, std::string mode,
                         OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu
