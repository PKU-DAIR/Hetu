#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"

namespace hetu {
namespace graph {

class InstanceNormOpImpl;
class InstanceNormOp;
class InstanceNormGradientOpImpl;
class InstanceNormGradientOp;

class InstanceNormOpImpl final : public OpInterface {
 public:
  InstanceNormOpImpl(double eps = 1e-7)
  : OpInterface(quote(InstanceNormOp)), _eps(eps) {
  }

  double get_eps() const {
    return _eps;
  }

protected:
  std::vector<NDArrayMeta>
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    HT_ASSERT_HAS_DIMS(inputs[0], 4);
    HTShape local_shape = inputs[0]->shape();
    local_shape[3] = 1;
    local_shape[2] = 1;
    NDArrayMeta output_meta = NDArrayMeta().set_dtype(inputs[0]->dtype())
                                           .set_shape(local_shape)
                                           .set_device(inputs[0]->device());
    return {inputs[0]->meta(), output_meta, output_meta};
  }

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta,
                      const InstantiationContext& inst_ctx) const override;

  TensorList DoGradient(Operator& op,
                        const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;

  double _eps;
 public:
  inline bool require_contig_inputs() const override {
    return false;
  }

  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const InstanceNormOpImpl&>(rhs);
      return (get_eps() == rhs_.get_eps());
    }
    return false;
  }
};

TensorList MakeInstanceNormOp(Tensor input, double eps = 1e-7,
                              OpMeta op_meta = OpMeta());

class InstanceNormGradientOpImpl final : public OpInterface {

 public:
  InstanceNormGradientOpImpl(double eps = 1e-7)
  : OpInterface(quote(InstanceNormGradientOp)),
  _eps(eps) {
  }

  double get_eps() const {
    return _eps;
  }

protected:
  std::vector<NDArrayMeta>
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    return {inputs[0]->meta()};
  }

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta,
                      const InstantiationContext& inst_ctx) const override;
  
  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;

  double _eps;
 public:
  inline bool require_contig_inputs() const override {
    return false;
  }

  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const InstanceNormOpImpl&>(rhs);
      return (get_eps() == rhs_.get_eps());
    }
    return false;
  }
};

Tensor MakeInstanceNormGradientOp(Tensor output_grad, Tensor input,
                                  Tensor save_mean, Tensor save_var,
                                  double eps = 1e-7,
                                  OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu
