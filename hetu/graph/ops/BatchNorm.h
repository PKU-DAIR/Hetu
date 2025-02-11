#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"

namespace hetu {
namespace graph {

class BatchNormOpImpl;
class BatchNormGradientOpImpl;

class BatchNormOpImpl final : public OpInterface {
 private:
  friend class BatchNormOp;
  struct constructor_access_key {};

 public:
  BatchNormOpImpl(double momentum = 0.1, double eps = 1e-5,
                  OpMeta op_meta = OpMeta())
  : OpInterface(quote(BatchNormOp)),
    _momentum(momentum),
    _eps(eps) {
  }

  double get_momentum() const {
    return _momentum;
  }

  double get_eps() const {
    return _eps;
  }

protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    int64_t channels = inputs[0]->shape(1);
    HTShape shape = {channels};
    if (inputs[0]->dtype() == DataType::FLOAT16 || inputs[0]->dtype() == DataType::BFLOAT16) {
      HT_ASSERT_TENSORS_SAME_DTYPE(TensorList(inputs.begin() + 1, inputs.end()));
    }
    else {
      HT_ASSERT_TENSORS_SAME_DTYPE(inputs);
    }
    HT_ASSERT_HAS_DIMS(inputs[0], 4);
    HT_ASSERT_TENSORS_SAME_SHAPE(inputs[1], inputs[2]);
    NDArrayMeta output_meta = NDArrayMeta().set_dtype(inputs[1]->dtype())
                                           .set_shape(shape)
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
  
  double _momentum;

  double _eps;

 public:
  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const BatchNormOpImpl&>(rhs);
      return (get_momentum() == rhs_.get_momentum()
              && get_eps() == rhs_.get_eps()); 
    }
    return false;
  }
};

TensorList MakeBatchNormOp(Tensor input, Tensor bn_scale, Tensor bn_bias,
                           Tensor running_mean, Tensor running_var,
                           double momentum = 0.1, double eps = 1e-5,
                           OpMeta op_meta = OpMeta());

class BatchNormGradientOpImpl final : public OpInterface {

 public:
  BatchNormGradientOpImpl(double eps,
                          OpMeta op_meta = OpMeta())
  : OpInterface(quote(BatchNormGradientOp)),
    _eps(eps) {
  }

  double get_eps() const {
    return _eps;
  }

protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    return {inputs[1]->meta(), inputs[2]->meta(), inputs[2]->meta()};
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
  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const BatchNormOpImpl&>(rhs);
      return (get_eps() == rhs_.get_eps()); 
    }
    return false;
  }
};

TensorList MakeBatchNormGradientOp(Tensor output_grad, Tensor input, Tensor bn_scale,
                                   Tensor save_mean, Tensor save_var, double eps,
                                   OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu
