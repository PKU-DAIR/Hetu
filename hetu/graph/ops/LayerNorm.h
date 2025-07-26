#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"

namespace hetu {
namespace graph {

class LayerNormOpImpl;
class LayerNormOp;
class LayerNormGradientOpImpl;
class LayerNormGradientOp;
class FusedLayerNormOpImpl;
class FusedLayerNormOp;
class FusedLayerNormGradientOpImpl;
class FusedLayerNormGradientOp;

class LayerNormOpImpl final : public OpInterface {
 private:
  friend class LayerNormOp;
  struct constructor_access_key {};

 public:
  LayerNormOpImpl(const HTShape& normalized_shape, double eps = 0.01)
  : OpInterface(quote(LayerNormOp)),
  _normalized_shape(normalized_shape),
  _eps(eps) {
  }

  HTShape normalized_shape() const {
    return _normalized_shape;
  }

  double get_eps() const {
    return _eps;
  }

protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    HT_ASSERT_TENSORS_SAME_DTYPE(inputs);
    HT_ASSERT_TENSORS_SAME_SHAPE(inputs[1], inputs[2]);
    size_t dim = normalized_shape().size();
    HTShape output_shape = inputs[0]->shape();
    if (inputs[0]->has_shape()) 
      for (size_t i = 0; i < dim; ++i) {
        HT_ASSERT(normalized_shape()[dim - 1 - i] == inputs[0]->shape(inputs[0]->ndim() - 1 - i))
        << "Normalized shape's last dims should equal to input shape's.But we have normalized shape:"
        << normalized_shape() << " and input shape:" << inputs[0]->shape();
        output_shape[inputs[0]->ndim() - 1 - i] = 1;
      }
    NDArrayMeta output_meta = NDArrayMeta().set_dtype(inputs[0]->dtype())
                                           .set_shape(output_shape)
                                           .set_device(inputs[0]->device());
    return {inputs[0]->meta(), output_meta, output_meta};
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

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;

  HTShape _normalized_shape;

  double _eps;

 public:
  inline bool require_contig_inputs() const override {
    return false;
  }

  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const LayerNormOpImpl&>(rhs);
      return (get_eps() == rhs_.get_eps()
              && normalized_shape() == rhs_.normalized_shape()); 
    }
    return false;
  }
};

TensorList MakeLayerNormOp(Tensor input, Tensor ln_scale, Tensor ln_bias, HTShape normalized_shape, 
                           double eps = 0.01, OpMeta op_meta = OpMeta());

class LayerNormGradientOpImpl final : public OpInterface {
 public:
  LayerNormGradientOpImpl(HTShape normalized_shape, double eps)
  : OpInterface(quote(LayerNormGradientOp)),
  _normalized_shape(normalized_shape),
  _eps(eps) {
  }

  HTShape normalized_shape() const {
    return _normalized_shape;
  }

  double get_eps() const {
    return _eps;
  }

protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    return {inputs[0]->meta(), inputs[2]->meta(), inputs[2]->meta()};
  }

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta,
                      const InstantiationContext& inst_ctx) const override;  

  void DoDeduceHeterProp(const std::vector<int32_t>& inputs_hetero_dim,
                         TensorList& outputs, const OpMeta& op_meta,
                         const InstantiationContext& inst_ctx) const override;  

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;

  HTShape _normalized_shape;

  double _eps;

 public:
  inline bool require_contig_inputs() const override {
    return false;
  }

  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const LayerNormGradientOpImpl&>(rhs);
      return (get_eps() == rhs_.get_eps()); 
    }
    return false;
  }
};

TensorList MakeLayerNormGradientOp(Tensor output_grad, Tensor input, Tensor ln_scale,
                                   Tensor save_mean, Tensor save_var, HTShape normalized_shape,
                                   double eps, OpMeta op_meta = OpMeta());

class FusedLayerNormOpImpl final : public OpInterface {
 private:
  friend class FusedLayerNormOp;
  struct constructor_access_key {};

 public:
  FusedLayerNormOpImpl(const HTShape& normalized_shape, double eps = 0.01, bool inplace = false)
  : OpInterface(quote(FusedLayerNormOp)),
  _normalized_shape(normalized_shape),
  _eps(eps),
  _inplace(inplace) {
  }

  HTShape normalized_shape() const {
    return _normalized_shape;
  }

  double get_eps() const {
    return _eps;
  }

  bool inplace() const {
    return _inplace;
  }

protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    HT_ASSERT_TENSORS_SAME_DTYPE(inputs);
    HT_ASSERT_TENSORS_SAME_SHAPE(inputs[1], inputs[2]);
    size_t dim = normalized_shape().size();
    HTShape output_shape = inputs[0]->shape();
    if (inputs[0]->has_shape()) 
      for (size_t i = 0; i < dim; ++i) {
        HT_ASSERT(normalized_shape()[dim - 1 - i] == inputs[0]->shape(inputs[0]->ndim() - 1 - i))
        << "Normalized shape's last dims should equal to input shape's.But we have normalized shape:"
        << normalized_shape() << " and input shape:" << inputs[0]->shape();
        output_shape[inputs[0]->ndim() - 1 - i] = 1;
      }
    NDArrayMeta output_meta = NDArrayMeta().set_dtype(kFloat32)
                                           .set_shape(output_shape)
                                           .set_device(inputs[0]->device());
    return {inputs[0]->meta(), output_meta, output_meta};
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

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;

  HTShape _normalized_shape;

  double _eps;

  bool _inplace;

 public:
  inline bool require_contig_inputs() const override {
    return false;
  }

  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const FusedLayerNormOpImpl&>(rhs);
      return (get_eps() == rhs_.get_eps()
              && normalized_shape() == rhs_.normalized_shape()); 
    }
    return false;
  }
};

TensorList MakeFusedLayerNormOp(Tensor input, Tensor ln_scale, Tensor ln_bias, HTShape normalized_shape, 
                                double eps = 0.01, bool inplace = false, OpMeta op_meta = OpMeta());

class FusedLayerNormGradientOpImpl final : public OpInterface {
 public:
  FusedLayerNormGradientOpImpl(HTShape normalized_shape, double eps, bool inplace)
  : OpInterface(quote(FusedLayerNormGradientOp)),
  _normalized_shape(normalized_shape),
  _eps(eps),
  _inplace(inplace) {
  }

  HTShape normalized_shape() const {
    return _normalized_shape;
  }

  double get_eps() const {
    return _eps;
  }

  bool inplace() const {
    return _inplace;
  }

protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    return {inputs[0]->meta(), inputs[2]->meta(), inputs[2]->meta()};
  }

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta,
                      const InstantiationContext& inst_ctx) const override;  

  void DoDeduceHeterProp(const std::vector<int32_t>& inputs_hetero_dim,
                         TensorList& outputs, const OpMeta& op_meta,
                         const InstantiationContext& inst_ctx) const override;  

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;

  HTShape _normalized_shape;

  double _eps;

  bool _inplace;

 public:
  inline bool require_contig_inputs() const override {
    return false;
  }

  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const FusedLayerNormGradientOpImpl&>(rhs);
      return (get_eps() == rhs_.get_eps()); 
    }
    return false;
  }
};

TensorList MakeFusedLayerNormGradientOp(Tensor output_grad, Tensor input, Tensor ln_scale, Tensor ln_bias,
                                        Tensor save_mean, Tensor save_var, HTShape normalized_shape,
                                        double eps, bool inplace, OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu
