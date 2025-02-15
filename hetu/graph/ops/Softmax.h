#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"

namespace hetu {
namespace graph {

class SoftmaxOpImpl;
class SoftmaxOp;
class SoftmaxGradientOpImpl;
class SoftmaxGradientOp;

class SoftmaxOpImpl final : public OpInterface {
 private:
  friend class SoftmaxOp;
  struct constructor_access_key {};

 public:
  SoftmaxOpImpl(int64_t dim)
  : OpInterface(quote(SoftmaxOp)),
  _dim(dim) {
  }

  int64_t get_dim() const {
    return _dim;
  }

 protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    HT_ASSERT(-int64_t(inputs[0]->ndim()) <= get_dim() && get_dim() < int64_t(inputs[0]->ndim()));
    return {inputs[0]->meta()};
  };

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta,
                      const InstantiationContext& inst_ctx) const override;  

  void DoDeduceHeterProp(const std::vector<int32_t>& inputs_hetero_dim,
                         TensorList& outputs, const OpMeta& op_meta,
                         const InstantiationContext& inst_ctx) const override;  

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

  TensorList DoGradient(Operator& op, const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const override;

  int64_t _dim;

 public:
  inline bool require_contig_inputs() const override {
    return false;
  }

  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const SoftmaxOpImpl&>(rhs);
      return (get_dim() == rhs_.get_dim());
    }
    return false;
  }
};

Tensor MakeSoftmaxOp(Tensor input, int64_t dim, OpMeta op_meta = OpMeta());

class SoftmaxGradientOpImpl final : public OpInterface {
 private:
  friend class SoftmaxGradientOp;
  struct constructor_access_key {};

 public:
  SoftmaxGradientOpImpl(int64_t dim)
  : OpInterface(quote(SoftmaxGradientOp)),
  _dim(dim) {
  }

  int64_t get_dim() const {
    return _dim;
  }

 protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    return {inputs[0]->meta()};
  };

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta,
                      const InstantiationContext& inst_ctx) const override;

  void DoDeduceHeterProp(const std::vector<int32_t>& inputs_hetero_dim,
                         TensorList& outputs, const OpMeta& op_meta,
                         const InstantiationContext& inst_ctx) const override;  
  
  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const override;

  int64_t _dim;

 public:
  inline bool require_contig_inputs() const override {
    return false;
  }

  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const SoftmaxGradientOpImpl&>(rhs);
      return (get_dim() == rhs_.get_dim());
    }
    return false;
  }
};

Tensor MakeSoftmaxGradientOp(Tensor input, Tensor grad_output,
                             int64_t dim, OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu
