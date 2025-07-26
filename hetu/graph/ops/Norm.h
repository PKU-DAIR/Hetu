#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"

namespace hetu {
namespace graph {

class NormOpImpl;
class NormOp;
class NormGradientOpImpl;
class NormGradientOp;

class NormOpImpl final : public OpInterface {
 public:
  NormOpImpl(int64_t p = 1, int64_t dim = 0, bool keepdim = false)
  : OpInterface(quote(NormOp)),
  _p(p),
  _dim(dim),
  _keepdim(keepdim) {
    HT_ASSERT(p > 0)
    << "p is " << p << "and it should be greater than 0.";
  }

  inline int64_t getp() const{
    return _p;
  }

  inline int64_t dim() const{
    return _dim;
  }

  inline bool keepdim() const{
    return _keepdim;
  }
  
 protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    HTShape outshape = inputs[0]->shape();
    if (inputs[0]->has_shape()) {
      int64_t axi = dim() >= 0? dim(): dim() + outshape.size();
      HT_ASSERT(axi >= 0 && axi < int64_t(outshape.size()))
      << "reduce dim should in [0, " << outshape.size() 
      << "], but now it is " << axi;
      if (keepdim()) 
        outshape[axi] = 1;
      else 
        outshape.erase(outshape.begin() + axi);
    }
    NDArrayMeta output_meta = NDArrayMeta().set_dtype(inputs[0]->dtype())
                                           .set_shape(outshape)
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

  int64_t _p;

  int64_t _dim;

  bool _keepdim;

 public:
  inline bool require_contig_inputs() const override {
    return false;
  }

  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const NormOpImpl&>(rhs);
      return (getp() == rhs_.getp()
              && dim() == rhs_.dim()
              && keepdim() == rhs_.keepdim());
    }
    return false;
  }
};

Tensor MakeNormOp(Tensor input, int64_t p = 1, int64_t dim = 0, 
                  bool keepdim = false, OpMeta op_meta = OpMeta());

class NormGradientOpImpl final : public OpInterface {
 public:
  NormGradientOpImpl(int64_t p, int64_t dim)
  : OpInterface(quote(NormGradientOp)),
  _p(p),
  _dim(dim) {
  }

  inline int64_t getp() const{
    return _p;
  }

  inline int64_t dim() const{
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

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const override;

  int64_t _p;

  int64_t _dim;

 public:
  inline bool require_contig_inputs() const override {
    return false;
  }

  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const NormGradientOpImpl&>(rhs);
      return (getp() == rhs_.getp()
              && dim() == rhs_.dim());
    }
    return false;
  }
};

Tensor MakeNormGradientOp(Tensor input, Tensor output, Tensor grad_output, int64_t p, 
                          int64_t dim, OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu
