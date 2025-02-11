#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"
#include "hetu/graph/ops/Unary.h"

namespace hetu {
namespace graph {

class QuantizationOpImpl;
class QuantizationOp;
class DeQuantizationOpImpl;
class DeQuantizationOp;

class QuantizationOpImpl final : public OpInterface {
 private:
  friend class QuantizationOp;
  struct constructor_access_key {};

 public:
  QuantizationOpImpl(DataType qtype, int64_t blocksize, bool stochastic)
  : OpInterface(quote(QuantizationOp)),
  _qtype(qtype),
  _blocksize(blocksize),
  _stochastic(stochastic) {}

 protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    NDArrayMeta out_meta = inputs.at(0)->meta();
    out_meta.set_dtype(qtype());
    HTShape absmax_shape = {int64_t(inputs.at(0)->numel() / blocksize())};
    NDArrayMeta absmax_meta = inputs.at(0)->meta();
    absmax_meta.set_dtype(kFloat32).set_shape(absmax_shape);
    return {out_meta, absmax_meta};
  }

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

  TensorList DoGradient(Operator& op,
                        const TensorList& grad_outputs) const override;
  
  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const override;

 public:
  DataType qtype() const {
    return _qtype;
  }

  int64_t blocksize() const {
    return _blocksize;
  }

  bool stochastic() const {
    return _stochastic;
  }

  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const QuantizationOpImpl&>(rhs);
      return (qtype() == rhs_.qtype()
            &&blocksize() == rhs_.blocksize()
            &&stochastic() == rhs_.stochastic()); 
    }
    return false;
  }

 protected:
  DataType _qtype;
  int64_t _blocksize;
  bool _stochastic;
};

TensorList MakeQuantizationOp(Tensor input, DataType qtype, 
                              int64_t blocksize, bool stochastic = false, 
                              OpMeta op_meta = OpMeta());

class DeQuantizationOpImpl final : public OpInterface {
 private:
  friend class DeQuantizationOp;
  struct constructor_access_key {};

 public:
  DeQuantizationOpImpl(DataType dqtype, int64_t blocksize)
  : OpInterface(quote(DeQuantizationOp)),
  _dqtype(dqtype),
  _blocksize(blocksize) {}

 protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    NDArrayMeta out_meta = inputs.at(0)->meta();
    return {out_meta.set_dtype(dqtype())};
  }

  TensorList DoGradient(Operator& op,
                        const TensorList& grad_outputs) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta,
                      const InstantiationContext& inst_ctx) const override;
  
  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const override;

 public:
  DataType dqtype() const {
    return _dqtype;
  }

  int64_t blocksize() const {
    return _blocksize;
  }


  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const DeQuantizationOpImpl&>(rhs);
      return (dqtype() == rhs_.dqtype()
            &&blocksize() == rhs_.blocksize()); 
    }
    return false;
  }

 protected:
  DataType _dqtype;
  int64_t _blocksize;
};

Tensor MakeDeQuantizationOp(Tensor input, Tensor absmax, DataType dqtype, 
                            int64_t blocksize, OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu
