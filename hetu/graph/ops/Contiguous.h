#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"

namespace hetu {
namespace graph {

class ContiguousOpImpl;
class ContiguousOp;
class ContiguousGradientOpImpl;
class ContiguousGradientOp;

class ContiguousOpImpl final : public OpInterface {
 public:
  ContiguousOpImpl()
  : OpInterface(quote(ContiguousOp)) {
  }

  inline uint64_t op_indicator() const noexcept override {
    return CONTIGUOUS_OP;
  }

 protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    NDArrayMeta output_meta = NDArrayMeta().set_dtype(inputs[0]->dtype())
                                           .set_shape(inputs[0]->shape())
                                           .set_device(inputs[0]->device());
    return {output_meta};       
  };

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta,
                      const InstantiationContext& inst_ctx) const override;
  
  NDArrayList DoCompute(Operator& op,
                        const NDArrayList& inputs,
                        RuntimeContext& ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

  TensorList DoGradient(Operator& op, const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const override;

 public:  
  inline bool require_contig_inputs() const override {
    return false;
  }

  bool operator==(const OpInterface& rhs) const override {
    return OpInterface::operator==(rhs); 
  }
};

Tensor MakeContiguousOp(Tensor input, OpMeta op_meta = OpMeta());

class ContiguousGradientOpImpl final : public OpInterface {
 public:
  ContiguousGradientOpImpl(HTStride stride)
  : OpInterface(quote(ContiguousGradientOp)), _stride(stride) {
  }

  HTStride fwd_stride() const {
    return _stride;
  }

 protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    NDArrayMeta output_meta = NDArrayMeta().set_dtype(inputs[0]->dtype())
                                           .set_shape(inputs[0]->shape())
                                           .set_stride(fwd_stride())
                                           .set_device(inputs[0]->device());
    return {output_meta};       
  };

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta,
                      const InstantiationContext& inst_ctx) const override;
  
  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const override;

  HTStride _stride;

 public:
  inline bool require_contig_inputs() const override {
    return false;
  }

  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const ContiguousGradientOpImpl&>(rhs);
      return (fwd_stride() == rhs_.fwd_stride());
    }
    return false;
  }
};

Tensor MakeContiguousGradientOp(Tensor input, const HTStride& stride, OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu
