#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"

namespace hetu {
namespace graph {

class BatchMatMulOpImpl;

class BatchMatMulOpImpl : public OpInterface {
 private:
  friend class BatchMatMulOp;
  struct constructor_access_key {};

 public:
  BatchMatMulOpImpl(bool trans_a = false, bool trans_b = false,
                    OpMeta op_meta = OpMeta())
  : OpInterface(quote(BatchMatMulOp)),
    _trans_a(trans_a),
    _trans_b(trans_b) {
  }

  inline bool trans_a() const {
    return _trans_a;
  }

  inline bool trans_b() const {
    return _trans_b;
  }

protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    HT_ASSERT_TENSORS_SAME_DTYPE(inputs);
    auto a = inputs[0];
    auto b = inputs[1];
    if (a->has_shape() && b->has_shape()) {
      HT_ASSERT(a->ndim() >= 2 && b->ndim() >= 2 && a->ndim() == b->ndim());
      int64_t ndims = a->ndim();
      int64_t dim_a = a->shape(trans_a() ? ndims - 2 : ndims - 1);
      int64_t dim_b = b->shape(trans_b() ? ndims - 1 : ndims - 2);
      HT_ASSERT(dim_a == -1 || dim_b == -1 || dim_a == dim_b)
      << "Incompatible batch dimensions: " << dim_a << " and " << dim_b;
    }
    HTShape shape = {};
    if (a->has_shape() && b->has_shape()) {
      int ndims = a->ndim();
      for (int i = 0; i < ndims - 2; ++i) {
        HT_ASSERT(a->shape(i) == b->shape(i));
        shape.emplace_back(a->shape(i));
      }
      shape.emplace_back(a->shape(trans_a() ? ndims - 1 : ndims - 2));
      shape.emplace_back(b->shape(trans_b() ? ndims - 2 : ndims - 1));
    }
    NDArrayMeta output_meta = NDArrayMeta().set_dtype(inputs[0]->dtype())
                                           .set_shape(shape)
                                           .set_device(inputs[0]->device());
    return {output_meta};
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
  
  bool _trans_a;

  bool _trans_b;
 public:
  inline bool require_contig_inputs() const override {
    return false;
  }

  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const BatchMatMulOpImpl&>(rhs);
      return (trans_a() == rhs_.trans_a()
              && trans_b() == rhs_.trans_b()); 
    }
    return false;
  }
};

Tensor MakeBatchMatMulOp(Tensor a, Tensor b, bool trans_a = false, bool trans_b = false,
                         OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu
