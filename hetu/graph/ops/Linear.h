#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"

namespace hetu {
namespace graph {

class LinearOpImpl;
class LinearOp;

class LinearOpImpl final : public OpInterface {

 public:
  LinearOpImpl(bool trans_a = false, bool trans_b = true,
              OpMeta op_meta = OpMeta())
  : OpInterface(quote(LinearOp)),
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
    const Tensor& a = inputs.at(0);
    const Tensor& b = inputs.at(1);
    if (a->has_shape() && b->has_shape()) {
      HT_ASSERT(a->ndim() == 2 && b->ndim() == 2)
        << "Failed to construct the \"MatMul\" op: "
        << "Dimensions must be 2. "
        << "Got " << a->ndim() << ", " << b->ndim() << ".";
      int64_t dim_a = a->shape(trans_a() ? 0 : 1);
      int64_t dim_b = b->shape(trans_b() ? 1 : 0);
      HT_ASSERT(dim_a == -1 || dim_b == -1 || dim_a == dim_b)
        << "Failed to construct the \"MatMul\" op: "
        << "Dimensions must be compatible. "
        << "Got " << dim_a << " vs. " << dim_b << ". "
        << "Input shapes: " << a->shape() << " vs. " << b->shape() << ".";
    }
    HTShape shape = {-1, -1};
    if (a->has_shape())
      shape[0] = a->shape(trans_a() ? 1 : 0);
    if (b->has_shape())
      shape[1] = b->shape(trans_b() ? 0 : 1);
    auto dst_dtype = (a->dtype() == kFloat4 || a->dtype() == kNFloat4)
                     ? b->dtype()
                     : a->dtype();
    return {NDArrayMeta().set_dtype(dst_dtype).set_shape(shape)};
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

  bool _trans_a;
  bool _trans_b;

 public:
  inline bool require_contig_inputs() const override {
    return false;
  }

  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const LinearOpImpl&>(rhs);
      return trans_a() == rhs_.trans_a() && trans_b() == rhs_.trans_b();
    }
    return false;
  }
};

Tensor MakeLinearOp(Tensor a, Tensor b, bool trans_a = false,
                    bool trans_b = true, OpMeta op_meta = OpMeta());

Tensor MakeLinearOp(Tensor a, Tensor b, Tensor bias, bool trans_a = false,
                    bool trans_b = true, OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu
