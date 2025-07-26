#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"
#include "hetu/graph/ops/Views.h"

namespace hetu {
namespace graph {

class TransposeOpImpl;
class TransposeOp;

class TransposeOpImpl final : public ViewsOpImpl {
 public:
  TransposeOpImpl(HTAxes perms)
  : ViewsOpImpl(quote(TransposeOp)), _perms{std::move(perms)} {}

  const HTAxes& get_perms() const {
    return _perms;
  }

  inline uint64_t op_indicator() const noexcept override {
    return INPLACE_OP;
  }

 protected:
  std::vector<NDArrayMeta>
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    HTShape res_shape = {};
    HTShape res_stride = {};
    if (inputs[0]->has_shape()) {
      HTShape ori_shape = inputs[0]->shape();
      HTShape ori_stride = inputs[0]->stride();
      HTAxes perm = _perms;
      HT_ASSERT(perm.size() == ori_shape.size())
      << "Invalid perm size: " << _perms << ", expected: " << inputs[0]->shape();
      int ndim = perm.size();

      HTShape vis(ndim);
      for (int i = 0; i < ndim; i++) {
        HT_ASSERT(perm[i] < ndim);
        HT_ASSERT(vis[perm[i]] == 0);
        vis[perm[i]]++;
      }

      for (int i = 0; i < ndim; i++) {
        res_shape.emplace_back(ori_shape[perm[i]]);
        res_stride.emplace_back(ori_stride[perm[i]]);
      }
    }
    NDArrayMeta output_meta = NDArrayMeta().set_dtype(inputs[0]->dtype())
                                           .set_shape(res_shape)
                                           .set_stride(res_stride)
                                           .set_device(inputs[0]->device());
    return {output_meta};
  }

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs,
                      const OpMeta& op_meta,
                      const InstantiationContext& inst_ctx) const override;

  void DoDeduceHeterProp(const std::vector<int32_t>& inputs_hetero_dim,
                         TensorList& outputs, const OpMeta& op_meta,
                         const InstantiationContext& inst_ctx) const override;
  
  TensorList DoGradient(Operator& op,
                        const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const override;
  
  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const {};

  NDArrayList DoCompute(Operator& op, const NDArrayList& inputs,
                        RuntimeContext& ctx) const override;

  HTAxes _perms;
 
 public:
  bool operator==(const OpInterface& rhs) const override {
    if (ViewsOpImpl::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const TransposeOpImpl&>(rhs);
      return (get_perms() == rhs_.get_perms());
    }
    return false;
  }
};

Tensor MakeTransposeOp(Tensor input, HTAxes perms, OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu
