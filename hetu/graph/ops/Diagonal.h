#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"
#include "hetu/graph/ops/Views.h"

namespace hetu {
namespace graph {

class DiagonalOpImpl;
class DiagonalOp;
class DiagonalGradientOpImpl;
class DiagonalGradientOp;

class DiagonalOpImpl final : public ViewsOpImpl {
 public:
  DiagonalOpImpl(int64_t offset, int64_t dim1, int64_t dim2)
  : ViewsOpImpl(quote(DiagonalOp)), _offset(offset), _dim1(dim1), _dim2(dim2) {}

  inline int64_t offset() const {
    return _offset;
  }

  inline int64_t dim1() const {
    return _dim1;
  }

  inline int64_t dim2() const {
    return _dim2;
  }

 protected:
  std::vector<NDArrayMeta>
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    HTShape ori_shape = inputs[0]->shape();
    int64_t ndim = ori_shape.size();
    int64_t dim1_ = NDArrayMeta::ParseAxis(dim1(), ndim);
    int64_t dim2_ = NDArrayMeta::ParseAxis(dim2(), ndim);
    HTStride ori_stride = inputs[0]->stride();
    HTShape res_shape(ori_shape.begin(), ori_shape.end());
    HTStride res_stride(ori_stride.begin(), ori_stride.end());
    res_shape.erase(res_shape.begin() + std::max(dim1_, dim2_));
    res_stride.erase(res_stride.begin() + std::max(dim1_, dim2_));
    res_shape.erase(res_shape.begin() + std::min(dim1_, dim2_));
    res_stride.erase(res_stride.begin() + std::min(dim1_, dim2_));
    if (offset() >= 0) {
      res_shape.emplace_back(
          std::min(ori_shape[dim1_], ori_shape[dim2_] - offset()));
    } else {
      res_shape.emplace_back(
          std::min(ori_shape[dim1_] + offset(), ori_shape[dim2_]));
    }
    res_stride.emplace_back(ori_stride[dim1_] + ori_stride[dim2_]);
    NDArrayMeta output_meta = NDArrayMeta().set_dtype(inputs[0]->dtype())
                                           .set_shape(res_shape)
                                           .set_stride(res_stride)
                                           .set_device(inputs[0]->device());
    return {output_meta};
  }

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs,
                      const OpMeta& op_meta,
                      const InstantiationContext& inst_ctx) const override;
  
  TensorList DoGradient(Operator& op,
                        const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const override;
  
  void DoSaveCtxForBackward(const TensorList& inputs, ContextStore& dst_ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const {};

  NDArrayList DoCompute(Operator& op, const NDArrayList& inputs,
                        RuntimeContext& ctx) const override;

  int64_t _offset;
  int64_t _dim1;
  int64_t _dim2;
 
 public:
  bool operator==(const OpInterface& rhs) const override {
    if (ViewsOpImpl::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const DiagonalOpImpl&>(rhs);
      return (offset() == rhs_.offset() && dim1() == rhs_.dim1()
           && dim2() == rhs_.dim2());
    }
    return false;
  }
};

Tensor MakeDiagonalOp(Tensor input, int64_t offset = 0, int64_t dim1 = 0,
                      int64_t dim2 = 1, OpMeta op_meta = OpMeta());

class DiagonalGradientOpImpl final : public ViewsOpImpl {
 public:
  DiagonalGradientOpImpl(int64_t offset, int64_t dim1, int64_t dim2)
  : ViewsOpImpl(quote(DiagonalGradientOp)), _offset(offset), _dim1(dim1), _dim2(dim2) {
  }

  inline int64_t offset() const {
    return _offset;
  }

  inline int64_t dim1() const {
    return _dim1;
  }

  inline int64_t dim2() const {
    return _dim2;
  }

 protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    HT_ASSERT_TENSORS_SAME_DTYPE(inputs);
    return {inst_ctx.get<NDArrayMeta>("in_meta")};
  };

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta,
                      const InstantiationContext& inst_ctx) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& ctx) const override;
  
  void DoLoadCtxForBackward(ContextStore& src_ctx, ContextStore& dst_ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

  int64_t _offset;
  int64_t _dim1;
  int64_t _dim2;

 public:
  bool operator==(const OpInterface& rhs) const override {
    if (ViewsOpImpl::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const DiagonalGradientOpImpl&>(rhs);
      return (offset() == rhs_.offset() &&
              dim1() == rhs_.dim1() &&
              dim2() == rhs_.dim2());
    }
    return false;
  }
};

Tensor MakeDiagonalGradientOp(Tensor grad_output, int64_t offset = 0,
                              int64_t dim1 = 0, int64_t dim2 = 1, OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu
