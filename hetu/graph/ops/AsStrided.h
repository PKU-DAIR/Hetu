#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"
#include "hetu/graph/ops/Views.h"

namespace hetu {
namespace graph {

class AsStridedOpImpl;
class AsStridedOp;
class AsStridedGradientOpImpl;
class AsStridedGradientOp;

class AsStridedOpImpl final : public ViewsOpImpl {

 public:
  AsStridedOpImpl(const HTShape& outshape, const HTStride& stride, int64_t storage_offset)
  : ViewsOpImpl(quote(AsStridedOp)),
  _outshape(outshape),
  _stride(stride),
  _storage_offset(storage_offset) {}

  inline HTShape outshape() const {
    return _outshape;
  }

  inline HTStride stride() const {
    return _stride;
  }

  inline int64_t storage_offset() const {
    return _storage_offset;
  }

  protected:
   std::vector<NDArrayMeta>
   DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
     HT_ASSERT_TENSORS_SAME_DTYPE(inputs);
     NDArrayMeta output_meta = NDArrayMeta().set_dtype(inputs[0]->dtype())
                                            .set_shape(outshape())
                                            .set_stride(stride())
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

   void DoSaveCtxForBackward(const TensorList& inputs, ContextStore& dst_ctx) const override;

   void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                  RuntimeContext& runtime_ctx) const {};
   
   NDArrayList DoCompute(Operator& op, const NDArrayList& inputs,
                         RuntimeContext& ctx) const override;
   
   HTShape _outshape;
   HTStride _stride;
   size_t _storage_offset;

  public:
   bool operator==(const OpInterface& rhs) const override {
     if (ViewsOpImpl::operator==(rhs)) {
       const auto& rhs_ = reinterpret_cast<const AsStridedOpImpl&>(rhs);
       return (outshape() == rhs_.outshape() &&
               stride() == rhs_.stride() &&
               storage_offset() == rhs_.storage_offset());
     }
     return false;
   }
};

Tensor MakeAsStridedOp(Tensor input, const HTShape& outshape, const HTStride& stride,
                       int64_t storage_offset = 0, OpMeta op_meta = OpMeta());

class AsStridedGradientOpImpl final : public ViewsOpImpl {

 public:
  AsStridedGradientOpImpl(const HTShape& outshape, const HTStride& stride, int64_t storage_offset)
  : ViewsOpImpl(quote(AsStridedGradientOp)),
  _outshape(outshape),
  _stride(stride),
  _storage_offset(storage_offset) {}

  inline HTShape outshape() const {
    return _outshape;
  }

  inline HTStride stride() const {
    return _stride;
  }

  inline int64_t storage_offset() const {
    return _storage_offset;
  }

 protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    HT_ASSERT_TENSORS_SAME_DTYPE(inputs);
    return {inst_ctx.get<NDArrayMeta>("in_meta")};
  }

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta,
                      const InstantiationContext& inst_ctx) const override;  

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;

  void DoLoadCtxForBackward(ContextStore& src_ctx, ContextStore& dst_ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;

  NDArrayList DoCompute(Operator& op, const NDArrayList& inputs,
                         RuntimeContext& ctx) const override;
  
  HTShape _outshape;
  HTStride _stride;
  int64_t _storage_offset;

 public:
  bool operator==(const OpInterface& rhs) const override {
    if (ViewsOpImpl::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const AsStridedGradientOpImpl&>(rhs);
      return (outshape() == rhs_.outshape()
           && stride() == rhs_.stride()
           && storage_offset() == rhs_.storage_offset()); 
    }
    return false;
  }
};

Tensor MakeAsStridedGradientOp(Tensor grad_output, const HTShape& outshape,
                               const HTStride& stride, int64_t storage_offset, OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu
