#pragma once

#include "hetu/graph/operator.h"

namespace hetu {
namespace graph {

class PoolOpImpl;
class PoolGradientOpImpl;

class PoolOpImpl : public OpInterface {
 protected:
  PoolOpImpl(OpType&& op_type,
             size_t kernel_H, size_t kernel_W,
             size_t padding, size_t stride)
  : OpInterface(std::move(op_type)),
    _kernel_H(kernel_H),
    _kernel_W(kernel_W),
    _padding(padding),
    _stride(stride) {
    HT_ASSERT(kernel_H >= 1)
    << "kernel_H < 1, kernel_H = " << kernel_H;
    HT_ASSERT(kernel_W >= 1)
    << "kernel_W < 1, kernel_W = " << kernel_W;
    HT_ASSERT(stride >= 1)
    << "stride < 1, stride = " << stride;
    HT_ASSERT(padding >= 0)
    << "padding < 0, padding = " << padding;
  }
 
 public:
  size_t get_kernel_H() const {
    return _kernel_H;
  }

  size_t get_kernel_W() const {
    return _kernel_W;
  }

  size_t get_padding() const {
    return _padding;
  }

  size_t get_stride() const {
    return _stride;
  }

  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const PoolOpImpl&>(rhs);
      return (get_kernel_H() == rhs_.get_kernel_H() &&
              get_kernel_W() == rhs_.get_kernel_W() &&
              get_padding() == rhs_.get_padding() &&
              get_stride() == rhs_.get_stride()); 
    }
    return false;
  }

 protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    HTShape shape = {-1, -1, -1, -1};
    if (inputs[0]->has_shape()) {
      HT_ASSERT_HAS_DIMS(inputs[0], 4);
      int64_t N = inputs[0]->shape(0);
      int64_t C = inputs[0]->shape(1);
      int64_t H = inputs[0]->shape(2);
      int64_t W = inputs[0]->shape(3);
      int64_t p_H = (H + 2 * get_padding() - get_kernel_H()) / get_stride() + 1;
      int64_t p_W = (W + 2 * get_padding() - get_kernel_W()) / get_stride() + 1;
      shape = {N, C, p_H, p_W};
    }
    HT_ASSERT_TENSORS_SAME_DTYPE(inputs);
    NDArrayMeta output_meta = NDArrayMeta().set_dtype(inputs[0]->dtype())
                                           .set_shape(shape)
                                           .set_device(inputs[0]->device());
    return {output_meta};
  }

  HTShapeList DoInferShape(Operator& op,
                           const HTShapeList& input_shapes,
                           RuntimeContext& ctx) const {
    int64_t N = input_shapes.at(0)[0];
    int64_t C = input_shapes.at(0)[1];
    int64_t H = input_shapes.at(0)[2];
    int64_t W = input_shapes.at(0)[3];
    int64_t p_H = (H + 2 * get_padding() - get_kernel_H()) / get_stride() + 1;
    int64_t p_W = (W + 2 * get_padding() - get_kernel_W()) / get_stride() + 1;
    return {{N, C, p_H, p_W}};
  }

  size_t _kernel_H;
  size_t _kernel_W;
  size_t _padding;
  size_t _stride;
};

class PoolGradientOpImpl : public OpInterface {
 protected:
  PoolGradientOpImpl(OpType&& op_type,
                     size_t kernel_H, size_t kernel_W,
                     size_t padding, size_t stride)
  : OpInterface(std::move(op_type)),
    _kernel_H(kernel_H),
    _kernel_W(kernel_W),
    _padding(padding),
    _stride(stride) {}
 
 public:
  size_t get_kernel_H() const {
    return _kernel_H;
  }

  size_t get_kernel_W() const {
    return _kernel_W;
  }

  size_t get_padding() const {
    return _padding;
  }

  size_t get_stride() const {
    return _stride;
  }

  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const PoolGradientOpImpl&>(rhs);
      return (get_kernel_H() == rhs_.get_kernel_H() &&
              get_kernel_W() == rhs_.get_kernel_W() &&
              get_padding() == rhs_.get_padding() &&
              get_stride() == rhs_.get_stride()); 
    }
    return false;
  }

 protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    HT_ASSERT_TENSORS_SAME_DTYPE(inputs);
    NDArrayMeta output_meta = inputs[2]->meta();
    return {output_meta};
  }

  HTShapeList
  DoInferShape(Operator& op,
               const HTShapeList& input_shapes,
               RuntimeContext& ctx) const {
    return {input_shapes.at(2)};
  }

  size_t _kernel_H;
  size_t _kernel_W;
  size_t _padding;
  size_t _stride;
};

} // namespace graph
} // namespace hetu