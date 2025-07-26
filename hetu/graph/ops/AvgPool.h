#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"
#include "hetu/graph/ops/Pool.h"

namespace hetu {
namespace graph {

class AvgPoolOpImpl;
class AvgPoolGradientOpImpl;

class AvgPoolOpImpl final : public PoolOpImpl {

 public:
  AvgPoolOpImpl(size_t kernel_H, size_t kernel_W, 
                size_t padding, size_t stride,
                OpMeta op_meta = OpMeta())
  : PoolOpImpl(quote(AvgPoolOp), kernel_H, kernel_W,
               padding, stride) {}

 protected:
  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta,
                      const InstantiationContext& inst_ctx) const override;

  TensorList DoGradient(Operator& op,
                        const TensorList& grad_outputs) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;
 public:
  bool operator==(const OpInterface& rhs) const override {
    return PoolOpImpl::operator==(rhs);
  }
};

class AvgPoolGradientOpImpl final : public PoolGradientOpImpl {
 public:
  AvgPoolGradientOpImpl(size_t kernel_H,
                        size_t kernel_W, size_t padding, size_t stride,
                        OpMeta op_meta = OpMeta())
  : PoolGradientOpImpl(quote(AvgPoolGradientOp), kernel_H, kernel_W,
                       padding, stride) {}

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

 protected:
  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta,
                      const InstantiationContext& inst_ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;

 public:
  bool operator==(const OpInterface& rhs) const override {
    return PoolGradientOpImpl::operator==(rhs);
  }
};

Tensor MakeAvgPoolOp(Tensor input, size_t kernel_H, size_t kernel_W, 
                     size_t padding, size_t stride,
                     OpMeta op_meta = OpMeta());

Tensor MakeAvgPoolGradientOp(Tensor output, Tensor output_grad, Tensor input,
                             size_t kernel_H, size_t kernel_W, size_t padding,
                             size_t stride, OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu
