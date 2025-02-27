#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"

namespace hetu {
namespace graph {

class ConcatOpImpl;
class ConcatOp;
class ConcatGradientOpImpl;
class ConcatGradientOp;

class ConcatOpImpl final : public OpInterface {
 public:
  ConcatOpImpl(size_t axis, OpMeta op_meta = OpMeta())
  : OpInterface(quote(ConcatOp)), _axis(axis) {
  }

  inline uint64_t op_indicator() const noexcept override {
    return CONCAT_OP;
  }

  size_t get_axis() const {
    return _axis;
  }

 protected:
  std::vector<NDArrayMeta>
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    HT_ASSERT_TENSORS_SAME_DTYPE(inputs);
    HTShape shape = inputs.at(0)->shape();
    shape[get_axis()] = 0;
    for (auto& input : inputs) {
      if (input->has_shape()) {
        for (size_t i = 0; i < input->ndim(); ++i) {
          if (i != get_axis()) {
            HT_ASSERT(input->shape(i) == shape[i])
                      << "input has different size at dim " << i
                      << ", input has " << input->shape(i) << ", concat_shape has " << shape[i];
          }
        }
        HT_ASSERT(input->shape(get_axis()) >= 0);
        shape[get_axis()] += input->shape(get_axis());
      }
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

  void DoSaveCtxForBackward(const TensorList& inputs, ContextStore& dst_ctx) const override;

  NDArrayList DoCompute(Operator& op, const NDArrayList& inputs,
                        RuntimeContext& runtime_ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;

  size_t _axis;


 public:
  inline bool require_contig_inputs() const override {
    return false;
  }

  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const ConcatOpImpl&>(rhs);
      return (get_axis() == rhs_.get_axis());
    }
    return false;
  }
};


Tensor MakeConcatOp(TensorList inputs, size_t axis, OpMeta op_meta = OpMeta());

class ConcatGradientOpImpl final : public OpInterface {
 public:
  ConcatGradientOpImpl(size_t axis, OpMeta op_meta = OpMeta())
  : OpInterface(quote(ConcatGradientOp)),
    _axis(axis) {
  }

  size_t get_axis() const {
    return _axis;
  }

 protected:
  std::vector<NDArrayMeta>
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    HT_ASSERT_TENSORS_SAME_DTYPE(inputs);
    NDArrayMeta output_meta = inputs[0]->meta();
    HTShape input_axis_size_list = inst_ctx.get<HTShape>("input_axis_size_list");
    std::vector<NDArrayMeta> ret;
    for (size_t i = 0; i < input_axis_size_list.size(); i++) {
      HTShape shape = inputs[0]->shape();
      shape[get_axis()] = input_axis_size_list[i];
      ret.push_back(NDArrayMeta().set_dtype(inputs[0]->dtype())
                                  .set_shape(shape)
                                  .set_device(inputs[0]->device()));
    }
    return ret;
  }

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta,
                      const InstantiationContext& inst_ctx) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;

  void DoLoadCtxForBackward(ContextStore& src_ctx, ContextStore& dst_ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;

  size_t _axis;

 public:
  inline bool require_contig_inputs() const override {
    return false;
  }

  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const ConcatGradientOpImpl&>(rhs);
      return (get_axis() == rhs_.get_axis());
    }
    return false;
  }
};

TensorList MakeConcatGradientOp(Tensor grad_output, size_t axis, OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu
