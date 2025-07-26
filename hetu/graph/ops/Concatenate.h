#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"

namespace hetu {
namespace graph {

class ConcatenateOpImpl;
class ConcatenateOp;
class ConcatenateGradientOpImpl;
class ConcatenateGradientOp;

class ConcatenateOpImpl final : public OpInterface {
 private:
  friend class ConcatenateOp;
  struct constructor_access_key {};

 public:
  ConcatenateOpImpl(size_t axis = 0, OpMeta op_meta = OpMeta())
  : OpInterface(quote(ConcatenateOp)), _axis(axis) {
  }

  inline uint64_t op_indicator() const noexcept override {
    return CONCAT_OP;
  }

  size_t get_axis() const {
    return _axis;
  }

  int64_t get_grad_offset(size_t idx) const {
    return grad_offsets[idx];
  }

  size_t grad_num() const {
    return grad_offsets.size();
  }

 protected:
  std::vector<NDArrayMeta>
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    HT_ASSERT_TENSORS_SAME_DTYPE(inputs);
    int len = inputs.size();
    bool flag = true;
    for (int i = 0; i < len; ++i) {
      if (!inputs.at(i)->has_shape()) {
        flag = false;
        break;
      }
    }
    HTShape out_shape = {};
    if (flag) {
      out_shape = inputs.at(0)->shape();
      int n_dim = out_shape.size();
      int out_dim = out_shape[get_axis()];
      int ind = 0;
      ind += 1;
      for (int i = 1; i < len; ++i) {
        HTShape shape = inputs.at(i)->shape();
        HT_ASSERT(shape.size() == out_shape.size());
        for (int j = 0; j < n_dim; ++j) {
          if (j != (int) _axis) {
            HT_ASSERT(shape[j] == out_shape[j] || shape[j] == -1 ||
                      out_shape[j] == -1)
            << "input0 and input" << i << " has different size at dim " << j
            << ", input0 has " << shape[j] <<  ", input" << i << " has "
            << out_shape[j];
          } else {
            out_dim += shape[j];
          }
        }
      }
      out_shape[_axis] = out_dim;
    }
    NDArrayMeta output_meta = NDArrayMeta().set_dtype(inputs[0]->dtype())
                                           .set_shape(out_shape)
                                           .set_device(inputs[0]->device());
    return {output_meta};
  }

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta, const InstantiationContext& inst_ctx) const override;

  TensorList DoGradient(Operator& op,
                        const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;

  NDArrayList DoCompute(Operator& op, const NDArrayList& inputs,
                        RuntimeContext& runtime_ctx) const override;
  
  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;

  size_t _axis;

  std::vector<int64_t> grad_offsets;


 public:
  inline bool require_contig_inputs() const override {
    return false;
  }

  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const ConcatenateOpImpl&>(rhs);
      return (get_axis() == rhs_.get_axis()
              && grad_offsets == rhs_.grad_offsets);
    }
    return false;
  }
};

Tensor MakeConcatenateOp(TensorList inputs, size_t axis = 0,
                         OpMeta op_meta = OpMeta());

class ConcatenateGradientOpImpl final : public OpInterface {
 public:
  ConcatenateGradientOpImpl(size_t axis, size_t offset,
                            OpMeta op_meta = OpMeta())
  : OpInterface(quote(ConcatenateGradientOp)),
    _axis(axis), _offset(offset) {
  }

  size_t get_axis() const {
    return _axis;
  }

  size_t get_offset() const {
    return _offset;
  }

  void set_offset(size_t offset) {
    _offset = offset;
  }

 protected:
  std::vector<NDArrayMeta>
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    HT_ASSERT_TENSORS_SAME_DTYPE(inputs);
    NDArrayMeta output_meta = inputs[0]->meta();
    return {output_meta};
  }

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta, const InstantiationContext& inst_ctx) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;

  size_t _axis;

  size_t _offset;


 public:
  inline bool require_contig_inputs() const override {
    return false;
  }

  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const ConcatenateGradientOpImpl&>(rhs);
      return (get_axis() == rhs_.get_axis()
              && get_offset() == rhs_.get_offset());
    }
    return false;
  }
};

Tensor MakeConcatenateGradientOp(Tensor input, Tensor output, Tensor grad_output, size_t axis, size_t offset,
                                 OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu
