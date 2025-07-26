#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"

namespace hetu {
namespace graph {

class DynamicConcatenateOpImpl;
class DynamicConcatenateOp;

class DynamicConcatenateOpImpl final : public OpInterface {
 private:
  friend class DynamicConcatenateOp;
  struct constructor_access_key {};

 public:
  DynamicConcatenateOpImpl(size_t axis = 0, OpMeta op_meta = OpMeta())
  : OpInterface(quote(DynamicConcatenateOp)), _axis(axis) {
  }

  size_t get_axis() const {
    return _axis;
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
      for (int i = 1; i < len; ++i) {
        HTShape shape = inputs.at(i)->shape();
        HT_ASSERT(shape.size() == out_shape.size());
        for (int j = 0; j < n_dim; ++j) {
          if (j != (int) _axis) {
            HT_ASSERT(shape[j] == out_shape[j] || shape[j] == -1 ||
                      out_shape[j] == -1)
            << "input0 and input" << i << " has different size at dim " << j
            << ", input0 has " << shape[j] <<  ",input" << i << " has "
            << out_shape[j];
          } else {
            out_dim = out_dim > shape[j] ? out_dim : shape[j];
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
                      const OpMeta& op_meta,
                      const InstantiationContext& inst_ctx) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;

  HTShapeList DoInferDynamicShape(Operator& op, const HTShapeList& input_shapes,
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
      const auto& rhs_ = reinterpret_cast<const DynamicConcatenateOpImpl&>(rhs);
      return get_axis() == rhs_.get_axis();
    }
    return false;
  }
};

Tensor MakeDynamicConcatenateOp(TensorList inputs, int64_t axis = 0,
                         OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu
