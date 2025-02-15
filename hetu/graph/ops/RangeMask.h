#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"

namespace hetu {
namespace graph {

class RangeMaskOpImpl;
class RangeMaskOp;

class RangeMaskOpImpl : public OpInterface {
 private:
  friend class RangeMaskOp;
  struct constructor_access_key {};

 public:
  // input value∈[min, max], mask set to 0, otherwise set to 1 
  RangeMaskOpImpl(int64_t min, int64_t max)
  : OpInterface(quote(RangeMaskOp)), _min(min), _max(max) {}

  inline int64_t min() const {
    return _min;
  }

  inline int64_t max() const {
    return _max;
  }

protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    return {inputs[0]->meta()};
  }

  TensorList DoGradient(Operator& op,
                        const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;

  int64_t _min;
  int64_t _max;
};

Tensor MakeRangeMaskOp(Tensor input, int64_t min, int64_t max,
                       OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu
