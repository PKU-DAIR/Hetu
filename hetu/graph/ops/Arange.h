#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"

namespace hetu {
namespace graph {

class ArangeOpImpl;
class ArangeOp;

class ArangeOpImpl final : public OpInterface {
 public:
  ArangeOpImpl(double start, double end, double step)
  : OpInterface(quote(ArangeOp)),
  _start(start),
  _end(end),
  _step(step) {
  }

  inline double start() const {
    return _start;
  }

  inline double end() const {
    return _end;
  }

  inline double step() const {
    return _step;
  }

 protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    HT_ASSERT_TENSORS_SAME_DTYPE(inputs);
    int64_t length = (end() - start()) / step();
    HT_ASSERT(length > 0)
    << "Arange length is " << length << ", but it should be greater than zero.";
    NDArrayMeta output_meta = NDArrayMeta().set_dtype(DataType::FLOAT64).set_shape({length});
    return {output_meta};
  }

  TensorList DoGradient(Operator& op,
                        const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;
  double _start;

  double _end;

  double _step;
 public:
  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const ArangeOpImpl&>(rhs);
      return (start() == rhs_.start()
              && end() == rhs_.end()
              && step() == rhs_.step()); 
    }
    return false;
  }

};

Tensor MakeArangeOp(double start, double end, double step,
                    OpMeta op_meta = OpMeta());


} // namespace graph
} // namespace hetu
