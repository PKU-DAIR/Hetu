// #pragma once

// #include "hetu/graph/operator.h"
// #include "hetu/graph/utils/tensor_utils.h"
// #include "hetu/graph/ops/Unary.h"

// namespace hetu {
// namespace graph {


// class MaskedSelectOpImpl;
// class MaskedSelectOp;

// class MaskedSelectOpImpl : public OpInterface {
//  protected:
//   MaskedSelectOpImpl(OpType&& op_type, bool inplace = true)
//   : OpInterface(std::move(op_type)), _inplace(inplace) {}
 
//  public:
//   inline bool require_contig_inputs() const override {
//     return false;
//   }

//   inline bool inplace() const {
//     return _inplace;
//   }

//   std::vector<NDArrayMeta> 
//   DoInferMeta(const TensorList& inputs) const override {
//     HTShape shape = inputs[0]->shape();
//     NDArrayMeta output_meta = NDArrayMeta().set_dtype(inputs[0]->dtype())
//                                            .set_shape(shape)
//                                            .set_device(inputs[0]->device());
//     return {output_meta};
//   }

//   HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
//                            RuntimeContext& runtime_ctx) const override {
//     return {input_shapes[0]};
//   }

//   TensorList DoGradient(Operator& op,
//                         const TensorList& grad_outputs) const override;

//   void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
//                  RuntimeContext& ctx) const override;

//   NDArrayList DoCompute(Operator& op,
//                         const NDArrayList& inputs,
//                         RuntimeContext& ctx) const override;

//   bool operator==(const OpInterface& rhs) const override {
//     return OpInterface::operator==(rhs);
//   }
//   protected:
//     bool _inplace;
// };

// Tensor MakeMaskedSelectOp(Tensor input, Tensor mask, Tensor source,
//                         OpMeta op_meta = OpMeta());

// }
// }