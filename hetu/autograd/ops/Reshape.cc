#include "hetu/autograd/ops/Reshape.h"
#include "hetu/autograd/ops/kernel_links.h"

namespace hetu {
namespace autograd {

void ArrayReshapeOpDef::DoCompute(const NDArrayList& inputs,
                                  NDArrayList& outputs, RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(),
                                  hetu::impl::Reshape, inputs.at(0),
                                  outputs.at(0), stream());
}

TensorList ArrayReshapeOpDef::DoGradient(const TensorList& grad_outputs) {
  auto& self = reinterpret_cast<ArrayReshapeOp&>(get_self());
  return {ArrayReshapeGradientOp(grad_outputs.at(0), self,
                                 grad_op_meta().set_name(grad_name()))
            ->output(0)};
}

HTShapeList ArrayReshapeOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  size_t input_size = 1;
  HTShape input_shape = input_shapes.at(0);
  size_t input_len = input_shape.size();
  for (size_t i = 0; i < input_len; ++i) {
    input_size *= input_shape[i];
  }
  // check if there exists -1 in output_shape
  int64_t idx = -1;
  size_t cnt = 0;
  size_t output_size = 1;
  HTShape output_shape = get_output_shape();
  int64_t output_len = output_shape.size();
  for (int64_t i = 0; i < output_len; ++i) {
    if (output_shape[i] == -1) {
      idx = i;
      cnt = cnt + 1;
      HT_ASSERT(cnt != 2) << "Output shape has more than one '-1' dims. ";
    }
    output_size *= output_shape[i];
  }
  if (idx == -1) {
    HT_ASSERT(input_size == output_size) << "Invalid output size.";
  } else {
    output_size = output_size * (-1);
    HT_ASSERT(input_size % output_size == 0) << "Invalid output size.";
    output_shape[idx] = input_size / output_size;
  }
  set_input_shape(input_shape);
  return {output_shape};
}

void ArrayReshapeGradientOpDef::DoCompute(const NDArrayList& inputs,
                                          NDArrayList& outputs,
                                          RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(placement().type(), type(),
                                  hetu::impl::Reshape, inputs.at(0),
                                  outputs.at(0), stream());
}

HTShapeList
ArrayReshapeGradientOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  return {get_input_node()->get_input_shape()};
}

} // namespace autograd
} // namespace hetu
