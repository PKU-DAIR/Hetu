#include "hetu/autograd/ops/Conv2d.h"
#include "hetu/autograd/ops/ReduceSum.h"
#include "hetu/autograd/ops/kernel_links.h"

namespace hetu {
namespace autograd {

void Conv2dOpDef::DoCompute(const NDArrayList& inputs, NDArrayList& outputs,
                            RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CUDA_ONLY(placement().type(), type(), hetu::impl::Conv2d,
                               inputs.at(0), inputs.at(1), outputs.at(0),
                               get_padding()[0], get_padding()[1],
                               get_stride()[0], get_stride()[1], stream());
}

TensorList Conv2dOpDef::DoGradient(const TensorList& grad_outputs) {
  auto g_op_meta = grad_op_meta();
  auto grad_input = Conv2dGradientofDataOp(
                      _inputs[1], grad_outputs.at(0), _inputs[0], get_padding(),
                      get_stride(), g_op_meta.set_name(grad_name(0)))
                      ->output(0);
  auto grad_filter =
    Conv2dGradientofFilterOp(_inputs[0], grad_outputs.at(0), _inputs[1],
                             get_padding(), get_stride(),
                             g_op_meta.set_name(grad_name(1)))
      ->output(0);
  return {grad_input, grad_filter};
}

HTShapeList Conv2dOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  int64_t N = input_shapes.at(0)[0];
  int64_t H = input_shapes.at(0)[2];
  int64_t W = input_shapes.at(0)[3];
  int64_t f_O = input_shapes.at(1)[0];
  int64_t f_H = input_shapes.at(1)[2];
  int64_t f_W = input_shapes.at(1)[3];
  HTShape padding = get_padding();
  HTShape stride = get_stride();
  int64_t out_H = (H + 2 * padding[0] - f_H) / stride[0] + 1;
  int64_t out_W = (W + 2 * padding[1] - f_W) / stride[1] + 1;
  return {{N, f_O, out_H, out_W}};
}

void Conv2dGradientofFilterOpDef::DoCompute(const NDArrayList& inputs,
                                            NDArrayList& outputs,
                                            RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CUDA_ONLY(
    placement().type(), type(), hetu::impl::Conv2dGradientofFilter,
    inputs.at(0), inputs.at(1), outputs.at(0), get_padding()[0],
    get_padding()[1], get_stride()[0], get_stride()[1], stream());
}

HTShapeList
Conv2dGradientofFilterOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  return {input_shapes.at(2)};
}

void Conv2dGradientofDataOpDef::DoCompute(const NDArrayList& inputs,
                                          NDArrayList& outputs,
                                          RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CUDA_ONLY(
    placement().type(), type(), hetu::impl::Conv2dGradientofData, inputs.at(0),
    inputs.at(1), outputs.at(0), get_padding()[0], get_padding()[1],
    get_stride()[0], get_stride()[1], stream());
}

HTShapeList
Conv2dGradientofDataOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  return {input_shapes.at(2)};
}

void Conv2dAddBiasOpDef::DoCompute(const NDArrayList& inputs,
                                   NDArrayList& outputs, RuntimeContext& ctx) {
  HT_DISPATCH_KERNEL_CUDA_ONLY(
    placement().type(), type(), hetu::impl::Conv2dAddBias, inputs.at(0),
    inputs.at(1), inputs.at(2), outputs.at(0), get_padding()[0],
    get_padding()[1], get_stride()[0], get_stride()[1], stream());
}

TensorList Conv2dAddBiasOpDef::DoGradient(const TensorList& grad_outputs) {
  auto g_op_meta = grad_op_meta();
  auto grad_input = Conv2dGradientofDataOp(
                      _inputs[1], grad_outputs.at(0), _inputs[0], get_padding(),
                      get_stride(), g_op_meta.set_name(grad_name(0)))
                      ->output(0);
  auto grad_filter =
    Conv2dGradientofFilterOp(_inputs[0], grad_outputs.at(0), _inputs[1],
                             get_padding(), get_stride(),
                             g_op_meta.set_name(grad_name(1)))
      ->output(0);
  auto grad_bias = ReduceSumOp(grad_outputs.at(0), {0, 2, 3}, {false},
                               g_op_meta.set_name(grad_name(2)))
                     ->output(0);
  return {grad_input, grad_filter, grad_bias};
}

HTShapeList Conv2dAddBiasOpDef::DoInferShape(const HTShapeList& input_shapes) {
  CheckNumInputsEqual(input_shapes.size());
  int64_t N = input_shapes.at(0)[0];
  int64_t H = input_shapes.at(0)[2];
  int64_t W = input_shapes.at(0)[3];
  int64_t f_O = input_shapes.at(1)[0];
  int64_t f_H = input_shapes.at(1)[2];
  int64_t f_W = input_shapes.at(1)[3];
  HTShape padding = get_padding();
  HTShape stride = get_stride();
  int64_t out_H = (H + 2 * padding[0] - f_H) / stride[0] + 1;
  int64_t out_W = (W + 2 * padding[1] - f_W) / stride[1] + 1;
  return {{N, f_O, out_H, out_W}};
}

} // namespace autograd
} // namespace hetu
