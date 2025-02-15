#include "hetu/graph/ops/Quantization.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

void QuantizationOpImpl::DoCompute(Operator& op, 
                                   const NDArrayList& inputs, 
                                   NDArrayList& outputs,
                                   RuntimeContext& ctx) const {
  if (qtype() == kInt8)
    NDArray::quantization(inputs.at(0), qtype(), blocksize(), stochastic(),
                          op->instantiation_ctx().stream_index, inputs.at(1),
                          outputs.at(1), outputs.at(0));
  else 
    NDArray::quantization(inputs.at(0), qtype(), blocksize(), stochastic(),
                          op->instantiation_ctx().stream_index, NDArray(), 
                          outputs.at(1), outputs.at(0));
}

TensorList QuantizationOpImpl::DoGradient(Operator& op, const TensorList& grad_outputs) const {
  if (op->input(0)->dtype() == kInt8)
    return {Tensor(), Tensor()};
  else
    return {Tensor()};
}

HTShapeList QuantizationOpImpl::DoInferShape(Operator& op, 
                                             const HTShapeList& input_shapes, 
                                             RuntimeContext& ctx) const {
  int64_t numel_ = 1;
  for (auto& item: input_shapes[0])
    numel_ *= item;
  HTShape absmax_shape = {numel_ / blocksize()};
  return {input_shapes.at(0), absmax_shape};
}

void DeQuantizationOpImpl::DoCompute(Operator& op, 
                                     const NDArrayList& inputs, 
                                     NDArrayList& outputs,
                                     RuntimeContext& ctx) const {
  if (inputs[0]->dtype() == kInt8)
    NDArray::dequantization(inputs.at(0), const_cast<NDArray&>(inputs.at(1)), dqtype(), 
                            blocksize(), op->instantiation_ctx().stream_index, inputs.at(2), outputs.at(0));
  else 
    NDArray::dequantization(inputs.at(0), const_cast<NDArray&>(inputs.at(1)), dqtype(), 
                            blocksize(), op->instantiation_ctx().stream_index, NDArray(), outputs.at(0));
}

TensorList DeQuantizationOpImpl::DoGradient(Operator& op, const TensorList& grad_outputs) const {
  if (op->input(0)->dtype() == kInt8)
    return {Tensor(), Tensor(), Tensor()};
  else
    return {Tensor(), Tensor()};
}

void DeQuantizationOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                          const OpMeta& op_meta,
                                          const InstantiationContext& inst_ctx) const {
  outputs.at(0)->set_distributed_states(inputs.at(0)->get_distributed_states());
}

HTShapeList DeQuantizationOpImpl::DoInferShape(Operator& op, 
                                               const HTShapeList& input_shapes, 
                                               RuntimeContext& ctx) const {
  return {input_shapes.at(0)};
}

TensorList MakeQuantizationOp(Tensor input, DataType qtype, 
                              int64_t blocksize, bool stochastic, 
                              OpMeta op_meta) {
  TensorList inputs = {std::move(input)};
  return Graph::MakeOp(
         std::make_shared<QuantizationOpImpl>(qtype, blocksize, stochastic),
         std::move(inputs),
         std::move(op_meta))->outputs();
}

TensorList MakeQuantizationOp(Tensor input, Tensor code, DataType qtype, 
                              int64_t blocksize, bool stochastic, 
                              OpMeta op_meta) {
  TensorList inputs = {std::move(input), std::move(code)};
  return Graph::MakeOp(
         std::make_shared<QuantizationOpImpl>(qtype, blocksize, stochastic),
         std::move(inputs),
         std::move(op_meta))->outputs();
}

Tensor MakeDeQuantizationOp(Tensor input, Tensor absmax, 
                            DataType dqtype, int64_t blocksize, 
                            OpMeta op_meta) {
  TensorList inputs = {std::move(input), std::move(absmax)};
  return Graph::MakeOp(
         std::make_shared<DeQuantizationOpImpl>(dqtype, blocksize),
         std::move(inputs),
         std::move(op_meta))->output(0);
}

Tensor MakeDeQuantizationOp(Tensor input, Tensor absmax, Tensor code,
                            DataType dqtype, int64_t blocksize,
                            OpMeta op_meta) {
  TensorList inputs = {std::move(input), std::move(absmax), std::move(code)};
  return Graph::MakeOp(
         std::make_shared<DeQuantizationOpImpl>(dqtype, blocksize),
         std::move(inputs),
         std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hetu
