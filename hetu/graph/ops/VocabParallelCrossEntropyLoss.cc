#include "hetu/graph/ops/VocabParallelCrossEntropyLoss.h"
#include "hetu/impl/communication/nccl_comm_group.h"
#include "hetu/impl/communication/comm_group.h"
#include "hetu/graph/distributed_states.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"
#include <numeric>

namespace hetu {
namespace graph {

// devices by dim for collective communication
DeviceGroup VocabParallelCrossEntropyOpImpl::get_devices_by_dim(const Tensor& input, int32_t dim) {
  const auto& placement_group = input->placement_group();
  const auto& placement = input->placement();
  HT_ASSERT(!placement_group.empty() && !placement.is_undetermined()) 
    << "Placement info should be assigned before get devices by dim " << dim;

  int32_t local_device_idx = placement_group.get_index(placement);
  const auto& src_ds = input->get_distributed_states();
  const auto& order = src_ds.get_order();
  const auto& states = src_ds.get_states();

  auto idx = std::find(order.begin(), order.end(), dim);
  int32_t interval = 1;
  for (auto cur_order = idx + 1; cur_order != order.end(); cur_order++) {
    interval *= states.at(*cur_order);
  }
  int32_t macro_interval = interval * src_ds.get_dim(dim);
  int32_t start = local_device_idx - local_device_idx % macro_interval + local_device_idx % interval;
  std::vector<Device> comm_group;
  for (auto i = start; i < start + macro_interval; i += interval) {
    comm_group.push_back(placement_group.get(i));
  }
  return std::move(DeviceGroup(comm_group));
}

// preds: [batch_size * seq_len, vocab_size], splited by tp in vocab_size dimension
// labels: [batch_size, seq_len], duplicate
// all compute and communicate in the same stream
void VocabParallelCrossEntropyOpImpl::DoCompute(
  Operator& op, const NDArrayList& inputs, 
  NDArrayList& outputs, RuntimeContext& ctx) const {
  const NDArray& preds = inputs.at(0);
  const NDArray& labels = inputs.at(1);
  DeviceGroup _comm_group = get_devices_by_dim(op->input(0), 1);
  auto ranks = hetu::impl::comm::DeviceGroupToWorldRanks(_comm_group);
  hetu::impl::comm::NCCLCommunicationGroup::GetOrCreate(ranks, op->instantiation_ctx().stream()); // do allreduce 3 times  
  // HT_LOG_INFO << hetu::impl::comm::GetLocalDevice() << ": VocabParallelCrossEntropyOp: comm_group: " << _comm_group;
  // loss per token = - log(e^x / sum(e^x)) = log(sum(e^x) / e^x)=log(sum(e^x)) - x; where x = x_ori - x_max
  // 1. x = x_ori - x_max
  HTShape reduce_max_shape = {preds->shape(0), 1};
  NDArray reduce_max_partial = NDArray::empty(reduce_max_shape, preds->device(), preds->dtype());
  NDArray::reduce(preds, kMAX, {-1}, true, op->instantiation_ctx().stream_index, reduce_max_partial); // split1 -> partial
  NDArray reduce_max = NDArray::empty(reduce_max_shape, preds->device(), preds->dtype());
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(), // partial -> dup
                                  hetu::impl::AllReduce, reduce_max_partial,
                                  reduce_max, kMAX, _comm_group,
                                  op->instantiation_ctx().stream());
  auto vocab_parallel_logits = preds - reduce_max;

  // 2. log(sum(e^x))
  NDArray exp_logits = NDArray::empty_like(vocab_parallel_logits);
  NDArray::exp(vocab_parallel_logits, op->instantiation_ctx().stream_index, exp_logits);
  HTShape reduce_sum_shape = {preds->shape(0), 1};
  NDArray sum_exp_logits_partial = NDArray::empty(reduce_sum_shape, preds->device(), preds->dtype());
  NDArray::reduce(exp_logits, kSUM, {-1}, true, op->instantiation_ctx().stream_index, sum_exp_logits_partial); // split1 -> partial
  NDArray sum_exp_logits = NDArray::empty(reduce_sum_shape, preds->device(), preds->dtype());
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(), // partial -> dup
                                  hetu::impl::AllReduce, sum_exp_logits_partial,
                                  sum_exp_logits, kSUM, _comm_group,
                                  op->instantiation_ctx().stream());
  NDArray log_sum_exp_logits = NDArray::empty_like(sum_exp_logits);
  NDArray::log(sum_exp_logits, op->instantiation_ctx().stream_index, log_sum_exp_logits);
  // store softmax for backward compute
  NDArray softmax = exp_logits / sum_exp_logits;
  OpRuntimeContext& op_ctx = ctx.get_or_create(op->id());
  op_ctx.put_ndarray("softmax", softmax);

  // 3. x[label]
  // Get the partition's vocab indecies, label should in range [vocab_start_index, vocab_end_index)]
  auto vocab_size_per_partition = preds->shape(1);
  auto local_device_index = op->placement_group().get_index(op->placement());
  auto vocab_range_index = op->input(0)->get_distributed_states().map_device_to_state_index(local_device_index)[1];
  auto vocab_start_index = vocab_size_per_partition * vocab_range_index;
  auto vocab_end_index = vocab_start_index + vocab_size_per_partition;
  HTShape predict_logits_shape = {preds->shape(0), 1};
  NDArray predict_logits_partial = NDArray::empty(predict_logits_shape, preds->device(), preds->dtype());
  HT_DISPATCH_KERNEL_CUDA_ONLY(op->instantiation_ctx().placement.type(), type(),
                               hetu::impl::VocabParallelCrossEntropy, vocab_parallel_logits,
                               labels, vocab_start_index, vocab_end_index, ignored_index(),
                               predict_logits_partial, log_sum_exp_logits, op->instantiation_ctx().stream());
  NDArray predict_logits = NDArray::empty(predict_logits_shape, preds->device(), preds->dtype());
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(), // partial -> dup
                                  hetu::impl::AllReduce, predict_logits_partial,
                                  predict_logits, kSUM, _comm_group,
                                  op->instantiation_ctx().stream());

  // 4. log(sum(e^x)) - x[label], shape = [batch_size * seq_len, 1]
  NDArray loss_unreduced = log_sum_exp_logits - predict_logits;
  // if use ignored_index, please use reduce SUM, and divide the sum by unignored num extraly
  if (reduction() != kNONE) {
    NDArray::reduce(loss_unreduced, reduction(), HTAxes(), false, 
                    op->instantiation_ctx().stream_index, outputs.at(0));
  } else {
    NDArray::copy(loss_unreduced, op->instantiation_ctx().stream_index, outputs.at(0));
  }
}

TensorList VocabParallelCrossEntropyOpImpl::DoGradient(
  Operator& op, const TensorList& grad_outputs) const {
  auto grad_input = op->requires_grad(0) ? MakeVocabParallelCrossEntropyGradientOp(
      op->input(0), op->input(1), grad_outputs.at(0), ignored_index(), reduction(),
      op->grad_op_meta().set_name(op->grad_name()))
    : Tensor();
  return {grad_input, Tensor()};
}

HTShapeList VocabParallelCrossEntropyOpImpl::DoInferShape(
  Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const {
  HT_ASSERT_EQ(input_shapes.at(0).size(), 2)
    << "Invalid shape for " << type() << ": " << input_shapes.at(0) 
    << "; shoulbe be [batch_size*seq_len, vocab_size]!";
  if (reduction() != kNONE)
    return {{1}};
  else {
    HTShape output_shape = {input_shapes.at(0)[0], 1};
    return {output_shape};
  }
}

void VocabParallelCrossEntropyOpImpl::DoDeduceStates(
  const TensorList& inputs, TensorList& outputs, const OpMeta& op_meta) const {
  const Tensor& preds = inputs.at(0);
  const Tensor& labels = inputs.at(1);                                
  const DistributedStates& ds_preds = preds->get_distributed_states();
  const DistributedStates& ds_labels = labels->get_distributed_states();

  // preds = ds_split01, labels = ds_split0_dup, loss = ds_split0_dup
  // _comm_group = get_devices_by_dim(preds, 1);
  // HT_LOG_INFO << hetu::impl::comm::GetLocalDevice() << ": VocabParallelCrossEntropyOp: comm_group: " << _comm_group;
  outputs.at(0)->set_distributed_states(ds_labels);
}

bool VocabParallelCrossEntropyOpImpl::DoInstantiate(
  Operator& op, const Device& placement, StreamIndex stream_index) const {
  bool ret = OpInterface::DoInstantiate(op, placement, stream_index);
  // auto ranks = hetu::impl::comm::DeviceGroupToWorldRanks(_comm_group);
  // hetu::impl::comm::NCCLCommunicationGroup::GetOrCreate(ranks, op->instantiation_ctx().stream()); // do allreduce 3 times
  return ret;
}

// inputs: preds, labels, grad_output
void VocabParallelCrossEntropyGradientOpImpl::DoCompute(
  Operator& op, const NDArrayList& inputs, 
  NDArrayList& outputs, RuntimeContext& ctx) const {
  const NDArray& preds = inputs.at(0);
  const NDArray& labels = inputs.at(1);
  const NDArray& grad_output = inputs.at(2);

  HTShape output_shape = {preds->shape(0), 1}; // [batch_size*seq_len, 1]
  NDArray broadcasted =
      reduction() == kNONE ? grad_output : NDArray::empty(output_shape, 
                                           preds->device(), preds->dtype());
  // grad = (softmax(prediction) - labels) / N
  if (reduction() == kMEAN) {
    HT_DISPATCH_KERNEL_CPU_AND_CUDA(
      op->instantiation_ctx().placement.type(), type(), hetu::impl::BroadcastShapeMul, grad_output,
      1.0f / broadcasted->numel(), broadcasted, HTAxes(), op->instantiation_ctx().stream());
  } else if (reduction() == kSUM) {
    HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(),
                                    hetu::impl::BroadcastShape, grad_output,
                                    broadcasted, HTAxes(), op->instantiation_ctx().stream());
  }

  auto vocab_size_per_partition = preds->shape(1);
  auto local_device_index = op->placement_group().get_index(op->placement());
  auto vocab_range_index = op->input(0)->get_distributed_states().map_device_to_state_index(local_device_index)[1];
  auto vocab_start_index = vocab_size_per_partition * vocab_range_index;
  auto vocab_end_index = vocab_start_index + vocab_size_per_partition;
  const NDArray& softmax = ctx.get_or_create(op->fw_op_id()).get_ndarray("softmax");

  HT_DISPATCH_KERNEL_CUDA_ONLY(op->instantiation_ctx().placement.type(), type(),
                               hetu::impl::VocabParallelCrossEntropyGradient, softmax,
                               labels, vocab_start_index, vocab_end_index, ignored_index(), 
                               broadcasted, outputs.at(0), op->instantiation_ctx().stream());
}

HTShapeList VocabParallelCrossEntropyGradientOpImpl::DoInferShape(
  Operator& op, const HTShapeList& input_shapes, RuntimeContext& ctx) const {
  HT_ASSERT_GE(input_shapes.at(0).size(), 2)
    << "Invalid shape for " << type() << ": " << input_shapes.at(0);
  return {input_shapes.at(0)};
}

void VocabParallelCrossEntropyGradientOpImpl::DoDeduceStates(
  const TensorList& inputs, TensorList& outputs, const OpMeta& op_meta) const {
  outputs.at(0)->set_distributed_states(inputs.at(0)->get_distributed_states());
}

Tensor MakeVocabParallelCrossEntropyOp(Tensor preds, Tensor labels, const int64_t ignored_index, 
                                       ReductionType reduction, OpMeta op_meta) {
  TensorList inputs = {preds, labels};
  DataType input_type = DataType::FLOAT32;
  AutoCast::Tensor_AutoCast(inputs, input_type);
  return Graph::MakeOp(
    std::make_shared<VocabParallelCrossEntropyOpImpl>(ignored_index, reduction),
    std::move(inputs),
    std::move(op_meta))->output(0);
}

Tensor MakeVocabParallelCrossEntropyOp(Tensor preds, Tensor labels, const int64_t ignored_index, 
                                       const std::string& reduction, OpMeta op_meta) {
  TensorList inputs = {preds, labels};
  DataType input_type = DataType::FLOAT32;
  AutoCast::Tensor_AutoCast(inputs, input_type);
  return Graph::MakeOp(
    std::make_shared<VocabParallelCrossEntropyOpImpl>(ignored_index, Str2ReductionType(reduction)),
    std::move(inputs),
    std::move(op_meta))->output(0);
}

Tensor MakeVocabParallelCrossEntropyGradientOp(Tensor preds, Tensor labels, Tensor grad_output,
                                               const int64_t ignored_index, ReductionType reduction,
                                               OpMeta op_meta) {
  return Graph::MakeOp(
    std::make_shared<VocabParallelCrossEntropyGradientOpImpl>(ignored_index, reduction),
    {std::move(preds), std::move(labels), std::move(grad_output)},
    std::move(op_meta))->output(0);
}

} // namespace graph
} // namespace hetu
