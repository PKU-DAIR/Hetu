#include "hetu/graph/headers.h"
#include "hetu/graph/ops/optimizer_update.h"
#include "hetu/graph/ops/Communication.h"
#include "hetu/impl/communication/nccl_comm_group.h"
#include "hetu/impl/communication/comm_group.h"
#include "hetu/graph/ops/kernel_links.h"

namespace hetu {
namespace graph {

bool OptimizerUpdateOpInterface::DoMapToParallelDevices(
  Operator& op, const DeviceGroupUnion& placement_group_union) const {
  // use comm_op instead
  // if (placement_group.num_devices() > 1) {
  //   // TODO
  //   HT_NOT_IMPLEMENTED << "Fill this up with AllReduceOpImpl";
  // }
  return OpInterface::DoMapToParallelDevices(op, placement_group_union);
}

void SGDUpdateOpImpl::DoCompute(Operator& op, const NDArrayList& inputs,
                                NDArrayList& outputs,
                                RuntimeContext& runtime_ctx) const {
  NDArray& param = outputs.at(0);
  const NDArray& grad = inputs.at(1);
  NDArray velocity;
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(),
                                  type(), hetu::impl::SGDUpdate, grad, param,
                                  velocity, learning_rate(), 0, false,
                                  op->instantiation_ctx().stream());
}

void SGDUpdateWithGradScalerOpImpl::DoCompute(Operator& op, const NDArrayList& inputs,
                                              NDArrayList& outputs,
                                              RuntimeContext& runtime_ctx) const {
  NDArray& param = outputs.at(0);
  const NDArray& grad = inputs.at(1);
  const NDArray& infinite_count = inputs.at(2);
  NDArray velocity;
  HT_DISPATCH_KERNEL_CUDA_ONLY(op->instantiation_ctx().placement.type(),
                               type(), hetu::impl::SGDUpdateWithGradScaler, grad, infinite_count, 
                               param, velocity, learning_rate(), 0, false,
                               op->instantiation_ctx().stream());
}

void MomentumUpdateOpImpl::DoCompute(Operator& op, const NDArrayList& inputs,
                                     NDArrayList& outputs,
                                     RuntimeContext& runtime_ctx) const {
  NDArray& param = outputs.at(0);
  const NDArray& grad = inputs.at(1);
  NDArray velocity = inputs.at(2);
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(),
                                  type(), hetu::impl::SGDUpdate, grad, param,
                                  velocity, learning_rate(), 0, false,
                                  op->instantiation_ctx().stream());
}

void AdamOpImpl::DoCompute(Operator& op, const NDArrayList& inputs,
                           NDArrayList& outputs,
                           RuntimeContext& runtime_ctx) const {
  HT_ASSERT(outputs.at(0)->is_equal(inputs.at(0)))
    << "AdamOp should be inplace, please ensure you have overrided the DoAllocOutputs()";
  NDArray& param = outputs.at(0);
  const NDArray& grad = inputs.at(1);
  NDArray& mean = const_cast<NDArray&>(inputs.at(2));
  NDArray& variance = const_cast<NDArray&>(inputs.at(3));
  NDArray& step = const_cast<NDArray&>(inputs.at(4));
  int64_t step_num = inputs.at(4)->item<int64_t>();
  // TODO: 目前只考虑grad和param的ds一致的情况
  // 后续要支持param冗余
  HT_ASSERT(op->input(0)->cur_ds_union().check_equal(op->input(1)->cur_ds_union()))
    << "Currently only support equal ds union for param and grad";
  // 这里直接更新即可
  // 如何得到符合ds的grad以及后续的transfer param则交给exec graph中的comm op来做
  HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(),
                                  type(), hetu::impl::Adam, grad, param,
                                  mean, variance, step, learning_rate(step_num), 
                                  beta1(), beta2(), eps(), weight_decay(step_num), true,
                                  op->instantiation_ctx().stream());
}

// old version: manually implement zero
/*
void AdamOpImpl::DoCompute(Operator& op, const NDArrayList& inputs,
                           NDArrayList& outputs,
                           RuntimeContext& runtime_ctx) const {
  NDArray& param = outputs.at(0);
  const NDArray& grad = inputs.at(1);
  NDArray& mean = const_cast<NDArray&>(inputs.at(2));
  NDArray& variance = const_cast<NDArray&>(inputs.at(3));
  NDArray& step = const_cast<NDArray&>(inputs.at(4));
  int64_t step_num = inputs.at(4)->item<int64_t>();
  // 不开zero
  if (!_multi_zero.at(op->graph().OPTIMIZE_STRATEGY_ID)) {
    HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(),
                                    type(), hetu::impl::Adam, grad, param,
                                    mean, variance, step, learning_rate(step_num), 
                                    beta1(), beta2(), eps(), weight_decay(step_num), true,
                                    op->instantiation_ctx().stream());
  } 
  // 开zero
  else {
    // homo
    if (is_reduce_scatter_op(op->input(1)->producer())) {
      // param is dup, should split as reduce-scatter
      // partial_grad -> reduce-scatter -> scatter_grad, use partial_grad distributed states to deduce scatter info (offset, index, comm_group...)
      auto& reduce_scatter_op = op->input(1)->producer();
      auto& reduce_scatter_impl = reinterpret_cast<ReduceScatterOpImpl&>(reduce_scatter_op->body());
      auto& partial_grad = reduce_scatter_op->input(0);
      DeviceGroup comm_group = reduce_scatter_impl.comm_group();
      int32_t scatter_dim = reduce_scatter_impl.get_scatter_dim();
      // HT_LOG_WARN << op << " comm group: " << comm_group;
      auto local_device_index = op->output(0)->local_placement_group().get_index(op->placement());
      auto scatter_num = comm_group.num_devices();
      HT_ASSERT(!partial_grad->cur_ds_union().is_hetero())
        << "Adam: reduce scatter shouln't have hetero grad";
      HT_ASSERT(scatter_num == partial_grad->get_local_distributed_states().get_dim(-2))
        << "Adam: comm_group num must equal to partial size!";
      auto param_size = param->numel();
      auto param_size_per_scatter = DIVUP(param_size, scatter_num); // todo: padding for reduce-scatter & all-gather
      auto scatter_index = partial_grad->get_local_distributed_states().map_device_to_state_index(local_device_index)[-2];
      auto param_start_index = param_size_per_scatter * scatter_index;
      auto param_end_index = param_start_index + param_size_per_scatter;
      HT_ASSERT(grad->numel() == param_size_per_scatter && param_end_index <= param_size) 
        << "now need param size can be div by dp group size! "
        << "got grad size = " << grad->numel() 
        << " vs. param_size_per_scatter = " 
        << param_size_per_scatter;
      auto param_scatter = NDArray(
        NDArrayMeta().set_shape(grad->shape())
                     .set_dtype(param->dtype())
                     .set_device(param->device()), 
        param->storage(), param->storage_offset() + param_start_index);
      // only update scatter part of param
      HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(),
                                      type(), hetu::impl::Adam, grad, param_scatter,
                                      mean, variance, step, learning_rate(step_num), 
                                      beta1(), beta2(), eps(), weight_decay(step_num), true,
                                      op->instantiation_ctx().stream());
      // in-place allgather
      // gather dim和scatter dim对齐
      HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(), 
                                      hetu::impl::AllGather, param_scatter, param, 
                                      comm_group, scatter_dim, op->instantiation_ctx().stream());
    }
    // hetero
    else if (is_split_reduce_scatter_op(op->input(1)->producer())) {
      auto& split_reduce_scatter_op = op->input(1)->producer();
      auto& split_reduce_scatter_impl = reinterpret_cast<SplitReduceScatterOpImpl&>(split_reduce_scatter_op->body());
      auto& partial_grad = split_reduce_scatter_op->input(0);
      const std::vector<DeviceGroupList>& comm_groups_list = split_reduce_scatter_impl.comm_groups_list();
      // HT_LOG_WARN << op << " comm group: " << comm_group;
      auto local_device_index = op->output(0)->local_placement_group().get_index(op->placement());
      auto scatter_num = comm_groups_list.at(0).at(0).num_devices();
      HT_ASSERT(partial_grad->cur_ds_union().hetero_dim() == -2)
        << "Adam: split reduce scatter should have hetero grad whose hetero dim is -2";
      HT_ASSERT(scatter_num == partial_grad->cur_ds_union().size())
        << "Adam: comm_group num must equal to hetero partial size!";
      auto split_num = split_reduce_scatter_impl.split_num();
      auto relative_param = param;
      if (is_reduce_scatter_op(op->input(1)->producer()->input(0)->producer())) {
        auto& reduce_scatter_op = op->input(1)->producer()->input(0)->producer();
        auto num_chunks = reduce_scatter_op->output(0)->local_placement_group().num_devices();
        auto idx = reduce_scatter_op->output(0)->local_placement_group().get_index(op->placement());
        relative_param = NDArray::split(relative_param, num_chunks).at(idx);
      }
      auto param_size_per_split = DIVUP(relative_param->numel(), split_num);
      auto param_size_per_scatter_per_split = DIVUP(param_size_per_split, scatter_num); // todo: padding for reduce-scatter & all-gather
      auto scatter_index = partial_grad->local_placement_group_idx();
      auto param_start_index = param_size_per_scatter_per_split * scatter_index;
      auto param_end_index = param_start_index + param_size_per_scatter_per_split;
      HT_ASSERT(grad->numel() / split_num == param_size_per_scatter_per_split && param_end_index <= param_size_per_split) 
        << "now need param size can be div by dp group size! "
        << "got grad size per split = " << grad->numel() 
        << " vs. param_size_per_scatter_per_split = " 
        << param_size_per_scatter_per_split;
      // 对每个split出来的micro block
      // 都进行Adam运算与AllGather
      NDArrayList split_param = NDArray::split(relative_param, split_num);
      NDArrayList split_grad = NDArray::split(grad, split_num);
      NDArrayList split_mean = NDArray::split(mean, split_num);
      NDArrayList split_variance = NDArray::split(variance, split_num);
      for (size_t i = 0; i < split_num; i++) {
        auto split_param_scatter = NDArray(
          NDArrayMeta().set_shape(split_grad.at(i)->shape())
                       .set_dtype(split_param.at(i)->dtype())
                       .set_device(split_param.at(i)->device()), 
          split_param.at(i)->storage(), split_param.at(i)->storage_offset() + param_start_index);
        // only update scatter part of param
        // 注意这里只有最后一次需要更新step
        HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(),
                                        type(), hetu::impl::Adam, split_grad.at(i), split_param_scatter,
                                        split_mean.at(i), split_variance.at(i), step, learning_rate(step_num), 
                                        beta1(), beta2(), eps(), weight_decay(step_num), i == split_num - 1 ? true : false,
                                        op->instantiation_ctx().stream());
        const auto& comm_groups = comm_groups_list.at(i);
        for (const auto& comm_group : comm_groups) {
          // in-place allgather
          // 目前只支持在第0维split scatter和split gather
          HT_DISPATCH_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(),
                                          hetu::impl::AllGather, split_param_scatter,
                                          split_param.at(i), comm_group, 0,
                                          op->instantiation_ctx().stream());
        }
      }
    }
    // 不可能有其他情形
    else {
      HT_RUNTIME_ERROR << "Adam: zero input grad must be (split)-reduce-scatter result!"
        << ", grad producer = " << op->input(1)->producer()
        << ", grad = " << op->input(1)
        << ", param = " << op->input(0)
        << ", grad ds = " << op->input(1)->get_distributed_states().ds_info()
        << ", param ds = " << op->input(0)->get_distributed_states().ds_info();
    }
  }
}
*/

// support zero
// update op needn't deduce states
void AdamOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                const OpMeta& op_meta,
                                const InstantiationContext& inst_ctx) const {
  HT_RUNTIME_ERROR << "optimizer op shouldn't deduce states";
  const DistributedStates& ds_param = inputs.at(0)->get_distributed_states();
  const DistributedStates& ds_grad = inputs.at(1)->get_distributed_states();
  const DistributedStates& ds_mean = inputs.at(2)->get_distributed_states();
  const DistributedStates& ds_variance = inputs.at(3)->get_distributed_states();
  const DistributedStates& ds_step = inputs.at(4)->get_distributed_states();
  if (_multi_zero.at(Graph::GetGraph(Graph::cur_graph_ctx()).OPTIMIZE_STRATEGY_ID)) {
    HT_ASSERT(ds_param.check_equal(ds_grad) && ds_mean.check_equal(ds_variance) && ds_param.check_equal(ds_mean))
      << "DistributedStates for param, grad, mean, variance should be equal!";
  } else {
    HT_ASSERT(ds_mean.check_equal(ds_variance) && ds_grad.check_equal(ds_mean))
      << "DistributedStates for grad, mean, variance should be equal for zero!";    
  }
  outputs.at(0)->set_distributed_states(ds_param);
}

// update op needn't deduce states
void AdamOpImpl::DoDeduceHeterProp(const std::vector<int32_t>& inputs_hetero_dim,
                                   TensorList& outputs, const OpMeta& op_meta,
                                   const InstantiationContext& inst_ctx) const {
  HT_RUNTIME_ERROR << "optimizer op shouldn't deduce states";
  outputs.at(0)->cur_ds_union().set_hetero_dim(inputs_hetero_dim.at(0));
}

void AdamOpImpl::DoSpecialMergeStrategy(Operator& op, Operator& another_op) {
  HT_ASSERT(is_adam_op(op) && is_adam_op(another_op))
    << "two ops should both be adam ops";
  auto& another_op_impl = dynamic_cast<AdamOpImpl&>(another_op->body());
  _multi_zero.insert(_multi_zero.end(), another_op_impl.multi_zero().begin(), another_op_impl.multi_zero().end());
  HT_ASSERT(_multi_zero.size() == op->graph().NUM_STRATEGY)
    << "size mismatch";
}

Tensor MakeSGDUpdateOp(Tensor param, Tensor grad, OptimizerParamScheduler param_scheduler,
                       OpMeta op_meta) {
  return Graph::MakeOp(std::make_shared<SGDUpdateOpImpl>(param_scheduler),
                       {std::move(param), std::move(grad)}, std::move(op_meta))
    ->output(0);
}

Tensor MakeSGDUpdateWithGradScalerOp(Tensor param, Tensor grad, Tensor infinite_count, 
                                     OptimizerParamScheduler param_scheduler, OpMeta op_meta) {
  return Graph::MakeOp(std::make_shared<SGDUpdateWithGradScalerOpImpl>(param_scheduler),
                       {std::move(param), std::move(grad), std::move(infinite_count)}, std::move(op_meta))
    ->output(0);
}


Tensor MakeMomentumUpdateOp(Tensor param, Tensor grad, Tensor velocity,
                            OptimizerParamScheduler param_scheduler, float momentum, bool nesterov,
                            OpMeta op_meta) {
  return Graph::MakeOp(std::make_shared<MomentumUpdateOpImpl>(
                         param_scheduler, momentum, nesterov),
                       {std::move(param), std::move(grad), std::move(velocity)},
                       std::move(op_meta))
    ->output(0);
}

Tensor MakeAdamOp(Tensor param, Tensor grad, Tensor mean, Tensor variance,
                   OptimizerParamScheduler param_scheduler, Tensor step, float beta1, float beta2, 
                  float eps, OpMeta op_meta) {
  // pure tp needn't zero 
  std::vector<bool> multi_zero; 
  multi_zero.reserve(param->ds_hierarchy().size());   
  for (const auto& ds_union : param->ds_hierarchy().raw_data()) {
    auto ds = ds_union.get(0);
    bool zero = (ds.get_dim(-1) > 1) && ds.zero();
    multi_zero.push_back(zero);
  }                 
  // HT_LOG_INFO << hetu::impl::comm::GetLocalDevice() << ": MakeAdamOp: param = " << param << ", multi zero = " << multi_zero;
  return Graph::MakeOp(std::make_shared<AdamOpImpl>(
                       param_scheduler, multi_zero, beta1, beta2, eps),
                       {std::move(param), std::move(grad), std::move(mean), std::move(variance), std::move(step)},
                       std::move(op_meta))
    ->output(0);
}

} // namespace graph
} // namespace hetu
