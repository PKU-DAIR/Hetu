#include "hetu/graph/ops/variable.h"
#include "hetu/graph/headers.h"

namespace hetu {
namespace graph {

bool VariableOpImpl::DoInstantiate(Operator& op, const Device& placement,
                                   StreamIndex stream_id) const {
  if (!OpInterface::DoInstantiate(op, placement, stream_id))
    return false;

  if (_init != nullptr) {
    Graph::AllocVariableData(op->output(0), *_init);
  } else {
    if (_copy_provided_data || dtype() != _provided_data->dtype() ||
        placement != _provided_data->device()) {
      Graph::AllocVariableData(op->output(0),
                               ProvidedInitializer(_provided_data));
      // TODO: free the provided data in order to save memory
    } else {
      Graph::RegisterVariableData(op->output(0), _provided_data);
    }
  }
  return true;
}

NDArrayList VariableOpImpl::DoAllocOutputs(Operator& op,
                                           const NDArrayList& inputs,
                                           RuntimeContext& runtime_ctx) const {
  return {Graph::GetVariableData(op->output(0))};
}

bool ParallelVariableOpImpl::DoInstantiate(Operator& op, 
                                           const Device& placement,
                                           StreamIndex stream_id) const {
  HT_ASSERT(_init == nullptr || _local_idx != -1)
    << "ParallelVariableOp: when use initializer, local_idx "
    << "must be assigned when local_device is in pipeline device_group!";
                                              
  if (!OpInterface::DoInstantiate(op, placement, stream_id))
    return false;

  if (_init != nullptr) {
    int32_t dup_group_idx = _ds.get_dup_group_index(_local_idx);
    // support 100 different duplicate group to set different seed
    uint64_t seed = 2023 + op->id() * 100 + dup_group_idx;
    HT_LOG_DEBUG << hetu::impl::comm::GetLocalDevice() << ": " << op << " inits by initializer.";
    // TODO: reset variable data also need parallel version
    Graph::AllocVariableData(op->output(0), *_init, seed, _global_shape);
  } else {
    if (_copy_provided_data || dtype() != _provided_data->dtype() ||
        placement != _provided_data->device()) {
      HT_LOG_DEBUG << hetu::impl::comm::GetLocalDevice() << ": " << op << " inits by provided data.";
      Graph::AllocVariableData(op->output(0),
                               ProvidedInitializer(_provided_data));
      // free the provided data in order to save memory, may still have some problem?
      _provided_data = hetu::NDArray();
    } else {
      Graph::RegisterVariableData(op->output(0), _provided_data);
    }
  }
  return true;
}

NDArrayList ParallelVariableOpImpl::DoAllocOutputs(Operator& op,
                                                   const NDArrayList& inputs,
                                                   RuntimeContext& runtime_ctx) const {
  return {Graph::GetVariableData(op->output(0))};
}

// in_degree=0, should set distributed states manually
Tensor MakeVariableOp(const Initializer& init, HTShape shape, 
                      DataType dtype, bool requires_grad, 
                      const DistributedStates& ds, OpMeta op_meta) {
  auto out = Graph::MakeOp(std::make_shared<VariableOpImpl>(
                           init, std::move(shape), dtype, requires_grad), TensorList(), 
                           std::move(op_meta.set_is_deduce_states(false)))->output(0);
  if (!ds.is_none()) {
    HT_ASSERT(ds.is_valid() && ds.get_dim(-2) == 1)
      << "DistributedStates for VariableOp must be valid! got: " 
      << ds.ds_info();
    out->set_distributed_states(ds);
  }
  return out;
}

// in_degree=0, should set distributed states manually
Tensor MakeVariableOp(NDArray provided_data, bool copy_provided_data, 
                      DataType dtype, bool requires_grad, 
                      const DistributedStates& ds, OpMeta op_meta) {
  auto out = Graph::MakeOp(std::make_shared<VariableOpImpl>(
                           std::move(provided_data), copy_provided_data, dtype,
                           requires_grad), TensorList(), 
                           std::move(op_meta.set_is_deduce_states(false)))->output(0);
  if (!ds.is_none()) {
    HT_ASSERT(ds.is_valid() && ds.get_dim(-2) == 1)
      << "DistributedStates for VariableOp must be valid! got: " 
      << ds.ds_info();
    out->set_distributed_states(ds);
  }
  return out;
}

Tensor MakeParameterOp(const Initializer& init, HTShape shape, 
                       DataType dtype, bool requires_grad, 
                       const DistributedStates& ds, OpMeta op_meta) {
  auto out = MakeVariableOp(init, std::move(shape), dtype, 
                            requires_grad, ds, std::move(op_meta));
  Graph::MarkAsParameter(out);
  return out;
}

Tensor MakeParameterOp(NDArray provided_data, bool copy_provided_data, 
                       DataType dtype, bool requires_grad, 
                       const DistributedStates& ds, OpMeta op_meta) {
  auto out = MakeVariableOp(std::move(provided_data), copy_provided_data, 
                            dtype, requires_grad, ds, std::move(op_meta));
  Graph::MarkAsParameter(out);
  return out;
}

Tensor MakeParallelVariableOp(const Initializer& init, HTShape global_shape, 
                              const DistributedStates& ds, int64_t local_idx,
                              DataType dtype, bool requires_grad, OpMeta op_meta) {
  // auto placement_group = op_meta.device_group;
  auto out = Graph::MakeOp(std::make_shared<ParallelVariableOpImpl>(
                           init, std::move(global_shape), ds, 
                           local_idx, dtype, requires_grad),
                           TensorList(), std::move(op_meta.set_is_deduce_states(false)))->output(0);
  HT_ASSERT(ds.is_valid() && ds.get_dim(-2) == 1)
    << "DistributedStates for ParallelVariableOp must be valid! got: " 
    << ds.ds_info();
  out->set_distributed_states(ds);
  // deprecated
  // If we set the placement group right now,
  // we can load the checkpoint easier in 3D parallel situation.
  // out->set_placement_group(placement_group); 
  // HT_LOG_TRACE << "MakeParallelVariableOp, placement group is: " << out->placement_group();
  return out;
}

Tensor MakeParallelVariableOp(NDArray provided_data, const DistributedStates& ds, 
                              bool copy_provided_data, DataType dtype, 
                              bool requires_grad, OpMeta op_meta) {
  // auto placement_group = op_meta.device_group;
  auto out = Graph::MakeOp(std::make_shared<ParallelVariableOpImpl>(
                           provided_data, copy_provided_data, 
                           ds, dtype, requires_grad),
                           TensorList(), std::move(op_meta.set_is_deduce_states(false)))->output(0);
  HT_ASSERT(ds.is_valid() && ds.get_dim(-2) == 1)
    << "DistributedStates for ParallelVariableOp must be valid! got: " 
    << ds.ds_info();
  out->set_distributed_states(ds);
  // deprecated
  // If we set the placement group right now,
  // we can load the checkpoint easier in 3D parallel situation.
  // out->set_placement_group(placement_group);
  // HT_LOG_TRACE << "MakeParallelVariableOp, placement group is: " << out->placement_group();
  return out;  
}

Tensor MakeParallelParameterOp(const Initializer& init, HTShape global_shape, 
                               const DistributedStates& ds, int64_t local_idx,
                               DataType dtype, bool requires_grad, OpMeta op_meta) {
  auto out = MakeParallelVariableOp(init, std::move(global_shape), ds, local_idx, 
                                    dtype, requires_grad, std::move(op_meta));
  // HT_LOG_INFO << out->producer() << ": device group = " << op_meta.device_group;                                    
  Graph::MarkAsParameter(out);
  return out;                                    
}

Tensor MakeParallelParameterOp(NDArray provided_data, const DistributedStates& ds, 
                               bool copy_provided_data, DataType dtype, 
                               bool requires_grad, OpMeta op_meta) {    
  auto out = MakeParallelVariableOp(std::move(provided_data), ds, copy_provided_data, 
                                    dtype, requires_grad, std::move(op_meta));
  // HT_LOG_INFO << out->producer() << ": device group = " << op_meta.device_group;                                    
  Graph::MarkAsParameter(out);
  return out;
}

} // namespace graph
} // namespace hetu
