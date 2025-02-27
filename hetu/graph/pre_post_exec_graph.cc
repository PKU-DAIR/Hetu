#include "hetu/graph/executable_graph.h"
#include "hetu/graph/switch_exec_graph.h"
#include "hetu/graph/operator.h"
#include "hetu/graph/subgraph.h"
#include "hetu/impl/communication/comm_group.h"
#include "hetu/impl/communication/nccl_comm_group.h"

namespace hetu {
namespace graph {

void ExecutableGraph::sort_optimize_compute_bridge_subgraph() {
  HT_ASSERT(_optimize_compute_bridge_subgraph_sorted.empty()) 
    << "can only sort once";
  for (const auto& entry : _optimize_compute_bridge_subgraph_map) {
    _optimize_compute_bridge_subgraph_sorted.push_back(entry);
  }
  // workaround: 按照键（param_id）进行降序排序是实际layer的升序
  std::sort(_optimize_compute_bridge_subgraph_sorted.begin(), _optimize_compute_bridge_subgraph_sorted.end(), 
    [](const std::pair<OpId, std::shared_ptr<SubGraph>>& a, const std::pair<OpId, std::shared_ptr<SubGraph>>& b) {
      return a.first > b.first;
  });
}

void ExecutableGraph::sort_compute_optimize_bridge_subgraph() {
  HT_ASSERT(_compute_optimize_bridge_subgraph_sorted.empty()) 
    << "can only sort once";
  for (const auto& entry : _compute_optimize_bridge_subgraph_map) {
    _compute_optimize_bridge_subgraph_sorted.push_back(entry);
  }
  // workaround: 按照键（param_id）进行升序排序是实际layer的降序
  std::sort(_compute_optimize_bridge_subgraph_sorted.begin(), _compute_optimize_bridge_subgraph_sorted.end(), 
    [](const std::pair<OpId, std::shared_ptr<SubGraph>>& a, const std::pair<OpId, std::shared_ptr<SubGraph>>& b) {
      return a.first < b.first;
  });
}


void ExecutableGraph::ResetVariableDataInner(const Tensor& tensor,
                                             const Initializer& init) {
  if (tensor->placement().is_undetermined()) {
    _add_on_inits[tensor->id()] = std::unique_ptr<Initializer>(init.copy());
  } else {
    init.Init(GetVariableDataInner(tensor));
  }
}

NDArray& ExecutableGraph::GetVariableDataInner(const Tensor& tensor) {
  auto it = _preserved_data.find(tensor->id());
  HT_RUNTIME_ERROR_IF(it == _preserved_data.end())
    << "Cannot find data for variable tensor " << tensor;
  return it->second;
}

NDArray ExecutableGraph::GetDetachedVariableDataInner(const Tensor& tensor) {
  // Question: store the data on different devices? For now, store all on CPU and return.
  auto it_1 = _preserved_data.find(tensor->id());
  if (it_1 == _preserved_data.end()) {
    auto it_2 = _add_on_inits.find(tensor->id());
    // haven't alloc yet
    if (it_2 != _add_on_inits.end()) {
      auto ret = NDArray::empty(tensor->shape(), Device(kCPU), tensor->dtype(), kBlockingStream);
      HT_LOG_TRACE << "The data is in executable graph, but not allocated yet, so getting the data of the variable from its initializer.";
      it_2->second->Init(ret);
      return ret;
    }
    else {
      HT_RUNTIME_ERROR << "Cannot find data in executable graph for variable tensor " << tensor;
    }
  }
  HT_LOG_TRACE << "Fetch the data from the executable graph.";
  return NDArray::to(it_1->second, Device(kCPU));
}

NDArray& ExecutableGraph::AllocVariableDataInner(const Tensor& tensor,
                                                 const Initializer& init,
                                                 uint64_t seed,
                                                 const HTShape& global_shape) {
  if (_preserved_data.find(tensor->id()) != _preserved_data.end()) {
    // HT_LOG_DEBUG << hetu::impl::comm::GetLocalDevice() << ": exec variable " << tensor << " already has the data, so we directly return it";
    return _preserved_data[tensor->id()];
  }
  HT_LOG_DEBUG << hetu::impl::comm::GetLocalDevice() << ": alloc exec variable " << tensor;
  // TODO: check meta is valid & maybe we can use non-blocking stream?
  bool is_param = (_parameter_ops.find(tensor->producer()->id()) != _parameter_ops.end() && tensor->requires_grad());
  bool is_optvar = (_optimizer_variable_ops.find(tensor->producer()->id()) != _optimizer_variable_ops.end());
  // TODO: better compatibility with hot switch and quantization
  if ((_use_origin_param_and_optimizer_buffer || _use_origin_param_and_optimizer_buckets) && (is_param || is_optvar)) {
    if (_use_origin_param_and_optimizer_buckets) {
      HT_ASSERT(_origin_param_and_optimizer_buckets_map[tensor->dtype()]->HasTensor(tensor))
        << "Cannot find param " << tensor << " in the origin param and optimizer buckets";
      // must have alloced in AllocRuntimeBuffer
      auto bucket = _origin_param_and_optimizer_buckets_map[tensor->dtype()]->GetTensorBucket(tensor);
      HT_ASSERT(bucket->IsAllocated())
        << "must have alloced in AllocRuntimeBuffer, but found " << tensor << " not allocated";
      _preserved_data[tensor->id()] = NDArray(tensor->meta(), 
                                              bucket->AsStorage(), 
                                              bucket->GetElementOffset(tensor));
    }
    // deprecated: 目前使用buckets
    else if (_use_origin_param_and_optimizer_buffer) {
      HT_RUNTIME_ERROR << "deprecated";
      /*
      HT_ASSERT(_origin_param_and_optimizer_buffer->HasTensor(tensor))
        << "Cannot find param " << tensor << " in the origin param and optimizer buffer";
      // alloc on-the-fly
      if (!_origin_param_and_optimizer_buffer->IsAllocated()) {
        _origin_param_and_optimizer_buffer->Alloc(Stream(tensor->placement(), kBlockingStream));
      }
      _preserved_data[tensor->id()] = NDArray(tensor->meta(), 
                                              _origin_param_and_optimizer_buffer->AsStorage(), 
                                              _origin_param_and_optimizer_buffer->GetElementOffset(tensor));
      */
    }
  } 
  // deprecated:
  // 目前一定会使用origin_param_and_optimizer_buffer或者buckets
  else if (!_use_origin_param_and_optimizer_buffer && !_use_origin_param_and_optimizer_buckets && is_param) {
    HT_RUNTIME_ERROR << "deprecated";
    /*
    HT_ASSERT(_origin_param_buffer->HasTensor(tensor))
      << "Cannot find param " << tensor << " in the origin param buffer";
    // alloc on-the-fly
    if (!_origin_param_and_optimizer_buckets_map[tensor->dtype()]->IsAllocated()) {
      _origin_param_and_optimizer_buckets_map[tensor->dtype()]->Alloc(Stream(tensor->placement(), kBlockingStream));
    }
    _preserved_data[tensor->id()] = NDArray(tensor->meta(), 
                                            _origin_param_buffer->AsStorage(), 
                                            _origin_param_buffer->GetElementOffset(tensor));
    */
  }
  // 其余不在buffer中
  else {
    // 另外一些是variable但不是param/optvar的正常走mempool
    // 分配的是碎片化的显存
    // mempool debug use
    HT_LOG_TRACE << hetu::impl::comm::GetLocalDevice() << ": on-the-fly alloc variable " << tensor
      << ", shape = " << tensor->shape() << ", placement = " << tensor->placement();
    _preserved_data[tensor->id()] = NDArray::empty(tensor->shape(), 
                                                   tensor->placement(), 
                                                   tensor->dtype(), 
                                                   kBlockingStream);
  }
  auto it = _add_on_inits.find(tensor->id());
  if (it != _add_on_inits.end()) {
    it->second->Init(_preserved_data[tensor->id()], seed, global_shape,
                     kBlockingStream);
  } else if (!init.vodify()) {
    init.Init(_preserved_data[tensor->id()], seed, global_shape,
              kBlockingStream);
  }
  return _preserved_data[tensor->id()];
}

void ExecutableGraph::RegisterVariableDataInner(const Tensor& tensor,
                                                NDArray data,
                                                const Initializer& init) {
  _preserved_data[tensor->id()] = std::move(data);
  auto it = _add_on_inits.find(tensor->id());
  if (it != _add_on_inits.end()) {
    it->second->Init(_preserved_data[tensor->id()]);
  } else if (!init.vodify()) {
    init.Init(_preserved_data[tensor->id()]);
  }
}

void ExecutableGraph::PreRun(std::vector<RuntimeContext>& runtime_ctx_list) {
  // some run-only-once (unrelated to pp) op could alloc and compute in advance
  // 1、fragile non-param varaible (alloc and compute)
  // 2、origin param (if needed, alloc and compute)
  // 3、transfer param (alloc and compute)
  // 4、grad (just alloc)
  auto local_device = hetu::impl::comm::GetLocalDevice();
  // ---------- param ----------
  if (_run_level != RunLevel::COMPUTE_ONLY 
      && !_is_transfer_param_hot_switch 
      && _use_origin_param_and_optimizer_buckets) {
    for (auto it = _origin_param_and_optimizer_buckets_map.begin();
         it != _origin_param_and_optimizer_buckets_map.end(); ++it) {
      const auto& buckets = it->second->buckets();
      for (const auto& bucket : buckets) {
        if (!bucket->IsEmpty() && !bucket->IsAllocated()) {
          // alloc origin param and opt var
          bucket->Alloc(Stream(local_device, kBlockingStream));
          HT_LOG_DEBUG << local_device << ": alloc origin param and opt var bucket"
            << ", the size is " << bucket->size();
        }
      }
    }
  }
  for (auto it = _transfer_param_buffer_map.begin();
       it != _transfer_param_buffer_map.end(); ++it) {
    if (!it->second->IsEmpty() && !it->second->IsAllocated()) {
      // alloc transfer param
      it->second->Alloc(Stream(local_device, kBlockingStream));
      HT_LOG_DEBUG << local_device << ": alloc transfer param buffer"
        << ", the size is " << it->second->size();
    }
  }
  // optimize-compute bridge subgraph单独处理
  for (const auto& [param_id, cur_subgraph] : _optimize_compute_bridge_subgraph_sorted) {
    bool is_param_local = false, is_transfer_param_local = false;
    auto& param_op = _op_indexing[param_id];
    auto it = _transfer_map.find(param_op->output(0)->id());
    HT_ASSERT(it != _transfer_map.end())
      << "The transfer map does not consist of " << param_op->output(0) << " " << param_op->output(0)->dtype();
    auto& transfer_param = it->second;
    // param是local的
    if (param_op->output(0)->placement() == local_device) {
      is_param_local = true;
    }
    // transfer param是local的
    if (transfer_param->placement() == local_device) {
      is_transfer_param_local = true;
      HT_ASSERT(!_transfer_param_buffer_map[transfer_param->dtype()]->IsEmpty())
        << "The transfer param buffer of " << DataType2Str(transfer_param->dtype()) << " should not be empty";
      auto transfer_param_data = NDArray(transfer_param->meta(),
                                          _transfer_param_buffer_map[transfer_param->dtype()]->AsStorage(), 
                                          _transfer_param_buffer_map[transfer_param->dtype()]->GetElementOffset(transfer_param));
      // 添加runtime allocation
      // HT_LOG_INFO << transfer_param << " id is " << transfer_param->id() << ", add runtime allocation";
      for (auto& runtime_ctx : runtime_ctx_list) {
        runtime_ctx.add_runtime_allocation(transfer_param->id(), transfer_param_data);
      }
      // 热切换
      if (_is_transfer_param_hot_switch) {
        HT_ASSERT(_preserved_data.find(transfer_param->id()) != _preserved_data.end())
          << "exec transfer param " << transfer_param << " should already has the data if it is hot switched";
      }
      // 冷启动
      else {
        _preserved_data[transfer_param->id()] = transfer_param_data;
      }
    }
    // 执行该subgraph
    // COMPUTE_ONLY模式或者热切换了transfer param后不需要再执行
    if (_run_level != RunLevel::COMPUTE_ONLY && !_is_transfer_param_hot_switch) {
      Tensor2NDArrayMap tensor2data = {};
      // param是local的则先执行
      if (is_param_local) {
        tensor2data[param_op->output(0)->id()] = param_op->Compute({}, runtime_ctx_list[0])[0];
      }
      // 实际执行
      // HT_LOG_INFO << cur_subgraph->global_name() << " run begin";
      cur_subgraph->run(tensor2data, {}, runtime_ctx_list[0]);
      // HT_LOG_INFO << cur_subgraph->global_name() << " run end";
      // subgraph彻底为空
      if (!is_param_local && !is_transfer_param_local) {
        HT_ASSERT(cur_subgraph->ops_topo().empty())
          << "optimize-compute bridge subgraph " << cur_subgraph->global_name() << " should be empty";
        /*
        ncclGroupStart_safe();
        ncclGroupEnd_safe();
        */
      }
      // origin param是本地但是transfer param不是
      else if (is_param_local && !is_transfer_param_local) {
        HT_ASSERT(cur_subgraph->ops_topo().back().get()->num_outputs() == 0
                  || cur_subgraph->ops_topo().back().get()->output(0)->num_consumers() == 0)
          << "nonlocal transfer param optimize-compute bridge subgraph " << cur_subgraph->global_name() << " should have no useful output tensor";
        if (cur_subgraph->ops_topo().back().get()->num_outputs() == 0) {
          HT_ASSERT(tensor2data.size() == 0)
            << "nonlocal transfer param optimize-compute bridge subgraph " << cur_subgraph->global_name() << " should have no input or output NDarray after running";
        } else {
          HT_ASSERT(tensor2data.size() == 1)
            << "nonlocal transfer param optimize-compute bridge subgraph " << cur_subgraph->global_name() << " should only have one unuseful output NDarray after running";
        }
      }
      // transfer param是本地的
      else {
        HT_ASSERT(cur_subgraph->ops_topo().back().get()->num_outputs() == 1
                  && cur_subgraph->ops_topo().back().get()->output(0)->num_consumers() >= 1)
          << "local transfer param optimize-compute bridge subgraph " << cur_subgraph->global_name() << " should have only one useful output tensor";
        HT_ASSERT(tensor2data.size() == 1 && tensor2data.find(transfer_param->id()) != tensor2data.end())
          << "local transfer param optimize-compute bridge subgraph " << cur_subgraph->global_name() << " should have a single output NDarray after running";
      }
      // local param不需要grad时可以删除
      if (is_param_local && !param_op->output(0)->requires_grad()) {
        auto data_it = _preserved_data.find(param_op->output(0)->id());
        HT_ASSERT(data_it != _preserved_data.end());
        _preserved_data.erase(data_it);
      }
    }
    // 最后集中添加runtime skipped
    for (auto& runtime_ctx : runtime_ctx_list) {
      runtime_ctx.add_runtime_skipped(param_op->id());
      // 这里将subgraph的所有op标记为skip
      for (auto& op_ref : cur_subgraph->ops_topo()) {
        runtime_ctx.add_runtime_skipped(op_ref.get()->id());
      }
    }
  }
  // 其余var直接正常compute
  for (const auto& op_ref : _execute_plan.local_placeholder_variable_ops) {
    auto& op = op_ref.get();
    if (is_variable_op(op) && _parameter_ops.find(op->id()) == _parameter_ops.end()) {
      // alloc阶段只分配param
      if (_run_level == RunLevel::ALLOC) {
        continue;
      }
      // compute_only和grad阶段optimizer相关的variable不用跑
      // 例如Adam的step、mean、variance
      if (_run_level == RunLevel::COMPUTE_ONLY || _run_level == RunLevel::GRAD) {
        if (op->output(0)->num_consumers() == 1 
            && is_optimizer_update_op(op->output(0)->consumer(0))) {
          continue;
        }
      }
      op->Compute({}, runtime_ctx_list[0]);
      // 添加runtime skipped
      for (auto& runtime_ctx : runtime_ctx_list) {
        runtime_ctx.add_runtime_skipped(op->id());
      }
    }
  }
  // ---------- grad ----------
  if (_run_level == RunLevel::GRAD || _run_level == RunLevel::UPDATE) {
    if (_use_current_grad_buffer) {
      for (auto it = _current_grad_buffer_map.begin();
           it != _current_grad_buffer_map.end(); ++it) {
        if (!it->second->IsEmpty() && !it->second->IsAllocated()) {
          // alloc current grad
          it->second->Alloc(Stream(local_device, kBlockingStream));
          HT_LOG_DEBUG << local_device << ": alloc current grad buffer "
            << ", the size is " << it->second->size();
        }
        for (const auto& current_grad : it->second->tensor_list()) {
          auto current_grad_data = NDArray(current_grad->meta(),
                                           it->second->AsStorage(), 
                                           it->second->GetElementOffset(current_grad));
          // 添加runtime allocation
          for (auto& runtime_ctx : runtime_ctx_list) {
            runtime_ctx.add_runtime_allocation(current_grad->id(), current_grad_data);
          }
          // 注意与param不同的是
          // 这里不能添加runtime skipped
          // 因为grad还是要计算的
        }
      }
    }
    // 使用accumulate_grad_buffer
    // 初始全为0
    else {
      if (_run_level == RunLevel::GRAD) {
        for (auto it = _accumulate_grad_buffer_map.begin();
             it != _accumulate_grad_buffer_map.end(); ++it) {
          if (!it->second->IsEmpty() && !it->second->IsAllocated()) {
            it->second->Alloc(Stream(local_device, kBlockingStream));
            HT_LOG_DEBUG << "accumulate_grad_buffer alloc.";
            auto accumulate_grad_buffer_data = it->second->AsNDArray();
            NDArray::zeros_(accumulate_grad_buffer_data, kBlockingStream);
          }
        }
      }
    }
  }
}

void ExecutableGraph::PostRun(std::vector<RuntimeContext>& runtime_ctx_list, Tensor2NDArrayMap& tensor2data) {
  // 主要负责grad reduce以及optimizer update相关操作
  HT_ASSERT(_run_level != RunLevel::COMPUTE_ONLY)
    << "RunLevel::COMPUTE_ONLY shouldn't call PostRun()";
  auto num_micro_batches = runtime_ctx_list.size();
  auto micro_batch_id = num_micro_batches - 1;
  for (const auto& [param_id, cur_subgraph] : _compute_optimize_bridge_subgraph_sorted) {
    // 执行该subgraph
    // HT_LOG_INFO << cur_subgraph->global_name() << " run begin";
    cur_subgraph->run(tensor2data, _preserved_data, runtime_ctx_list[micro_batch_id], micro_batch_id, SubGraphOpType::UPDATE, true,
                      [this](Operator& op, Tensor2NDArrayMap& tensor2data, size_t micro_batch_id) { return PostOpHandler(op, tensor2data, micro_batch_id); });
    // HT_LOG_INFO << cur_subgraph->global_name() << " run end";
  }
  _terminate_subgraph->run(tensor2data, _preserved_data, runtime_ctx_list[micro_batch_id], micro_batch_id, SubGraphOpType::UPDATE, true,
                           [this](Operator& op, Tensor2NDArrayMap& tensor2data, size_t micro_batch_id) { return PostOpHandler(op, tensor2data, micro_batch_id); });
}

OpHandlerStatus ExecutableGraph::PostOpHandler(Operator& op, Tensor2NDArrayMap& tensor2data, size_t micro_batch_id) {
  // HT_LOG_INFO << "PostOpHandler for " << op << " begin to run";
  OpHandlerStatus status;
  if (is_grad_reduce_op(op) && _overlap_grad_reduce) {
    // overlap_grad_reduce情形下把grad reduce op放入到了ComputeFunc中去执行
    // 因此这里直接跳过即可
    status.need_skip = true;
    return status;
  }
  if (is_group_op(op) && _run_level == RunLevel::GRAD) {
    status.need_skip = true;
    return status;
  }
  if (is_optimizer_update_op(op)) {
    // 只用得到grad而不需要进行update
    if (_run_level == RunLevel::GRAD) {
      auto& grad = op->input(1);
      auto& grad_op = grad->producer();
      // HT_LOG_INFO << "grad op " << grad_op << " placement is " << grad_op->placement();
      if (_use_current_grad_buffer) {
        // 什么都不用操作
      }
      // 不使用current_grad_buffer的话需要在这里直接将grad加到accumulate_grad_buffer上
      else {
        HT_ASSERT(tensor2data.find(grad->id()) != tensor2data.end());
        auto current_grad_data = tensor2data[grad->id()];
        HT_ASSERT(_accumulate_grad_buffer_map.find(grad->dtype()) != _accumulate_grad_buffer_map.end());
        auto accumulate_grad_data = NDArray(grad->meta(), 
                                            _accumulate_grad_buffer_map[grad->dtype()]->AsStorage(), 
                                            _accumulate_grad_buffer_map[grad->dtype()]->GetElementOffset(grad));
        auto grad_stream = grad_op->instantiation_ctx().stream(); 
        if (_grad_scale != 1) {
          NDArray::mul(current_grad_data,
                       _grad_scale,
                       grad_stream.stream_index(),
                       current_grad_data);
        }
        // 如果有一些累计梯度是switch过来的
        // 那么我们这里进行实际的sync
        auto event_it = _switch_grad_events.find(grad->id());
        if (event_it != _switch_grad_events.end()) {
          event_it->second->Block(grad_stream);
        } 
        NDArray::add(current_grad_data, 
                     accumulate_grad_data, 
                     grad_stream.stream_index(),
                     accumulate_grad_data);                                    
      }
      // 需要记录grad op的event来在结束时同步
      auto event = std::make_unique<hetu::impl::CUDAEvent>(grad_op->placement());
      event->Record(grad_op->instantiation_ctx().stream());
      _run_grad_events[grad->id()] = std::move(event);
      tensor2data.erase(grad); // 清除tensor2data中该grad的引用计数
      status.need_skip = true;
      return status;
    }
    // 要进行梯度更新
    else if (_run_level == RunLevel::UPDATE) {
      // 如果有累积梯度那么此时要加上
      // 这里的逻辑和上面的正好反过来
      if (_accumulate_grad_buffer_map[op->input(1)->dtype()]->IsAllocated()) {
        auto& grad = op->input(1);
        auto& grad_op = grad->producer();
        HT_ASSERT(tensor2data.find(grad->id()) != tensor2data.end());
        auto current_grad_data = tensor2data[grad->id()];
        auto accumulate_grad_data = NDArray(grad->meta(), 
                                            _accumulate_grad_buffer_map[grad->dtype()]->AsStorage(), 
                                            _accumulate_grad_buffer_map[grad->dtype()]->GetElementOffset(grad));
        auto grad_stream = Stream(grad_op->placement(),
                                  grad_op->instantiation_ctx().stream_index);
        if (_grad_scale != 1) {
          NDArray::mul(current_grad_data,
                       _grad_scale,
                       grad_stream.stream_index(),
                       current_grad_data);
        }
        // 如果有一些累计梯度是switch过来的
        // 那么我们这里进行实际的sync
        auto event_it = _switch_grad_events.find(grad->id());
        if (event_it != _switch_grad_events.end()) {
          event_it->second->Block(grad_stream);
        } 
        NDArray::add(current_grad_data, 
                     accumulate_grad_data, 
                     grad_stream.stream_index(),
                     current_grad_data);
        // 需要重新设置grad op的stop event来保证update算子的输入是sync的
        grad->producer()->instantiation_ctx().stop[micro_batch_id]->Record(grad_stream);
      }
    }
    // 其余情况不可能发生
    else {
      HT_RUNTIME_ERROR << "run level error";
    }
  }
  return status;
}

void ExecutableGraph::AllocMemory(size_t& memory_size, MemoryPlan& memory_plan,
                                  MemoryBlockList& temporary_free_memory, MemoryBlockList& free_memory, MicroBatchTensorId tensor_id,
                                  size_t alloc_memory_size) {
  // Best Fit strategy
  sort(temporary_free_memory.begin(), temporary_free_memory.end(),
       [&](MemoryBlock a, MemoryBlock b) { return a.second < b.second; });

  // 找temp free memory中最小的能容纳下的块
  // 并进行split
  for (auto block_iter = temporary_free_memory.begin(); block_iter != temporary_free_memory.end(); block_iter++) {
    auto block_size = block_iter->second;
    if (block_size >= alloc_memory_size) {
      auto block_ptr = block_iter->first;
      temporary_free_memory.erase(block_iter);
      memory_plan[tensor_id] = {block_ptr, alloc_memory_size};
      auto remain_size = block_size - alloc_memory_size;
      if (remain_size > 0) {
        temporary_free_memory.push_back({block_ptr + alloc_memory_size, remain_size});
      }
      return;
    }
  }

  sort(free_memory.begin(), free_memory.end(),
       [&](MemoryBlock a, MemoryBlock b) { return a.second < b.second; });

  // 同上
  // free memory更珍贵
  for (auto block_iter = free_memory.begin(); block_iter != free_memory.end(); block_iter++) {
    auto block_size = block_iter->second;
    if (block_size >= alloc_memory_size) {
      auto block_ptr = block_iter->first;
      free_memory.erase(block_iter);
      memory_plan[tensor_id] = {block_ptr, alloc_memory_size};
      auto remain_size = block_size - alloc_memory_size;
      if (remain_size > 0) {
        free_memory.push_back({block_ptr + alloc_memory_size, remain_size});
      }
      return;
    }
  }

  memory_plan[tensor_id] = {memory_size, alloc_memory_size};
  memory_size += alloc_memory_size;
}

void ExecutableGraph::FreeMemory(MemoryPlan& memory_plan, MemoryBlockList& free_memory,
                                 MicroBatchTensorId tensor_id) {
  // free memory space and merge with adjacent free blocks
  auto free_block_ptr = memory_plan[tensor_id].first;
  auto free_block_size = memory_plan[tensor_id].second;
  for (auto i = 0; i < free_memory.size(); i++) {
    auto block_head = free_memory[i].first;
    auto block_tail = block_head + free_memory[i].second;
    if (block_tail == free_block_ptr) {
      free_block_ptr = block_head;
      free_block_size += free_memory[i].second;
      free_memory.erase(free_memory.begin() + i);
      i--;
    } else if (free_block_ptr + free_block_size == block_head) {
      free_block_size += free_memory[i].second;
      free_memory.erase(free_memory.begin() + i);
      i--;
    }
  }
  free_memory.push_back({free_block_ptr, free_block_size});
}

MemoryPlan ExecutableGraph::GenerateMemoryPlan(size_t& memory_size, std::vector<std::pair<bool, size_t>> tasks,
                                               std::vector<Tensor2IntMap> tensor2degrees_list,    
                                               const FeedDict& feed_dict){
  memory_size = 0;
  MemoryPlan memory_plan;
  MemoryBlockList temporary_free_memory[HT_NUM_STREAMS_PER_DEVICE];
  MemoryBlockList free_memory;
  std::map<MemoryBlock, int> storage_use_count;

  auto& subgraphs = GetAllSubGraphs();
  std::vector<OpList> block_ops;
  std::vector<std::string> block_name = {"GPTBlock", "LLamaBlock"};
  for (auto [name, subgraph] : subgraphs) {
    std::function<OpList(std::shared_ptr<SubGraph>)> get_all_ops = [&](std::shared_ptr<SubGraph> subgraph){
      OpList ops;
      for (auto [name, op] : subgraph->ops()) 
        ops.push_back(op);  
      for (auto [name, op] : subgraph->bwd_ops()) 
        ops.push_back(op); 
      for (auto [name, child] : subgraph->subgraphs()) {
        auto child_ops = get_all_ops(child);
        for (auto op : child_ops) 
          ops.push_back(op);
      }
      return ops;
    };
    for (auto bname : block_name) {
      if (subgraph->subgraph_type() == SubGraphType::MODULE && subgraph->module_type() == bname) {
        block_ops.push_back(get_all_ops(subgraph));
      }
    }
  }
  if (block_ops.size() <= 0) {
    HT_LOG_WARN << "The topology graph only supports segmentation using GPTBlock or LLamaBlock as the block, but find none of them. If you define other types of block, please add the class name to the list above.";
  }
  std::set<OpId> fw_block_start_op, fw_block_end_op;
  for (auto ops : block_ops) {
    std::vector<int> in_degree(ops.size(), 0), out_degree(ops.size(), 0);
    for (int i = 0; i < ops.size(); i++) {
      for (auto& output : ops[i]->outputs()) {
        for (auto consumer : output->consumers()) {
          for (int j = 0; j < ops.size(); j++) {
            if (consumer.get()->graph_id() == ops[j]->graph_id() && consumer.get()->id() == ops[j]->id()) {
              in_degree[j]++;
              out_degree[i]++;
            }
          }
        }
      }
    }
    int start_op_cnt = 0, end_op_cnt = 0;
    for (int i = 0; i < ops.size(); i++) {
      // workaround: comm op还会出现在subgraph中而被当成是block start/end op
      // 后续应该将comm op单独处理成bridge subgraph中（python端定义会用户不友好）
      if (is_comm_op(ops[i])) {
        HT_RUNTIME_ERROR << "subgraphs should not consists of comm op";
      }
      if (in_degree[i] == 0) {
        start_op_cnt++;
        fw_block_start_op.insert(ops[i]->id());
        HT_LOG_DEBUG << ops[i] << " is a block start op, the inputs are " << ops[i]->inputs();
      }
      if (out_degree[i] == 0) {
        end_op_cnt++;
        fw_block_end_op.insert(ops[i]->id());
        HT_LOG_DEBUG << ops[i] << " is a block end op, the inputs are " << ops[i]->inputs();
      }
    }
    HT_ASSERT(start_op_cnt == 1 && end_op_cnt == 1) 
      << "Each block only has a start operator and an end operator.";
  }

  for (size_t i = 0; i < tasks.size(); i++) {
    auto& task = tasks[i];
    bool is_forward = task.first;
    size_t& micro_batch_id = task.second;
    auto& tensor2degrees = tensor2degrees_list[micro_batch_id];
    bool grad_accumulation_finished = ((i == tasks.size() - 1) && is_forward == false);
    OpRefList &topo = is_forward ? _execute_plan.local_fw_topo : _execute_plan.local_bw_topo;
    const TensorIdSet& dtype_transfer_tensor = _execute_plan.dtype_transfer_tensor;
    const TensorIdSet& shared_weight_tensor = _execute_plan.shared_weight_tensor;
    const OpIdSet& shared_weight_p2p = _execute_plan.shared_weight_p2p;
    const OpIdSet& shared_weight_grad_p2p = _execute_plan.shared_weight_grad_p2p;
    const TensorIdSet& accumulated_tensor = _execute_plan.accumulated_tensor;
    const OpIdSet& accumulated_ops = _execute_plan.accumulated_ops;

    OpRefList executable_topo;
    for (auto& op_ref : topo) {
      auto& op = op_ref.get();
      bool computed = Operator::all_output_tensors_of(op, [&](Tensor& tensor) {
        return feed_dict.find(tensor->id()) != feed_dict.end();
      });
      if (computed ||
          op->num_outputs() > 0 && dtype_transfer_tensor.find(op->output(0)->id()) != dtype_transfer_tensor.end() && micro_batch_id > 0 ||
          !shared_weight_p2p.empty() && shared_weight_p2p.find(op->id()) != shared_weight_p2p.end() && micro_batch_id > 0 || 
          !grad_accumulation_finished && accumulated_ops.find(op->id()) != accumulated_ops.end()) {
        continue;
      }
      executable_topo.push_back(op_ref);
    }

    std::vector<MicroBatchTensorId> release_tensor;
    for (auto& op_ref : executable_topo) {
      auto& op = op_ref.get();
      auto clear_tensor_and_merge_space = [&](){
        for (auto& micro_tensor_id : release_tensor) {
          FreeMemory(memory_plan, free_memory, micro_tensor_id);
          storage_use_count[memory_plan[micro_tensor_id]] = 0;
          storage_use_count.erase(memory_plan[micro_tensor_id]);
        }
        release_tensor.clear();
        // 将所有stream上的temp free memory全部放回到free memory
        for (auto stream_id = 0; stream_id < HT_NUM_STREAMS_PER_DEVICE; stream_id++) {
          for (auto& space : temporary_free_memory[stream_id]) {
            auto free_block_ptr = space.first;
            auto free_block_size = space.second;
            for (auto i = 0; i < free_memory.size(); i++) {
              auto block_head = free_memory[i].first;
              auto block_tail = block_head + free_memory[i].second;
              if (block_tail == free_block_ptr) {
                free_block_ptr = block_head;
                free_block_size += free_memory[i].second;
                free_memory.erase(free_memory.begin() + i);
                i--;
              } else if (free_block_ptr + free_block_size == block_head) {
                free_block_size += free_memory[i].second;
                free_memory.erase(free_memory.begin() + i);
                i--;
              }
            }
            free_memory.push_back({free_block_ptr, free_block_size});
          }
          temporary_free_memory[stream_id].clear();
        }
      };

      // fw block的开头和bw block的结尾
      // 全部进行清空
      if (is_forward && fw_block_start_op.find(op->id()) != fw_block_start_op.end() || !is_forward && fw_block_end_op.find(op->fw_op_id()) != fw_block_end_op.end()) {
        clear_tensor_and_merge_space();
      }
      if (is_optimizer_update_op(op) || is_data_transfer_op(op)) {
        continue;
      }
      // TODO: maybe too heuristic
      // 可以reuse的会累积storage_use_count
      if (op->type() == "TransposeOp"|| is_slice_op(op) ||
          (op->type() == "ArrayReshapeOp" || op->type() == "ArrayReshapeGradientOp") && op->inputs().at(0)->is_contiguous() || 
          is_inplace_op(op) || is_all_reduce_op(op) || is_reduce_scatter_op(op)) {
        auto input_id = op->inputs().at(0)->id();
        auto output_id = op->outputs().at(0)->id();
        if (memory_plan.find({micro_batch_id, input_id}) != memory_plan.end()
            && storage_use_count.find(memory_plan[{micro_batch_id, input_id}]) != storage_use_count.end()
            && storage_use_count[memory_plan[{micro_batch_id, input_id}]] > 0) {
          memory_plan[{micro_batch_id, output_id}] = memory_plan[{micro_batch_id, input_id}];
          storage_use_count[memory_plan[{micro_batch_id, input_id}]] += tensor2degrees[output_id];
        }
      } 
      // 其余情况需要分配
      else {
        for (auto& output : op->outputs()) {
          auto tensor_id = output->id();
          int64_t numElem = output->numel();
          numElem = DIVUP(numElem * DataType2Size(output->dtype()), 256) * 256 / DataType2Size(kInt64);
          AllocMemory(memory_size, memory_plan, temporary_free_memory[op->stream_index()], free_memory, {micro_batch_id, tensor_id}, numElem);
          storage_use_count[memory_plan[{micro_batch_id, tensor_id}]] = tensor2degrees[tensor_id];
        }
      }

      for (size_t i = 0; i < op->num_outputs(); i++) {
        auto tensor_id = op->output(i)->id();
        if (memory_plan.find({micro_batch_id, tensor_id}) == memory_plan.end()) {
          // HT_LOG_WARN << op << " output: micro batch " << micro_batch_id << " tensor " << op->output(i) << " is not in memory_plan";
          continue;
        }
        if (storage_use_count.find(memory_plan[{micro_batch_id, tensor_id}]) == storage_use_count.end()) {
          // HT_LOG_WARN << op << " output: micro batch " << micro_batch_id << " tensor " << op->output(i) << " is not in storage_use_count";
          continue;
        }
        if (accumulated_tensor.find(tensor_id) != accumulated_tensor.end() 
            || storage_use_count[memory_plan[{micro_batch_id, tensor_id}]] == 0) {
          FreeMemory(memory_plan, temporary_free_memory[op->stream_index()], {micro_batch_id, tensor_id});
          storage_use_count[memory_plan[{ micro_batch_id, tensor_id}]] = 0;
          storage_use_count.erase(memory_plan[{ micro_batch_id, tensor_id}]);
          // if (op->placement().index() == 0) std::cout << "tensor " << tensor_id << ' ' << "free" << ' ' << memory_plan[{micro_batch_id, tensor_id}].first << ' ' << memory_plan[{micro_batch_id, tensor_id}].second << std::endl;
        }
      }

      for (const auto& input : op->inputs()) {
        auto used_by_multi_stream = [&](const Tensor& tensor) {
          for (auto &consumer_ref : tensor->consumers()) {
            auto& consumer = consumer_ref.get();
            if (consumer->stream_index() != tensor->producer()->stream_index()) {
              return true;
            }
          }
          return false;
        };
        if (memory_plan.find({micro_batch_id, input->id()}) == memory_plan.end()) {
          // HT_LOG_WARN << op << " input: micro batch " << micro_batch_id << " tensor " << input << " is not in memory_plan";
          continue;
        }
        if (storage_use_count.find(memory_plan[{micro_batch_id, input->id()}]) == storage_use_count.end()) {
          // HT_LOG_WARN << op << " input: micro batch " << micro_batch_id << " tensor " << input << " is not in storage_use_count";
          continue;
        }
        if (accumulated_tensor.find(input->id()) != accumulated_tensor.end()) {
          continue;
        }
        if (--storage_use_count[memory_plan[{micro_batch_id, input->id()}]] == 0
            && !is_pipeline_stage_recv_op(input->producer()) && !is_pipeline_stage_send_op(op)) {
          if (used_by_multi_stream(input) == false) {
            FreeMemory(memory_plan, temporary_free_memory[op->stream_index()], {micro_batch_id, input->id()});
            storage_use_count[memory_plan[{micro_batch_id, input->id()}]] = 0;
            storage_use_count.erase(memory_plan[{micro_batch_id, input->id()}]);
            // if (op->placement().index() == 0) std::cout << "tensor " << input->id() << ' ' << "free" << ' ' << memory_plan[{micro_batch_id, input->id()}].first << ' ' << memory_plan[{micro_batch_id, input->id()}].second << std::endl;
          }
          // multi-stream的memory plan暂时无法处理
          // 只能是下一个block再去全部free
          else {
            release_tensor.push_back({micro_batch_id, input->id()});
          }
        }
      }
      // fw block的结尾或者bw block的开头
      // 再次清空
      if (is_forward && fw_block_end_op.find(op->id()) != fw_block_end_op.end() || !is_forward && fw_block_start_op.find(op->fw_op_id()) != fw_block_start_op.end()) {
        clear_tensor_and_merge_space();
      }
    }
  }
  return memory_plan;
}

} // namespace graph
} // namespace hetu
