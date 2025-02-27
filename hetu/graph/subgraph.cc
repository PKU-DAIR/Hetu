#include "hetu/graph/headers.h"
#include "hetu/graph/profiler.h"
#include "hetu/graph/subgraph.h"
#include "hetu/graph/ops/Concat.h"
#include "hetu/graph/ops/ParallelAttention.h"
#include <queue>

namespace hetu {
namespace graph {

void SubGraph::alloc_concat_memory(Operator& final_concat_op, RuntimeContext& runtime_ctx, std::vector<TensorId>& alloc_concat_tensor_id_list) {
  HT_ASSERT(_already_topo_sorted)
    << "must ensure it is topo sorted";
  HT_ASSERT(final_concat_op->num_outputs() == 1)
    << "there should only be one output of final_concat_op " << final_concat_op;
  // 说明已经分配过memory了
  // 直接返回即可
  if (std::find(alloc_concat_tensor_id_list.begin(), alloc_concat_tensor_id_list.end(), final_concat_op->output(0)->id()) != alloc_concat_tensor_id_list.end()) {
    return;
  }
  const auto& final_concat_op_shape = runtime_ctx.get_runtime_shape(final_concat_op->output(0)->id());
  NDArray final_ndarray;
  if (runtime_ctx.has_runtime_allocation(final_concat_op->output(0)->id())) {
    final_ndarray = runtime_ctx.get_runtime_allocation(final_concat_op->output(0)->id());
  } else {
    auto final_ndarray = NDArray::empty(final_concat_op_shape,
                                        final_concat_op->instantiation_ctx().placement,
                                        final_concat_op->output(0)->dtype(),
                                        final_concat_op->instantiation_ctx().stream_index);
    runtime_ctx.add_runtime_allocation(final_concat_op->output(0)->id(), final_ndarray);
    alloc_concat_tensor_id_list.push_back(final_concat_op->output(0)->id());
  }
  std::queue<Operator> concat_op_queue;
  concat_op_queue.push(final_concat_op);
  while(!concat_op_queue.empty()) {
    auto cur_op = concat_op_queue.front();
    // HT_LOG_INFO << "trying to find out the allocation of " << cur_op;
    concat_op_queue.pop();
    HT_ASSERT(cur_op->num_outputs() == 1 && runtime_ctx.has_runtime_allocation(cur_op->output(0)->id()))
      << "cannot find the runtime allocation of " << cur_op
      << ", it should already generated";
    auto cur_ndarray = runtime_ctx.get_runtime_allocation(cur_op->output(0)->id());
    auto concat_axis = dynamic_cast<ConcatOpImpl&>(cur_op->body()).get_axis();
    auto concat_num = cur_op->num_inputs();
    if (concat_axis >= 1) {
      // 无法处理在非第0维存在切分的情况
      if (concat_num == 1) {
        // 直接inplace
        runtime_ctx.add_runtime_allocation(cur_op->input(0)->id(), cur_ndarray);
        alloc_concat_tensor_id_list.push_back(cur_op->input(0)->id());
      }
      continue;
    }
    HTShape begin_pos(cur_ndarray->shape().size(), 0);
    int32_t offset = 0;
    for (size_t i = 0; i < cur_op->num_inputs(); i++) {
      auto& input = cur_op->input(i);
      const auto& input_shape = runtime_ctx.get_runtime_shape(input->id()); 
      auto input_ndarray = NDArray::slice(cur_ndarray, begin_pos, input_shape);
      runtime_ctx.add_runtime_allocation(input->id(), input_ndarray);
      alloc_concat_tensor_id_list.push_back(input->id());
      HT_ASSERT(input_ndarray->is_contiguous())
        << "found the runtime allocation of " << input << " uncontiguous"
        << ", which shouldn't happen";
      if (is_concat_op(input->producer())) {
        concat_op_queue.push(input->producer());
      }
      offset += input_shape.at(0);
      begin_pos.at(0) = offset;
    }
  }
}

void SubGraph::topo_sort(bool only_local) {
  HT_ASSERT(!_already_topo_sorted)
    << "cannot topo sort subgraph " << _global_name << "twice";
  auto local_device = hetu::impl::comm::GetLocalDevice();
  // 检查当前operator是否不在_ops中
  auto ops_stop_at = [this](const Operator& op) -> bool {
    return _ops.find(op->id()) == _ops.end();
  };
  auto bwd_ops_stop_at = [this](const Operator& op) -> bool {
    return _bwd_ops.find(op->id()) == _bwd_ops.end();
  };
  auto update_ops_stop_at = [this](const Operator& op) -> bool {
    return _update_ops.find(op->id()) == _update_ops.end();
  };
  OpRefList init_ops, init_bwd_ops, init_update_ops;
  for (auto& pair : _ops) {
    init_ops.emplace_back(std::ref(pair.second));
  }
  for (auto& pair : _bwd_ops) {
    init_bwd_ops.emplace_back(std::ref(pair.second));
  }
  for (auto& pair : _update_ops) {
    init_update_ops.emplace_back(std::ref(pair.second));
  }
  _ops_topo = Graph::TopoSort(init_ops, -1, ops_stop_at);
  _bwd_ops_topo = Graph::TopoSort(init_bwd_ops, -1, bwd_ops_stop_at);
  _update_ops_topo = Graph::TopoSort(init_update_ops, -1, update_ops_stop_at);
  _ops_topo.erase(std::remove_if(_ops_topo.begin(), _ops_topo.end(), [&](OpRef op_ref) {
    return (only_local && op_ref.get()->placement() != local_device) || _ops.find(op_ref.get()->id()) == _ops.end();
  }), _ops_topo.end());
  _bwd_ops_topo.erase(std::remove_if(_bwd_ops_topo.begin(), _bwd_ops_topo.end(), [&](OpRef op_ref) {
    return (only_local && op_ref.get()->placement() != local_device) || _bwd_ops.find(op_ref.get()->id()) == _ops.end();
  }), _bwd_ops_topo.end());
  _update_ops_topo.erase(std::remove_if(_update_ops_topo.begin(), _update_ops_topo.end(), [&](OpRef op_ref) {
    return (only_local && op_ref.get()->placement() != local_device) || _update_ops.find(op_ref.get()->id()) == _ops.end();
  }), _update_ops_topo.end());
  _tensor2degrees.clear();
  _bwd_tensor2degrees.clear();
  _update_tensor2degrees.clear();
  for (auto& op_ref : _ops_topo) {
    for (auto& input : op_ref.get()->inputs()) {
      _tensor2degrees[input->id()]++;
    }
  }
  for (auto& op_ref : _bwd_ops_topo) {
    for (auto& input : op_ref.get()->inputs()) {
      _bwd_tensor2degrees[input->id()]++;
    }
  }
  for (auto& op_ref : _update_ops_topo) {
    for (auto& input : op_ref.get()->inputs()) {
      _update_tensor2degrees[input->id()]++;
    }
  }
  _already_topo_sorted = true;
}

// 按照topo顺序运行整个subgraph
// 输入输出全部存放在tensor2data中
// 注意runtime_ctx中的allocation与skip仍然是适用的
void SubGraph::run(Tensor2NDArrayMap& tensor2data, const Tensor2NDArrayMap& preserved_data, RuntimeContext& runtime_ctx,
                   size_t micro_batch_id, SubGraphOpType subgraph_op_type, bool use_concat_memory_optimization, const OpHandler& op_handler) {
  HT_ASSERT(_already_topo_sorted)
    << "cannot run before subgraph " << _global_name << " topo sort";
  std::reference_wrapper<OpRefList> topo_ref = std::ref(_ops_topo);
  std::reference_wrapper<Tensor2IntMap> tensor2degrees_ref = std::ref(_tensor2degrees);
  switch (subgraph_op_type)
  {
    case SubGraphOpType::FORWARD:
      topo_ref = std::ref(_ops_topo);
      tensor2degrees_ref = std::ref(_tensor2degrees);
      break;
    case SubGraphOpType::BACKWARD:
      topo_ref = std::ref(_bwd_ops_topo);
      tensor2degrees_ref = std::ref(_bwd_tensor2degrees);
      break;
    case SubGraphOpType::UPDATE:
      topo_ref = std::ref(_update_ops_topo);
      tensor2degrees_ref = std::ref(_update_tensor2degrees);
      break;
    default:
      HT_NOT_IMPLEMENTED << "unsupported subgraph op type";
  }

  // workaround: 针对batched-send-recv通信pattern的内存优化
  std::vector<TensorId> alloc_concat_tensor_id_list;
  if (use_concat_memory_optimization) {
    for (auto it = topo_ref.get().rbegin(); it != topo_ref.get().rend(); ++it) {
      if (is_concat_op(it->get())) {
        alloc_concat_memory(it->get(), runtime_ctx, alloc_concat_tensor_id_list);
      }
    }
  }
  Tensor2IntMap tensor2degrees = tensor2degrees_ref.get();
  // HT_LOG_INFO << "subgraph " << _global_name << " begin run local topo";
  for (auto& op_ref : topo_ref.get()) {
    auto& op = op_ref.get();
    if (runtime_ctx.has_runtime_skipped(op->id())) {
      continue; 
    }
    if (is_peer_to_peer_send_op(op) || is_peer_to_peer_recv_op(op)) {
      HT_RUNTIME_ERROR << "p2p op in subgraph is currently forbidden, because we can't know how to wrap them with ncclGroup";
    }

    // 执行回调函数
    if (op_handler) {
      // HT_LOG_INFO << "subgraph " << _global_name << " running call back func";
      auto status = op_handler(op, tensor2data, micro_batch_id);
      if (status.need_skip) {
        continue;
      }
    }

    NDArrayList input_vals;
    input_vals.reserve(op->num_inputs());
    for (const auto& input : op->inputs()) {
      HT_ASSERT(input.is_defined())
        << op << " has an undefined input, it cannot run";
      NDArray data;
      auto preserved_it = preserved_data.find(input->id());
      if (preserved_it != preserved_data.end()) {
        data = preserved_it->second;
      } 
      // 只可能在preserved data或者tensor2data中
      else {
        auto it = tensor2data.find(input->id());
        HT_ASSERT(it != tensor2data.end() && it->second.is_defined())
          << "Failed to execute the \"" << op->type() << "\" operation "
          << "(with name \"" << op->name() << "\") in subgraph " << _global_name << ": "
          << "Cannot find input " << input;
        data = it->second;
      }
      input_vals.emplace_back(data);
      HT_ASSERT(data->device() == input->placement() && data->dtype() == input->dtype())
        << input << " placement should be " << input->placement() << " and dtype should be " << input->dtype()
        << ", but found actual data placement " << data->device() << " and dtype " << data->dtype();
      if (--tensor2degrees[input->id()] == 0) {
        // HT_LOG_INFO << "subgraph " << _global_name << " release " << input;
        tensor2data.erase(input->id());
      }
    }
    // 调用op计算
    // debug stuck bug use
    // HT_LOG_INFO << "subgraph " << _global_name << " execute " << op << " begin";
    NDArrayList output_vals = op->Compute(input_vals, runtime_ctx, micro_batch_id);
    checkOutputsMemory(op, micro_batch_id, input_vals, output_vals);
    // op->instantiation_ctx().stream().Sync();
    /*
    TensorIdList output_ids;
    for (auto& output : op->outputs()) {
      output_ids.emplace_back(output->id());
    }
    HT_LOG_INFO << op->outputs() << " ids are " << output_ids;
    */
    // HT_LOG_INFO << "subgraph " << _global_name << " execute " << op << " end";
    // Note: The usage should be marked inside kernels, 
    // but we still mark here in case we forget to do so in some kernels. 
    NDArray::MarkUsedBy(input_vals, op->instantiation_ctx().stream());
    NDArray::MarkUsedBy(output_vals, op->instantiation_ctx().stream());
    for (size_t i = 0; i < op->num_outputs(); i++) {
      const auto& output = op->output(i);
      tensor2data[output->id()] = output_vals.at(i);
    }
  }
  // 跑完之后就删除来清空NDArray的引用计数
  // 否则mempool将一直无法回收这些显存
  // 直到runtime_ctx析构为止
  if (use_concat_memory_optimization) {
    for (const auto& tensor_id : alloc_concat_tensor_id_list) {
      runtime_ctx.delete_runtime_allocation(tensor_id);
    }
  }
}

std::ostream& operator<<(std::ostream& os, SubGraph& subgraph) {
  if (subgraph.already_topo_sorted()) {
    os << "subgraph(name=" << subgraph.name() << ", type=" << static_cast<int32_t>(subgraph.subgraph_type())
       << ", ops_topo=" << subgraph.ops_topo() << ", bwd_ops_topo=" << subgraph.bwd_ops_topo() << 
       ", update_ops_topo=" << subgraph.update_ops_topo() << ", subgraphs=" << subgraph.subgraph_info().size() << "-"
       << subgraph.subgraph_info();
    return os;
  }
  os << "subgraph(name=" << subgraph.name() << ", type=" << static_cast<int32_t>(subgraph.subgraph_type())
     << ", ops=" << subgraph.ops() << ", bwd_ops=" << subgraph.bwd_ops() << 
     ", update_ops=" << subgraph.update_ops() << ", subgraphs=" << subgraph.subgraph_info().size() << "-"
     << subgraph.subgraph_info();
  return os;
}

} // namespace graph
} // namespace hetu

