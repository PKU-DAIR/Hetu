#pragma once

#include "hetu/graph/graph.h"
#include "hetu/graph/operator.h"
#include "hetu/graph/profiler.h"
#include "hetu/graph/init/initializer.h"
#include "hetu/graph/ops/Communication.h"
#include "hetu/graph/ops/ParallelAttention.h"
#include "hetu/graph/ops/group.h"

namespace hetu {
namespace graph {

struct ExecutePlan {
  OpRefList local_placeholder_variable_ops;
  OpRefList local_fw_topo;
  OpRefList local_bw_topo;
  OpRefList local_topo;
  TensorIdSet dtype_transfer_tensor;
  TensorIdSet shared_weight_tensor;
  OpIdSet shared_weight_p2p;
  OpIdSet shared_weight_grad_p2p;
  TensorIdSet accumulated_tensor;
  OpIdSet accumulated_ops;

  void update(OpRefList& _local_placeholder_variable_ops, 
              OpRefList& _local_fw_topo, OpRefList& _local_bw_topo, 
              OpRefList& _local_topo, TensorIdSet& _dtype_transfer_tensor, 
              TensorIdSet& _shared_weight_tensor, OpIdSet& _shared_weight_p2p, 
              OpIdSet& _shared_weight_grad_p2p, TensorIdSet& _accumulated_tensor, 
              OpIdSet& _accumulated_ops) {
    local_placeholder_variable_ops = _local_placeholder_variable_ops;
    local_fw_topo = _local_fw_topo;
    local_bw_topo = _local_bw_topo;
    local_topo = _local_topo;
    dtype_transfer_tensor = _dtype_transfer_tensor;
    shared_weight_tensor = _shared_weight_tensor;
    shared_weight_p2p = _shared_weight_p2p;
    shared_weight_grad_p2p = _shared_weight_grad_p2p;
    accumulated_tensor = _accumulated_tensor;
    accumulated_ops = _accumulated_ops;
  }
};

class ExecutableGraph : public Graph {
 protected:
  friend class Graph;
  friend class Tensor;
  friend class DefineAndRunGraph;
  friend class SwitchExecGraph;
  friend class AttnCommRing;

  ExecutableGraph(GraphName name, size_t init_capacity)
  : Graph(name, init_capacity) {}

 public:
  ExecutableGraph(const constructor_access_key&, GraphName name,
                  size_t init_capacity = DEFAULT_GRAPH_INITIAL_CAPACITY)
  : ExecutableGraph(name, init_capacity) {}

  // bool MapOpsToParallelDevices(const DeviceGroup& placement_group);

  bool Instantiate(const TensorList& fetches, const Device& placement);

  NDArrayList Run(const TensorList& fetches, 
                  const FeedDict& feed_dict = {});

  NDArrayList Run(const Tensor& loss, const TensorList& fetches, 
                  const FeedDict& feed_dict = {}, const IntSymbolDict& int_symbol_dict = {}, 
                  const int num_micro_batches = 1,
                  RunLevel run_level = RunLevel::UPDATE, const double grad_scale = 1,
                  const RuntimeContext& ctx = RuntimeContext());

  GraphType type() const {
    return GraphType::EXECUTABLE;
  }

  bool p2p_single_communicator() const {
    return _p2p_single_communicator;
  }

  int32_t shape_mismatch_flag() const {
    return _shape_mismatch_flag;
  }

  bool is_pipeline_stage_send_op(Operator& op);

  bool is_pipeline_stage_recv_op(Operator& op);

  void sort_optimize_compute_bridge_subgraph();

  void sort_compute_optimize_bridge_subgraph();

  void SetPipeline(const Device2PipelineMap& pipeline_map) {
    _pipeline_map = pipeline_map;
  }

  const std::vector<int>& GetUsedRanks() const {
    return _used_ranks;
  }

  void SetUsedRanks(const std::vector<int>& used_ranks) {
    _used_ranks = used_ranks;
  }

  bool NeedRank(int rank) {
    return std::find(_used_ranks.begin(), _used_ranks.end(), rank) != _used_ranks.end();
  }

  void SetRunLevel(RunLevel run_level) {
    _run_level = run_level;
  }

  void SetShapePlan(size_t num) {
    HT_ASSERT(num < _shape_plan_pool.size())
      << "plan number shouldn't exceed the size of the plan pool";
    _active_shape_plan = num;
  }

  void SetShapePlanList(std::vector<size_t>&& micro_batch_plan_list) {
    for (auto micro_batch_plan_idx : micro_batch_plan_list) {
      HT_ASSERT(micro_batch_plan_idx < _shape_plan_pool.size())
        << "plan number shouldn't exceed the size of the plan pool";
    }
    _active_shape_plan_list = std::move(micro_batch_plan_list);
  }

  void AddShapePlan(const Tensor2ShapeMap& shape_plan) {
    _shape_plan_pool.emplace_back(shape_plan);
  }

  void AddShapePlan(Tensor2ShapeMap&& shape_plan) {
    _shape_plan_pool.emplace_back(std::move(shape_plan));
  }

  void TopoSortExecTensors() {
    // 对_record_exec_tensors进行拓扑排序
    std::vector<Tensor> sorted_exec_tensors;
    std::unordered_map<TensorId, bool> visited;
    std::function<void(const Tensor&)> topo_sort_by_dep = [&](const Tensor& tensor) {
      if (visited[tensor->id()]) return;
      visited[tensor->id()] = true;
  
      auto& exec_op = tensor->producer();
      for (const auto& input : exec_op->inputs()) {
        auto it = std::find_if(_record_exec_tensors.begin(), _record_exec_tensors.end(),
          [&input](const Tensor& t) { return t->id() == input->id(); });
        if (it != _record_exec_tensors.end()) {
          topo_sort_by_dep(*it);
        }
      }
      sorted_exec_tensors.emplace_back(tensor);
    };
  
    for (const auto& tensor : _record_exec_tensors) {
      if (!visited[tensor->id()]) {
        topo_sort_by_dep(tensor);
      }
    }
    
    _record_exec_tensors = std::move(sorted_exec_tensors);
  }

  void UpdateExecShapePlan(RuntimeContext& runtime_ctx) {
    auto& exec_shape_plan = runtime_ctx.shape_plan();
    // topo sort the recorded exec tensors in case the input is also recorded
    // but emplaced after the output
    TopoSortExecTensors();
    for (const auto& exec_tensor : _record_exec_tensors) {
      if (exec_shape_plan.find(exec_tensor->id()) != exec_shape_plan.end())
        continue;
      auto& exec_op = exec_tensor->producer();
      HTShapeList exec_input_shapes;
      exec_input_shapes.reserve(exec_op->num_inputs());
      for (const auto& exec_input : exec_op->inputs()) {
        auto it = exec_shape_plan.find(exec_input->id());
        HT_ASSERT(it != exec_shape_plan.end()) 
          << "Something wrong, can't find the shape of input " << exec_input
          << " of op " << exec_op
          << " from the current exec shape plan!";
        exec_input_shapes.push_back(it->second);
      }
      HTShapeList exec_output_shapes = exec_op->InferShape(exec_input_shapes, runtime_ctx);
      auto exec_output_shapes_size = exec_output_shapes.size();
      for (size_t i = 0; i < exec_output_shapes_size; i++) {
        if (exec_op->output(i)->symbolic()) {
          if (is_SyShape_leaf(exec_op->output(i)->symbolic_shape())) {
            exec_op->output(i)->set_symbolic_shape(exec_output_shapes[i]);
          }
        }
        exec_shape_plan.insert(std::make_pair(exec_op->output(i)->id(), std::move(exec_output_shapes[i]))); // move constructor
      }
    }
  }

  // 目前主要功能是
  // 1、记录exec graph相较define graph新插入的tensor
  // 2、记录新插入tensor的shape到当前的shape plan
  void RecordExecTensor(const Tensor& tensor) {
    const auto& shape = tensor->shape();
    // need to record the shape for all shape plans in the shape plan pool
    // so we leverage _record_exec_tensor and do it lazily
    // workaround: batched_isend_irecv_op需要保证其不会在concat算子之后
    // 不然deduce shape plan时顺序错误会导致无法推导
    _record_exec_tensors.emplace_back(tensor);
    if (is_batched_isend_irecv_op(tensor->producer()) && _record_exec_tensors.size() > 1) {
      // HT_LOG_INFO << "record_exec_tensors: " << _record_exec_tensors;
      for (size_t i = _record_exec_tensors.size() - 2; i >= 0; i--) {
        if (is_concat_op(_record_exec_tensors[i]->producer())) {
          _record_exec_tensors[i + 1] = _record_exec_tensors[i];
        } else {
          _record_exec_tensors[i + 1] = tensor;
          break;
        }
      }
    }
    if (!_shape_plan_pool.empty()) {
      auto& shape_plan = _shape_plan_pool.at(_active_shape_plan);
      auto it = shape_plan.find(tensor->id());
      if (it != shape_plan.end()) {
        // already existed, then must be equal
        HT_ASSERT(it->second.size() == shape.size())
          << "Tensor " << tensor << " is already existed in shape plan but is unequal";
        for (size_t i = 0; i < shape.size(); i++) { 
          HT_ASSERT(it->second[i] == shape[i])
            << "Tensor " << tensor << " is already existed in shape plan but is unequal";
        }
        return;
      }
      shape_plan.insert(std::make_pair(tensor->id(), shape));
    }
  }

  // 将op从exec graph中删除
  // 1、将其从对应的subgraph中删除（如果存在）
  // 2、将其的输出从record_exec_tensors中删除（如果存在）
  void DeleteExecOp(Operator& op) {
    DeleteOpFromSubGraph(op);
    for (auto& output : op->outputs()) {
      _record_exec_tensors.erase(
        std::remove_if(_record_exec_tensors.begin(), _record_exec_tensors.end(),
          [&output](const Tensor& tensor) {
            return tensor->id() == output->id();
          }),
        _record_exec_tensors.end()
      );
    }
  }

  bool HasTensorShape(const Tensor& tensor) const {
    const auto& shape_plan = _shape_plan_pool.at(_active_shape_plan);
    return shape_plan.find(tensor->id()) != shape_plan.end();
  }

  const HTShape& GetTensorShape(const Tensor& tensor) const {
    const auto& shape_plan = _shape_plan_pool.at(_active_shape_plan);
    auto it = shape_plan.find(tensor->id());
    HT_ASSERT(it != shape_plan.end())
      << "Tensor " << tensor << " is not existed in current shape plan";
    return it->second;
  }

  void AddLeafSymbolicTensor(const Tensor& tensor) {
    _leaf_symbolic_tensor_list.emplace_back(tensor);
  }

 protected:
  DeviceGroup GetPrevStage();

  DeviceGroup GetNextStage();

  std::unordered_map<size_t, std::vector<std::pair<bool, size_t>>>
  GenerateGpipeSchedule(size_t num_stages, size_t num_micro_batches, bool is_inference);

  std::unordered_map<size_t, std::vector<std::pair<int32_t, size_t>>>
  GeneratePipedreamFlushSchedule(size_t num_stages, size_t num_micro_batches, bool is_inference);

  void ComputeFunc(size_t& micro_batch_id, const OpRefList& topo, RuntimeContext& runtime_ctx,
                                    Tensor2NDArrayMap& tensor2data, Tensor2IntMap& tensor2degrees, 
                                    Tensor2NDArrayMap& grad_accumulation, bool grad_accumulation_finished,
                                    const FeedDict& feed_dict, const TensorList& fetches,
                                    const std::unordered_map<TensorId, size_t>& fetch_indices, bool& is_continuous_p2p);

  void SubstituteCommOp(const OpRefList& topo_order);

  void InsertContiguousOp(const OpRefList& topo_order);

  Operator& MakeOpInner(std::shared_ptr<OpInterface> body, TensorList inputs,
                        OpMeta op_meta);

  void ResetVariableDataInner(const Tensor& tensor,
                              const Initializer& init) override;

  NDArray& GetVariableDataInner(const Tensor& tensor) override;

  NDArray GetDetachedVariableDataInner(const Tensor& tensor, bool gpu = false) override;

  NDArray& AllocVariableDataInner(
    const Tensor& tensor,
    const Initializer& init = VoidifiedInitializer(),
    uint64_t seed = 0, const HTShape& global_shape = HTShape()) override;

  void RegisterVariableDataInner(
    const Tensor& tensor, NDArray data,
    const Initializer& init = VoidifiedInitializer()) override;

  void PreRun(std::vector<RuntimeContext>& runtime_ctx_list);

  void PostRun(std::vector<RuntimeContext>& runtime_ctx_list, Tensor2NDArrayMap& tensor2data);

  OpHandlerStatus PostOpHandler(Operator& op, Tensor2NDArrayMap& tensor2data, size_t micro_batch_id);

  NDArrayList CrucialRun(const TensorList& fetches, 
                         const FeedDict& feed_dict, 
                         const IntSymbolDict& int_symbol_dict,
                         const int num_micro_batches,
                         const RuntimeContext& ctx);

  void GetExecEnvs();
  
  // memory plan相关
  void AllocMemory(size_t& memory_size, MemoryPlan& memory_plan,
                            MemoryBlockList& temporary_free_memory, MemoryBlockList& free_memory, MicroBatchTensorId tensor_id,
                            size_t alloc_memory_size);
  
  void FreeMemory(MemoryPlan& memory_plan, MemoryBlockList& free_memory,
                  MicroBatchTensorId tensor_id);
  
  MemoryPlan GenerateMemoryPlan(size_t& memory_size, std::vector<std::pair<bool, size_t>> tasks,
                                std::vector<Tensor2IntMap> tensor2degrees_list,
                                const FeedDict& feed_dict);

  // plan相关
  ExecutePlan _execute_plan;
  std::vector<Tensor2ShapeMap> _shape_plan_pool;
  size_t _active_shape_plan;
  std::vector<size_t> _active_shape_plan_list;
  std::vector<Tensor> _record_exec_tensors;

  // run相关
  std::unordered_map<TensorId, std::unique_ptr<Initializer>> _add_on_inits;
  Device2PipelineMap _pipeline_map;
  std::vector<int> _used_ranks;
  int _num_micro_batches;
  std::vector<std::unique_ptr<Event>> _p2p_events;

  // switch相关
  /*
  std::shared_ptr<ParamBuffer> _origin_param_buffer;
  std::shared_ptr<ParamBuffer> _transfer_param_buffer;
  std::shared_ptr<ParamBuffer> _origin_param_and_optimizer_buffer; // deprecated
  std::shared_ptr<ParamBuckets> _origin_param_and_optimizer_buckets; 
  std::shared_ptr<ParamBuffer> _current_grad_buffer; 
  std::shared_ptr<ParamBuffer> _accumulate_grad_buffer; 
  */
  std::unordered_map<DataType, std::shared_ptr<ParamBuckets>> _origin_param_and_optimizer_buckets_map;
  std::unordered_map<DataType, std::shared_ptr<ParamBuffer>> _transfer_param_buffer_map;
  std::unordered_map<DataType, std::shared_ptr<ParamBuffer>> _current_grad_buffer_map;
  std::unordered_map<DataType, std::shared_ptr<ParamBuffer>> _accumulate_grad_buffer_map;
  TensorList _leaf_symbolic_tensor_list;
  Tensor2TensorMap _transfer_map; // origin param到transfer param的映射（注意substitute comm op后会对其进行修正）
  Tensor2TensorMap _grad_map; // origin param到grad的映射（注意substitute comm op后会对其进行修正）
  bool _use_current_grad_buffer{false};
  bool _use_origin_param_and_optimizer_buffer{false};
  bool _use_origin_param_and_optimizer_buckets{true};
  double _grad_scale; 
  bool _is_transfer_param_hot_switch{false};

  // 记录上一个图的param切换完的event
  std::unordered_map<TensorId, std::unique_ptr<Event>> _switch_param_events;
  // 记录上一个图的grad切换完的event
  std::unordered_map<TensorId, std::unique_ptr<Event>> _switch_grad_events; // 注意这里的TensorId是未substitue comm op前的grad
  // 记录当前图的param不再用的event
  // 即意味着可以开始切换param了
  std::unordered_map<TensorId, std::unique_ptr<Event>> _run_param_events; 
  // 记录当前图的grad计算完的event
  // 即意味着可以开始切换grad了
  std::unordered_map<TensorId, std::unique_ptr<Event>> _run_grad_events; // 注意这里的TensorId是未substitue comm op后的grad

  // 分别记录param op到两个bridge的subgraph的映射
  std::unordered_map<OpId, std::shared_ptr<SubGraph>> _optimize_compute_bridge_subgraph_map;
  std::unordered_map<OpId, std::shared_ptr<SubGraph>> _compute_optimize_bridge_subgraph_map;
  // 排序后的
  std::vector<std::pair<OpId, std::shared_ptr<SubGraph>>> _optimize_compute_bridge_subgraph_sorted;
  std::vector<std::pair<OpId, std::shared_ptr<SubGraph>>> _compute_optimize_bridge_subgraph_sorted;
  // 存放group op的subgraph
  // 以及后面可能会有一些别的后处理
  std::shared_ptr<SubGraph> _terminate_subgraph;
  // grad reduce算子的input到compute_optimize_bridge_subgraph的映射
  // 用于overlap grad reduce
  std::unordered_map<TensorId, std::shared_ptr<SubGraph>> _grad_reduce_subgraph_map;

  // env相关
  bool _p2p_single_communicator;
  bool _bridge_single_communicator;
  bool _overlap_grad_reduce;
  int32_t _shape_mismatch_flag;
  int32_t _straggler_flag;
  std::string _straggler_log_file_path;
  MEMORY_PROFILE_LEVEL _memory_profile_level;
  std::string _memory_log_file_path;
  std::vector<std::shared_ptr<MicroBatchMemoryInfo>> _all_micro_batches_memory_info;
  int32_t _parallel_attn_flag;
  std::string _parallel_attn_log_file_path;
};

} // namespace graph
} // namespace hetu
