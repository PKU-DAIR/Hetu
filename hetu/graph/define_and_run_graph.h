#pragma once

#include "hetu/graph/graph.h"
#include "hetu/graph/executable_graph.h"
#include "hetu/graph/init/initializer.h"

namespace std {

template <>
struct hash<std::pair<size_t, size_t>> {
  std::size_t operator()(const std::pair<size_t, size_t>& key) const noexcept {
    auto hash_1 = std::hash<size_t>()(key.first);
    auto hash_2 = std::hash<size_t>()(key.second);
    return hash_1 ^ hash_2;
  }
};

} // namespace std

namespace hetu {
namespace graph {

class ExecGraphPlan {
 public:
  std::shared_ptr<ExecutableGraph> exec_graph;
  size_t compute_strategy_id;
  size_t optimize_strategy_id;
  Op2OpMap op_to_exec_op_mapping;
  Tensor2TensorMap tensor_to_exec_tensor_mapping;
  OpRefList global_topo; // cache the global topo to accelerate ineferring new shape plan
  std::vector<Tensor2ShapeMap> shape_plan_pool; // single exec graph with multi shape plan
  TensorList fetches; // most likey useless

  // forbid copy constructor to avoid high cost
  /*
  ExecGraphPlan(const std::shared_ptr<ExecutableGraph>& _exec_graph, 
                const Op2OpMap& _op_to_exec_op_mapping, 
                const Tensor2TensorMap& _tensor_to_exec_tensor_mapping,
                const OpRefList& _global_topo,
                const std::vector<Tensor2ShapeMap>& _shape_plan_pool,
                size_t _strategy_id = 0)
  : exec_graph(_exec_graph), 
    op_to_exec_op_mapping(_op_to_exec_op_mapping),
    tensor_to_exec_tensor_mapping(_tensor_to_exec_tensor_mapping),
    global_topo(_global_topo),
    shape_plan_pool(_shape_plan_pool),
    strategy_id(_strategy_id) {}
  */
  
  ExecGraphPlan(std::shared_ptr<ExecutableGraph>&& _exec_graph, 
                Op2OpMap&& _op_to_exec_op_mapping, 
                Tensor2TensorMap&& _tensor_to_exec_tensor_mapping,
                OpRefList&& _global_topo,
                std::vector<Tensor2ShapeMap>&& _shape_plan_pool,
                size_t _compute_strategy_id = 0,
                size_t _optimize_strategy_id = 0)
  : exec_graph(std::move(_exec_graph)), 
    op_to_exec_op_mapping(std::move(_op_to_exec_op_mapping)),
    tensor_to_exec_tensor_mapping(std::move(_tensor_to_exec_tensor_mapping)),
    global_topo(std::move(_global_topo)),
    shape_plan_pool(std::move(_shape_plan_pool)),
    compute_strategy_id(_compute_strategy_id),
    optimize_strategy_id(_optimize_strategy_id) {}
};

/*
class SwitcherPoolKey {
 public:
  size_t before_num;
  size_t after_num;
  
  SwitcherPoolKey(const size_t& _before_num, const size_t& _after_num)
  : before_num(_before_num),
    after_num(_after_num) {}
};
*/

class DefineAndRunGraph : public Graph {
 protected:
  friend class Graph;
  friend class Tensor;
  friend class SwitchExecGraph;

  DefineAndRunGraph(GraphName name, size_t init_capacity)
  : Graph(name, init_capacity),
    _init_capacity(init_capacity) {
    std::srand(std::time(0));
  }

 public:
  DefineAndRunGraph(const constructor_access_key&, GraphName name,
                    size_t init_capacity = DEFAULT_GRAPH_INITIAL_CAPACITY)
  : DefineAndRunGraph(name, init_capacity) {}

  NDArrayList Run(const TensorList& fetches, const FeedDict& feed_dict = {});

  NDArrayList Run(const Tensor& loss, const TensorList& fetches, 
                  const FeedDict& feed_dict = {}, const IntSymbolDict& int_symbol_dict = {}, const int num_micro_batches = 1,
                  const int compute_strategy_id = 0, const int optimize_strategy_id = 0, RunLevel run_level = RunLevel::UPDATE,
                  bool save_checkpoint = false, const double grad_scale = 1,
                  const RuntimeContext& ctx = RuntimeContext());

  GraphType type() const {
    return GraphType::DEFINE_AND_RUN;
  }

  bool use_optimizer_strategy(Operator& op) const;

  const ExecGraphPlan& GetPlan(size_t num) const {
    HT_ASSERT(num < _exec_graph_plan_pool.size());
    return _exec_graph_plan_pool[num];
  }

  void RecordBeforeZero(const Tensor& tensor, const DistributedStatesHierarchy& ds_hierarchy) {
    HT_ASSERT(_ds_hierarchy_before_zero.find(tensor->id()) ==_ds_hierarchy_before_zero.end())
      << tensor << " is already recorded in the ds hierarchy before zero mapping";
    _ds_hierarchy_before_zero[tensor->id()] = ds_hierarchy;
  }

  void MergeGraph(DefineAndRunGraph& another_graph);

 protected:
  Operator& MakeOpInner(std::shared_ptr<OpInterface> body, TensorList inputs,
                        OpMeta op_meta);

  void DeducePipeline(size_t cur_strategy_id, int32_t pipeline_num);

  void DeduceShapePlan(ExecGraphPlan& exec_graph_plan,
                       const FeedDict& feed_dict,
                       Tensor2ShapeMap& feed_dict_shape);

  DeviceGroupUnion DeducePlacementGroup(Operator& op, Op2DGUnionMap& dg_union_map);

  void Instantiate(OpRefList&& global_topo,
                   Tensor2ShapeMap&& shape_plan,
                   int32_t pipeline_num);

  void ResetVariableDataInner(const Tensor& tensor,
                              const Initializer& init) override;

  NDArray GetDetachedVariableDataInner(const Tensor& tensor, bool gpu = false) override;

  DeviceGroupUnion GetVariableDeviceGroupUnionInner(const Tensor& tensor) override;

  void RemoveOp(Operator& op) override {
    auto& op_to_exec_op_mapping = _exec_graph_plan_pool[_active_exec_plan].op_to_exec_op_mapping;
    auto& tensor_to_exec_tensor_mapping = _exec_graph_plan_pool[_active_exec_plan].tensor_to_exec_tensor_mapping;
    op_to_exec_op_mapping.erase(op->id());
    Operator::for_each_output_tensor(op, [&](Tensor& tensor) {
      tensor_to_exec_tensor_mapping.erase(tensor->id());
    });
    Graph::RemoveOp(op);
  }

  void Clear() override {
    _add_on_inits.clear();
    _ops_with_device_group_hierarchy.clear();
    _multi_pipeline_maps.clear();
    _param_switcher_pool.clear();
    _grad_switcher_pool.clear();
    _exec_graph_plan_pool.clear();
    Graph::Clear();
  }
  
  void SetExecPlan(size_t num) {
    HT_ASSERT(num < _exec_graph_plan_pool.size())
      << "plan number shouldn't exceed the size of the plan pool";
    _active_exec_plan = num;
    _is_active = true;
  }

  size_t _init_capacity;
  std::unordered_map<TensorId, std::unique_ptr<Initializer>> _add_on_inits;
  // deprecated: now support heterogenous pipeline parallel
  // std::vector<DeviceGroupList> _multi_device_groups; // all the device groups of ops, in the order of MakeOp calls
  std::vector<Operator> _ops_with_device_group_hierarchy; // all ops with device groups, in the order of MakeOp calls
  std::unordered_map<TensorId, DistributedStatesHierarchy> _ds_hierarchy_before_zero; // all tensors (most likely parameters) with ds hierarchy changed due to zero
  // Note: here Device2PipelineMap record the mapping from device to the pipeline that it belongs to
  // and each strategy has a specified mapping
  // To be specific, for each pipeline, each stage is a tp group 
  // therefore each pipeline could be regard as a DeviceGroupList
  std::unordered_map<size_t, Device2PipelineMap> _multi_pipeline_maps;

  std::unordered_map<std::pair<size_t, size_t>, std::unordered_map<DataType, std::vector<std::shared_ptr<SwitchExecGraph>>>> _param_and_opt_var_bucket_switcher_pool;
  std::unordered_map<std::pair<size_t, size_t>, std::unordered_map<DataType, std::shared_ptr<SwitchExecGraph>>> _param_switcher_pool; // 目前其实只会有transfer param
  std::unordered_map<std::pair<size_t, size_t>, std::unordered_map<DataType, std::shared_ptr<SwitchExecGraph>>> _grad_switcher_pool; // 目前其实只会有accumulate grad
  std::vector<ExecGraphPlan> _exec_graph_plan_pool;
  // deprecated: now support single exec graph with multi shape plan
  // and we store multi shape plan into ExecGraphPlan
  // std::vector<Tensor2ShapeMap> _shape_plan_pool; 
  size_t _active_exec_plan;
  bool _is_active = false;

  // 如果判断不需要进行grad的热切换
  // 此值为true时仍会进行grad热切换的topo计算
  // 为false时则什么都不做
  bool _need_grad_switch_topo = false;

 public: 
  /* deprecated: utils for parallel plan changing test case */
  /*
  static void dp2tp(Operator& op);
  static void tp2dp(Operator& op);
  void SetVariableDistributedStates(Operator& op, int32_t dp, int32_t tp);
  */
  void InstantiateTestCase(const OpRefList& topo,
                           Tensor2ShapeMap& shape_plan);
};

} // namespace graph
} // namespace hetu
