#pragma once

#include "hetu/common/macros.h"
#include "hetu/core/ndarray.h"
#include "hetu/graph/common.h"
#include "hetu/graph/tensor.h"
#include "hetu/utils/context_store.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/stream/CPUStream.h"
#include "hetu/impl/communication/comm_group.h"

namespace hetu {
namespace graph {

class OpMeta {
 public:
  OpMeta() = default;

  OpMeta(const OpMeta&) = default;
  OpMeta(OpMeta&&) = default;
  OpMeta& operator=(OpMeta&&) = default;
  OpMeta& operator=(const OpMeta&) = default;

  inline OpMeta& set_name(const OpName& n) {
    name = n;
    return *this;
  }

  inline OpMeta& set_name(OpName&& n) {
    name = std::move(n);
    return *this;
  }

  inline OpMeta& set_stream_index(StreamIndex si) {
    // TODO: check whether the stream index is valid
    stream_index = si;
    return *this;
  }

  inline OpMeta& set_eager_device(const Device& device) {
    eager_device = device;
    return *this;
  }

  inline OpMeta& set_eager_device(Device&& device) {
    eager_device = std::move(device);
    return *this;
  }

  inline OpMeta& set_device_group_hierarchy(const DeviceGroupHierarchy& hierarchy) {
    device_group_hierarchy = hierarchy;
    return *this;
  }

  inline OpMeta& set_device_group_hierarchy(DeviceGroupHierarchy&& hierarchy) {
    device_group_hierarchy = std::move(hierarchy);
    return *this;
  }

  inline OpMeta& set_extra_deps(const TensorList& deps) {
    extra_deps = deps;
    return *this;
  }

  inline OpMeta& set_extra_deps(TensorList&& deps) {
    extra_deps = std::move(deps);
    return *this;
  }

  inline OpMeta& set_is_deduce_states(bool deduce_states) {
    is_deduce_states = deduce_states;
    return *this;
  }

  inline OpMeta& set_is_cpu(bool _is_cpu) {
    is_cpu = _is_cpu;
    return *this;
  }

  inline OpMeta& set_origin_op_id(OpId id) {
    origin_op_id = id;
    return *this;
  }

  inline OpMeta& set_fw_op_id(OpId id) {
    fw_op_id = id;
    return *this;
  }

  inline OpMeta& set_multi_recompute(const std::vector<std::vector<bool>>& multi_recompute) {
    multi_is_recompute = multi_recompute;
    return *this;
  }

  bool get_recompute(size_t strategy_id, size_t hetero_id) {
    std::vector<bool> is_recompute;
    size_t multi_len = multi_is_recompute.size();
    HT_ASSERT(multi_len > 0)
      << name << " multi recompute is empty, something wrong";
    if (multi_is_recompute.size() == 1) {
      is_recompute = multi_is_recompute.at(0);
    } else {
      HT_ASSERT(multi_len > strategy_id)
        << name << " multi recompute size is wrong"
        << ", can't fetch strategy id " << strategy_id << " within len " << multi_len;
      is_recompute = multi_is_recompute.at(strategy_id);
    }
    size_t hetero_len = is_recompute.size();
    if (is_recompute.size() == 1) {
      return is_recompute.at(0);
    }
    HT_ASSERT(hetero_len > hetero_id)
      << name << " hetero recompute size is wrong"
      << ", can't fetch hetero id " << hetero_id << " within len " << hetero_len;
    return is_recompute.at(hetero_id); 
  }

  inline OpMeta& set_multi_cpu_offload(const std::vector<std::vector<bool>>& multi_cpu_offload) {
    multi_is_cpu_offload = multi_cpu_offload;
    return *this;
  }

  bool get_cpu_offload(size_t strategy_id, size_t hetero_id) {
    std::vector<bool> is_cpu_offload;
    size_t multi_len = multi_is_cpu_offload.size();
    HT_ASSERT(multi_len > 0)
      << name << " multi cpu offload is empty, something wrong";
    if (multi_is_cpu_offload.size() == 1) {
      is_cpu_offload = multi_is_cpu_offload.at(0);
    } else {
      HT_ASSERT(multi_len > strategy_id)
        << name << " multi cpu offload size is wrong"
        << ", can't fetch strategy id " << strategy_id << " within len " << multi_len;
      is_cpu_offload = multi_is_cpu_offload.at(strategy_id);
    }
    size_t hetero_len = is_cpu_offload.size();
    if (is_cpu_offload.size() == 1) {
      return is_cpu_offload.at(0);
    }
    HT_ASSERT(hetero_len > hetero_id)
      << name << " hetero cpu offload size is wrong"
      << ", can't fetch hetero id " << hetero_id << " within len " << hetero_len;
    return is_cpu_offload.at(hetero_id);
  }

  // TODO: support multi-strategies offload
  inline OpMeta& set_is_offload(bool offload) {
    is_offload = offload;
    return *this;
  }

  inline OpMeta& set_parameter_dict(ParameterDict param_dict) {
    parameter_dict = param_dict;
    return *this;
  }

  bool need_dequantization() {
    return parameter_dict.find("tensor_id") != parameter_dict.end()
        && parameter_dict.find("blocksize") != parameter_dict.end();
  }

  inline OpMeta& set(const OpMeta& other) {
    operator=(other);
    return *this;
  }

  inline OpMeta& set(OpMeta&& other) {
    operator=(std::move(other));
    return *this;
  }

  static OpMeta Merge(const OpMeta& base_meta, const OpMeta& new_meta) {
    OpMeta ret = base_meta;
    if (!new_meta.name.empty())
      ret.set_name(new_meta.name);
    if (new_meta.stream_index != kUndeterminedStream)
      ret.set_stream_index(new_meta.stream_index);
    if (!new_meta.eager_device.is_undetermined())
      ret.set_eager_device(new_meta.eager_device);
    if (new_meta.device_group_hierarchy.size() != 0)
      ret.set_device_group_hierarchy(new_meta.device_group_hierarchy);
    if (!new_meta.extra_deps.empty())
      ret.set_extra_deps(new_meta.extra_deps);
    return ret;
  }

  OpName name;
  StreamIndex stream_index{kUndeterminedStream};
  Device eager_device{kUndeterminedDevice};
  // deprecated: DeviceGroupList device_groups; // for multi ds deduce
  DeviceGroupHierarchy device_group_hierarchy{}; // for multi ds multi hetero-dp deduce
  TensorList extra_deps;
  OpId origin_op_id{-1}; // for recomputation only
  OpId fw_op_id{-1};
  std::vector<std::vector<bool>> multi_is_recompute{{false}}; // for multi recomputation strategy multi pipeline
  std::vector<std::vector<bool>> multi_is_cpu_offload{{false}}; // for multi cpu offload strategy multi pipeline
  bool is_offload{false}; // for offload D2H op only
  bool is_deduce_states{true};  
  bool is_cpu{false};
  ParameterDict parameter_dict;
};

std::ostream& operator<<(std::ostream&, const OpMeta&);

using OpRuntimeContext = ContextStore;
using InstantiationContext = ContextStore;

class RuntimeContext {
 public:
  RuntimeContext(): _shape_plan(std::nullopt) {}

  RuntimeContext(size_t init_capacity): _shape_plan(std::nullopt) {
    _ctxs.reserve(init_capacity);
  }
  
  RuntimeContext(size_t init_capacity, Tensor2ShapeMap& shape_plan): _shape_plan(shape_plan) {
    _ctxs.reserve(init_capacity);
  }

  ~RuntimeContext() {
    for (auto& kv : _ctxs)
      delete kv.second;
    _ctxs.clear();
  }

  OpRuntimeContext& get_or_create(OpId id) {
    auto it = _ctxs.find(id);
    if (it != _ctxs.end()) {
      return *it->second;
    } else {
      auto* op_ctx = new OpRuntimeContext();
      _ctxs[id] = op_ctx;
      return *op_ctx;
    }
  }

  OpRuntimeContext& get(OpId id) {
    HT_ASSERT(_ctxs.find(id) != _ctxs.end())
      << "Op " << id << " is not found in runtime context";
    return *_ctxs.at(id);
  }

  const OpRuntimeContext& get(OpId id) const {
    HT_ASSERT(_ctxs.find(id) != _ctxs.end())
      << "Op " << id << " is not found in runtime context";
    return *_ctxs.at(id);
  }

  void remove(OpId id) {
    _ctxs.erase(id);
  }

  void clear() {
    _ctxs.clear();
  }

  const HTShape& get_runtime_shape(const TensorId& tensor_id) const {
    HT_ASSERT(_shape_plan)
      << "The shape plan is null, ensure that you've used a define graph to instantiate a exec graph";
    auto it = _shape_plan->get().find(tensor_id);
    HT_ASSERT(it != _shape_plan->get().end())
      << "Tensor " << tensor_id << " is not existed in runtime shape plan";
    return it->second;
  }
  
  Tensor2ShapeMap& shape_plan() {
    HT_ASSERT(_shape_plan)
      << "The shape plan is null, ensure that you've used a define graph to instantiate a exec graph";
    return _shape_plan->get();
  }

  bool has_shape_plan() {
    if (_shape_plan)
      return true;
    return false;
  }

  const Tensor2NDArrayMap& allocation_plan() const {
    return _allocation_plan;
  }

  bool has_runtime_allocation(const TensorId& tensor_id) const {
    return _allocation_plan.find(tensor_id) != _allocation_plan.end();
  }

  void add_runtime_allocation(const TensorId& tensor_id, const NDArray& allocation) {
    auto it = _allocation_plan.find(tensor_id);
    HT_ASSERT(it == _allocation_plan.end())
      << "Tensor " << tensor_id << " is already existed in runtime allocation plan";
    _allocation_plan[tensor_id] = allocation;
  }

  const NDArray& get_runtime_allocation(const TensorId& tensor_id) const {
    auto it = _allocation_plan.find(tensor_id);
    HT_ASSERT(it != _allocation_plan.end())
      << "Tensor " << tensor_id << " is not existed in runtime allocation plan";
    return it->second;
  }

  void delete_runtime_allocation(const TensorId& tensor_id) {
    auto it = _allocation_plan.find(tensor_id);
    HT_ASSERT(it != _allocation_plan.end())
      << "Tensor " << tensor_id << " is not existed in runtime allocation plan";
    _allocation_plan.erase(it);
  }

  bool has_runtime_skipped(const OpId& op_id) const {
    return _skipped_plan.find(op_id) != _skipped_plan.end();
  }

  void add_runtime_skipped(const OpId& op_id) {
    auto it = _skipped_plan.find(op_id);
    HT_ASSERT(it == _skipped_plan.end())
      << "Op " << op_id << " is already existed in runtime skipped plan";
    _skipped_plan.insert(op_id);
  }

 private:
  std::unordered_map<OpId, OpRuntimeContext*> _ctxs; // 初始化时进行赋值
  std::optional<std::reference_wrapper<Tensor2ShapeMap>> _shape_plan; // 初始化时进行赋值，每个tensor必须有一个对应的shape，没有则报错
  Tensor2NDArrayMap _allocation_plan; // 初始化后进行赋值，部分tensor可以有一个对应的allocation，没有则临时分配
  std::unordered_set<OpId> _skipped_plan; // 初始化后进行赋值，部分op不需要sync
};

struct OpInstantiationContext {
  bool has_placement_group{false};
  DeviceGroupUnion placement_group_union{};
  Device placement{};
  StreamIndex stream_index;
  InstantiationContext ctx{};
  std::unique_ptr<Event> start[HT_MAX_NUM_MICRO_BATCHES];
  std::unique_ptr<Event> stop[HT_MAX_NUM_MICRO_BATCHES];

  Stream stream() const {
    // Question: create stream inside kernels?
    return Stream(placement, stream_index);
  }
};

class OpInterface : public shared_ptr_target {
 protected:
  friend class OpDef;
  friend class Operator;
  OpInterface(OpType&& op_type) : _type(op_type) {}

 public:
  ~OpInterface() = default;

  // disable copy constructor and move constructor
  OpInterface(const OpInterface&) = delete;
  OpInterface& operator=(const OpInterface&) = delete;
  OpInterface(OpInterface&&) = delete;
  OpInterface& operator=(OpInterface&&) = delete;

  inline const OpType& type() const {
    return _type;
  }

  virtual bool require_contig_inputs() const {
    return true;
  }

  virtual uint64_t inplace_pos() const {
    return 0;
  }

  virtual uint64_t op_indicator() const noexcept {
    return 0;
  }

  virtual bool operator==(const OpInterface& rhs) const {
    return op_indicator() == rhs.op_indicator() && type() == rhs.type()
        && require_contig_inputs() == rhs.require_contig_inputs();
  }

  bool operator!=(const OpInterface& rhs) const {
    return !(*this == rhs);
  }

  virtual bool inplace_at(size_t input_position) const {
    return false;
  }

  inline std::vector<NDArrayMeta> InferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const {
    return DoInferMeta(inputs, inst_ctx);
  }

  inline void DeduceStates(const TensorList& inputs, TensorList& outputs, 
                           const OpMeta& op_meta, const InstantiationContext& inst_ctx) const {
    return DoDeduceStates(inputs, outputs, op_meta, inst_ctx);
  }

  inline void DeduceHeterProp(const std::vector<int32_t>& inputs_hetero_dim,
                              TensorList& outputs, const OpMeta& op_meta,
                              const InstantiationContext& inst_ctx) const {
    bool is_all_homo = true;
    for (const auto& input_hetero_dim : inputs_hetero_dim) {
      if (input_hetero_dim != NULL_HETERO_DIM) {
        is_all_homo = false;
        break;
      }
    }
    if (is_all_homo) {
      for (auto& output : outputs) {
        output->cur_ds_union().set_hetero_dim(NULL_HETERO_DIM);
      }
      return;
    }
    return DoDeduceHeterProp(inputs_hetero_dim, outputs, op_meta, inst_ctx);
  }

  inline void DeduceStatesHierarchy(const TensorList& inputs, TensorList& outputs, 
                                    const OpMeta& op_meta, const InstantiationContext& inst_ctx,
                                    Graph& graph) const {
    DoDeduceStatesHierarchy(inputs, outputs, op_meta, inst_ctx, graph);
  }

  inline void LoadCtxForBackward(ContextStore& src_ctx, ContextStore& dst_ctx) const {
    DoLoadCtxForBackward(src_ctx, dst_ctx);
  }

  inline void SaveCtxForBackward(const TensorList& inputs, ContextStore& dst_ctx) const {
    DoSaveCtxForBackward(inputs, dst_ctx);
  }

  inline TensorList Gradient(Operator& op,
                             const TensorList& grad_outputs) const {
    return DoGradient(op, grad_outputs);
  }

  inline bool MapToParallelDevices(Operator& op,
                                   const DeviceGroupUnion& placement_group_union) const {
    return DoMapToParallelDevices(op, placement_group_union);
  }

  inline void MergeStrategy(Operator& op, Operator& another_op) {
    DoMergeStrategy(op, another_op); 
  }

  inline bool Instantiate(Operator& op, const Device& placement,
                          StreamIndex stream_id) const {
    return DoInstantiate(op, placement, stream_id);
  }

  void LoadAndSaveCtxForBackward(Operator& op, RuntimeContext& runtime_ctx) const;

  HTShapeList InferShape(Operator& op, const HTShapeList& shapes,
                         RuntimeContext& runtime_ctx) const;

  inline NDArrayList AllocOutputs(Operator& op, const NDArrayList& inputs,
                                  RuntimeContext& runtime_ctx) const {
    return DoAllocOutputs(op, inputs, runtime_ctx);
  }

  inline void Compute(Operator& op, const NDArrayList& inputs,
                      NDArrayList& outputs, RuntimeContext& runtime_ctx) const {
    DoCompute(op, inputs, outputs, runtime_ctx);
  }

  inline NDArrayList Compute(Operator& op, const NDArrayList& inputs,
                             RuntimeContext& runtime_ctx) const {
    return DoCompute(op, inputs, runtime_ctx);
  }

 protected:
  virtual std::vector<NDArrayMeta>
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const = 0;

  virtual void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                              const OpMeta& op_meta, const InstantiationContext& inst_ctx) const;

  virtual void DoDeduceHeterProp(const std::vector<int32_t>& inputs_hetero_dim, 
                                 TensorList& outputs, const OpMeta& op_meta,
                                 const InstantiationContext& inst_ctx) const;

  virtual void DoDeduceStatesHierarchy(const TensorList& inputs, TensorList& outputs, 
                                       const OpMeta& op_meta, const InstantiationContext& inst_ctx,
                                       Graph& graph) const;

  virtual void DoLoadCtxForBackward(ContextStore& src_ctx, ContextStore& dst_ctx) const {
    return;
  }

  virtual void DoSaveCtxForBackward(const TensorList& inputs,  ContextStore& dst_ctx) const {
    return;
  }

  virtual TensorList DoGradient(Operator&, const TensorList&) const {
    HT_RUNTIME_ERROR << "Op with type " << type() << "is not differentiable";
    __builtin_unreachable();
  }

  virtual bool DoMapToParallelDevices(Operator& op, const DeviceGroupUnion& placement_group_union) const;

  virtual void DoMergeStrategy(Operator& op, Operator& another_op);

  virtual void DoSpecialMergeStrategy(Operator& op, Operator& another_op);

  virtual bool DoInstantiate(Operator& op, const Device& placement,
                             StreamIndex stream_id) const;

  virtual HTShapeList DoInferShape(Operator& op,
                                   const HTShapeList& input_shapes,
                                   RuntimeContext& runtime_ctx) const;

  virtual HTShapeList DoInferDynamicShape(Operator& op,
                                   const HTShapeList& input_shapes,
                                   RuntimeContext& runtime_ctx) const {
    return DoInferShape(op, input_shapes, runtime_ctx);
  }

  virtual NDArrayList DoAllocOutputs(Operator& op, const NDArrayList& inputs,
                                     RuntimeContext& runtime_ctx) const;

  NDArrayList DoAllocOutputs(Operator& op, const NDArrayList& inputs,
                             RuntimeContext& runtime_ctx, const Device& device) const;

  virtual NDArray DoAllocOutput(Operator& op, const NDArrayList& inputs,
                                size_t idx, RuntimeContext& runtime_ctx) const;

  virtual void DoCompute(Operator& op, const NDArrayList& inputs,
                         NDArrayList& outputs,
                         RuntimeContext& runtime_ctx) const = 0;

  virtual NDArrayList DoCompute(Operator& op, const NDArrayList& inputs,
                                RuntimeContext& runtime_ctx) const {
    auto outputs = DoAllocOutputs(op, inputs, runtime_ctx);
    DoCompute(op, inputs, outputs, runtime_ctx);
    return outputs;
  }

  const OpType _type;
};

struct OpIdentifier {
  GraphId graph_id;
  OpId op_id;
};

class OpDef : public shared_ptr_target {
 protected:
  friend class Operator;
  friend class Graph;
  friend class DefineByRunGraph;
  friend class DefineAndRunGraph;
  friend class ExecutableGraph;
  friend class Recompute;
  struct constructor_access_key {};

 public:
  OpDef(const constructor_access_key&, OpIdentifier ids,
        std::shared_ptr<OpInterface> body, TensorList inputs, OpMeta op_meta);

  ~OpDef() = default;

  // disable copy constructor and move constructor
  OpDef(const OpDef&) = delete;
  OpDef& operator=(const OpDef&) = delete;
  OpDef(OpDef&&) = delete;
  OpDef& operator=(OpDef&&) = delete;

  inline TensorList Gradient(const TensorList& grad_outputs) {
    return _body->Gradient(get_self(), grad_outputs);
  }

  inline bool MapToParallelDevices(const DeviceGroupUnion& placement_group_union) {
    return _body->MapToParallelDevices(get_self(), placement_group_union);
  }

  inline void MergeStrategy(Operator& another_op) {
    return _body->MergeStrategy(get_self(), another_op);
  }

  inline void DeduceStates() {
    return _body->DeduceStates(inputs(), outputs(), op_meta(), instantiation_ctx().ctx);
  }

  inline void DeduceStatesHierarchy() {
    return _body->DeduceStatesHierarchy(inputs(), outputs(), op_meta(), instantiation_ctx().ctx, graph());
  }

  inline bool Instantiate(const Device& placement, StreamIndex stream_id) {
    return _body->Instantiate(get_self(), placement, stream_id);
  }

  void LoadAndSaveCtxForBackward(RuntimeContext& runtime_ctx) {
    _body->LoadAndSaveCtxForBackward(get_self(), runtime_ctx);
  }

  HTShapeList InferShape(const HTShapeList& input_shapes,
                         RuntimeContext& runtime_ctx) {
    return _body->InferShape(get_self(), input_shapes, runtime_ctx);
  }

  void Compute(const NDArrayList& inputs, NDArrayList& outputs, RuntimeContext& runtime_ctx, size_t micro_batch_id = 0) {
    HT_ASSERT(micro_batch_id < HT_MAX_NUM_MICRO_BATCHES)
      << "Num micro batches muse <= " << HT_MAX_NUM_MICRO_BATCHES 
      << ", got micro batch id: " << micro_batch_id;
    BlockOrSyncAllInputs(runtime_ctx, micro_batch_id);
    instantiation_ctx().start[micro_batch_id]->Record(stream());
    _body->Compute(get_self(), inputs, outputs, runtime_ctx);
    instantiation_ctx().stop[micro_batch_id]->Record(stream());
  }

  NDArrayList Compute(const NDArrayList& inputs, RuntimeContext& runtime_ctx, size_t micro_batch_id = 0) {
    HT_ASSERT(micro_batch_id < HT_MAX_NUM_MICRO_BATCHES)
      << "Num micro batches muse <= " << HT_MAX_NUM_MICRO_BATCHES 
      << ", got micro batch id: " << micro_batch_id;
    BlockOrSyncAllInputs(runtime_ctx, micro_batch_id);
    // correctness debug
    /*
    HTShapeList input_shapes, input_strides;
    for (auto& input : inputs) {
      input_shapes.push_back(input->shape());
      input_strides.push_back(input->stride());
    }
    HT_LOG_INFO << hetu::impl::comm::GetLocalDevice() << " micro batch: " << micro_batch_id << ", compute op: " << name()
      << ", the inputs are " << this->inputs() << " and the input shapes are " << input_shapes << ", strides are " << input_strides;
    */
    // precision debug
    /*
    NDArrayList input_sums;
    for (auto& input : inputs) {
      input_sums.push_back(NDArray::sum(input));
    }
    HT_LOG_INFO << hetu::impl::comm::GetLocalDevice() << " micro batch: " << micro_batch_id << ", compute op: " << name()
      << ", the input vals are " << input_sums;
    */
    instantiation_ctx().start[micro_batch_id]->Record(stream());
    auto rets = _body->Compute(get_self(), inputs, runtime_ctx);
    instantiation_ctx().stop[micro_batch_id]->Record(stream());
    // stream().Sync();
    // precision debug
    /*
    NDArrayList ret_sums;
    for (auto& ret : rets) {
      ret_sums.push_back(NDArray::sum(ret));
    }
    HT_LOG_INFO << hetu::impl::comm::GetLocalDevice() << " micro batch: " << micro_batch_id << ", compute op: " << name()
      << ", the result is " << ret_sums;
    */
    // correctness debug
    /*
    HTShapeList ret_shapes, ret_strides;
    for (auto& ret : rets) {
      ret_shapes.push_back(ret->shape());
      ret_strides.push_back(ret->stride());
    }
    HT_LOG_INFO << hetu::impl::comm::GetLocalDevice() << " micro batch: " << micro_batch_id << ", compute op: " << name()
      << ", the return shapes are " << ret_shapes << ", strides are " << ret_strides;
    */
    // for some ops that rely on symbolic shape
    // 2024.9.5 Update:
    // now move this part into executable graph before ComputeFunc
    /*
    auto output_size = num_outputs();
    for (size_t i = 0; i < output_size; i++) {
      if (output(i)->symbolic()) {
        HT_LOG_TRACE << "exec op " << name()
          << " output " << i << " has " << output(i)->symbolic_shape();
        if (is_SyShape_leaf(output(i)->symbolic_shape())) {
          output(i)->set_symbolic_shape(rets[i]->shape());
          HT_LOG_TRACE << "set symbolic shape of exec op " << name()
            << " output " << i << " to " << rets[i]->shape();
        }
      }
    }
    */
    return rets;
  }

  void Sync(size_t micro_batch_id = 0) {
    instantiation_ctx().stop[micro_batch_id]->Sync();
  }

  inline int64_t TimeCost(size_t micro_batch_id = 0) {
    return instantiation_ctx().stop[micro_batch_id]->TimeSince(
      *instantiation_ctx().start[micro_batch_id]);
  }

  OpId id() const noexcept {
    return _ids.op_id;
  }

  GraphId graph_id() const noexcept {
    return _ids.graph_id;
  }

  const Graph& graph() const;

  Graph& graph();

  const OpType& type() const noexcept {
    return _body->type();
  }

  const OpInterface& body() const {
    return *_body;
  }

  // for op interface specific func call
  OpInterface& body() {
    return *_body;
  }  

  const OpName& name() const noexcept {
    return _op_meta.name;
  }

  OpName grad_name(size_t input_id = 0) const {
    return name() + "_grad_" + input(input_id)->name();
  }

  const Device& eager_device() const noexcept {
    return _op_meta.eager_device;
  }

  const DeviceGroupHierarchy& device_group_hierarchy() const noexcept {
    return _op_meta.device_group_hierarchy;
  }

  void set_device_group_hierarchy(const DeviceGroupHierarchy& hierarchy) {
    _op_meta.device_group_hierarchy = hierarchy;
  }

  bool is_deduce_states() const noexcept {
    return _op_meta.is_deduce_states;
  }

  DeviceGroupUnion& device_group_union();

  DeviceGroup& device_group();

  const OpMeta& op_meta() const noexcept {
    return _op_meta;
  }

  OpMeta& op_meta() {
    return _op_meta;
  }

  OpMeta grad_op_meta() const {
    return OpMeta()
      .set_stream_index(stream_index())
      .set_device_group_hierarchy(device_group_hierarchy())
      .set_fw_op_id(id());
  }

  void set_fw_op_id(OpId id) {
    _op_meta.fw_op_id = id;
  }

  OpId fw_op_id() const {
    return _op_meta.fw_op_id;
  }

  bool inplace_at(size_t input_position) const {
    return _body->inplace_at(input_position);
  }

  TensorId inplace_tensor_id() const {
    return _inputs[_body->inplace_pos()]->id();
  }

  const TensorList& inputs() const noexcept {
    return _inputs;
  }

  TensorList& inputs() noexcept {
    return _inputs;
  }

  const TensorList& outputs() const noexcept {
    return _outputs;
  }

  TensorList& outputs() noexcept {
    return _outputs;
  }

  const Tensor& input(size_t i) const {
    return _inputs[i];
  }

  Tensor& input(size_t i) {
    return _inputs[i];
  }

  const Tensor& output(size_t i) const {
    return _outputs[i];
  }

  Tensor& output(size_t i) {
    return _outputs[i];
  }

  size_t num_inputs() const {
    return _inputs.size();
  }

  size_t num_outputs() const {
    return _outputs.size();
  }

  const TensorList& in_dep_linkers() const noexcept {
    return _extra_in_dep_linkers;
  }

  TensorList& in_dep_linkers() noexcept {
    return _extra_in_dep_linkers;
  }

  const Tensor& in_dep_linker(size_t i) const {
    return _extra_in_dep_linkers[i];
  }

  Tensor& in_dep_linker(size_t i) {
    return _extra_in_dep_linkers[i];
  }

  size_t num_in_dep_linkers() const {
    return _extra_in_dep_linkers.size();
  }

  void add_in_dep_linker(Tensor in_dep) {
    _extra_in_dep_linkers.push_back(in_dep);
  }

  const Tensor& out_dep_linker() const noexcept {
    return _extra_out_dep_linkers.front();
  }

  Tensor& out_dep_linker() noexcept {
    return _extra_out_dep_linkers.front();
  }

  size_t in_degrees() const {
    return num_inputs() + _extra_in_dep_linkers.size();
  }

  uint64_t op_indicator() const noexcept {
    return _body->op_indicator();
  }

  bool has_placement_group() const {
    return _inst_ctx.has_placement_group;
  }

  const DeviceGroupUnion& placement_group_union() const {
    HT_RUNTIME_ERROR << "Currently it is forbidden to call placement_group_union() of op " << _op_meta.name
      << ", only tensor has placement_group_union(), that is because the placement_group_union() is undefined for ops such as comm op"
      << ", if you want to judge whether op is local, please use placement_group() instead";
    return _inst_ctx.placement_group_union;
  }

  const DeviceGroup placement_group() const {
    HT_ASSERT(has_placement_group())
      << _op_meta.name << " should set placement group in advance";
    return _inst_ctx.placement_group_union.all();
  }

  const DeviceGroup local_placement_group() const {
    HT_RUNTIME_ERROR << "Currently it is forbidden to call local_placement_group() of op " << _op_meta.name
      << ", only tensor has local_placement_group(), that is because the local_placement_group() is undefined for ops such as comm op";
    HT_ASSERT(!placement().is_undetermined())
      << "local_placement_group should be called only after instantiated the placement";
    return _inst_ctx.placement_group_union.get(placement());
  }

  size_t local_placement_group_idx() const {
    HT_RUNTIME_ERROR << "Currently it is forbidden to call local_placement_group_idx() of op " << _op_meta.name
      << ", only tensor has local_placement_group_idx(), that is because the local_placement_group_idx() is undefined for ops such as comm op";
    HT_ASSERT(!placement().is_undetermined())
      << "local_placement_group_idx should be called only after instantiated the placement";
    return _inst_ctx.placement_group_union.get_index(placement());
  }

  size_t suggested_hetero_id() const {
    return _suggested_hetero_id;
  }

  void set_suggested_hetero_id(size_t suggested_hetero_id) {
    _suggested_hetero_id = suggested_hetero_id;
  }

  const Device& placement() const noexcept {
    return _inst_ctx.placement;
  }

  Stream stream() const noexcept {
    return Stream(placement(), stream_index());
  }

  StreamIndex stream_index() const noexcept {
    return _inst_ctx.stream_index;
  }

  const OpInstantiationContext& instantiation_ctx() const {
    return _inst_ctx;
  }

  OpInstantiationContext& instantiation_ctx() {
    return _inst_ctx;
  }

  bool is_computed() const {
    return false;
  }

  bool is_parameter() const;

  bool requires_grad(size_t i) const {
    return _inputs[i]->requires_grad();
  }

  bool is_bw_op() const {
    return _op_meta.fw_op_id != -1;
  }

 protected:
  // Walkaround methods to get the corresponding wrapper
  Operator& get_self();

  const Operator& get_self() const;

  void BlockOrSyncAllInputs(RuntimeContext& runtime_ctx, size_t micro_batch_id = 0);
  
  void BlockOrSyncInput(Tensor& input, RuntimeContext& runtime_ctx, size_t micro_batch_id = 0);

  const OpIdentifier _ids;
  std::shared_ptr<OpInterface> _body;

  TensorList _inputs;
  TensorList _outputs;
  TensorList _extra_in_dep_linkers;
  TensorList _extra_out_dep_linkers;

  OpMeta _op_meta;
  OpInstantiationContext _inst_ctx;

  size_t _suggested_hetero_id{0}; // suggest which hetero id should use for non-local op
};

class Operator : public shared_ptr_wrapper<OpDef> {
 protected:
  friend class Graph;
  friend class DefineByRunGraph;
  friend class DefineAndRunGraph;
  friend class ExecutableGraph;

  Operator(OpIdentifier ids, std::shared_ptr<OpInterface> body,
           TensorList inputs, OpMeta op_meta = OpMeta());

 public:
  Operator() = default;

  /******************************************************
   * Helper functions
   ******************************************************/
 public:
  template <typename UnaryFunction>
  static void for_each_input_tensor(Operator& op, UnaryFunction fn) {
    for (auto& tensor : op->_inputs)
      fn(tensor);
    for (auto& tensor : op->_extra_in_dep_linkers)
      fn(tensor);
  }

  template <typename UnaryFunction>
  static void for_each_input_tensor(const Operator& op, UnaryFunction fn) {
    for (const auto& tensor : op->_inputs)
      fn(tensor);
    for (const auto& tensor : op->_extra_in_dep_linkers)
      fn(tensor);
  }

  template <typename UnaryFunction>
  static auto transform_each_input_tensor(Operator& op, UnaryFunction fn) {
    std::vector<decltype(fn(op->_inputs.front()))> transformed_inputs;
    std::vector<decltype(fn(op->_inputs.front()))> transformed_in_deps;
    transformed_inputs.reserve(op->_inputs.size());
    transformed_in_deps.reserve(op->_extra_in_dep_linkers.size());
    for (auto& tensor : op->_inputs)
      transformed_inputs.push_back(fn(tensor));
    for (auto& tensor : op->_extra_in_dep_linkers)
      transformed_in_deps.push_back(fn(tensor));
    return std::make_tuple(transformed_inputs, transformed_in_deps);
  }

  template <typename UnaryFunction>
  static auto transform_each_input_tensor(const Operator& op,
                                          UnaryFunction fn) {
    std::vector<decltype(fn(op->_inputs.front()))> transformed_inputs;
    std::vector<decltype(fn(op->_inputs.front()))> transformed_in_deps;
    transformed_inputs.reserve(op->_inputs.size());
    transformed_in_deps.reserve(op->_extra_in_dep_linkers.size());
    for (const auto& tensor : op->_inputs)
      transformed_inputs.push_back(fn(tensor));
    for (const auto& tensor : op->_extra_in_dep_linkers)
      transformed_in_deps.push_back(fn(tensor));
    return std::make_tuple(transformed_inputs, transformed_in_deps);
  }

  template <typename BinaryFunction>
  static void for_each_input_tensor_pair(Operator& op1, Operator& op2,
                                         BinaryFunction fn) {
    for (size_t i = 0; i < op1->_inputs.size(); i++)
      fn(op1->_inputs.at(i), op2->_inputs.at(i));
    for (size_t i = 0; i < op1->_extra_in_dep_linkers.size(); i++)
      fn(op1->_extra_in_dep_linkers.at(i), op2->_extra_in_dep_linkers.at(i));
  }

  template <typename BinaryFunction>
  static void for_each_input_tensor_pair(const Operator& op1,
                                         const Operator& op2,
                                         BinaryFunction fn) {
    for (size_t i = 0; i < op1->_inputs.size(); i++)
      fn(op1->_inputs.at(i), op2->_inputs.at(i));
    for (size_t i = 0; i < op1->_extra_in_dep_linkers.size(); i++)
      fn(op1->_extra_in_dep_linkers.at(i), op2->_extra_in_dep_linkers.at(i));
  }

  template <typename UnaryFunction>
  static void for_each_output_tensor(Operator& op, UnaryFunction fn) {
    if (op->_outputs.empty()) {
      fn(op->_extra_out_dep_linkers.front());
    } else {
      for (auto& tensor : op->_outputs)
        fn(tensor);
    }
  }

  template <typename UnaryFunction>
  static void for_each_output_tensor(const Operator& op, UnaryFunction fn) {
    if (op->_outputs.empty()) {
      fn(op->_extra_out_dep_linkers.front());
    } else {
      for (const auto& tensor : op->_outputs)
        fn(tensor);
    }
  }

  template <typename BinaryFunction>
  static void for_each_output_tensor_pair(Operator& op1, Operator& op2,
                                          BinaryFunction fn) {
    if (op1->_outputs.empty()) {
      fn(op1->_extra_out_dep_linkers.front(),
         op2->_extra_out_dep_linkers.front());
    } else {
      for (size_t i = 0; i < op1->_outputs.size(); i++)
        fn(op1->_outputs.at(i), op2->_outputs.at(i));
    }
  }

  template <typename BinaryFunction>
  static void for_each_output_tensor_pair(const Operator& op1,
                                          const Operator& op2,
                                          BinaryFunction fn) {
    if (op1->_outputs.empty()) {
      fn(op1->_extra_out_dep_linkers.front(), op2->_extra_out_dep_linkers.front());
    } else {
      for (size_t i = 0; i < op1->_outputs.size(); i++)
        fn(op1->_outputs.at(i), op2->_outputs.at(i));
    }
  }

  template <typename UnaryPredicate>
  static bool all_input_tensors_of(Operator& op, UnaryPredicate pred) {
    for (auto& tensor : op->_inputs)
      if (!pred(tensor))
        return false;
    for (auto& tensor : op->_extra_in_dep_linkers)
      if (!pred(tensor))
        return false;
    return true;
  }

  template <typename UnaryPredicate>
  static bool all_input_tensors_of(const Operator& op, UnaryPredicate pred) {
    for (const auto& tensor : op->_inputs)
      if (!pred(tensor))
        return false;
    for (const auto& tensor : op->_extra_in_dep_linkers)
      if (!pred(tensor))
        return false;
    return true;
  }

  template <typename UnaryPredicate>
  static bool any_input_tensor_of(Operator& op, UnaryPredicate pred) {
    for (auto& tensor : op->_inputs)
      if (pred(tensor))
        return true;
    for (auto& tensor : op->_extra_in_dep_linkers)
      if (pred(tensor))
        return true;
    return false;
  }

  template <typename UnaryPredicate>
  static bool any_input_tensor_of(const Operator& op, UnaryPredicate pred) {
    for (const auto& tensor : op->_inputs)
      if (pred(tensor))
        return true;
    for (const auto& tensor : op->_extra_in_dep_linkers)
      if (pred(tensor))
        return true;
    return false;
  }

  template <typename UnaryPredicate>
  static bool all_output_tensors_of(Operator& op, UnaryPredicate pred) {
    if (op->_outputs.empty()) {
      return pred(op->_extra_out_dep_linkers.front());
    } else {
      for (auto& tensor : op->_outputs)
        if (!pred(tensor))
          return false;
    }
    return true;
  }

  template <typename UnaryPredicate>
  static bool all_output_tensors_of(const Operator& op, UnaryPredicate pred) {
    if (op->_outputs.empty()) {
      return pred(op->_extra_out_dep_linkers.front());
    } else {
      for (const auto& tensor : op->_outputs)
        if (!pred(tensor))
          return false;
    }
    return true;
  }

  template <typename UnaryPredicate>
  static bool any_output_tensor_of(Operator& op, UnaryPredicate pred) {
    if (op->_outputs.empty()) {
      return pred(op->_extra_out_dep_linkers.front());
    } else {
      for (auto& tensor : op->_outputs)
        if (pred(tensor))
          return true;
    }
    return false;
  }

  template <typename UnaryPredicate>
  static bool any_output_tensor_of(const Operator& op, UnaryPredicate pred) {
    if (op->_outputs.empty()) {
      return pred(op->_extra_out_dep_linkers.front());
    } else {
      for (const auto& tensor : op->_outputs)
        if (pred(tensor))
          return true;
    }
    return false;
  }
};

std::ostream& operator<<(std::ostream&, const Operator&);

/******************************************************
 * Indicators of Operators
 ******************************************************/

static const uint64_t PLACEHOLDER_OP = 1ul << 1;
static const uint64_t VARIABLE_OP = 1ul << 2;
static const uint64_t HOST_TO_DEVICE_OP = 1ul << 3;
static const uint64_t DEVICE_TO_HOST_OP = 1ul << 4;
static const uint64_t PEER_TO_PEER_SEND_OP = 1ul << 5;
static const uint64_t PEER_TO_PEER_RECV_OP = 1ul << 6;
static const uint64_t ALL_TO_ALL_OP = 1ul << 7;
static const uint64_t ALL_REDUCE_OP = 1ul << 8;
static const uint64_t ALL_GATHER_OP = 1ul << 9;
static const uint64_t REDUCE_SCATTER_OP = 1ul << 10;
static const uint64_t BROADCAST_OP = 1ul << 11;
static const uint64_t REDUCE_OP = 1ul << 12;
static const uint64_t P2P_OP = 1ul << 13;
static const uint64_t BATCHED_ISEND_IRECV_OP = 1ul << 14;
static const uint64_t GATHER_OP = 1ul << 15;
static const uint64_t SCATTER_OP = 1ul << 16;
static const uint64_t INPLACE_OP = 1ul << 17;
static const uint64_t COMM_SPLIT_OP = 1ul << 19;
static const uint64_t SPLIT_ALL_REDUCE_OP = 1ul << 20;
static const uint64_t SPLIT_ALL_GATHER_OP = 1ul << 21;
static const uint64_t SPLIT_REDUCE_SCATTER_OP = 1ul << 22;
static const uint64_t COMM_OP = 1ul << 23;
static const uint64_t UNKNOWN_OP = 1ul << 24;
static const uint64_t UNUSED_OP = 1ul << 25;
static const uint64_t PARALLEL_ATTN_OP = 1ul << 26;
static const uint64_t PARALLEL_ATTN_GRAD_OP = 1ul << 27;
static const uint64_t BINARY_OP = 1ul << 28;
static const uint64_t INDEX_ADD_OP = 1ul << 29;
static const uint64_t FUSED_GROUP_OP = 1ul << 53;
static const uint64_t CONCAT_OP = 1ul << 54;
static const uint64_t CONTIGUOUS_OP = 1ul << 55;
static const uint64_t DATA_TRANSFER_OP = 1ul << 56;
static const uint64_t ADAM_OP = 1ul << 57;
static const uint64_t SUM_OP = 1ul << 58;
static const uint64_t SLICE_OP = 1ul << 59;
static const uint64_t LOSS_OP = 1ul << 60;
static const uint64_t LOSS_GRADIENT_OP = 1ul << 61;
static const uint64_t OPTIMIZER_UPDATE_OP = 1ul << 62;
static const uint64_t GROUP_OP = 1ul << 63;

#define DECLARE_OP_INDICATOR_CHECKER(type, indicator)                          \
  inline bool is_##type##_op(const OpInterface& x) {                           \
    return (x.op_indicator() & (indicator)) != 0;                              \
  }                                                                            \
  inline bool is_##type##_op(const Operator& x) {                              \
    return is_##type##_op(x->body());                                          \
  }                                                                            \
  inline bool is_##type##_op(const OpRef& x) {                                 \
    return is_##type##_op(x.get());                                            \
  }                                                                            \
  inline bool is_##type##_op(const OpCRef& x) {                                \
    return is_##type##_op(x.get());                                            \
  }

DECLARE_OP_INDICATOR_CHECKER(placeholder, PLACEHOLDER_OP)
DECLARE_OP_INDICATOR_CHECKER(variable, VARIABLE_OP)
DECLARE_OP_INDICATOR_CHECKER(host_to_device, HOST_TO_DEVICE_OP)
DECLARE_OP_INDICATOR_CHECKER(device_to_host, DEVICE_TO_HOST_OP)
DECLARE_OP_INDICATOR_CHECKER(peer_to_peer_send, PEER_TO_PEER_SEND_OP)
DECLARE_OP_INDICATOR_CHECKER(peer_to_peer_recv, PEER_TO_PEER_RECV_OP)
DECLARE_OP_INDICATOR_CHECKER(all_to_all, ALL_TO_ALL_OP)
DECLARE_OP_INDICATOR_CHECKER(all_reduce, ALL_REDUCE_OP)
DECLARE_OP_INDICATOR_CHECKER(all_gather, ALL_GATHER_OP)
DECLARE_OP_INDICATOR_CHECKER(reduce_scatter, REDUCE_SCATTER_OP)
DECLARE_OP_INDICATOR_CHECKER(split_all_reduce, SPLIT_ALL_REDUCE_OP)
DECLARE_OP_INDICATOR_CHECKER(split_all_gather, SPLIT_ALL_GATHER_OP)
DECLARE_OP_INDICATOR_CHECKER(split_reduce_scatter, SPLIT_REDUCE_SCATTER_OP)
DECLARE_OP_INDICATOR_CHECKER(broadcast, BROADCAST_OP)
DECLARE_OP_INDICATOR_CHECKER(reduce, REDUCE_OP)
DECLARE_OP_INDICATOR_CHECKER(p2p, P2P_OP)
DECLARE_OP_INDICATOR_CHECKER(batched_isend_irecv, BATCHED_ISEND_IRECV_OP)
DECLARE_OP_INDICATOR_CHECKER(gather, GATHER_OP)
DECLARE_OP_INDICATOR_CHECKER(scatter, SCATTER_OP)
DECLARE_OP_INDICATOR_CHECKER(inplace, INPLACE_OP)
DECLARE_OP_INDICATOR_CHECKER(grad_reduce, 
                             ALL_REDUCE_OP | REDUCE_SCATTER_OP |
                             SPLIT_ALL_REDUCE_OP | SPLIT_REDUCE_SCATTER_OP)
DECLARE_OP_INDICATOR_CHECKER(comm_split, COMM_SPLIT_OP)
DECLARE_OP_INDICATOR_CHECKER(comm, COMM_OP)
DECLARE_OP_INDICATOR_CHECKER(parallel_attn, PARALLEL_ATTN_OP)
DECLARE_OP_INDICATOR_CHECKER(parallel_attn_grad, PARALLEL_ATTN_GRAD_OP)
DECLARE_OP_INDICATOR_CHECKER(binary, BINARY_OP)
DECLARE_OP_INDICATOR_CHECKER(unknown, UNKNOWN_OP)
DECLARE_OP_INDICATOR_CHECKER(index_add, INDEX_ADD_OP)
DECLARE_OP_INDICATOR_CHECKER(communication,
                             PEER_TO_PEER_SEND_OP | PEER_TO_PEER_RECV_OP |
                             ALL_TO_ALL_OP | ALL_REDUCE_OP |
                             ALL_GATHER_OP | REDUCE_SCATTER_OP |
                             SPLIT_ALL_REDUCE_OP | SPLIT_REDUCE_SCATTER_OP |
                             SPLIT_ALL_GATHER_OP |
                             BROADCAST_OP | REDUCE_OP |
                             P2P_OP | BATCHED_ISEND_IRECV_OP |
                             GATHER_OP | SCATTER_OP)
DECLARE_OP_INDICATOR_CHECKER(data_transfer, DATA_TRANSFER_OP)
DECLARE_OP_INDICATOR_CHECKER(adam, ADAM_OP)
DECLARE_OP_INDICATOR_CHECKER(sum, SUM_OP)                               
DECLARE_OP_INDICATOR_CHECKER(slice, SLICE_OP)
DECLARE_OP_INDICATOR_CHECKER(concat, CONCAT_OP)
DECLARE_OP_INDICATOR_CHECKER(contiguous, CONTIGUOUS_OP)
DECLARE_OP_INDICATOR_CHECKER(loss, LOSS_OP)
DECLARE_OP_INDICATOR_CHECKER(loss_gradient, LOSS_GRADIENT_OP)
DECLARE_OP_INDICATOR_CHECKER(optimizer_update, OPTIMIZER_UPDATE_OP)
DECLARE_OP_INDICATOR_CHECKER(group, GROUP_OP)

inline StreamIndex get_suggested_stream_index(const Operator& op) {
  if (is_host_to_device_op(op)) {
    return kH2DStream;
  } else if (is_device_to_host_op(op)) {
    return kD2HStream;
  } else if (is_peer_to_peer_send_op(op) || is_peer_to_peer_recv_op(op)) {
    return kP2PStream;
  } else if (is_communication_op(op)) {
    return kCollectiveStream;
  } else {
    return kComputingStream;
  }
}

} // namespace graph
} // namespace hetu
