#include "hetu/graph/headers.h"
#include "hetu/graph/graph.h"
#include "hetu/graph/ops/group.h"
#include "hetu/graph/ops/variable.h"
#include "hetu/graph/ops/Communication.h"
#include "hetu/impl/stream/CPUStream.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/communication/comm_group.h"

namespace hetu {
namespace graph {

void OpInterface::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                 const OpMeta& op_meta) const {
  // default: distributed states of output tensor directly copy from input tensor
  // check input states is valid & check distributed states of all input tensor are the same.
  HT_ASSERT(inputs.size() > 0) << op_meta.name << ": distributed states should be manually set when in_degree=0!";
  HT_LOG_TRACE << hetu::impl::comm::GetLocalDevice() << " " << op_meta.name << ": default copy states from inputs";
  DistributedStates default_ds;
  for (auto& input : inputs) {
    const auto& input_ds = input->get_distributed_states(); 
    HT_ASSERT(input_ds.is_valid()) << op_meta.name << ": input states must be valid! and " 
                                    << "input: " << input << ", input_ds: " << input_ds.ds_info();
    HT_ASSERT(input_ds.get_dim(-2) == 1) << op_meta.name << ": input shouldn't be partial!";      
    if (!default_ds.is_valid()) {
      default_ds.set_distributed_states(input_ds);
    } else {
      HT_ASSERT(default_ds.check_equal(input_ds))
        << op_meta.name << ": in Default DoDeduceStates: distributed states of all input tensor must be same"
        << ", but found at least two different ds: " << default_ds.ds_info() << " vs. " << input_ds.ds_info();
    }
  }
  for (auto& output : outputs) {
    output->set_distributed_states(default_ds);
  }
}

void OpInterface::DoDeduceHeterProp(const std::vector<int32_t>& inputs_hetero_dim,
                                    TensorList& outputs, const OpMeta& op_meta) const {
  HT_ASSERT(inputs_hetero_dim.size() > 0) 
    << op_meta.name << ": distributed states hetero dim should be manually set when in_degree is 0";
  HT_LOG_TRACE << hetu::impl::comm::GetLocalDevice() << " " << op_meta.name << ": default copy states hetero dim from inputs";
  int32_t default_hetero_dim = -3;
  for (auto& hetero_dim : inputs_hetero_dim) {
    if (default_hetero_dim == -3) {
      default_hetero_dim = hetero_dim;
    }
    HT_ASSERT(default_hetero_dim == hetero_dim)
      << "hetero dim should be equal for all inputs by default"
      << ", or you need to implement special DoDeduceHeterProp() method for " << op_meta.name;
  }
  for (auto& output : outputs) {
    output->cur_ds_union().set_hetero_dim(default_hetero_dim);
  }
}

void OpInterface::DoDeduceStatesHierarchy(const TensorList& inputs, TensorList& outputs, 
                                          const OpMeta& op_meta, Graph& graph) const {
  int32_t hetero_size = 1;
  std::vector<int32_t> inputs_hetero_dim;
  for (const auto& input : inputs) {
    HT_ASSERT(input->has_cur_ds_union())
      << "Ds union of inputs should be created before deducing hetero dim"
      <<  ", but cur strategy id is " << input->cur_strategy_id() 
      << ", and size of its ds hierarchy is " << input->ds_hierarchy().size();
    const auto& cur_ds_union = input->cur_ds_union();
    inputs_hetero_dim.emplace_back(cur_ds_union.hetero_dim());
    int32_t cur_hetero_size = cur_ds_union.size();
    if (cur_hetero_size == 1) {
      HT_ASSERT(cur_ds_union.hetero_dim() == NULL_HETERO_DIM)
        << "Union size = 1 means homo";
      continue;
    }
    if (hetero_size == 1) {
      hetero_size = cur_hetero_size;
    }
    HT_ASSERT(cur_ds_union.hetero_dim() != NULL_HETERO_DIM)
      << "Op input " << input << " has an illegal hetero ds";
    HT_ASSERT(hetero_size == cur_hetero_size)
      << "Op " << op_meta.name << " has an unaligned hetero ds size";
  }
  for (const auto& input : inputs) {
    const auto& cur_ds_union = input->cur_ds_union();
    HT_LOG_WARN_IF(cur_ds_union.size() != hetero_size)
      << input << " has " << cur_ds_union.ds_union_info()
      << ", although we aim to support homo & hetero deducing"
      << ", but we currently don't suggest you to do so"
      << ", there may be some unknown bugs remain";
  }
  for (auto& output : outputs) {
    HT_ASSERT(!output->has_cur_ds_union())
      << "Ds union of outputs shouldn't be created before deducing hetero dim";
  }
  // 创建所有outputs的ds union并确定hetero dim
  graph.CREATE_STRATEGY = true;
  // HT_LOG_WARN << "DeduceHeterProp for " << op_meta.name;
  DeduceHeterProp(inputs_hetero_dim, outputs, op_meta);
  graph.CREATE_STRATEGY = false;
  // 依据是否为hetero
  // 创建所有outputs的ds union中的ds
  for (auto& output : outputs) {
    auto& ds_union = output->cur_ds_union();
    if (ds_union.hetero_dim() == NULL_HETERO_DIM) {
      ds_union.add(DistributedStates());
    } else {
      HT_ASSERT(hetero_size != 1)
        << "There is a hetero dim, so the hetero size shouldn't equal to 1";
      for (int32_t i = 0; i < hetero_size; i++) {
        ds_union.add(DistributedStates());
      }
    }
  }
  // 推导所有outputs的ds union中的ds
  graph.USE_HETERO_ID = true;
  for (size_t cur_hetero_id = 0; cur_hetero_id < hetero_size; cur_hetero_id++) {
    graph.CUR_HETERO_ID = cur_hetero_id;
    DeduceStates(inputs, outputs, op_meta);
  }
  /*
  for (auto& output : outputs) {
    HT_LOG_WARN << output << " ds union is " << output->cur_ds_union().ds_union_info();
  }
  */
  graph.CUR_HETERO_ID = 0;
  graph.USE_HETERO_ID = false;
}

bool OpInterface::DoMapToParallelDevices(Operator& op,
                                         const DeviceGroupUnion& pg_union) const {
  HT_ASSERT(pg_union.size() > 0) 
    << "Placement Group Union shouldn't be empty";
  op->instantiation_ctx().placement_group_union = pg_union;
  op->instantiation_ctx().has_placement_group = true;
  // set output statuses
  Operator::for_each_output_tensor(
    op, [&](Tensor& tensor) { tensor->set_placement_group_union(pg_union); });
  // TODO: add P2P communication ops for pipeline parallel
  return true;
}

void OpInterface::DoMergeStrategy(Operator& op, Operator& another_op) {
  // merge op dg_hierarchy & merge tensor ds_hierarchy
  // some ops like comm and variable need special treatment
  HT_ASSERT(op->type() == another_op->type())
    << "cannot merge two ops with different types: "
    << op << " and " << another_op;
  // merge op dg_hierarchy
  DeviceGroupHierarchy new_dg_hierarchy(op->device_group_hierarchy());
  for (const auto& dg_union : another_op->device_group_hierarchy().raw_data()) {
    new_dg_hierarchy.add(dg_union);
  }
  HT_ASSERT(new_dg_hierarchy.size() == 0 || new_dg_hierarchy.size() == op->graph().NUM_STRATEGY)
    << "please set the correct NUM_STRATEGY of the graph before merge strategy"
    << ", if dg hierarchy size is not zero";
  op->set_device_group_hierarchy(new_dg_hierarchy);
  // merge tensor ds_hierarchy
  HT_ASSERT(op->num_outputs() == another_op->num_outputs())
    << "output size should be equal";
  for (size_t i = 0; i < another_op->num_outputs(); i++) {
    op->output(i)->merge_strategy(another_op->output(i));
  }
  /*
  Operator::for_each_output_tensor_pair(
    op, another_op, 
    [&](Tensor& tensor, Tensor& another_tensor) { tensor->merge_strategy(another_tensor); }
  );
  */
  DoSpecialMergeStrategy(op, another_op);
}

void OpInterface::DoSpecialMergeStrategy(Operator& op, Operator& another_op) {
  return;
}

bool OpInterface::DoInstantiate(Operator& op, const Device& placement,
                                StreamIndex stream_index) const {
  auto& inst_ctx = op->instantiation_ctx();
  inst_ctx.placement = placement;
  inst_ctx.stream_index = stream_index;
  auto enable_timing = op->graph().EVENT_TIMING;
  if (placement.is_cuda()) {
    for (size_t i = 0; i < HT_MAX_NUM_MICRO_BATCHES; i++) { 
      inst_ctx.start[i] = std::make_unique<hetu::impl::CUDAEvent>(placement, enable_timing);
      inst_ctx.stop[i] = std::make_unique<hetu::impl::CUDAEvent>(placement, enable_timing);
    }
  } else {
    for (size_t i = 0; i < HT_MAX_NUM_MICRO_BATCHES; i++) {     
      inst_ctx.start[i] = std::make_unique<hetu::impl::CPUEvent>(enable_timing);
      inst_ctx.stop[i] = std::make_unique<hetu::impl::CPUEvent>(enable_timing);
    }
  }
  Operator::for_each_output_tensor(
    op, [&](Tensor& tensor) { tensor->set_placement(placement); });
  return true;
}

HTShapeList OpInterface::DoInferShape(Operator& op,
                                      const HTShapeList& input_shapes,
                                      RuntimeContext& runtime_ctx) const {
  if (op->num_outputs() == 0)
    return HTShapeList();
  HT_NOT_IMPLEMENTED << "InferShape fn of op \"" << op->type()
                     << "\" is not defined";
  __builtin_unreachable();
}

NDArrayList OpInterface::DoAllocOutputs(Operator& op, const NDArrayList& inputs,
                                        RuntimeContext& runtime_ctx) const {
  NDArrayList outputs;
  auto output_size = op->num_outputs();
  if (output_size > 0) {
    outputs.reserve(output_size);
    // 动态图
    // 无runtime_ctx
    // 现推output_shapes
    if (!runtime_ctx.has_shape_plan()) {
      HTShapeList input_shapes;
      input_shapes.reserve(op->num_inputs());
      for (auto& input : inputs) {
        input_shapes.push_back(input->shape());
      }
      auto output_shapes = DoInferShape(op, input_shapes, runtime_ctx);
      for (size_t i = 0; i < output_size; i++) {
        outputs.push_back(NDArray::empty(output_shapes[i],
                          op->instantiation_ctx().placement,
                          op->output(i)->dtype(),
                          op->instantiation_ctx().stream_index));
      }
    }
    // 静态图
    // 有runtime_ctx
    // output_shapes全部提前设置好
    // 部分output的allocation也会设置好
    // 在使用memory plan后大部分output的allocation也会设置好
    else {
      for (size_t i = 0; i < output_size; i++) {
        // question: will tensor shape != NDArray shape happen in any situation
        auto output_id = op->output(i)->id();
        // HT_LOG_INFO << hetu::impl::comm::GetLocalDevice() << ": get runtime shape for " << op->output(i);
        const auto& output_shape = runtime_ctx.get_runtime_shape(output_id);
        HT_LOG_TRACE << hetu::impl::comm::GetLocalDevice() << ": exec op " << op
          << " output " << i << " shape = " << output_shape << " ds = " << op->output(i)->get_distributed_states().ds_info();
        if (runtime_ctx.has_runtime_allocation(output_id)) {
          // HT_LOG_INFO << op->output(i) << " has runtime allocation";
          outputs.push_back(runtime_ctx.get_runtime_allocation(output_id));
        } 
        // alloc on-the-fly
        else {
          // mempool debug use
          /*
          HT_LOG_INFO << hetu::impl::comm::GetLocalDevice() << ": exec op " << op
            << " on-the-fly alloc output " << i << " shape = " << output_shape << " stream = " << op->instantiation_ctx().stream();
          if (op->name() == "OnesLikeOp" || op->name() == "VocabParallelCrossEntropyOp") {
            HT_LOG_INFO << "--- sync point ---";
            op->instantiation_ctx().stream().Sync();
          }
          */
          auto output = NDArray::empty(output_shape,
                                       op->instantiation_ctx().placement,
                                       op->output(i)->dtype(),
                                       op->instantiation_ctx().stream_index);
          outputs.push_back(output);
        }
      }
    }
  }
  return outputs;
}

NDArrayList OpInterface::DoAllocOutputs(Operator& op, const NDArrayList& inputs,
                                        RuntimeContext& runtime_ctx, const Device& device) const {
  NDArrayList outputs;
  auto output_size = op->num_outputs();
  if (output_size > 0) {
    outputs.reserve(output_size);
    // 动态图
    // 无runtime_ctx
    // 现推output_shapes
    if (!runtime_ctx.has_shape_plan()) {
      HTShapeList input_shapes;
      input_shapes.reserve(op->num_inputs());
      for (auto& input : inputs) {
        input_shapes.push_back(input->shape());
      }
      auto output_shapes = DoInferShape(op, input_shapes, runtime_ctx);
      for (size_t i = 0; i < output_size; i++) {
        outputs.push_back(NDArray::empty(output_shapes[i],
                          device,
                          op->output(i)->dtype(),
                          op->instantiation_ctx().stream_index));
      }
    }
    // 静态图
    // 有runtime_ctx
    // output_shapes全部提前设置好
    // 部分output的allocation也会设置好
    // 在使用memory plan后大部分output的allocation也会设置好
    else {
      for (size_t i = 0; i < output_size; i++) {
        // question: will tensor shape != NDArray shape happen in any situation
        auto output_id = op->output(i)->id();
        // HT_LOG_INFO << hetu::impl::comm::GetLocalDevice() << ": get runtime shape for " << op->output(i);
        const auto& output_shape = runtime_ctx.get_runtime_shape(output_id);
        HT_LOG_TRACE << hetu::impl::comm::GetLocalDevice() << ": exec op " << op
          << " output " << i << " shape = " << output_shape << " ds = " << op->output(i)->get_distributed_states().ds_info();
        if (runtime_ctx.has_runtime_allocation(output_id)) {
          outputs.push_back(runtime_ctx.get_runtime_allocation(output_id));
        } 
        // alloc on-the-fly
        else {
          outputs.push_back(NDArray::empty(output_shape,
                            device,
                            op->output(i)->dtype(),
                            op->instantiation_ctx().stream_index));
        }
      }
    }
  }
  return outputs;
}


NDArray OpInterface::DoAllocOutput(Operator& op, const NDArrayList& inputs,
                                   size_t idx, RuntimeContext& runtime_ctx) const {
  auto output_size = op->num_outputs();
  HT_ASSERT(idx < output_size)
    << "Output index " << idx << " is out of range for op " << op
    << " while allocating outputs";

  // 动态图
  // 无runtime_ctx
  // 现推output_shapes
  if (!runtime_ctx.has_shape_plan()) {
    HTShapeList input_shapes;
    input_shapes.reserve(op->num_inputs());
    for (auto& input : inputs) {
      input_shapes.push_back(input->shape());
    }
    auto output_shapes = DoInferShape(op, input_shapes, runtime_ctx);
    return NDArray::empty(output_shapes[idx],
                          op->instantiation_ctx().placement,
                          op->output(idx)->dtype(),
                          op->instantiation_ctx().stream_index);
  }
  // 静态图
  // 有runtime_ctx
  // output_shapes全部提前设置好
  // 在使用memory plan后大部分output的allocation也会设置好
  else {
    // question: will tensor shape != NDArray shape happen in any situation
    auto output_id = op->output(idx)->id();
    // HT_LOG_INFO << hetu::impl::comm::GetLocalDevice() << ": get runtime shape for " << op->output(idx);
    const auto& output_shape = runtime_ctx.get_runtime_shape(output_id);
    HT_LOG_TRACE << hetu::impl::comm::GetLocalDevice() << ": exec op " << op
      << " output " << idx << " shape = " << output_shape << " ds = " << op->output(idx)->get_distributed_states().ds_info();
    if (runtime_ctx.has_runtime_allocation(output_id)) {
      return runtime_ctx.get_runtime_allocation(output_id);
    } 
    // alloc on-the-fly
    else {
      auto output = NDArray::empty(output_shape,
                                   op->instantiation_ctx().placement,
                                   op->output(idx)->dtype(),
                                   op->instantiation_ctx().stream_index);
      return output;
    }
  }
}

OpDef::OpDef(const constructor_access_key&, OpIdentifier ids,
             std::shared_ptr<OpInterface> body, TensorList inputs,
             OpMeta op_meta)
: _ids{std::move(ids)},
  _body(std::move(body)),
  _inputs(std::move(inputs)),
  _op_meta(std::move(op_meta)) {
  auto& graph = Graph::GetGraph(_ids.graph_id);
  // Question: Is op name really necessary?
  if (_op_meta.name.empty()) {
    auto cnt = graph.get_op_type_cnt(_body->type());
    if (cnt == 0)
      _op_meta.name = _body->type();
    else
      _op_meta.name = _body->type() + '(' + std::to_string(cnt) + ')';
  }
  // All inputs must be tensors
  for (size_t i = 0; i < _inputs.size(); i++) {
    HT_VALUE_ERROR_IF((!_inputs[i].is_defined()) ||
                      _inputs[i]->is_out_dep_linker())
      << "Failed to construct the \"" << _body->type() << "\" op: "
      << "Cannot convert input " << i << " to a tensor: " << _inputs[i] << ".";
  }
  // Extra input depencenies. May be tensors or output dependency linkers
  auto& extra_deps = _op_meta.extra_deps;
  if (extra_deps.size() <= 1 || is_group_op(*_body)) {
    // Walkaround: if we are constructing a group op,
    // we shall not construct another group op to handle the dependecies.
    _extra_in_dep_linkers = extra_deps;
  } else {
    // Merge dependencies into a group op
    _extra_in_dep_linkers.push_back(
      MakeGroupOp(OpMeta()
                    .set_extra_deps(extra_deps)
                    .set_name(_op_meta.name + "_extra_deps")));
  }
  // Deduce requires grad
  bool requires_grad = false;
  if (is_variable_op(*_body)) {
    if (_body->type() == "VariableOp")
      requires_grad = reinterpret_cast<VariableOpImpl&>(*_body).requires_grad();
    else
      requires_grad = reinterpret_cast<ParallelVariableOpImpl&>(*_body).requires_grad();
  } else {
    requires_grad =
      std::any_of(_inputs.begin(), _inputs.end(),
                  [](const Tensor& tensor) { return tensor->requires_grad(); });
  }
  // Outputs of this op
  auto output_meta_list = _body->InferMeta(_inputs);
  if (output_meta_list.size() == 1) {
    auto& output_meta = output_meta_list.front();
    HT_ASSERT(output_meta.dtype != kUndeterminedDataType)
      << "Data type is not provided for output " << _outputs.size()
      << " of the \"" << _body->type() << "\" op.";
    _outputs.emplace_back(
      TensorIdentifier{_ids.graph_id, _ids.op_id, 0, graph.next_tensor_id()},
      _op_meta.name, requires_grad, output_meta);
  } else if (output_meta_list.size() > 1) {
    _outputs.reserve(output_meta_list.size());
    for (int i = 0; i < static_cast<int>(output_meta_list.size()); i++) {
      // (TensorIdentifier ids, TensorName name, NDArrayMeta meta)
      auto& output_meta = output_meta_list[i];
      HT_ASSERT(output_meta.dtype != kUndeterminedDataType)
        << "Data type is not provided for output " << i << " of the \""
        << _body->type() << "\" op.";
      _outputs.emplace_back(
        TensorIdentifier{_ids.graph_id, _ids.op_id, i, graph.next_tensor_id()},
        _op_meta.name + '_' + std::to_string(i), requires_grad, output_meta);
    }
  } else {
    _extra_out_dep_linkers.emplace_back(
      TensorIdentifier{_ids.graph_id, _ids.op_id, -1, graph.next_tensor_id()},
      _op_meta.name, requires_grad);
  }
  
  // Deduce states for output tensor
  if (op_meta.is_deduce_states) {
    auto deduce_states = [&]() {
      bool exist_ds = false;
      bool exist_none_ds = false;
      for (auto& input : _inputs) {
        if (input->has_distributed_states()) {
          exist_ds = true;
        } else {
          exist_none_ds = true;
        }
      }
      // if ds of all inputs are not none, then do deduce states;
      // if ds of all inputs are none, then do not deduce states; 
      // if ds of partial inputs are none, return error
      if (!exist_none_ds) {
        _body->DeduceStatesHierarchy(_inputs, _outputs, _op_meta, graph);
      } else if (exist_ds) {
        HT_RUNTIME_ERROR << "Only part of " << name() << " inputs has distributed states!";
      }
    };
    if (graph.type() == GraphType::DEFINE_AND_RUN) {
      for (size_t cur_strategy_id = 0; cur_strategy_id < graph.NUM_STRATEGY; cur_strategy_id++) {
        graph.CUR_STRATEGY_ID = cur_strategy_id;
        deduce_states();
      }
      graph.CUR_STRATEGY_ID = 0;
    } else if (graph.type() == GraphType::EXECUTABLE) {
      deduce_states();
    } else if (graph.type() == GraphType::EAGER) {
      // eager graph no need to deduce states
    } else {
      HT_RUNTIME_ERROR << "deduce states do not support this graph type: " << graph.type();
    }
  }
}

const Graph& OpDef::graph() const {
  return Graph::GetGraph(graph_id());
}

Graph& OpDef::graph() {
  return Graph::GetGraph(graph_id());
}

Operator& OpDef::get_self() {
  return Graph::GetGraph(graph_id()).GetOp(id());
}

const Operator& OpDef::get_self() const {
  return Graph::GetGraph(graph_id()).GetOp(id());
}

void OpDef::BlockOrSyncAllInputs(RuntimeContext& runtime_ctx, size_t micro_batch_id) {
  for (auto& input : _inputs)
    BlockOrSyncInput(input, runtime_ctx, micro_batch_id);
  for (auto& in_dep : _extra_in_dep_linkers)
    BlockOrSyncInput(in_dep, runtime_ctx, micro_batch_id);
}

void OpDef::BlockOrSyncInput(Tensor& input, RuntimeContext& runtime_ctx, size_t micro_batch_id) {
  if (!input.is_defined())
    return;
  // for commom case
  auto& input_op = input->producer();
  // in_degree=0 op shouldn't blocked
  if (is_placeholder_op(input_op) || is_variable_op(input_op))
    return;
  // p2p ops are all gathered in group start/end, so the start/stop events for p2p ops is invalid, should not be used any more!
  // another case: shared weight p2p ops will not execute in micro batch i>0, so these ops will not record start/stop events.
  // besides, switch exec graph handles this event by itself
  auto is_pipeline_stage_recv_op = [](const Operator& op) -> bool {
    if (is_peer_to_peer_recv_op(op)) {
      return true;
    }
    if (is_batched_isend_irecv_op(op)) {
      const auto& batched_isend_irecv_op_impl = dynamic_cast<BatchedISendIRecvOpImpl&>(op->body());
      // 只收不发
      if (batched_isend_irecv_op_impl.dst_devices().empty()) {
        HT_ASSERT(!batched_isend_irecv_op_impl.src_devices().empty())
          << "only one side could be empty";
        return true;
      }
    }
    return false;
  };
  if (is_pipeline_stage_recv_op(input_op))
    return;
  const auto& input_placement = input_op->instantiation_ctx().placement;
  const auto& current_placement = instantiation_ctx().placement;
  if (input_placement.is_undetermined() || input_placement != current_placement) {
    // We cannot block different devices.
    /*
    HT_LOG_DEBUG << hetu::impl::comm::GetLocalDevice() << ": op " << name() 
      << " placement = " << current_placement
      << " but input " << input_op->name() << " placement = " << input_placement;
    */
    return;
  } else if (input_op->instantiation_ctx().stream_index !=
             instantiation_ctx().stream_index) {
    // Both ops are on the same device. We can block the current op
    // by waiting for the stop event of the dependency.
    if (!input_op->instantiation_ctx().stop[micro_batch_id]->IsRecorded()) {
      HT_ASSERT(runtime_ctx.has_runtime_skipped(input_op->id()))
        << "input op " << input_op << " is not skipped and doesn't record the stop event of "
        << micro_batch_id << " micro batch, which is not permitted";
      return;
    }
    input_op->instantiation_ctx().stop[micro_batch_id]->Block(instantiation_ctx().stream());
    /*
    if (is_split_all_reduce_op(input_op)) {
      HT_LOG_INFO << "input op " << input_op << " stream is " << input_op->instantiation_ctx().stream() 
        << " and current op " << name() << " stream is " << instantiation_ctx().stream()
        << ", block on micro batch " << micro_batch_id;
      input_op->instantiation_ctx().stream().Sync();
    }
    */
  }
}

bool OpDef::is_parameter() const {
  const auto& graph = Graph::GetGraph(graph_id());
  return graph._parameter_ops.find(id()) != graph._parameter_ops.end();
}

DeviceGroupUnion& OpDef::device_group_union() {
  if (graph().CREATE_STRATEGY) {
    while (graph().CUR_STRATEGY_ID >= _op_meta.device_group_hierarchy.size()) {
      _op_meta.device_group_hierarchy.add(DeviceGroupUnion());
    }
  }
  HT_ASSERT(graph().CUR_STRATEGY_ID < _op_meta.device_group_hierarchy.size())
    << "Strategy id out of range";
  return _op_meta.device_group_hierarchy.get(graph().CUR_STRATEGY_ID);
}

DeviceGroup& OpDef::device_group() {
  auto& dg_union = device_group_union();
  if (graph().USE_HETERO_ID) {
    if (graph().CREATE_HETERO) {
      while (graph().CUR_HETERO_ID >= dg_union.size()) {
        dg_union.add(DeviceGroup());
      }
      return dg_union.get(graph().CUR_HETERO_ID);
    }
    // 其余情况暂时不支持使用
    else {
      HT_RUNTIME_ERROR << "Currently not support directly use device group";
    }
  }
  // device_group都使用CUR_HETERO_ID的方式去用
  // 否则请使用placement_group
  else {
    HT_RUNTIME_ERROR << "Ensure you have instantiate the CUR_HETERO_ID and set USE_HETERO_ID to true";
  }
}

size_t OpDef::inferred_local_placement_group_idx() const {
  HT_ASSERT(!graph().USE_HETERO_ID)
    << "inferred_local_placement_group_idx should only used when hetero id is not provided";
  auto inferred = hetu::impl::comm::GetLocalDevice();
  HT_ASSERT(placement().is_undetermined() || placement() == inferred)
    << "inferred_local_placement_group_idx is really a bad idea! "
    << "It is mainly used when the op is not instantiated";
  HT_ASSERT(_inst_ctx.has_placement_group)
    << "inferred_local_placement_group_idx should at least guarantee there is a placement group union";
  if (_inst_ctx.placement_group_union.has(inferred)) {
    return _inst_ctx.placement_group_union.get_index(inferred);
  }
  if (graph().type() == GraphType::DEFINE_AND_RUN) {
    HT_ASSERT(graph().SUGGESTED_HETERO_ID == 0)
      << "Shouldn't change the SUGGESTED_HETERO_ID of the define graph";
  }
  return _inst_ctx.placement_group_union.size() > 1 ? graph().SUGGESTED_HETERO_ID : 0;
}

Operator::Operator(OpIdentifier ids, std::shared_ptr<OpInterface> body,
                   TensorList inputs, OpMeta op_meta) {
  this->_ptr =
    make_ptr<OpDef>(OpDef::constructor_access_key(), std::move(ids),
                    std::move(body), std::move(inputs), std::move(op_meta));
}

std::ostream& operator<<(std::ostream& os, const OpMeta& meta) {
  os << "{";
  bool first = true;
  if (meta.stream_index != kUndeterminedStream) {
    if (!first)
      os << ", ";
    os << "stream_index=" << meta.stream_index;
    first = false;
  }
  if (!meta.eager_device.is_undetermined()) {
    if (!first)
      os << ", ";
    os << "eager_device=" << meta.eager_device;
    first = false;
  }
  if (meta.device_group_hierarchy.size() != 0) {
    if (!first)
      os << ", ";
    os << "device_group_hierarchy = " << meta.device_group_hierarchy;
    first = false;
  }
  if (!meta.extra_deps.empty()) {
    if (!first)
      os << ", ";
    os << "extra_deps=" << meta.extra_deps;
    first = false;
  }
  if (!meta.name.empty()) {
    if (!first)
      os << ", ";
    os << "name=" << meta.name;
    first = false;
  }
  os << "}";
  return os;
}

std::ostream& operator<<(std::ostream& os, const Operator& op) {
  if (op.is_defined())
    os << op->name();
  else
    os << "Operator()";
  return os;
}

} // namespace graph
} // namespace hetu
