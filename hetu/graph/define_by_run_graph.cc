#include "hetu/graph/define_by_run_graph.h"
#include "hetu/graph/eager_graph.h"
#include "hetu/graph/executable_graph.h"
#include "hetu/graph/ops/placeholder.h"

namespace hetu {
namespace graph {

Operator& DefineByRunGraph::MakeOpInner(std::shared_ptr<OpInterface> body,
                                        TensorList inputs, OpMeta op_meta) {
  auto handle_inputs = [&](TensorList& inputs_or_in_deps) mutable -> void {
    for (size_t i = 0; i < inputs_or_in_deps.size(); i++) {
      auto& input = inputs_or_in_deps[i];
      auto& input_graph = Graph::GetGraph(input->graph_id());
      if (input_graph.type() == GraphType::EAGER) {
        inputs_or_in_deps[i] = DetachEagerTensor(input);
      } else {
        HT_RUNTIME_ERROR_IF(input_graph.type() != type())
          << "A " << type() << " graph cannot accept tensor " << input->name()
          << " from a " << input_graph.type() << " graph";
        HT_RUNTIME_ERROR_IF(input_graph.id() != id())
          << "Graph " << id() << " cannot accept tensor " << input->name()
          << " since it belongs to another graph " << input_graph.id();
      }
    }
  };

  handle_inputs(inputs);
  handle_inputs(op_meta.extra_deps);
  OpId op_id = FindReusableOp(body, inputs, op_meta);
  if (op_id == std::numeric_limits<OpId>::max()) {
    HT_LOG_TRACE << "Cannot find a reusable op for " << body->type()
                 << ", creating it";
    auto& op =
      MakeAndAddOp(std::move(body), std::move(inputs), std::move(op_meta));
    op_id = op->id();
  } else {
    HT_LOG_TRACE << "Found reusable op: " << _op_indexing[op_id]->name();
    ReuseOp(_op_indexing[op_id]);
  }

  return _op_indexing[op_id];
}

void DefineByRunGraph::ResetVariableDataInner(const Tensor& tensor,
                                              const Initializer& init) {
  auto it = _tensor_to_exec_tensor_mapping.find(tensor->id());
  if (it == _tensor_to_exec_tensor_mapping.end()) {
    // The op is not instantiated yet. Mark an add-on initializer.
    _add_on_inits[tensor->id()] = std::unique_ptr<Initializer>(init.copy());
  } else {
    // The op has been instantiated. Let the executable graph handle it.
    Graph::ResetVariableData(it->second, init);
  }
}

Tensor& DefineByRunGraph::DetachEagerTensor(const Tensor& tensor) {
  auto it = _detached_ops.find(tensor->id());
  if (it != _detached_ops.end())
    return _op_indexing[it->second]->output(0);

  // Question: `get_or_compute` will sync the op but we do not need to sync.
  // Instead, we only need the memory. But if not synced here, then when?
  auto data = tensor->get_or_compute();
  std::shared_ptr<OpInterface> body =
    std::make_shared<PlaceholderOpImpl>(data->meta());
  auto op_meta = OpMeta().set_name(tensor->name());
  OpId op_id = FindReusableOp(body, TensorList(), op_meta);
  if (op_id == std::numeric_limits<OpId>::max()) {
    HT_LOG_TRACE << "Cannot find a reusable op for " << body->type()
                 << ", creating it";
    auto& op = MakeAndAddOp(std::move(body), TensorList(), std::move(op_meta));
    _detached_ops[tensor->id()] = op;
    op_id = op->id();
  } else {
    HT_LOG_TRACE << "Found reusable op: " << _op_indexing[op_id]->name();
    ReuseOp(_op_indexing[op_id]);
  }
  _preserved_data[_op_indexing[op_id]->output(0)->id()] = data;
  return _op_indexing[op_id]->output(0);
}

OpId DefineByRunGraph::FindReusableOp(std::shared_ptr<OpInterface>& body,
                                      const TensorList& inputs,
                                      const OpMeta& op_meta) {
  if (is_variable_op(*body))
    return std::numeric_limits<OpId>::max();

  if (is_placeholder_op(*body)) {
    // TODO: trace reusable placeholder ops?
    for (auto& candidate_op_id : _reusable_ops) {
      auto& candidate_op = _op_indexing[candidate_op_id];
      if (candidate_op->body() != *body)
        continue;
      // A reusable op is found
      return candidate_op->id();
    }
    return std::numeric_limits<OpId>::max();
  }

  // TODO: support P2PRecvOp?
  if (inputs.empty() && op_meta.extra_deps.empty())
    return std::numeric_limits<OpId>::max();

  // Note: It's unlikely, but possible, that there are multiple usable ops.
  // Currently we cannot determine which one is the best to opt in.
  // For simplicity, we choose the first usable op.
  const Tensor& first_in =
    inputs.empty() ? op_meta.extra_deps.front() : inputs.front();
  for (auto& candidate_op_ref : first_in->consumers()) {
    auto& candidate_op = candidate_op_ref.get();
    // Req 1: the candidate op is in the reusable set
    if (_reusable_ops.find(candidate_op->id()) == _reusable_ops.end())
      continue;
    // Req 2: the candidate op has the same body
    if (candidate_op->body() != *body)
      continue;
    // Req 3: all inputs and dependencies are matched and aligned
    bool matched = std::equal(
      inputs.begin(), inputs.end(), candidate_op->inputs().begin(),
      candidate_op->inputs().end(),
      [](const Tensor& x, const Tensor& y) { return x->id() == y->id(); });
    matched = matched &&
      std::equal(op_meta.extra_deps.begin(), op_meta.extra_deps.end(),
                 candidate_op->in_dep_linkers().begin(),
                 candidate_op->in_dep_linkers().end(),
                 [](const Tensor& x, const Tensor& y) {
                   return x->id() == y->id();
                 });
    if (!matched)
      continue;
    // A reusable op is found
    return candidate_op->id();
  }
  return std::numeric_limits<OpId>::max();
}

void DefineByRunGraph::PruneTensor(const Tensor& tensor) {
  auto& producer = _op_indexing[tensor->producer_id()];
  size_t num_outputs = MAX(producer->num_outputs(), 1);
  if (num_outputs != (++_op_to_num_destructed_outputs[tensor->producer_id()]))
    return;

  OpRefDeque prunable_queue;
  std::unordered_set<OpId> prunable_visited;
  std::unordered_set<OpId> removable_op_ids;
  prunable_queue.push_back(std::ref(producer));
  prunable_visited.insert(producer->id());

  // A tensor is prunable if all consumers are prunable
  // and it is no longer referred by the user.
  auto is_tensor_prunable = [&](const Tensor& tensor) -> bool {
    for (const auto& consumer_ref : tensor->consumers()) {
      if (_reusable_ops.find(consumer_ref.get()->id()) == _reusable_ops.end())
        return false;
    }
    return Graph::get_tensor_referrence_count(tensor) == 0;
  };

  // An operator is prunable if all outputs are prunable
  auto is_op_prunable = [&](const Operator& op) -> bool {
    return Operator::all_output_tensors_of(op, is_tensor_prunable);
  };

  while (!prunable_queue.empty()) {
    auto& op = prunable_queue.front().get();
    prunable_queue.pop_front();
    PruneOp(op);

    Operator::for_each_input_tensor(op, [&](Tensor& tensor) {
      if (prunable_visited.find(tensor->producer_id()) ==
          prunable_visited.end()) {
        auto& producer_op = tensor->producer();
        if (is_op_prunable(producer_op)) {
          prunable_queue.push_back(std::ref(producer_op));
          prunable_visited.insert(tensor->producer_id());
        }
      }
    });

    if (_parameter_ops.find(op->id()) != _parameter_ops.end()) {
      removable_op_ids.insert(op->id());
    }
  }

  if (!removable_op_ids.empty()) {
    // TODO: find all descending ops and remove them
    // (and remove the corresponding ops in the executable graph)
  }
}

NDArrayList DefineByRunGraph::Run(const Tensor& loss, const TensorList& fetches,
                                  const FeedDict& feed_dict, const IntSymbolDict& int_symbol_dict, const int num_micro_batches,
                                  const int compute_strategy_id, const int optimize_strategy_id, RunLevel run_level, 
                                  bool save_checkpoint, const double grad_scale,
                                  const RuntimeContext& ctx) {
  TensorList referred_tensors, exec_referred_tensors;
  FeedDict exec_feed_dict;
  std::tie(referred_tensors, exec_referred_tensors, exec_feed_dict) =
    GenerateExecutionTargets(fetches, feed_dict);
  NDArrayList exec_results =
    _exec_graph->Run(loss, exec_referred_tensors, exec_feed_dict, int_symbol_dict, num_micro_batches);

  // For all referred tensors, store their values for future use
  for (size_t i = 0; i < exec_results.size(); i++) {
    _preserved_data[referred_tensors[i]->id()] = exec_results[i];
  }

  // Return the values of fetches
  NDArrayList results;
  results.reserve(fetches.size());
  for (const auto& fetch : fetches) {
    auto it = _preserved_data.find(fetch->id());
    HT_RUNTIME_ERROR_IF(it == _preserved_data.end())
      << "Result of tensor " << fetch << " cannot be found";
    results.push_back(it->second);
  }
  return results;
}

NDArray DefineByRunGraph::GetOrCompute(Tensor& tensor) {
  Run({tensor});
  return _preserved_data[tensor->id()];
}

std::tuple<TensorList, TensorList, FeedDict>
DefineByRunGraph::GenerateExecutionTargets(const TensorList& fetches,
                                           const FeedDict& feed_dict) {
  if (_exec_graph == nullptr) {
    _exec_graph =
      Graph::_make_new_graph<ExecutableGraph>(name() + "_executable");
  }

  TensorList referred_tensors, exec_referred_tensors;
  FeedDict exec_feed_dict;

  auto get_exec_input = [&](const Tensor& input) -> Tensor {
    auto it = _tensor_to_exec_tensor_mapping.find(input->id());
    HT_RUNTIME_ERROR_IF(it == _tensor_to_exec_tensor_mapping.end())
      << "Cannot find the executable version of Tensor " << input;
    return it->second;
  };

  auto put_exec_output = [&](Tensor& tensor, Tensor& exec_tensor) -> void {
    _tensor_to_exec_tensor_mapping[tensor->id()] = exec_tensor;
    auto it = _add_on_inits.find(tensor->id());
    if (it != _add_on_inits.end()) {
      Graph::ResetVariableData(exec_tensor, *it->second);
      _add_on_inits.erase(tensor->id());
    }
  };

  auto generate_exec_target = [&](const Tensor& output) -> void {
    if (feed_dict.find(output->id()) != feed_dict.end()) {
      auto& exec_output = _tensor_to_exec_tensor_mapping[output->id()];
      exec_feed_dict[exec_output->id()] = feed_dict.at(output->id());
    } else if (_preserved_data.find(output->id()) != _preserved_data.end()) {
      auto& exec_output = _tensor_to_exec_tensor_mapping[output->id()];
      exec_feed_dict[exec_output->id()] = {_preserved_data[output->id()]};
    } else {
      auto ref_cnt = Graph::get_tensor_referrence_count(output);
      if (ref_cnt > 0) {
        referred_tensors.push_back(output);
        auto& exec_output = _tensor_to_exec_tensor_mapping[output->id()];
        exec_referred_tensors.push_back(exec_output);
      }
    }
  };

  auto is_op_computed = [&](const Operator& op) -> bool {
    return Operator::all_output_tensors_of(op, [&](const Tensor& tensor) {
      return _preserved_data.find(tensor->id()) != _preserved_data.end();
    });
  };
  OpRefList topo = Graph::TopoSort(fetches, num_ops(), is_op_computed);
  HT_LOG_TRACE << "Generating execution target for topo " << topo;
  for (auto& op_ref : topo) {
    auto& op = op_ref.get();
    if (_op_to_exec_op_mapping.find(op->id()) == _op_to_exec_op_mapping.end()) {
      HT_LOG_TRACE << "Op " << op
                   << " does not have an executable version, creating it";
      TensorList exec_inputs, exec_in_deps;
      std::tie(exec_inputs, exec_in_deps) =
        Operator::transform_each_input_tensor(op, get_exec_input);
      auto& exec_op = Graph::MakeOp(
        op->_body, std::move(exec_inputs),
        OpMeta().set(op->op_meta()).set_extra_deps(std::move(exec_in_deps)),
        *_exec_graph);
      if (_parameter_ops.find(op->id()) != _parameter_ops.end())
        Graph::MarkAsParameter(exec_op);

      Operator::for_each_output_tensor_pair(op, exec_op, put_exec_output);
      _op_to_exec_op_mapping[op->id()] = exec_op;
    }
    Operator::for_each_output_tensor(op, generate_exec_target);
  }

  return {referred_tensors, exec_referred_tensors, exec_feed_dict};
}

} // namespace graph
} // namespace hetu
