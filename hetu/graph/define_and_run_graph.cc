#include "hetu/graph/define_and_run_graph.h"
#include "hetu/graph/executable_graph.h"

namespace hetu {
namespace graph {

Operator& DefineAndRunGraph::MakeOpInner(std::shared_ptr<OpInterface> body,
                                         TensorList inputs, OpMeta op_meta) {
  _check_all_inputs_in_graph(inputs, op_meta.extra_deps);
  return MakeAndAddOp(std::move(body), std::move(inputs), std::move(op_meta));
}

void DefineAndRunGraph::ResetVariableDataInner(const Tensor& tensor,
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

void DefineAndRunGraph::Instantiate() {
  if (_exec_graph == nullptr)
    _exec_graph =
      Graph::_make_new_graph<ExecutableGraph>(name() + "_executable");

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

  OpRefList topo = topo_order();
  HT_LOG_TRACE << "Instantiating a " << type() << " graph with topo " << topo;
  for (auto& op_ref : topo) {
    auto& op = op_ref.get();
    if (_op_to_exec_op_mapping.find(op->id()) != _op_to_exec_op_mapping.end())
      continue;
    HT_LOG_TRACE << "Creating an executable version of op " << op;

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
}

NDArrayList DefineAndRunGraph::Run(const TensorList& fetches,
                                   const FeedDict& feed_dict) {
  bool has_uninstantiated_ops =
    std::any_of(fetches.begin(), fetches.end(), [&](const Tensor& fetch) {
      return _op_to_exec_op_mapping.find(fetch->producer_id()) ==
        _op_to_exec_op_mapping.end();
    });
  if (has_uninstantiated_ops)
    Instantiate();
  TensorList exec_fetches;
  exec_fetches.reserve(fetches.size());
  for (const auto& fetch : fetches) {
    exec_fetches.push_back(_tensor_to_exec_tensor_mapping[fetch->id()]);
  }
  FeedDict exec_feed_dict;
  exec_feed_dict.reserve(feed_dict.size());
  for (const auto& kv : feed_dict)
    exec_feed_dict[_tensor_to_exec_tensor_mapping[kv.first]->id()] = kv.second;
  return _exec_graph->Run(exec_fetches, exec_feed_dict);
}

} // namespace graph
} // namespace hetu
