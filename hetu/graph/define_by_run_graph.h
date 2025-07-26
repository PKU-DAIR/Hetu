#pragma once

#include "hetu/graph/graph.h"
#include "hetu/graph/executable_graph.h"

namespace hetu {
namespace graph {

class DefineByRunGraph : public Graph {
 protected:
  friend class Graph;
  friend class Tensor;

  DefineByRunGraph(GraphName name, size_t init_capacity)
  : Graph(name, init_capacity) {
    _detached_ops.reserve(init_capacity);
    _reusable_ops.reserve(init_capacity);
    _op_to_exec_op_mapping.reserve(init_capacity);
    _tensor_to_exec_tensor_mapping.reserve(init_capacity);
    _op_to_num_destructed_outputs.reserve(init_capacity);
  }

 public:
  DefineByRunGraph(const constructor_access_key&, GraphName name,
                   size_t init_capacity = DEFAULT_GRAPH_INITIAL_CAPACITY)
  : DefineByRunGraph(name, init_capacity) {}

  NDArrayList Run(const TensorList& fetches, 
                  const FeedDict& feed_dict = {}) {}

  NDArrayList Run(const Tensor& loss, const TensorList& fetches, 
                  const FeedDict& feed_dict = {}, const IntSymbolDict& int_symbol_dict = {}, const int num_micro_batches = 1,
                  const int compute_strategy_id = 0, const int optimize_strategy_id = 0, RunLevel run_level = RunLevel::UPDATE, 
                  bool save_checkpoint = false, const double grad_scale = 1, const RuntimeContext& ctx = RuntimeContext());

  GraphType type() const {
    return GraphType::DEFINE_BY_RUN;
  }

 protected:
  Operator& MakeOpInner(std::shared_ptr<OpInterface> body, TensorList inputs,
                        OpMeta op_meta);

  void ResetVariableDataInner(const Tensor& tensor,
                              const Initializer& init) override;

  NDArray GetOrCompute(Tensor& tensor);

  NDArray GetOrCompute(const Tensor& loss, Tensor& tensor, 
                       const int num_micro_batches = 1) {}

  std::tuple<TensorList, TensorList, FeedDict>
  GenerateExecutionTargets(const TensorList& fetches,
                           const FeedDict& feed_dict);

  void RemoveOp(Operator& op) override {
    _detached_ops.erase(op->id());
    _reusable_ops.erase(op->id());
    _op_to_exec_op_mapping.erase(op->id());
    Operator::for_each_output_tensor(op, [&](Tensor& tensor) {
      _tensor_to_exec_tensor_mapping.erase(tensor->id());
    });
    _op_to_num_destructed_outputs.erase(op->id());
    Graph::RemoveOp(op);
  }

  void PruneOp(Operator& op) {
    _reusable_ops.insert(op->id());
    _source_ops.erase(op->id());
    _sink_ops.erase(op->id());
    Operator::for_each_output_tensor(op, [&](Tensor& tensor) {
      _detached_ops.erase(tensor->id());
      _preserved_data.erase(tensor->id());
    });
    Operator::for_each_input_tensor(op, [&](Tensor& tensor) {
      if ((--_op_out_degrees[tensor->producer_id()]) == 0) {
        _sink_ops.insert(tensor->producer_id());
      }
    });
  }

  void ReuseOp(Operator& op) {
    _reusable_ops.erase(op->id());
    _op_out_degrees[op->id()] = 0;
    _sink_ops.insert(op->id());
    if (op->in_degrees() == 0) {
      _source_ops.insert(op->id());
    } else {
      Operator::for_each_input_tensor(op, [&](Tensor& tensor) {
        _op_out_degrees[tensor->producer_id()]++;
        _sink_ops.erase(tensor->producer_id());
      });
    }
    _op_to_num_destructed_outputs[op->id()] = 0;
  }

  void PruneTensor(const Tensor& tensor);

  OpId FindReusableOp(std::shared_ptr<OpInterface>& body,
                      const TensorList& inputs, const OpMeta& op_meta);

  Tensor& DetachEagerTensor(const Tensor& tensor);

  void Clear() override {
    _detached_ops.clear();
    _reusable_ops.clear();
    _op_to_exec_op_mapping.clear();
    _tensor_to_exec_tensor_mapping.clear();
    _op_to_num_destructed_outputs.clear();
    Graph::Clear();
  }

  std::unordered_map<TensorId, OpId> _detached_ops;
  std::unordered_set<OpId> _reusable_ops;
  std::shared_ptr<ExecutableGraph> _exec_graph;
  Op2OpMap _op_to_exec_op_mapping;
  Tensor2TensorMap _tensor_to_exec_tensor_mapping;
  std::unordered_map<TensorId, std::unique_ptr<Initializer>> _add_on_inits;
  std::unordered_map<OpId, size_t> _op_to_num_destructed_outputs;
};

} // namespace graph
} // namespace hetu
