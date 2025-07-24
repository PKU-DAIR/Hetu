#pragma once

#include "hetu/common/macros.h"
#include "hetu/core/ndarray.h"
#include "hetu/graph/common.h"
#include "hetu/graph/tensor.h"
#include "hetu/graph/operator.h"
#include <functional>

namespace hetu {
namespace graph {

struct OpHandlerStatus {
  bool need_skip = false;
};

using OpHandler = std::function<OpHandlerStatus(Operator&, Tensor2NDArrayMap&, size_t)>;

enum class SubGraphType : int8_t {
  MODULE = 0,
  CROSS_MODAL_COMM_OP,
  PIPELINE,
  OPTIMIZE_COMPUTE_BRIDGE,
  COMPUTE_OPTIMIZE_BRIDGE,
  TERMINATE,
  NUM_SUBGRAPH_TYPES
};

inline std::ostream& operator<<(std::ostream& os, const SubGraphType& type) {
  switch (type) {
    case SubGraphType::MODULE:
      os << "MODULE";
      break;
    case SubGraphType::CROSS_MODAL_COMM_OP:
      os << "CROSS_MODAL_COMM_OP";
      break;
    case SubGraphType::PIPELINE:
      os << "PIPELINE";
      break;
    case SubGraphType::OPTIMIZE_COMPUTE_BRIDGE:
      os << "OPTIMIZE_COMPUTE_BRIDGE";
      break;
    case SubGraphType::COMPUTE_OPTIMIZE_BRIDGE:
      os << "COMPUTE_OPTIMIZE_BRIDGE";
      break;
    case SubGraphType::TERMINATE:
      os << "TERMINATE";
      break;
    case SubGraphType::NUM_SUBGRAPH_TYPES:
      os << "NUM_SUBGRAPH_TYPES";
      break;
    default:
      os << "UNKNOWN_SUBGRAPH_TYPE";
      break;
  }
  return os;
}

enum class SubGraphOpType : int8_t {
  FORWARD = 0,
  BACKWARD,
  UPDATE,
  UNKNOWN,
  NUM_SUBGRAPH_OP_TYPES
};

class SubGraph {
  private:
    Op2OpMap _ops;
    Op2OpMap _bwd_ops;
    Op2OpMap _update_ops;
    OpRefList _ops_topo;
    OpRefList _bwd_ops_topo;
    OpRefList _update_ops_topo;
    Tensor2IntMap _tensor2degrees;
    Tensor2IntMap _bwd_tensor2degrees;
    Tensor2IntMap _update_tensor2degrees;
    std::unordered_map<std::string, std::shared_ptr<SubGraph>> _subgraphs;
    std::shared_ptr<SubGraph> _parent_graph;
    SubGraphType _subgraph_type;
    std::string _name; // 相对于parent的sub graph的name
    std::string _global_name; // 全局的name大部分时候就是parent的global name加上当前的name
    std::string _module_type;
    int64_t _fwd_time;
    int64_t _bwd_time;
    int64_t _update_time;
    bool _already_profiled;
    bool _already_topo_sorted;

  public:
    SubGraph() {
      _subgraphs = {};
      _ops = {};
      _already_profiled = false;
      _already_topo_sorted = false;
    }

    SubGraph(std::string name) {
      _name = name;
      _subgraphs = {};
      _ops = {};
      _already_profiled = false;
      _already_topo_sorted = false;
    }

    SubGraph(SubGraphType subgraph_type, std::string name, std::string global_name = "", std::string module_type = "") {
      _name = name;
      _subgraph_type = subgraph_type;
      _global_name = global_name;
      _module_type = module_type;
      _subgraphs = {};
      _ops = {};
      _already_profiled = false;
      _already_topo_sorted = false;
    }

    bool already_topo_sorted() {
      return _already_topo_sorted;
    }

    void add_op(Operator& op) {
      HT_ASSERT(!_already_topo_sorted)
        << "cannot add new op after subgraph " << _global_name << " topo sort";
      if (_ops.find(op->id()) == _ops.end()) {
        _ops.emplace(op->id(), op);
      }
    }

    void add_bwd_op(Operator& op) {
      HT_ASSERT(!_already_topo_sorted)
        << "cannot add new op after subgraph " << _global_name << " topo sort";
      if (_bwd_ops.find(op->id()) == _bwd_ops.end()) {
        _bwd_ops.emplace(op->id(), op);
      }
    }

    void add_update_op(Operator& op) {
      HT_ASSERT(!_already_topo_sorted)
        << "cannot add new op after subgraph " << _global_name << " topo sort";
      if (_update_ops.find(op->id()) == _update_ops.end()) {
        _update_ops.emplace(op->id(), op);
      }
    }

    void delete_op(Operator& op) {
      HT_ASSERT(!_already_topo_sorted)
        << "cannot delete op after subgraph " << _global_name << " topo sort";
      auto it = _ops.find(op->id());
      HT_ASSERT(it != _ops.end())
        << "cannot find " << op << " in ops of subgraph " << _global_name;
      _ops.erase(it);
    }

    void delete_bwd_op(Operator& op) {
      HT_ASSERT(!_already_topo_sorted)
        << "cannot delete op after subgraph " << _global_name << " topo sort";
      auto it = _bwd_ops.find(op->id());
      HT_ASSERT(it != _bwd_ops.end())
        << "cannot find " << op << " in bwd_ops of subgraph " << _global_name;
      _bwd_ops.erase(it);
    }

    void delete_update_op(Operator& op) {
      HT_ASSERT(!_already_topo_sorted)
        << "cannot delete op after subgraph " << _global_name << " topo sort";
      auto it = _update_ops.find(op->id());
      HT_ASSERT(it != _update_ops.end())
        << "cannot find " << op << " in update_ops of subgraph " << _global_name;
      _update_ops.erase(it);
    }

    void add_subgraph(std::shared_ptr<SubGraph> subgraph) {
      if (_subgraphs.find(subgraph->global_name()) != _subgraphs.end())
        _subgraphs.emplace(subgraph->global_name(), subgraph);
    }

    std::shared_ptr<SubGraph> parent_graph() {
      return _parent_graph;
    }

    void set_parent_graph(std::shared_ptr<SubGraph> parent_graph) {
      _parent_graph = parent_graph;
    }

    std::string name() const {
      return _name;
    }

    SubGraphType subgraph_type() const {
      return _subgraph_type;
    }

    std::string global_name() const {
      return _global_name;
    }

    std::string module_type() const {
      HT_ASSERT(_subgraph_type == SubGraphType::MODULE)
        << "only MODULE subgraph has module_type"
        << ", but " << _global_name << " subgraph type is " << static_cast<int32_t>(_subgraph_type);
      return _module_type;
    }

    const Op2OpMap& ops() const {
      return _ops;
    }

    const Op2OpMap& bwd_ops() const {
      return _bwd_ops;
    }

    const Op2OpMap& update_ops() const {
      return _update_ops;
    }

    const OpRefList& ops_topo() const {
      return _ops_topo;
    }

    const OpRefList& bwd_ops_topo() const {
      return _bwd_ops_topo;
    }

    const OpRefList& update_ops_topo() const {
      return _update_ops_topo;
    }

    void alloc_concat_memory(Operator& final_concat_op, RuntimeContext& runtime_ctx, std::vector<TensorId>& alloc_concat_tensor_id_list);

    void topo_sort(bool only_local = true);

    void run(Tensor2NDArrayMap& tensor2data, const Tensor2NDArrayMap& preserved_data, RuntimeContext& runtime_ctx,
             size_t micro_batch_id = 0, SubGraphOpType subgraph_op_type = SubGraphOpType::FORWARD,
             bool use_concat_memory_optimization = true, const OpHandler& = {});

    std::vector<std::string> subgraph_info() {
      std::vector<std::string> output = {};
      output.reserve(_subgraphs.size());
      for (auto it = _subgraphs.begin(); it != _subgraphs.end(); ++it) {
        output.push_back("name=" + it->second->name() + 
                         ", type=" + std::to_string(static_cast<int32_t>(it->second->subgraph_type())));
      }
      return output;
    }

    int64_t fwd_time() const {
      return _fwd_time;
    }

    int64_t bwd_time() const {
      return _bwd_time;
    }

    int64_t update_time() const {
      return _update_time;
    }

    int64_t total_time() const {
      return _fwd_time + _bwd_time + _update_time;
    }

    void set_fwd_time(int64_t fwd_time) {
      _fwd_time = fwd_time;
    }

    void set_bwd_time(int64_t bwd_time) {
      _bwd_time = bwd_time;
    }

    void set_update_time(int64_t update_time) {
      _update_time = update_time;
    }

    void profiling_ops(std::unordered_map<OpId, int64_t> op_execute_map, int num_micro_batches = 1) {
      _fwd_time = 0;
      _bwd_time = 0;
      _update_time = 0;
      for (auto it = _ops.begin(); it != _ops.end(); ++it) {
        if (op_execute_map.find(it->second->id()) == op_execute_map.end())
          continue;
        for (int i = 0; i < num_micro_batches; ++i) {
          _fwd_time += it->second->TimeCost(i);
        }
      }
      for (auto it = _bwd_ops.begin(); it != _bwd_ops.end(); ++it) {
        if (op_execute_map.find(it->second->id()) == op_execute_map.end())
          continue;
        for (int i = 0; i < num_micro_batches; ++i) {
          _bwd_time += it->second->TimeCost(i);
        }
      }
      for (auto it = _update_ops.begin(); it != _update_ops.end(); ++it) {
        if (op_execute_map.find(it->second->id()) == op_execute_map.end())
          continue;
        for (int i = 0; i < num_micro_batches; ++i) {
          _update_time += it->second->TimeCost(i);
        }
      }
    }

    std::unordered_map<std::string, std::shared_ptr<SubGraph>> subgraphs() {
      return _subgraphs;
    }

    void profile(std::unordered_map<OpId, int64_t> op_execute_map, int num_micro_batches = 1) {
      if (_already_profiled) {
        return;
      }
      profiling_ops(op_execute_map, num_micro_batches);
      for (auto it = _subgraphs.begin(); it != _subgraphs.end(); ++it) {
        it->second->profile(op_execute_map, num_micro_batches);
        _fwd_time += it->second->fwd_time();
        _bwd_time += it->second->bwd_time();
        _update_time += it->second->update_time();
      }
      _already_profiled = true;
    }

    void profile_reset() {
      _fwd_time = 0;
      _bwd_time = 0;
      _update_time = 0;
      _already_profiled = false;
    }
};

std::ostream& operator<<(std::ostream&, SubGraph&);

} // namespace graph
} // namespace hetu