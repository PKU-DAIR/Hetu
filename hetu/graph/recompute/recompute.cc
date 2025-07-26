#include "hetu/graph/recompute/recompute.h"
#include "hetu/impl/communication/comm_group.h"

namespace hetu {
namespace graph {

std::stack<std::vector<std::vector<bool>>> Recompute::_multi_recompute_stack{{std::vector<std::vector<bool>>{{false}}}};

// helper functions
namespace {
  inline int get_out_idx_of(const Operator& op, const Tensor& tensor) {
    auto num_output = op->num_outputs();
    for (int out_pos = 0; out_pos < num_output; out_pos++) {
      if (op->output(out_pos)->id() == tensor->id()) {
        return out_pos;
      }
    }
    return -1;
  }
} // namespace

// 尽量确保op在local device上有之后再调用
bool Recompute::IsNoRecomputedOp(Operator& op) {
  if (is_comm_op(op)) {
    auto& comm_op = op;
    auto& comm_op_impl = reinterpret_cast<CommOpImpl&>(comm_op->body());
    uint64_t comm_type = comm_op_impl.get_comm_type(comm_op, hetu::impl::comm::GetLocalDevice());
    return comm_type == PEER_TO_PEER_RECV_OP ||
           comm_type == PEER_TO_PEER_SEND_OP ||
           comm_type == BATCHED_ISEND_IRECV_OP;
  } else {
    return is_variable_op(op) ||
           is_placeholder_op(op);
  }
}

void Recompute::GetMaxRecomputeSubGraph(Op2OpRefMap& recompute_subgraph, bool get_inputs, bool get_outputs) {
  auto& local_device = hetu::impl::comm::GetLocalDevice();
  OpRefQueue to_visit;
  for (auto& op_opref : recompute_subgraph) {
    to_visit.push(op_opref.second);
  }
  recompute_subgraph.clear();

  while (!to_visit.empty()) {
    auto& op_ref = to_visit.front();
    to_visit.pop();
    if (recompute_subgraph.find(op_ref.get()->id()) != recompute_subgraph.end()) {
      continue;
    }
    recompute_subgraph.insert({op_ref.get()->id(), op_ref});
    // add recomputed inputs to recompute subgraph
    if (get_inputs) {
      auto& op_inputs = op_ref.get()->inputs();
      for (auto& input : op_inputs) {
        auto& op = input->producer();
        // HT_LOG_TRACE << op_ref.get() << " input op " << op;
        if (op->placement_group().contains(local_device) && op->op_meta().get_recompute(op->graph().COMPUTE_STRATEGY_ID, op->suggested_hetero_id())
            && !IsNoRecomputedOp(op) && recompute_subgraph.find(op->id()) == recompute_subgraph.end()) {
          to_visit.push(std::ref(op));
        }
      }
    }
    // add recomputed outputs to recompute subgraph
    if (get_outputs) {
      auto& op_outputs = op_ref.get()->outputs();
      for (auto& output : op_outputs) {
        auto& out_consumers = output->consumers();
        for (auto& op_ref : out_consumers) {
          auto& op = op_ref.get();
          // HT_LOG_TRACE << "and output op " << op;
          if (op->placement_group().contains(local_device) && op->op_meta().get_recompute(op->graph().COMPUTE_STRATEGY_ID, op->suggested_hetero_id()) 
              && !IsNoRecomputedOp(op) && recompute_subgraph.find(op->id()) == recompute_subgraph.end()) {
            to_visit.push(op_ref);
          }
        }
      }
    }
  }
}

bool Recompute::HasFilterOpInPath(const Operator& op, std::function<bool(const Operator &)> filter_fn,
                                  std::unordered_map<OpId, bool>& has_filter_op_map) {
  if (has_filter_op_map.find(op->id()) != has_filter_op_map.end()) {
    return has_filter_op_map[op->id()];
  }
  if (filter_fn(op)) {
    has_filter_op_map.insert({op->id(), true});
    return true;
  }
  if (op->is_bw_op()) {
    auto& inputs = op->inputs();
    for (auto& input : inputs) {
      if (HasFilterOpInPath(input->producer(), filter_fn, has_filter_op_map)) {
        has_filter_op_map.insert({op->id(), true});
        return true;
      }
    }
  }
  has_filter_op_map.insert({op->id(), false});
  return false;
}

Operator& Recompute::DuplicateRecomputedOp(const Operator& origin_op, const Op2OpRefMap& filtered_recomputed_ops,
                                           const TensorList& first_mapped_grad_inputs, Op2OpMap& origin_to_recomputed_map,
                                           ExecutableGraph& cur_exec_graph) {
  auto& local_device = hetu::impl::comm::GetLocalDevice();
  auto iter = origin_to_recomputed_map.find(origin_op->id());
  if (iter != origin_to_recomputed_map.end()) {
    return iter->second;
  }
  TensorList new_inputs;
  bool has_recomputed_input = false;
  auto& origin_inputs = origin_op->inputs();
  for (auto& input : origin_inputs) {
    auto& input_op = input->producer();
    if (filtered_recomputed_ops.find(input_op->id()) == filtered_recomputed_ops.end()) {
      new_inputs.push_back(input);
    } else {
      has_recomputed_input = true;
      // find the output position of input in input_op
      auto out_idx = get_out_idx_of(input_op, input);
      HT_ASSERT(out_idx != -1)
        << "Cannot find input " << input << " in the outputs of recomputed op " << input_op;
      auto& new_input_op = DuplicateRecomputedOp(input_op, filtered_recomputed_ops,
                                              first_mapped_grad_inputs, origin_to_recomputed_map, cur_exec_graph);
      new_inputs.push_back(new_input_op->output(out_idx));
    }
  }
  auto new_op_meta = OpMeta().set(origin_op->op_meta())
                             .set_multi_recompute({{false}})
                             .set_name(origin_op->name() + "_recompute")
                             .set_is_deduce_states(false)
                             .set_origin_op_id(origin_op->id())
                             .set_is_recompute(true);
  // add the execution dependency
  // 即对于输入op全都不在recompute subgraph中的
  // 需要添加一条之前grad op到该op的依赖边
  if (!has_recomputed_input) {
    new_op_meta.set_extra_deps(first_mapped_grad_inputs);
  }
  // HT_LOG_TRACE << "making a duplicate op for " << origin_op;
  // 注意MakeCommOp在InferMeta时不得不特殊处理
  // 需要从外面把CUR_HETERO_ID传进去
  if (is_comm_op(origin_op) || is_parallel_attn_op(origin_op)) {
    HT_ASSERT(origin_op->output(0)->placement_group_union().has(local_device))
      << "something wrong, new duplicated op should all be local";
    origin_op->graph().CUR_HETERO_ID = origin_op->output(0)->placement_group_union().get_index(local_device);
  }
  auto& new_op = Graph::MakeOp(origin_op->_body, std::move(new_inputs),
                               std::move(new_op_meta), cur_exec_graph);
  if (is_comm_op(origin_op) || is_parallel_attn_op(origin_op)) {
    origin_op->graph().CUR_HETERO_ID = 0;
  }
  if (origin_op->graph().GetSubGraph(const_cast<Operator&>(origin_op)) != nullptr) {
    origin_op->graph().AddOpToSubGraph(new_op, origin_op->graph().GetSubGraph(const_cast<Operator&>(origin_op))->global_name(), 
                                       origin_op->graph().GetSubGraphOpType(const_cast<Operator&>(origin_op)));
  }
  // HT_LOG_TRACE << "make op done, the output 0 shape is " << new_op->output(0)->shape();
  for (size_t i = 0; i < new_op->num_outputs(); i++) {
    const auto& new_output = new_op->output(i);
    const auto& old_output = origin_op->output(i);
    if (old_output->symbolic()) {
      new_output->copy_symbolic_shape(old_output->symbolic_shape());
      if (is_SyShape_leaf(new_output->symbolic_shape())) {
        new_output->set_symbolic_shape(new_output->shape());
      }
    }
    cur_exec_graph.RecordExecTensor(new_output);
  }
  if (origin_op->num_outputs() > 0) {
    new_op->MapToParallelDevices(origin_op->output(0)->placement_group_union());
  } else {
    new_op->MapToParallelDevices(origin_op->out_dep_linker()->placement_group_union());
  }
  new_op->Instantiate(origin_op->instantiation_ctx().placement, 
                      origin_op->instantiation_ctx().stream_index);
  for (auto i = 0; i < origin_op->num_outputs(); i++) {
    // new_op->output(i)->set_ds_hierarchy(origin_op->output(i)->ds_hierarchy());
    new_op->output(i)->set_cur_ds_union(origin_op->output(i)->cur_ds_union());
  }
  origin_to_recomputed_map.insert({origin_op->id(), new_op});
  return origin_to_recomputed_map[origin_op->id()];
}

void Recompute::InsertRecomputedOps(const OpRefList& topo_order) {
  auto& local_device = hetu::impl::comm::GetLocalDevice();
  // Find candidate recomputed ops.
  auto has_grad_consumer = [](const Tensor& tensor) {
    return Tensor::any_consumer_of(tensor, [](const OpRef& op_ref) -> bool {
      return op_ref.get()->is_bw_op();
    });
  };
  HT_LOG_TRACE << "[Recompute] find candidate recomputed ops begin...";
  OpRefList candidate_recomputed_ops;
  for (auto& op_ref : topo_order) {
    auto& op = op_ref.get();
    HT_LOG_TRACE << "[Recompute] " << op << " recompute is " << op->op_meta().get_recompute(op->graph().COMPUTE_STRATEGY_ID, op->suggested_hetero_id());
    if (!op->placement_group().contains(local_device) 
        || !op->op_meta().get_recompute(op->graph().COMPUTE_STRATEGY_ID, op->suggested_hetero_id())
        || IsNoRecomputedOp(op)) {
      continue;
    }
    // No recomputation if there is no grad op in the outputs
    if (Operator::any_output_tensor_of(op, has_grad_consumer)) {
      candidate_recomputed_ops.push_back(op_ref);
    }
  }
  HT_LOG_TRACE << "[Recompute] Found " << candidate_recomputed_ops.size()
               << " candidate recomputed ops: " << candidate_recomputed_ops;
  // Iterate over all candidate recomputed ops and
  // construct recompute subgraph with continuous recomputed ops.
  OpIdSet visited_ops;
  for (auto& candidate_recomputed_op_ref : candidate_recomputed_ops) {
    auto& candidate_recomputed_op = candidate_recomputed_op_ref.get();
    if (visited_ops.find(candidate_recomputed_op->id()) != visited_ops.end()) {
      continue;
    }
    // Get max continuous recompute subgraph first.
    HT_LOG_TRACE << "[Recompute] get max recomputed subgraph begin...";
    Op2OpRefMap max_recompute_subgraph = {{candidate_recomputed_op->id(), candidate_recomputed_op_ref}};
    GetMaxRecomputeSubGraph(max_recompute_subgraph, true, true);
    OpIdList max_recompute_subgraph_ids;
    for (auto& op : max_recompute_subgraph) {
      max_recompute_subgraph_ids.push_back(op.first);
    }
    visited_ops.insert(max_recompute_subgraph_ids.begin(), max_recompute_subgraph_ids.end());
    // Filter recomputed ops that directly output to grad ops.
    HT_LOG_TRACE << "[Recompute] filter recomputed ops that directly output to grad ops begin...";
    Op2OpRefMap filtered_recomputed_ops; // 记录recompute subgraph中下一个算子是反向传播算子的算子
    Op2OpRefMap mapped_grad_ops; // 记录recompute subgraph的那些邻接的反向传播算子（不在recompute subgraph中）
    for (auto& op_opref : max_recompute_subgraph) {
      bool inserted = false;
      auto op_id = op_opref.first;
      auto& op_ref = op_opref.second;
      auto& op_outputs = op_ref.get()->outputs();
      for (auto& output : op_outputs) {
        auto& out_consumers = output->consumers();
        // 只要有一个consumer是grad就要算在里头
        for (auto& out_op_ref : out_consumers) {
          if (!out_op_ref.get()->is_bw_op()) {
            continue;
          }
          mapped_grad_ops.insert({out_op_ref.get()->id(), out_op_ref});
          if (!inserted) {
            inserted = true;
            if (filtered_recomputed_ops.find(op_ref.get()->id()) != filtered_recomputed_ops.end()) {
              continue;
            }
            filtered_recomputed_ops.insert({op_id, op_ref});
          }
        }
      }
    }
    // Get inputs of filtered recomputed ops which eventually output to grad ops.
    HT_LOG_TRACE << "[Recompute] get inputs of filtered recomputed ops which eventually output to grad ops begin...";
    GetMaxRecomputeSubGraph(filtered_recomputed_ops, true, false);
    HT_LOG_TRACE << "[Recompute] found continuous recompute subgraph: " << filtered_recomputed_ops;
    // Find inputs of the first mapped grad op in the topo order, these
    // inputs will be the execution dependencies of the recompute subgraph.
    HT_LOG_TRACE << "[Recompute] find inputs of the first mapped grad op in the topo order begin...";
    TensorList first_mapped_grad_inputs; // 记录在进行recompute之前的邻接的grad op
    std::unordered_map<OpId, bool> has_filter_op_map; // 记录依赖于recompute subgraph的recompute op和grad op
    auto filter_fn = [&filtered_recomputed_ops, &mapped_grad_ops](const Operator& op) -> bool {
      return filtered_recomputed_ops.find(op->id()) != filtered_recomputed_ops.end() ||
             mapped_grad_ops.find(op->id()) != mapped_grad_ops.end();
    };
    for (auto& op_ref : topo_order) {
      if (mapped_grad_ops.find(op_ref.get()->id()) == mapped_grad_ops.end()) {
        continue;
      }
      auto& op_inputs = op_ref.get()->inputs();
      for (auto& input : op_inputs) {
        auto& op = input->producer();
        if (!op->is_bw_op()) {
          continue;
        }
        // 目前获得了一个反向传播图中某一个依赖recompute subgraph（即位于mapped_grad_ops）的grad op的前一个input grad op
        // As execution dependency, `input` should not rely on
        // any recomputed op that will be executed later and
        // any mapped grad op that relies on the recompute subgraph.
        if (HasFilterOpInPath(op, filter_fn, has_filter_op_map)) {
          continue;
        }
        first_mapped_grad_inputs.push_back(input);
      }
      if (!first_mapped_grad_inputs.empty()) {
        break;
      }
    }
    // Duplicate recomputed ops
    HT_LOG_TRACE << "[Recompute] duplicate recomputed ops begin...";
    Op2OpMap origin_to_recomputed_map;
    // recompute pass is executed after instantiating exec graph,
    // so we ensure current graph is an exec graph
    auto& cur_exec_graph = dynamic_cast<ExecutableGraph&>(Graph::GetGraph(Graph::cur_graph_ctx()));
    for (auto& op_opref : mapped_grad_ops) {
      auto& mapped_grad_op = op_opref.second.get();
      auto num_inputs = mapped_grad_op->num_inputs();
      for (int i = 0; i < num_inputs; i++) {
        auto& input = mapped_grad_op->input(i);
        auto& input_op = input->producer();
        if (filtered_recomputed_ops.find(input_op->id()) != filtered_recomputed_ops.end()) {
          auto out_idx = get_out_idx_of(input_op, input);
          HT_ASSERT(out_idx != -1)
            << "Cannot find input " << input << " in the outputs of recomputed op " << input_op;
          auto& recomputed_op = DuplicateRecomputedOp(input_op, filtered_recomputed_ops,
                                                      first_mapped_grad_inputs, origin_to_recomputed_map, cur_exec_graph);
          HT_LOG_TRACE << "[Recompute] replacing mapped_grad_op " << mapped_grad_op << " input[" << i
                      << "] with recomputed_op " << recomputed_op << " output[" << out_idx << "]...";
          Graph::ReplaceInput(mapped_grad_op, i, recomputed_op->output(out_idx));
        }
      }
    }
  }
}

} // namespace graph
} // namespace hetu