#include "hetu/graph/headers.h"
#include "hetu/graph/define_by_run_graph.h"
#include "hetu/graph/define_and_run_graph.h"
#include "hetu/graph/eager_graph.h"
#include "hetu/graph/executable_graph.h"
#include "hetu/graph/ops/Contiguous.h"
#include "hetu/graph/ops/ones_like.h"
#include "hetu/graph/ops/sum.h"
#include "hetu/graph/ops/Contiguous.h"
#include "hetu/impl/communication/comm_group.h"
#include <thread>

namespace hetu {
namespace graph {

std::once_flag Graph::_init_flag;
std::vector<std::shared_ptr<Graph>> Graph::_global_graphs;
std::unordered_map<GraphName, std::shared_ptr<Graph>> Graph::_name_to_graphs;
std::shared_ptr<Graph> Graph::_default_eager_graph;
std::shared_ptr<Graph> Graph::_default_define_by_run_graph;
std::shared_ptr<Graph> Graph::_default_define_and_run_graph;
thread_local std::stack<GraphId> Graph::_cur_graph_ctx;

GraphId Graph::_next_graph_id() {
  static std::atomic<GraphId> _global_graph_id{0};
  return _global_graph_id++;
}

void Graph::Init() {
  // exit handler
  auto status = std::atexit([]() {
    HT_LOG_DEBUG << "Clearing and destructing all graphs...";
    Graph::_name_to_graphs.clear();
    Graph::_default_eager_graph = nullptr;
    Graph::_default_define_by_run_graph = nullptr;
    Graph::_default_define_and_run_graph = nullptr;
    for (auto& graph : Graph::_global_graphs) {
      if (graph == nullptr)
        continue;
      graph->Clear();
    }
    Graph::_global_graphs.clear();
    HT_LOG_DEBUG << "Destructed all graphs";
  });
  HT_ASSERT(status == 0)
    << "Failed to register the exit function for memory pools.";
  auto concurrency = std::thread::hardware_concurrency();
  Graph::_global_graphs.reserve(MIN(concurrency, 16) * 2);
  Graph::_name_to_graphs.reserve(MIN(concurrency, 16) * 2);
  Graph::_default_eager_graph =
    Graph::_make_new_graph<EagerGraph>("default_eager");
  Graph::_default_define_by_run_graph =
    Graph::_make_new_graph<DefineByRunGraph>("default_define_by_run");
  Graph::_default_define_and_run_graph =
    Graph::_make_new_graph<DefineAndRunGraph>("default_define_and_run");
}

Operator& Graph::MakeOp(std::shared_ptr<OpInterface> body, TensorList inputs,
                        OpMeta op_meta) {
  Graph::InitOnce();
  if (inputs.empty() && op_meta.extra_deps.empty()) {
    HT_VALUE_ERROR_IF(Graph::_cur_graph_ctx.empty())
      << "The target graph must be explicitly passed or enqueued to ctx "
      << "when making a new op with zero inputs";
    HT_LOG_TRACE << "Make variable op on a " << Graph::GetGraph(Graph::_cur_graph_ctx.top()).type() << " graph";
    return MakeOp(std::move(body), std::move(inputs), std::move(op_meta),
                  Graph::GetGraph(Graph::_cur_graph_ctx.top()));
  } else {
    GraphId target_graph_id = std::numeric_limits<GraphId>::max();
    bool input_graph_changed = false;
    auto find_target_graph = [&](const Tensor& input) mutable {
      auto& input_graph = Graph::GetGraph(input->graph_id());
      if (target_graph_id == std::numeric_limits<GraphId>::max()) {
        target_graph_id = input->graph_id();
      } else if (target_graph_id != input->graph_id()) {
        input_graph_changed = true;
        if (input_graph.type() == GraphType::DEFINE_BY_RUN) {
          target_graph_id = input->graph_id();
        }
      }
      HT_VALUE_ERROR_IF(input_graph_changed &&
                        input_graph.type() != GraphType::EAGER &&
                        input_graph.type() != GraphType::DEFINE_BY_RUN && 
                        input_graph.type() != GraphType::DEFINE_AND_RUN)
        << "The target graph must be explicitly passed "
        << "when making new op to a " << input_graph.type() << " graph";
    };
    for (auto& input : inputs)
      find_target_graph(input);
    for (auto& in_dep : op_meta.extra_deps)
      find_target_graph(in_dep);
    return MakeOp(std::move(body), std::move(inputs), std::move(op_meta),
                  Graph::GetGraph(target_graph_id));
  }
}

Operator& Graph::MakeOp(std::shared_ptr<OpInterface> body, TensorList inputs,
                        OpMeta op_meta, Graph& graph) {
  Graph::InitOnce();
  Operator& op = graph.MakeOpInner(std::move(body), std::move(inputs),
                                   std::move(op_meta));
  // workaround: 暂时不把pipeline插入的comm op放到任何subgraph中
  if (op->name().find("pipeline_layer") == std::string::npos) { 
    if (graph.get_cur_subgraph_name() != "") {
      graph.AddOpToSubGraph(op, graph.get_cur_subgraph_name());
    }
    if (is_optimizer_update_op(op)) {
      std::shared_ptr<SubGraph> subgraph = graph.GetSubGraph(op->input(0)->producer());
      if (subgraph != nullptr) {
        graph.AddOpToSubGraph(op, subgraph->global_graph_name(), SubGraphType::UPDATE);
      }
    }
  }
  return op;
}

TensorList Graph::Gradients(const TensorList& ys, const TensorList& xs,
                            const TensorList& grad_ys, int32_t num_ops_hint) {
  for (const auto& y : ys) {
    HT_VALUE_ERROR_IF(!y.is_defined()) << "Passed an undefined tensor";
    HT_VALUE_ERROR_IF(y->is_out_dep_linker())
      << "Cannot compute the gradient of " << y << " with operator "
      << y->producer()->type();
  }

  // Fill gradients
  TensorList filled_grads;
  filled_grads.reserve(ys.size());
  if (grad_ys.empty()) {
    for (const auto& y : ys) {
      // TODO: check whether requires grad
      filled_grads.emplace_back(MakeOnesLikeOp(y));
      filled_grads.back()->set_is_grad(true);
      filled_grads.back()->producer()->set_fw_op_id(y->producer()->id());
    }
  } else {
    HT_VALUE_ERROR_IF(ys.size() != grad_ys.size())
      << "Provided " << grad_ys.size() << " gradients for " << ys.size()
      << " tensors";
    for (size_t i = 0; i < ys.size(); i++) {
      if (!grad_ys[i].is_defined()) {
        filled_grads.emplace_back(MakeOnesLikeOp(ys[i]));
      } else {
        filled_grads.push_back(grad_ys[i]);
      }
      filled_grads.back()->set_is_grad(true);
      filled_grads.back()->producer()->set_fw_op_id(ys[i]->producer()->id());      
    }
  }

  Tensor2TensorListMap tensor_to_grads;
  Tensor2TensorMap tensor_to_reduced_grad;
  for (size_t i = 0; i < ys.size(); i++) {
    auto it = tensor_to_grads.find(ys[i]->id());
    if (it == tensor_to_grads.end())
      tensor_to_grads[ys[i]->id()] = {filled_grads[i]};
    else
      it->second.push_back(filled_grads[i]);
  }

  auto reduce_grad = [](const TensorList& unreduced_grads) -> Tensor {
    TensorList filtered;
    filtered.reserve(unreduced_grads.size());
    for (const auto& grad : unreduced_grads)
      if (grad.is_defined())
        filtered.push_back(grad);
    if (filtered.empty()) {
      return Tensor();
    } else if (filtered.size() == 1) {
      return filtered.front();
    } else {
      // Question: How to set op_meta properly?
      // if grad in filtered are all allreduce/reduce-scatter
      // should use last bw grad here, correspond to first use in fw, 
      // if the tensor was shared in different pp stages
      OpId fw_op_id = filtered.back()->producer()->fw_op_id();
      // comm op的multi ds中只要有一组满足reduce通信的条件, 
      // 就要走先sum再comm的路径
      auto& graph = filtered[0]->graph();
      DistributedStatesHierarchy dst_ds_hierarchy;
      bool is_need_sum_before_reduce = false;
      for (size_t cur_strategy_id = 0; cur_strategy_id < graph.NUM_STRATEGY; cur_strategy_id++) {
        graph.CUR_STRATEGY_ID = cur_strategy_id;
        size_t all_allreduce_num = 0;
        size_t reduce_scatter_num = 0;
        auto cur_dst_ds_union = DistributedStatesUnion();
        std::unordered_map<int, Tensor> patch_comm_grad_list;
        // 24.12.20 [Workaround] for LoRA corner case:
        // For grads in different branches, they may be dup or partial state.
        // The goal of both `sum -> reduce` and `reduce -> sum` is to merge branches into dup state.
        // Thus, if there exists dup branches, dup branches should have the same dup state,
        // and partial branches should turn into dup states by inserting CommOps.
        // If there is no dup branch, we will apply `sum -> reduce`.
        for (int i = 0; i < filtered.size(); i++) {
          auto& grad = filtered.at(i);
          if (is_comm_op(grad->producer())) {
            auto& comm_op_impl = dynamic_cast<CommOpImpl&>(grad->producer()->body());
            auto src_ds_union = grad->producer()->input(0)->cur_ds_union();
            auto dst_ds_union = comm_op_impl.get_dst_ds_union(grad->producer());
            bool is_homo = (src_ds_union.hetero_dim() == NULL_HETERO_DIM && dst_ds_union.hetero_dim() == NULL_HETERO_DIM);
            if (!is_homo) {
              HT_ASSERT(src_ds_union.size() == dst_ds_union.size())
                << "Currently only support equal ds union size";
            }
            if ((is_homo && src_ds_union.get(0).check_allreduce(dst_ds_union.get(0)))
                || (!is_homo && src_ds_union.hetero_dim() == -2 && dst_ds_union.hetero_dim() == -1)) {
              all_allreduce_num += 1;
              if (cur_dst_ds_union.is_empty()) {
                cur_dst_ds_union = dst_ds_union;
              } else {
                HT_ASSERT(cur_dst_ds_union.check_equal(dst_ds_union));
              }
            }
            if ((is_homo && src_ds_union.get(0).check_reducescatter(dst_ds_union.get(0)))
                || (!is_homo && src_ds_union.hetero_dim() == -2 && dst_ds_union.hetero_dim() == 0)) {
              reduce_scatter_num += 1;
              if (cur_dst_ds_union.is_empty()) {
                cur_dst_ds_union = dst_ds_union;
              } else {
                HT_ASSERT(cur_dst_ds_union.check_equal(dst_ds_union));
              }
            }
          } else {
            auto src_ds_union = grad->cur_ds_union();
            bool is_homo = (src_ds_union.hetero_dim() == NULL_HETERO_DIM);
            if ((is_homo && src_ds_union.get(0).states(-2) > 1)
                || (!is_homo && src_ds_union.hetero_dim() == -2)) {
              if ((is_homo && src_ds_union.get(0).check_allreduce(cur_dst_ds_union.get(0)))
                  || (!is_homo && src_ds_union.hetero_dim() == -2 && cur_dst_ds_union.hetero_dim() == -1)) {
                all_allreduce_num += 1;
              }
              if ((is_homo && src_ds_union.get(0).check_reducescatter(cur_dst_ds_union.get(0)))
                  || (!is_homo && src_ds_union.hetero_dim() == -2 && cur_dst_ds_union.hetero_dim() == 0)) {
                reduce_scatter_num += 1;
              }
              patch_comm_grad_list[i] = grad;
            } else {
              if (cur_dst_ds_union.is_empty()) {
                cur_dst_ds_union = src_ds_union;
              } else {
                HT_ASSERT(cur_dst_ds_union.check_equal(src_ds_union));
              }
            }
          }
        }
        if (!cur_dst_ds_union.is_empty()) {
          DistributedStatesHierarchy ds_hierarchy;
          ds_hierarchy.add(cur_dst_ds_union);
          for (auto& [i, grad] : patch_comm_grad_list) {
            Tensor comm_grad = MakeCommOp(grad, ds_hierarchy, OpMeta().set_name("workaround_patch_comm_of_" + grad->name()));
            filtered[i] = std::move(comm_grad);
          }
        }
        HT_ASSERT ((all_allreduce_num == filtered.size() || all_allreduce_num == 0)
                    && (reduce_scatter_num == filtered.size() || reduce_scatter_num == 0))
          << "Not support all_allreduce_num = " << all_allreduce_num
          << ", and reduce_scatter_num = " << reduce_scatter_num
          << ", considering " << filtered 
          << ", please check strategy " << graph.CUR_STRATEGY_ID;
        dst_ds_hierarchy.add(filtered[0]->cur_ds_union());
        if (!(all_allreduce_num == 0 && reduce_scatter_num == 0)) {
          is_need_sum_before_reduce = true;
        }
      }
      graph.CUR_STRATEGY_ID = 0;

      Tensor grad_sum;
      if (is_need_sum_before_reduce) {
        TensorList partial_grad_list;
        std::vector<OpName> partial_grad_name_list;
        for (const auto& grad : filtered) {
          Tensor partial_grad = grad->producer()->input(0);
          partial_grad_list.push_back(partial_grad);
          partial_grad_name_list.push_back(partial_grad->name());
        }
        // concatenate names of partial grads
        OpName concat_name_of_partial_grad = std::accumulate(
          partial_grad_name_list.begin() + 1, partial_grad_name_list.end(), partial_grad_name_list[0],
          [](const std::string& a, const std::string& b) {
            return a + ", " + b;
        });
        // if allreduce/reduce-scatter group is different between input grads,
        // then assert error in placement group deduce process.
        Tensor partial_grad_sum = MakeSumOp(
          partial_grad_list,
          OpMeta().set_name("sum_op_for_partial_[" + concat_name_of_partial_grad + "]")
        );
        partial_grad_sum->set_is_grad(true);
        // 原地的comm
        grad_sum = MakeCommOp(partial_grad_sum, dst_ds_hierarchy, OpMeta().set_name("comm_op_after_partial_grad_sum"));
      } else {
        std::vector<OpName> grad_name_list;
        for (const auto& grad : filtered) {
          grad_name_list.push_back(grad->name());
        }
        OpName concat_name_of_grad = std::accumulate(
          grad_name_list.begin() + 1, grad_name_list.end(), grad_name_list[0],
          [](const std::string& a, const std::string& b) {
            return a + ", " + b;
        });
        grad_sum = MakeSumOp(filtered, OpMeta().set_name("sum_op_for_[" + concat_name_of_grad + "]"));
      }
      grad_sum->set_is_grad(true);
      return grad_sum;
    }
  };

  // traverse the forward graph in the reversed topo order
  auto topo = Graph::TopoSort(ys, num_ops_hint);
  for (auto it = topo.rbegin(); it != topo.rend(); ++it) {  
    auto& op = it->get();
    TensorList grad_outputs;
    if (op->num_outputs() > 0) {
      grad_outputs.reserve(op->num_outputs());
      for (auto& output : op->outputs()) {
        auto grad = reduce_grad(tensor_to_grads[output->id()]);
        tensor_to_reduced_grad[output->id()] = grad;
        grad_outputs.push_back(grad);
      }
    }

    if (op->num_inputs() > 0) {
      TensorList grad_inputs;
      // workaround
      // 两个连续的comm op部分情况下需要进行等价变换
      // 要先做inter group的comm再做intra group的comm
      // 目前主要针对share weight grad
      // 例如要先BatchedIsendIrecv再SplitAllReduce/SplitReduceScatter
      // 更多情形目前还未遇到
      bool skip_actual_gradient_op = false;
      if (is_comm_op(op) && grad_outputs.size() > 0 && grad_outputs.at(0).is_defined()) {
        auto& comm_op_impl = dynamic_cast<CommOpImpl&>(op->body());
        HT_ASSERT(grad_outputs.size() == 1)
          << "Comm op outputs size should be 1";
        auto& prev_grad_op = grad_outputs.at(0)->producer();
        // 判断comm op是inter op且prev op是在对gradient进行reduce时临时引入的intra op
        if (comm_op_impl.dst_group_hierarchy().size() != 0 
            && is_comm_op(prev_grad_op) 
            && prev_grad_op->fw_op_id() == -1) {
          skip_actual_gradient_op = true;
          // 修正prev op
          HT_ASSERT(prev_grad_op->num_inputs() == 1)
            << "Comm op inputs size should be 1";
          HT_ASSERT(is_variable_op(op->input(0)->producer()))
            << "This workaround is target for sharing weight like wte"
            << ", we are not sure if it still works for other weird situations";
          /*
          HT_LOG_WARN << op->input(0)->cur_ds_union().ds_union_info()
            << " and " << prev_grad_op->input(0)->cur_ds_union().ds_union_info();
          */
          auto& graph = prev_grad_op->graph();
          DistributedStatesHierarchy inter_ds_hierarchy, intra_ds_hierarchy;
          bool is_need_comm_op = false;
          for (size_t cur_strategy_id = 0; cur_strategy_id < graph.NUM_STRATEGY; cur_strategy_id++) {
            graph.CUR_STRATEGY_ID = cur_strategy_id;
            DistributedStatesUnion inter_ds_union, intra_ds_union;
            HT_ASSERT(op->input(0)->has_cur_ds_union())
              << "something wrong when initializing or deducing the ds for " << op->input(0);
            for (size_t cur_hetero_id = 0; cur_hetero_id < op->input(0)->cur_ds_union().size(); cur_hetero_id++) {
              auto& original_ds = op->input(0)->cur_ds_union().get(cur_hetero_id);
              HT_ASSERT(original_ds.is_valid())
                << "something wrong when initializing or deducing the ds for " << op->input(0);
              // original_ds是wte的ds
              // grad的partial与variable的dup是一致的
              if (original_ds.get_dim(-1) > 1) { 
                int32_t device_num = original_ds.get_device_num();
                std::pair<std::vector<int32_t>, int32_t> inter_src2dst({{-1}, -2});
                std::pair<std::vector<int32_t>, int32_t> intra_src2dst({{-1}, 0});
                if (op->input(0)->cur_ds_union().hetero_dim() == -1) {
                  inter_ds_union.set_hetero_dim(-2);
                  intra_ds_union.set_hetero_dim(-1);
                  if (original_ds.zero()) {
                    intra_ds_union.set_hetero_dim(0);
                    intra_ds_union.set_split_pattern(SplitPattern(false));
                  }
                } else {
                  HT_ASSERT(op->input(0)->cur_ds_union().hetero_dim() == NULL_HETERO_DIM)
                    << "only support original comm op input hetero on -1";
                }
                std::unordered_map<int32_t, int32_t> new_states = original_ds.combine_states(inter_src2dst);
                std::vector<int32_t> new_order = original_ds.combine_order(inter_src2dst);
                DistributedStates new_ds({device_num, new_states, new_order});
                inter_ds_union.add(new_ds);
                // SplitReduceScatter
                if (original_ds.zero()) {
                  new_states = original_ds.combine_states(intra_src2dst);
                  new_order = original_ds.combine_order(intra_src2dst);
                  new_ds = {device_num, new_states, new_order};
                  intra_ds_union.add(new_ds);
                } 
                // SplitAllReduce
                else {
                  intra_ds_union.add(original_ds);
                }
                is_need_comm_op = true;
              } 
              else {
                inter_ds_union.add(original_ds);
                intra_ds_union.add(original_ds);
              }
            }
            inter_ds_hierarchy.add(inter_ds_union);
            intra_ds_hierarchy.add(intra_ds_union);
          }
          graph.CUR_STRATEGY_ID = 0;
          HT_ASSERT(is_need_comm_op)
            << "something wrong, if two comm op appears, that means there must have partial->dup";
          // 1、make inter comm op
          // BacthedIsendIrecv/p2p
          grad_outputs.at(0) = MakeCommOp(prev_grad_op->input(0), inter_ds_hierarchy, 
            OpMeta().set_name("workaround_share_weight_grad_inter_comm")); // inter group op
          grad_outputs.at(0)->set_is_grad(true);
          grad_outputs.at(0)->producer()->set_fw_op_id(op->id());
          // 2、make intra comm op
          // SplitAllReduce/SplitReduceScatter
          // 将partial再都转化为dup（SplitAllReduce）或者split0（SplitReduceScatter）
          grad_outputs.at(0) = MakeCommOp(grad_outputs.at(0), intra_ds_hierarchy, 
            OpMeta().set_name("workaround_share_weight_grad_intra_comm")); // intra group op
          grad_inputs = {grad_outputs.at(0)};
        }
      }

      // 实际求导操作
      // 生成相应的gradient算子
      // workaround: share weight grad
      HT_LOG_TRACE << "Make gradient op for " << op << " with inputs " << grad_outputs;
      if (!skip_actual_gradient_op) {
        grad_inputs = op->Gradient(grad_outputs);
      } 
      HT_LOG_TRACE << "Gradient op outputs are " << grad_inputs;

      // states deduce
      // 如出现partial需要自动将其转化为dup或split
      for (size_t i = 0; i < op->num_inputs(); i++) {
        if (!grad_inputs[i].is_defined())
          continue;
        // bw subgraph deduce
        // TODO: subgraph need to contain other inserted intermediate op (e.g. sum)
        auto& cur_graph = op->graph();
        std::shared_ptr<SubGraph> subgraph = cur_graph.GetSubGraph(op);
        if (subgraph != nullptr) {
          cur_graph.AddOpToSubGraph(grad_inputs[i]->producer(), 
                                    subgraph->global_graph_name(), 
                                    SubGraphType::BACKWARD);
        }
        grad_inputs[i]->set_is_grad(true);
        grad_inputs[i]->producer()->set_fw_op_id(op->id());
        auto& grad_op = grad_inputs[i]->producer();
        Tensor final_grad = grad_inputs[i];
        auto& graph = grad_inputs[i]->graph();
        DistributedStatesHierarchy dst_ds_hierarchy;
        bool is_need_comm_op = false;
        for (size_t cur_strategy_id = 0; cur_strategy_id < graph.NUM_STRATEGY; cur_strategy_id++) {
          graph.CUR_STRATEGY_ID = cur_strategy_id;
          DistributedStatesUnion dst_ds_union;
          if (!grad_inputs[i]->has_cur_ds_union()) {
            HT_RUNTIME_ERROR << "something wrong when initializing or deducing the ds for " << grad_inputs[i];
          }
          for (size_t cur_hetero_id = 0; cur_hetero_id < grad_inputs[i]->cur_ds_union().size(); cur_hetero_id++) {
            auto& ds_grad = grad_inputs[i]->cur_ds_union().get(cur_hetero_id);
            HT_ASSERT(ds_grad.is_valid())
              << "something wrong when initializing or deducing the ds for " << grad_inputs[i];
            // HT_LOG_DEBUG << local_device << ": " << "grad_op: " << grad_op << ": states: " << ds_grad.ds_info() << ", shape: " << grad_inputs[i]->shape();
            // partial->duplicate 
            if (ds_grad.get_dim(-2) > 1) { 
              int32_t device_num = ds_grad.get_device_num();
              // std::pair<std::vector<int32_t>, int32_t> src2dst({{-2}, -1});
              std::pair<std::vector<int32_t>, int32_t> src2dst;
              // zero
              bool is_dp_reduce = false;
              bool is_zero = false;
              // 正常的grad
              if (is_variable_op(op->input(i)->producer())) {
                is_dp_reduce = true;
                is_zero = op->input(i)->get_distributed_states().zero();
              } 
              // share weight
              // TODO: comm op后续要支持变成一个p2p和一个reduce
              // 这样的话如果is_comm_op(op->input(i)->producer()
              // 那么可以全部交给这个comm op进行处理
              if (is_comm_op(op->input(i)->producer())
                  && is_variable_op(op->input(i)->producer()->input(0)->producer())) {
                is_dp_reduce = true;
                is_zero = op->input(i)->producer()->input(0)->get_distributed_states().zero();
              }
              // 去除partial
              // zero情形
              if (is_dp_reduce && is_zero) {
                // attention: the result tensor was dp grouped split0, not really split0!
                // so should do allgather still within the same dp group later! 
                src2dst = {{-2}, 0}; // reduce-scatter
                // Question: consider exec graph switch
                // for row parallel whose split is at dim 1
                // maybe it's a better option to set src2dst = {{-2}, 1}
                // 1、sync the gradients for dp
                if (grad_inputs[i]->cur_ds_union().hetero_dim() == -2) {
                  HT_ASSERT(dst_ds_union.hetero_dim() == 0 || dst_ds_union.hetero_dim() == NULL_HETERO_DIM)
                    << "hetero dim in the union should be consistent";
                  dst_ds_union.set_hetero_dim(0);
                  dst_ds_union.set_split_pattern(SplitPattern(false));
                } 
                // 暂时不会有其余情况
                else {
                  HT_ASSERT(grad_inputs[i]->cur_ds_union().hetero_dim() == NULL_HETERO_DIM)
                    << "only support grad hetero on -2 for variables"
                    << ", but " << grad_inputs[i] << " cur ds union is " 
                    << grad_inputs[i]->cur_ds_union().ds_union_info();
                }
                HT_ASSERT(dst_ds_union.hetero_dim() == 0 || dst_ds_union.hetero_dim() == NULL_HETERO_DIM)
                  << "hetero dim in the union should be consistent";
              } 
              // 非zero情形
              else {
                src2dst = {{-2}, -1}; // allreduce
                // 1、sync the gradients for dp
                if (is_dp_reduce) {
                  if (grad_inputs[i]->cur_ds_union().hetero_dim() == -2) {
                    HT_ASSERT(dst_ds_union.hetero_dim() == -1 || dst_ds_union.hetero_dim() == NULL_HETERO_DIM)
                      << "hetero dim in the union should be consistent";
                    dst_ds_union.set_hetero_dim(-1);
                  } else {
                    HT_ASSERT(grad_inputs[i]->cur_ds_union().hetero_dim() == NULL_HETERO_DIM)
                      << "only support grad hetero on -2 or 0 when zero is off"
                      << ", but " << grad_inputs[i] << " cur ds union is " 
                      << grad_inputs[i]->cur_ds_union().ds_union_info();
                  }
                }
                // 2、tp/sp reduce
                else {
                  // sp会显式地在正向传播时插入all gather
                  // 因此这里不需要做reduce
                  // 交给之后comm op的DoGradient去自动处理
                  if (is_comm_op(op->input(i)->producer())) {
                    continue;
                  }
                  if (grad_inputs[i]->cur_ds_union().hetero_dim() == 0) {
                    HT_ASSERT(dst_ds_union.hetero_dim() == 0 || dst_ds_union.hetero_dim() == NULL_HETERO_DIM)
                      << "hetero dim in the union should be consistent";
                    dst_ds_union.set_hetero_dim(0);
                  }
                  else {
                    HT_ASSERT(grad_inputs[i]->cur_ds_union().hetero_dim() == NULL_HETERO_DIM)
                      << "only support grad hetero on -2 or 0 when zero is off"
                      << ", but " << grad_inputs[i] << " cur ds union is " 
                      << grad_inputs[i]->cur_ds_union().ds_union_info();
                  }
                }
              }
              std::unordered_map<int32_t, int32_t> res_states = ds_grad.combine_states(src2dst);
              std::vector<int32_t> res_order = ds_grad.combine_order(src2dst);
              DistributedStates ds_dst({device_num, res_states, res_order});
              HT_LOG_TRACE << hetu::impl::comm::GetLocalDevice() << ": " 
                << "backward: partial to duplicate: " << grad_inputs[i]
                << ", src states: " << ds_grad.ds_info()
                << ", dst states: " << ds_dst.ds_info();
              dst_ds_union.add(ds_dst);
              is_need_comm_op = true;
            } else {
              HT_ASSERT(dst_ds_union.hetero_dim() == NULL_HETERO_DIM || dst_ds_union.hetero_dim() == grad_inputs[i]->cur_ds_union().hetero_dim())
                << "hetero dim in the union should be consistent";
              dst_ds_union.set_hetero_dim(grad_inputs[i]->cur_ds_union().hetero_dim());
              dst_ds_union.add(ds_grad);
            }
          }
          dst_ds_hierarchy.add(dst_ds_union);
        }
        graph.CUR_STRATEGY_ID = 0;
        HT_LOG_TRACE << grad_inputs[i] << " need_comm_op is " << is_need_comm_op;
        if (is_need_comm_op) {
          final_grad = MakeCommOp(grad_inputs[i], dst_ds_hierarchy, 
            OpMeta().set_name("comm_op_after_" + grad_op->name())); // allreduce
          final_grad->set_is_grad(true);
          // 这里一定要在原地做
          // 所以不能设置fw_op_id
          // 否则会导致推导placement group错误
          // final_grad->producer()->set_fw_op_id(op->id());
        }
        auto input = op->input(i);
        auto it = tensor_to_grads.find(input->id());
        if (it == tensor_to_grads.end())
          tensor_to_grads[input->id()] = {final_grad};
        else
          it->second.push_back(final_grad);
      }
    }
  }

  TensorList ret;
  ret.reserve(xs.size());
  for (auto& x : xs) {
    auto it = tensor_to_reduced_grad.find(x->id());
    if (it != tensor_to_reduced_grad.end())
      ret.push_back(it->second);
    else
      ret.emplace_back(Tensor());
  }
  return ret;
}

std::string GraphType2Str(GraphType type) {
  if (type == GraphType::EAGER) {
    return "eager";
  } else if (type == GraphType::DEFINE_BY_RUN) {
    return "define_by_run";
  } else if (type == GraphType::DEFINE_AND_RUN) {
    return "define_and_run";
  } else if (type == GraphType::EXECUTABLE) {
    return "executable";
  } else {
    HT_VALUE_ERROR << "Unrecognizable graph type: " << static_cast<int>(type);
    __builtin_unreachable();
  }
}

void Graph::SubGraphProfiling(std::unordered_map<OpId, int64_t> op_execute_map, int num_micro_batches) {
  for (auto it = _subgraphs.begin(); it != _subgraphs.end(); ++it) {
    it->second->profile_reset();  
  }
  for (auto it = _subgraphs.begin(); it != _subgraphs.end(); ++it) {
    it->second->profile(op_execute_map, num_micro_batches);
    HT_LOG_INFO << it->first << ":" 
    << "fwd_time: " << it->second->fwd_time() * 1.0 / 1e6 << "ms,"
    << "bwd_time: " << it->second->bwd_time() * 1.0 / 1e6 << "ms,"
    << "update_time: " << it->second->update_time() * 1.0 / 1e6 << "ms,"
    << "total_time: " << it->second->total_time() * 1.0 / 1e6 << "ms.";  
  }
}

std::ostream& operator<<(std::ostream& os, GraphType type) {
  os << GraphType2Str(type);
  return os;
}

std::ostream& operator<<(std::ostream& os, const Graph& graph) {
  os << "graph(name=" << graph.name() << ", id=" << graph.id()
     << ", type=" << graph.type() << ")";
  return os;
}

std::ostream& operator<<(std::ostream& os, SubGraph& subgraph) {
  os << "subgraph(name=" << subgraph.name() << ", type=" << subgraph.subgraph_type()
     << ", ops=" << subgraph.ops() << ", bwd_ops=" << subgraph.bwd_ops() << 
     ", update_ops=" << subgraph.update_ops() << ", subgraphs=" << subgraph.subgraph_info().size() << "-"
     << subgraph.subgraph_info();
  return os;
}

} // namespace graph
} // namespace hetu
