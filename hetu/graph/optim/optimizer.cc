#include "hetu/graph/optim/optimizer.h"
#include "hetu/graph/ops/group.h"
#include "hetu/graph/ops/variable.h"
#include "hetu/graph/ops/Arithmetics.h"
#include "hetu/graph/ops/ones_like.h"
#include "hetu/graph/ops/optimizer_update.h"
#include "hetu/graph/graph.h"
#include "hetu/graph/define_and_run_graph.h"

namespace hetu {
namespace graph {

Tensor Optimizer::Minimize(const Tensor& loss, const TensorList& var_list,
                           const Tensor& grad_loss, const OpName& name) {
  GradAndVarList grads_and_vars = ComputeGradients(loss, var_list, grad_loss);
  GradAndVarList filtered_grads_and_vars;
  filtered_grads_and_vars.reserve(grads_and_vars.size());
  std::copy_if(grads_and_vars.begin(), grads_and_vars.end(),
               std::back_inserter(filtered_grads_and_vars),
               [](const GradAndVar& n) { return n.first.is_defined(); });
  // 这里补充对zero param的ds hierarchy的修正
  // 否则fp32 origin param并没有进行zero切分
  ApplyZero(filtered_grads_and_vars);
  return ApplyGradients(filtered_grads_and_vars, name);
}

StateDict Optimizer::GetStates(const Tensor& var) {
  HT_ASSERT(state_dict.find(var->id()) != state_dict.end());
  return state_dict[var->id()];
}

void Optimizer::SetStates(const Tensor& var, const OpName state_name, const NDArray& value) {
  HT_ASSERT(state_dict.find(var->id()) != state_dict.end());
  ResetVariableData(state_dict[var->id()][state_name], value);
}

Tensor Optimizer::ApplyGradients(const GradAndVarList& grads_and_vars,
                                 const OpName& name, const Tensor& infinite_count) {
  TensorList updated_params;
  updated_params.reserve(grads_and_vars.size());
  std::transform(
    grads_and_vars.begin(), grads_and_vars.end(),
    std::back_inserter(updated_params),
    [&](const GradAndVar& grad_and_var) { return ApplyDense(grad_and_var, infinite_count); });
  return MakeGroupOp(OpMeta().set_extra_deps(updated_params)
                      .set_name(name).set_is_deduce_states(false)); // group op needn't deduce states
}

// TODO: 后续可以考虑支持冗余zero存储
void Optimizer::ApplyZero(const GradAndVarList& grads_and_vars) {
  for (const auto& grad_and_var : grads_and_vars) {
    const Tensor& grad = grad_and_var.first;
    const Tensor& var = grad_and_var.second;
    if (var->graph().type() == GraphType::EAGER) {
      // workaround for unit tests
      break;
    }
    // 需要记录未应用zero前的var的ds hierarchy
    // 后续define graph实例化具体的optimize-compute bridge subgraph时需要
    dynamic_cast<DefineAndRunGraph&>(var->graph()).RecordBeforeZero(var, var->ds_hierarchy());
    for (size_t cur_strategy_id = 0; cur_strategy_id < var->ds_hierarchy().size(); cur_strategy_id++) {
      var->graph().CUR_STRATEGY_ID = cur_strategy_id;
      auto& ds_union = var->cur_ds_union();
      auto& target_ds_union = grad->cur_ds_union();
      HT_ASSERT(ds_union.size() >= 1)
        << var << " has an empty ds union when strategy id is " << cur_strategy_id;
      bool is_zero = ds_union.get(0).zero();
      for (const auto& ds : ds_union.raw_data()) {
        HT_ASSERT(is_zero == ds.zero())
          << "all ds in the ds union of " << var << " should have the same zero state";
      }
      if (is_zero) {
        // 这里其实可以直接把grad对应的ds赋值给var
        // 但保险起见我们这里多加一次check
        if (ds_union.is_hetero()) {
          ds_union.set_split_pattern(SplitPattern{false});
          ds_union.set_hetero_dim(0);
        }
        for (size_t cur_hetero_id = 0; cur_hetero_id < ds_union.size(); cur_hetero_id++) {
          // 直接把所有的-1维度转化成0
          std::pair<std::vector<int32_t>, int32_t> src2dst = {{-1}, 0};
          auto new_states = ds_union.get(cur_hetero_id).combine_states(src2dst);
          auto new_order = ds_union.get(cur_hetero_id).combine_order(src2dst);
          DistributedStates new_ds({ds_union.get(cur_hetero_id).get_device_num(), new_states, new_order});
          ds_union.get(cur_hetero_id) = new_ds;
        }
        HT_ASSERT(ds_union.check_equal(target_ds_union))
          << "currently final var (after apply zero) and final grad ds should be equal, but found "
          << ds_union.ds_union_info() << " vs " << target_ds_union.ds_union_info()
          << ", for var " << var << " on strategy " << cur_strategy_id;
      }
    }
    var->graph().CUR_STRATEGY_ID = 0;
    dynamic_cast<ParallelVariableOpImpl&>(var->producer()->body()).set_ds_hierarchy(var->ds_hierarchy());
  }
}

// the distributed states for adam mean/variance was dummy, just do allgather for dp groups in later execution
Tensor Optimizer::MakeStates(const Tensor& variable, const Tensor& grad, const OpName& state_name) {
  const auto& producer = variable->producer();
  HT_VALUE_ERROR_IF(!producer->is_parameter());
  // special case: Varibale States should be set distributed_states as ds_grad (whether zero or not)
  // const DistributedStates& ds_variable = variable->get_distributed_states(); 
  // const DistributedStates& ds_grad = grad->get_distributed_states();
  const auto& variable_ds_hierarchy = variable->ds_hierarchy(); 
  const auto& grad_ds_hierarchy = grad->ds_hierarchy(); 
  // HT_ASSERT (ds_variable.is_valid() && ds_grad.is_valid()) 
  //   << "Diastributed States for varibale " << variable << " must be valid!";  
  // HT_LOG_INFO << variable->name() + "_" + state_name << " directly use grad " << grad << ": " << grad_ds_hierarchy.get(1).ds_union_info();
  Tensor states = MakeParallelVariableOp(ZerosInitializer(), grad->global_shape(),
                                         grad_ds_hierarchy, {0}, grad->dtype(), false, {},
                                         OpMeta()
                                          .set_device_group_hierarchy(producer->device_group_hierarchy())
                                          .set_eager_device(producer->eager_device())
                                          .set_name(variable->name() + "_" + state_name));  
  Graph::MarkAsOptimizerVariable(states);
  if (state_dict.find(variable->id()) == state_dict.end()) {
    state_dict[variable->id()] = {};
  }
  state_dict[variable->id()][state_name] = states;

  return std::move(states);
}

GradAndVarList Optimizer::ComputeGradients(const Tensor& loss,
                                           const TensorList& var_list,
                                           const Tensor& grad_loss) {
  TensorList vars = var_list;
  if (vars.empty()) {
    auto topo_order = Graph::TopoSort(loss);
    for (auto& op_ref : topo_order)
      if (op_ref.get()->is_parameter())
        vars.push_back(op_ref.get()->output(0));
  }
  TensorList grads = Graph::Gradients(loss, vars, grad_loss);
  GradAndVarList grads_and_vars;
  grads_and_vars.reserve(grads.size());
  for (size_t i = 0; i < grads.size(); i++)
    grads_and_vars.emplace_back(grads[i], vars[i]);
  return grads_and_vars;
}

Tensor SGDOptimizer::ApplyDense(const GradAndVar& grad_and_var, const Tensor& infinite_count) {
  const Tensor& grad = grad_and_var.first;
  const Tensor& var = grad_and_var.second;
  auto update_op_meta = OpMeta()
                          .set_device_group_hierarchy(var->producer()->device_group_hierarchy())
                          .set_name("Update_" + var->name())
                          .set_is_deduce_states(false);
  if (momentum() == 0) {
    if (infinite_count != Tensor())
      return MakeSGDUpdateWithGradScalerOp(var, grad, infinite_count, param_scheduler(), update_op_meta);
    return MakeSGDUpdateOp(var, grad, param_scheduler(), update_op_meta);
  } else {
    return MakeMomentumUpdateOp(var, grad, MakeStates(var, grad, "velocity"),
                                param_scheduler(), momentum(), nesterov(),
                                update_op_meta);
  }
}

Tensor AdamOptimizer::ApplyDense(const GradAndVar& grad_and_var, const Tensor& infinite_count) {
  const Tensor& grad = grad_and_var.first;
  const Tensor& var = grad_and_var.second;
  auto update_op_meta = OpMeta()
                          .set_device_group_hierarchy(var->producer()->device_group_hierarchy())
                          .set_name("Update_" + var->name())
                          .set_is_deduce_states(false); // update op needn't deduce states
  HTShape step_shape = {1};
  // 在cpu上不需要设置dstates
  Tensor step = MakeVariableOp(OnesInitializer(), step_shape, kInt64,
                                false, var->ds_hierarchy(), 
                                OpMeta()
                                  .set_device_group_hierarchy(var->producer()->device_group_hierarchy())
                                  .set_eager_device(kCPU)
                                  .set_name(var->name() + "_step")
                                  .set_is_deduce_states(false)
                                  .set_is_cpu(true));
  if (state_dict.find(var->id()) == state_dict.end()) {
    state_dict[var->id()] = {};
  }
  state_dict[var->id()]["step"] = step;
  // variable: dup in dp group, grad: reduce-scatter in dp group, mean & variance: same as grad



  return MakeAdamOp(var, grad, MakeStates(var, grad, "mean"),
                    MakeStates(var, grad, "variance"),
                    param_scheduler(), step, beta1(), beta2(),
                    eps(), update_op_meta);
}

} // namespace graph
} // namespace hetu

