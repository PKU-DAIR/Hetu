#include "hetu/graph/headers.h"
#include "hetu/graph/eager_graph.h"
#include "hetu/graph/define_by_run_graph.h"

namespace hetu {
namespace graph {

TensorDef::TensorDef(const constructor_access_key&, TensorIdentifier ids,
                     TensorName name, bool requires_grad, NDArrayMeta meta)
: _ids(std::move(ids)),
  _name(std::move(name)),
  _requires_grad(requires_grad),
  _meta(std::move(meta)),
  _symbolic(false) {
  auto& graph = Graph::GetGraph(_ids.graph_id);
  _inform_graph_on_destruction = (graph.type() == GraphType::EAGER ||
                                  graph.type() == GraphType::DEFINE_BY_RUN);
  _symbolic_shape = SyShape(_meta.shape.size());
}

Tensor::Tensor(TensorIdentifier ids, TensorName name, bool requires_grad,
               NDArrayMeta meta)
: shared_ptr_wrapper<TensorDef>() {
  _ptr =
    make_ptr<TensorDef>(TensorDef::constructor_access_key(), std::move(ids),
                        std::move(name), requires_grad, std::move(meta));
}

Tensor::~Tensor() {
  if (!is_defined())
    return;

  if (_ptr->_inform_graph_on_destruction && _ptr->num_consumers() == 0 &&
      get_referrence_count() == 1) {
    // To avoid a second entrance
    _ptr->_inform_graph_on_destruction = false;
    // Inform graph to move or prune
    auto& graph = _ptr->graph();
    if (graph.type() == GraphType::EAGER) {
      reinterpret_cast<EagerGraph&>(graph).RemoveTensor(*this);
    } else if (graph.type() == GraphType::DEFINE_BY_RUN) {
      reinterpret_cast<DefineByRunGraph&>(graph).PruneTensor(*this);
    }
  }
  _ptr = nullptr;
}

void TensorDef::AddConsumer(Operator& op) {
  _consumers.push_back(std::ref(op));
}

void TensorDef::DelConsumer(const Operator& op) {
  _consumers.erase(
    std::remove_if(_consumers.begin(), _consumers.end(),
                   [&](const OpRef& x) { return x.get()->id() == op->id(); }), _consumers.end());
}

const Graph& TensorDef::graph() const {
  return Graph::GetGraph(graph_id());
}

Graph& TensorDef::graph() {
  return Graph::GetGraph(graph_id());
}

const Operator& TensorDef::producer() const {
  return graph().GetOp(producer_id());
}

Operator& TensorDef::producer() {
  return graph().GetOp(producer_id());
}

const Operator& TensorDef::consumer(size_t i) const {
  return _consumers[i];
}

Operator& TensorDef::consumer(size_t i) {
  return _consumers[i];
}

Tensor& TensorDef::get_self() {
  return output_id() >= 0 ? producer()->output(output_id())
                          : producer()->out_dep_linker();
}

const Tensor& TensorDef::get_self() const {
  return output_id() >= 0 ? producer()->output(output_id())
                          : producer()->out_dep_linker();
}

bool TensorDef::is_variable() const {
  return is_variable_op(producer());
}

bool TensorDef::is_parameter() const {
  const auto& graph = Graph::GetGraph(graph_id());
  return graph._parameter_ops.find(producer_id()) != graph._parameter_ops.end();
}

NDArray TensorDef::get_or_compute() {
  return graph().GetOrCompute(get_self());
}

void TensorDef::merge_strategy(Tensor& tensor) {
  for (const auto& ds_union : tensor->ds_hierarchy().raw_data()) {
    _ds_hierarchy.add(ds_union);
  }
  HT_ASSERT(_ds_hierarchy.size() == 0 || _ds_hierarchy.size() == graph().NUM_STRATEGY)
    << name() << ": please set the correct NUM_STRATEGY of the graph before merge strategy"
    << ", found _ds_hierarchy.size() = " << _ds_hierarchy.size()
    << " but graph().NUM_STRATEGY = " << graph().NUM_STRATEGY;
}

size_t TensorDef::num_strategy() const {
  return graph().NUM_STRATEGY;
}

size_t TensorDef::cur_strategy_id() const {
  return graph().CUR_STRATEGY_ID;
}

size_t TensorDef::cur_hetero_id() const {
  return graph().CUR_HETERO_ID;
}

bool TensorDef::has_cur_ds_union() {
  if (graph().type() == GraphType::EXECUTABLE) {
    HT_ASSERT(_ds_hierarchy.size() <= 1)
      << "ExecutableGraph only has at most one single strategy"
      << ", but found " << name() << " has " << _ds_hierarchy.size() << " strategy";
    return _ds_hierarchy.size() == 1;
  } else if (graph().type() == GraphType::EAGER) {
    // workaround for unit test
    return true;
  }
  return cur_strategy_id() < _ds_hierarchy.size();
}

DistributedStatesUnion& TensorDef::cur_ds_union() {
  // 2024.9.25 Update
  // 限制executable graph只有一个strategy
  if (graph().type() == GraphType::EXECUTABLE) {
    if (_ds_hierarchy.size() == 0) {
      HT_ASSERT(graph().CREATE_STRATEGY)
        << "ExecutableGraph ds union is constructed only through define graph instantiate or make op in exec graph"
        << ", but found " << name() << " disobey the rule";
      _ds_hierarchy.add(DistributedStatesUnion());
      return _ds_hierarchy.get(0);
    }
    HT_ASSERT(_ds_hierarchy.size() == 1)
      << "ExecutableGraph only has one single strategy"
      << ", but found " << name() << " has " << _ds_hierarchy.size() << " strategy";
    return _ds_hierarchy.get(0);
  }
  if (graph().CREATE_STRATEGY) {
    while (graph().CUR_STRATEGY_ID >= _ds_hierarchy.size()) {
      _ds_hierarchy.add(DistributedStatesUnion());
    }
  }
  // 没有开deduce states的op的输出
  if (_ds_hierarchy.size() == 0) {
    return _dummy_ds_union;
  }
  HT_ASSERT(graph().CUR_STRATEGY_ID < _ds_hierarchy.size())
    << name() << " strategy id out of range: " 
    << graph().CUR_STRATEGY_ID << " is larger than " << _ds_hierarchy.size() 
    << ", the CREATE_STRATEGY is " << graph().CREATE_STRATEGY;
  return _ds_hierarchy.get(graph().CUR_STRATEGY_ID);
}

void TensorDef::set_cur_ds_union(const DistributedStatesUnion& ds_union) {
  // 2024.9.25 Update
  // 限制executable graph只有一个strategy
  if (graph().type() == GraphType::EXECUTABLE) {
    if (_ds_hierarchy.size() == 0) {
      _ds_hierarchy.add(ds_union);
      return;
    }
    HT_ASSERT(_ds_hierarchy.size() == 1)
      << "ExecutableGraph only has one single strategy";
    _ds_hierarchy.get(0) = ds_union;
    return;
  }
  // 未在InferMeta时DeduceStatesHierarchy的tensor
  if (_ds_hierarchy.size() == 0) {
    while (graph().CUR_STRATEGY_ID >= _ds_hierarchy.size()) {
      _ds_hierarchy.add(DistributedStatesUnion());
    }
    _ds_hierarchy.get(graph().CUR_STRATEGY_ID) = ds_union;
  }
  // 其余情况必须要求_hierarchy已准备妥当
  HT_ASSERT(graph().CUR_STRATEGY_ID < _ds_hierarchy.size())
    << "Strategy id out of range";
  _ds_hierarchy.get(graph().CUR_STRATEGY_ID) = ds_union;
}

size_t TensorDef::inferred_local_placement_group_idx() const {
  HT_ASSERT(!graph().USE_HETERO_ID)
    << "inferred_local_placement_group_idx should only used when hetero id is not provided";
  auto inferred = hetu::impl::comm::GetLocalDevice();
  HT_ASSERT(placement().is_undetermined() || placement() == inferred)
    << "inferred_local_placement_group_idx is mainly used when the op is not instantiated";
  HT_ASSERT(has_placement_group())
    << "inferred_local_placement_group_idx should at least guarantee there is a placement group union";
  if (placement_group_union().has(inferred)) {
    return placement_group_union().get_index(inferred);
  }
  if (graph().type() == GraphType::DEFINE_AND_RUN) {
    HT_ASSERT(producer()->suggested_hetero_id() == 0)
      << "All ops in define graph should have the suggested_hetero_id with 0"
      << ", make sure you haven't changed the COMPUTE_SUGGESTED_HETERO_ID of the define graph";
  }
  return placement_group_union().size() > 1 ? producer()->suggested_hetero_id() : 0;
}

// 对于不同阶段下的tensor
// infer方式不同
DistributedStates& TensorDef::inferred_cur_ds() {
  auto& ds_union = cur_ds_union();
  // 给定了hetero id
  // 说明是在DeduceStates
  // 或者是保证InferMeta具有明确的指向
  if (graph().USE_HETERO_ID) {
    if (graph().CREATE_HETERO) {
      while (cur_hetero_id() >= ds_union.size()) {
        ds_union.add(DistributedStates());
      }
      return ds_union.get(cur_hetero_id());
    }
    // homo情形相当于一个所有hetero ds都一样的union
    if (!ds_union.is_hetero()) {
      return ds_union.get(0);
    }
    HT_ASSERT(cur_hetero_id() < ds_union.size())
      << "Hetero id is out of range";
    return ds_union.get(cur_hetero_id());
  } 
  // 其余情况下
  // 1、define graph
  // 比如在InferMeta
  // 但实际上我们并不关心define graph里的ds、dg、shape等
  // 因此默认使用第0个
  if (graph().type() == GraphType::DEFINE_AND_RUN) {
    HT_ASSERT(producer()->suggested_hetero_id() == 0)
      << "All ops in define graph should have the suggested_hetero_id with 0"
      << ", make sure you haven't changed the COMPUTE_SUGGESTED_HETERO_ID of the define graph";
    return ds_union.get(0);
  }
  // 2、exec graph
  // 应保证已经有_placement_group_union了
  else if (graph().type() == GraphType::EXECUTABLE) {
    HT_ASSERT(_has_placement_group)
      << name() << " should already MapToParallelDevices";
    // 已经instantiate过了
    if (!placement().is_undetermined()) {
      HT_ASSERT(ds_union.size() == placement_group_union().size())
        << name() << " size of ds union and pg union should be equal";
      return ds_union.get(local_placement_group_idx());
    }
    // 没instantiate或者是op不在当前device上
    else {
      return ds_union.get(inferred_local_placement_group_idx());
    }
  }
  else if (graph().type() == GraphType::EAGER) {
    // workaround for unit tests
    return _dummy_ds;
  }
  // 其他graph类型暂不支持
  else {
    HT_RUNTIME_ERROR << "Currently not support";
  }
}

DistributedStates TensorDef::get_local_distributed_states() {
  auto& ds_union = cur_ds_union();
  auto& ds = inferred_cur_ds();
  if (!ds_union.is_hetero()) {
    return ds;
  }
  // 需要从hetero dim剔除掉hetero size
  int32_t cur_hetero_dim = ds_union.hetero_dim();
  HT_ASSERT(ds.states(cur_hetero_dim) % ds_union.size() == 0)
    << "Hetero dim of ds need to be divided by the size of union";
  auto new_device_num = ds.get_device_num() / ds_union.size();
  auto new_states = ds.get_states();
  auto new_order = ds.get_order();
  auto new_zero = ds.zero();
  HT_ASSERT(new_states.find(cur_hetero_dim) != new_states.end())
    << "The ds must consists of the hetero dim";
  new_states[cur_hetero_dim] /= ds_union.size();
  if (new_states[cur_hetero_dim] == 1 && cur_hetero_dim >= 0) {
    new_states = ds.reduce_states(cur_hetero_dim);
    new_order = ds.reduce_order(cur_hetero_dim);
  }
  return DistributedStates(new_device_num, new_states, new_order, new_zero);
}

std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
  if (tensor.is_defined())
    os << tensor->name();
  else
    os << "Tensor()";
  return os;
}

} // namespace graph
} // namespace hetu
