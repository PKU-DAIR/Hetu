#pragma once

#include "hetu/common/macros.h"
#include "hetu/core/ndarray.h"
#include "hetu/core/symbol.h"
#include "hetu/graph/common.h"
#include "hetu/graph/distributed_states.h"


namespace hetu {
namespace graph {

struct TensorIdentifier {
  GraphId graph_id;
  mutable OpId producer_id;
  int32_t output_id;
  TensorId tensor_id;
};

class TensorDef : public shared_ptr_target {
 protected:
  friend class OpDef;
  friend class Operator;
  friend class Tensor;
  friend class Graph;
  friend class ExecutableGraph;
  struct constructor_access_key {};

 public:
  TensorDef(const constructor_access_key&, TensorIdentifier ids,
            TensorName name, bool requires_grad, NDArrayMeta meta);

  ~TensorDef() = default;

  // disable copy constructor and move constructor
  TensorDef(const TensorDef&) = delete;
  TensorDef& operator=(const TensorDef&) = delete;
  TensorDef(TensorDef&&) = delete;
  TensorDef& operator=(TensorDef&&) = delete;

  TensorId id() const {
    return _ids.tensor_id;
  }

  const TensorName& name() const {
    return _name;
  }

  GraphId graph_id() const {
    return _ids.graph_id;
  }

  OpId producer_id() const {
    return _ids.producer_id;
  }

  // unsafe, may cause chaos of topo, so try to avoid using it
  void set_producer_id(const OpId& id) {
    _ids.producer_id = id;
  }

  const Graph& graph() const;

  Graph& graph();

  size_t num_strategy() const;

  size_t cur_strategy_id() const;

  size_t cur_hetero_id() const;
  
  const Operator& producer() const;

  Operator& producer();

  int32_t output_id() const noexcept {
    return _ids.output_id;
  }

  bool is_out_dep_linker() const noexcept {
    return output_id() < 0;
  }

  bool is_leaf() const {
    return is_variable();
  }

  bool is_variable() const;
  
  bool is_parameter() const;

  size_t num_consumers() const {
    return _consumers.size();
  }

  const OpRefList& consumers() const {
    return _consumers;
  }

  OpRefList& consumers() {
    return _consumers;
  }

  const Operator& consumer(size_t i) const;

  Operator& consumer(size_t i);

  const NDArrayMeta& meta() const noexcept {
    return _meta;
  }

  size_t ndim() const {
    return _meta.ndim();
  }

  size_t numel() const {
    return _meta.numel();
  }

  DataType dtype() const {
    return _meta.dtype;
  }

  const Device& device() const noexcept {
    return _meta.device;
  }

  bool is_cpu() const {
    return _meta.device.is_cpu();
  }

  bool is_cuda() const {
    return _meta.device.is_cuda();
  }

  const HTShape& shape() const {
    /*
    // 优先使用symbolic的value
    if (_symbolic) 
      return get_HTShape_from_SyShape(_symbolic_shape);
    */
    return _meta.shape;
  }

  int64_t shape(size_t axis) const {
    /*
    // 优先使用symbolic的value
    if (_symbolic) {
      auto shape = get_HTShape_from_SyShape(_symbolic_shape);
      return shape[axis];
    }
    */
    return _meta.shape[axis];
  }

  const HTStride& stride() const {
    /*
    // 优先使用symbolic的value
    if (_symbolic) {
      auto shape = get_HTShape_from_SyShape(_symbolic_shape);
      HTStride stride(shape.size());
      if (stride.size() > 0) {
        stride[stride.size() - 1] = 1;
        for (auto d = stride.size() - 1; d > 0; d--)
          stride[d - 1] = stride[d] * shape[d];
      }
      return stride;
    }
    */
    return _meta.stride;
  }

  int64_t stride(size_t axis) const {
    /*
    // 优先使用symbolic的value
    if (_symbolic) {
      auto shape = get_HTShape_from_SyShape(_symbolic_shape);
      HTStride stride(shape.size());
      if (stride.size() > 0) {
        stride[stride.size() - 1] = 1;
        for (auto d = stride.size() - 1; d > 0; d--)
          stride[d - 1] = stride[d] * shape[d];
      }
      return stride[axis];
    }
    */
    return _meta.stride[axis];
  }

  const HTShape& temp_shape() const {
    HT_ASSERT(has_temp_shape())
      << _name << " doesn't have temporal shape";
    return _temp_shape;
  }

  int64_t temp_shape(size_t axis) const {
    HT_ASSERT(has_temp_shape())
      << _name << " doesn't have temporal shape";
    return _temp_shape[axis];
  }

  bool has_shape() const {
    return _meta.shape.size() > 0;
  }

  bool has_temp_shape() const {
    return _temp_shape.size() > 0;
  }

  bool has_global_shape() const {
    return _global_shape.size() > 0;
  }

  const Device& placement() const noexcept {
    return _placement;
  }

  void set_placement(const Device& p) {
    _meta.set_device(p);
    _placement = p;
  }

  const bool requires_grad() const noexcept {
    return _requires_grad;
  }

  void set_requires_grad(bool new_requires_grad) {
    _requires_grad = new_requires_grad;
  }

  const bool is_grad() const noexcept {
    return _is_grad;
  }

  void set_is_grad(bool is_grad) {
    _is_grad = is_grad;
  }

  bool is_contiguous() const {
    if (ndim() < 1 || numel() <= 1) { return true; }
    int64_t ndim_ = ndim();
    int64_t contiguous_stride = 1;
    for (int64_t i = ndim_ - 1; i >= 0; i--) {
      if (stride(i) != contiguous_stride)
        return false;
      contiguous_stride *= shape(i);
    }
    return true;
  }

  std::optional<OpId> get_contiguous_op_id() const {
    return _contiguous_op_id;
  }

  void set_contiguous_op_id(OpId contiguous_op_id) {
    _contiguous_op_id = std::make_optional<OpId>(contiguous_op_id);
  }

  NDArray get_or_compute();

  bool has_cur_ds_union();

  DistributedStatesUnion& cur_ds_union();

  void set_cur_ds_union(const DistributedStatesUnion& ds_union);

  DistributedStates& inferred_cur_ds();

  const DistributedStatesHierarchy& ds_hierarchy() const {
    return _ds_hierarchy;
  }

  // executable graph now only need set_cur_ds_union() method
  /*
  void set_ds_hierarchy(const DistributedStatesHierarchy& ds_hierarchy) {
    _ds_hierarchy = ds_hierarchy;
  }
  */

  bool check_ds_hierarchy_equal(const DistributedStatesHierarchy& ds_hierarchy) {
    if (_ds_hierarchy.size() != ds_hierarchy.size()) {
      return false;
    }
    for (size_t i = 0; i < _ds_hierarchy.size(); i++) {
      if (_ds_hierarchy.get(i).size() != ds_hierarchy.get(i).size()) {
        return false;
      }
      for (size_t j = 0; j < _ds_hierarchy.get(i).size(); j++) {
        if (!_ds_hierarchy.get(i).get(j).check_equal(ds_hierarchy.get(i).get(j))) {
          return false;
        }
      }
    }
    return true;
  }

  // used in context parallel
  bool check_ds_hierarchy_equal_except_split(const DistributedStatesHierarchy& ds_hierarchy) {
    if (_ds_hierarchy.size() != ds_hierarchy.size()) {
      return false;
    }
    for (size_t i = 0; i < _ds_hierarchy.size(); i++) {
      if (_ds_hierarchy.get(i).size() != ds_hierarchy.get(i).size()) {
        return false;
      }
      for (size_t j = 0; j < _ds_hierarchy.get(i).size(); j++) {
        if (_ds_hierarchy.get(i).get(j).get_device_num() != ds_hierarchy.get(i).get(j).get_device_num()) {
          return false;
        }
        if (_ds_hierarchy.get(i).get(j).states(-1) != ds_hierarchy.get(i).get(j).states(-1)) {
          return false;
        }
        if (_ds_hierarchy.get(i).get(j).states(-2) != ds_hierarchy.get(i).get(j).states(-2)) {
          return false;
        }
      }
    }
    return true;
  }

  bool has_distributed_states() {
    if (has_cur_ds_union())
      return true;
    return false;
  }

  const DistributedStates& get_distributed_states() {
    return inferred_cur_ds();
  }

  // 去除ds union中的hetero dim
  // 获得局部placement group下的local ds
  DistributedStates get_local_distributed_states();

  void set_distributed_states(const DistributedStates& distributed_states) {
    inferred_cur_ds() = distributed_states;
  }

  const HTShape& global_shape() {
    if (has_global_shape()) {
      return _global_shape;
    }
    HTShape local_shape = shape();
    if (!has_distributed_states()) {
      return local_shape;
    }
    HTShape global_shape(local_shape.size());
    for (size_t d = 0; d < local_shape.size(); d++) {
      global_shape[d] = local_shape[d] * inferred_cur_ds().get_dim(d);
    }
    _global_shape = global_shape;
    return _global_shape;
  }

  bool has_placement_group() const {
    return _has_placement_group;
  }

  const DeviceGroupUnion& placement_group_union() const {
    HT_ASSERT(_has_placement_group)
      << "Ensure you MapToParallelDevices before calling placement_group_union for " << name();
    return _placement_group_union;
  }

  const DeviceGroup placement_group() const {
    HT_LOG_WARN << "It's better to use placement_group_union instead of placement_group";
    return _placement_group_union.all();
  }

  const DeviceGroup local_placement_group() const {
    HT_ASSERT(!_placement.is_undetermined())
      << "local_placement_group should be called only after instantiated the placement for " << name();
    return _placement_group_union.get(_placement);
  }

  size_t local_placement_group_idx() const {
    HT_ASSERT(!_placement.is_undetermined())
      << "local_placement_group_idx should be called only after instantiated the placement for " << name();
    return _placement_group_union.get_index(_placement);
  }

  // 支持placement还未instantiate时候进行推导
  // 主要用在未instantiate的variable和comm中
  size_t inferred_local_placement_group_idx() const;

  void set_placement_group_union(const DeviceGroupUnion& placement_group_union) {
    _placement_group_union = placement_group_union;
    _has_placement_group = true;
  }  

  bool symbolic() const {
    return _symbolic;
  }

  const SyShape& symbolic_shape() const {
    HT_ASSERT(_symbolic) 
      << name() << ": symbolic_shape() can only work after calling copy/set/init symbolic_shape";
    return _symbolic_shape;
  }

  SyShape symbolic_global_shape() {
    HT_ASSERT(_symbolic) 
      << name() << ": symbolic_global_shape() can only work after calling copy/set/init symbolic_shape";
    if (!has_distributed_states()) {
      return _symbolic_shape;
    }
    SyShape sy_global_shape;
    for (size_t d = 0; d < _symbolic_shape.size(); d++) {
      sy_global_shape.emplace_back(_symbolic_shape[d] * inferred_cur_ds().get_dim(d));
    }
    return sy_global_shape;
  }

  void set_shape(const HTShape& shape) {
    _meta.set_shape(shape);
    // no need to set _global_shape
    // since global_shape() method will automatically calc it
  }

  // only used in InferShape
  void set_temp_shape(const HTShape& temp_shape) {
    _temp_shape = temp_shape;
  }
  
  // copy constructor
  // don't know if it is a leaf
  void copy_symbolic_shape(const SyShape& symbolic_shape) {
    _symbolic = true;
    _symbolic_shape = symbolic_shape;
  }

  // leaf
  void set_symbolic_shape(const HTShape& shape) {
    _symbolic = true;
    set_HTShape_to_SyShape(shape, _symbolic_shape);
  }

  // leaf
  void init_symbolic_shape(const HTShape& shape) {
    _symbolic = true;
    for (auto x : _symbolic_shape) {
      x.reset();
    }
    set_HTShape_to_SyShape(shape, _symbolic_shape);
  }

  // leaf
  void init_symbolic_shape() {
    _symbolic = true;
    for (auto x : _symbolic_shape) {
      x.reset();
    }
    set_HTShape_to_SyShape(_meta.shape, _symbolic_shape);
  }

  void merge_strategy(Tensor& tensor);

  bool use_compute_suggested_hetero_id() const {
    return _use_compute_suggested_hetero_id;
  }

  void set_to_use_compute_suggested_hetero_id() {
    _use_compute_suggested_hetero_id = true;
  }

 protected:
  void AddConsumer(Operator& op);

  void DelConsumer(const Operator& op);

  // Walkaround methods to get the corresponding wrapper
  Tensor& get_self();

  const Tensor& get_self() const;
  
  const TensorIdentifier _ids;
  const TensorName _name;
  bool _requires_grad;
  NDArrayMeta _meta;
  OpRefList _consumers;
  bool _inform_graph_on_destruction;
  // deprecated: DistributedStatesList _distributed_states; // for multi ds deduce
  DistributedStatesHierarchy _ds_hierarchy; // for multi ds multi hetero-dp deduce
  DistributedStatesUnion _dummy_ds_union{}; // for deduce_states=False op outputs
  DistributedStates _dummy_ds{}; // for deduce_states=False op outputs
  HTShape _global_shape{}; // almost deprecated (only used in initialization)
  HTShape _temp_shape{}; // only used in InferShape
  bool _is_grad{false};
  bool _use_compute_suggested_hetero_id{true};

  // Used when the tensor's shape is not fixed
  bool _symbolic;
  SyShape _symbolic_shape;

  // Used only in exec graph
  // *We move the attribute of placement group and placement from DistributedStates to TensorDef
  bool _has_placement_group{false};
  DeviceGroupUnion _placement_group_union{};
  Device _placement{};

  std::optional<OpId> _contiguous_op_id{std::nullopt};
};

class Tensor : public shared_ptr_wrapper<TensorDef> {
 public:
  friend class Operator;
  friend class Graph;
  friend class ExecutableGraph;

  Tensor(TensorIdentifier ids, TensorName name, bool requires_grad,
         NDArrayMeta meta = {});

  Tensor() = default;

  ~Tensor();

 protected:
  size_t get_referrence_count() const {
    auto use_cnt_ = use_count();
    auto num_consumers_ = _ptr->num_consumers();
    HT_VALUE_ERROR_IF(use_cnt_ < (num_consumers_ + 1))
      << "Tensor " << _ptr->name() << " with " << num_consumers_ 
      << " consumers should have at least " << (num_consumers_ + 1) 
      << " use counts, but got " << use_cnt_;
    return use_cnt_ - (num_consumers_ + 1);
  }
  
  /******************************************************
   * Helper functions
   ******************************************************/ 
 public: 
  template <typename UnaryFunction>
  static void for_each_consumer(Tensor& tensor, UnaryFunction fn) {
    for (auto& op : tensor->_consumers)
      fn(op.get());
  }

  template <typename UnaryFunction>
  static void for_each_consumer(const Tensor& tensor, UnaryFunction fn) {
    for (const auto& op : tensor->_consumers)
      fn(op.get());
  }

  template <typename UnaryPredicate>
  static bool all_consumers_of(Tensor& tensor, UnaryPredicate pred) {
    for (auto& op : tensor->_consumers)
      if (!pred(op))
        return false;
    return true;
  }

  template <typename UnaryPredicate>
  static bool all_consumers_of(const Tensor& tensor, UnaryPredicate pred) {
    for (const auto& op : tensor->_consumers)
      if (!pred(op))
        return false;
    return true;
  }

  template <typename UnaryPredicate>
  static bool any_consumer_of(Tensor& tensor, UnaryPredicate pred) {
    for (auto& op : tensor->_consumers)
      if (pred(op))
        return true;
    return false;
  }

  template <typename UnaryPredicate>
  static bool any_consumer_of(const Tensor& tensor, UnaryPredicate pred) {
    for (const auto& op : tensor->_consumers)
      if (pred(op))
        return true;
    return false;
  }
};

std::ostream& operator<<(std::ostream&, const Tensor&);

} // namespace graph
} // namespace hetu
