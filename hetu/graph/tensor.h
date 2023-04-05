#pragma once

#include "hetu/common/macros.h"
#include "hetu/core/ndarray.h"
#include "hetu/graph/common.h"

namespace hetu {
namespace graph {

struct TensorIdentifier {
  GraphId graph_id;
  OpId producer_id;
  int32_t output_id;
  TensorId tensor_id;
};

class TensorDef : public shared_ptr_target {
 protected:
  friend class OperatorDef;
  friend class Operator;
  friend class Tensor;
  friend class Graph;
  friend class ExecutableGraph;
  struct constrcutor_access_key {};

 public:
  TensorDef(const constrcutor_access_key&, TensorIdentifier ids,
            TensorName name, NDArrayMeta meta);

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

  const Graph& graph() const;

  Graph& graph();
  
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
    return _meta.shape;
  }

  int64_t shape(size_t axis) const {
    return _meta.shape[axis];
  }

  const HTStride& stride() const {
    return _meta.stride;
  }

  int64_t stride(size_t axis) const {
    return _meta.stride[axis];
  }

  bool has_shape() const {
    return _meta.shape.size() > 0;
  }

  const Device& placement() const noexcept {
    return device();
  }

  void set_placement(const Device& p) {
    _meta.set_device(p);
  }

  const bool require_grad() const noexcept {
    return _require_grad;
  }

  void set_require_grad(bool new_require_grad) {
    _require_grad = new_require_grad;
  }

  NDArray get_or_compute();

 protected:
  void AddConsumer(Operator& op);

  void DelConsumer(const Operator& op);

  // Walkaround methods to get the corresponding wrapper
  Tensor& get_self();

  const Tensor& get_self() const;
  
  const TensorIdentifier _ids;
  const TensorName _name;
  NDArrayMeta _meta;
  OpRefList _consumers;
  bool _require_grad;
  bool _inform_graph_on_destruction;
};

class Tensor : public shared_ptr_wrapper<TensorDef> {
 public:
  friend class Operator;
  friend class Graph;
  friend class ExecutableGraph;

  Tensor(TensorIdentifier ids, TensorName name, NDArrayMeta meta = {});

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
};

std::ostream& operator<<(std::ostream&, const Tensor&);

} // namespace graph
} // namespace hetu
