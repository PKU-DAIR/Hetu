#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/graph.h"
#include "hetu/graph/init/initializer.h"
#include "hetu/graph/distributed_states.h"

namespace hetu {
namespace graph {

namespace {
inline DataType _InferDataType(const NDArray& data, DataType dtype) {
  return dtype != kUndeterminedDataType
    ? dtype
    : (data.is_defined() ? data->dtype() : kUndeterminedDataType);
}
} // namespace

// class VariableOpImpl;
// class VariableOp;
// class ParallelVariableOpImpl;
// class ParallelVariableOp;

class VariableOpImpl : public OpInterface {
 protected:
  VariableOpImpl(OpType&& type, const Initializer& init, HTShape shape,
                 DataType dtype, bool requires_grad)
  : OpInterface(std::move(type)),
    _init(init.copy()),
    _shape(std::move(shape)),
    _dtype(dtype),
    _device(kUndeterminedDevice),
    _requires_grad(requires_grad) {
    _check_init();
  }

  VariableOpImpl(OpType&& type, NDArray provided_data, bool copy_provided_data,
                 DataType dtype, bool requires_grad)
  : OpInterface(std::move(type)),
    _provided_data(std::move(provided_data)),
    _copy_provided_data(copy_provided_data),
    _shape(provided_data->shape()),
    _dtype(_InferDataType(provided_data, dtype)),
    _device(provided_data->device()),
    _requires_grad(requires_grad) {
    _check_init();
  }

  void _check_init() {
    HT_VALUE_ERROR_IF(std::find(_shape.begin(), _shape.end(), -1) !=
                      _shape.end())
      << "Shape of " << _type << " is undetermined: " << _shape;
    HT_VALUE_ERROR_IF(_dtype == kUndeterminedDataType)
      << "Data type of " << _type << " is undetermined";
  }

 public:
  VariableOpImpl(const Initializer& init, HTShape shape,
                 DataType dtype = kFloat32, bool requires_grad = false)
  : VariableOpImpl(quote(VariableOp), init, std::move(shape), dtype,
                   requires_grad) {}

  VariableOpImpl(NDArray provided_data, bool copy_provided_data, DataType dtype,
                 bool requires_grad)
  : VariableOpImpl(quote(VariableOp), provided_data, copy_provided_data, dtype,
                   requires_grad) {}

  uint64_t op_indicator() const noexcept override {
    return VARIABLE_OP;
  }

 protected:
  std::vector<NDArrayMeta>
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    return {NDArrayMeta().set_shape(shape()).set_dtype(dtype()).set_device(device())};
  }

  TensorList DoGradient(Operator& op,
                        const TensorList& grad_outputs) const override {
    return {Tensor()};
  }

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override {
    return {shape()};
  }

  NDArrayList DoAllocOutputs(Operator& op, const NDArrayList& inputs,
                             RuntimeContext& runtime_ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override {}

 public:
  bool operator==(const OpInterface& rhs) const override {
    return false;
  }

  const Initializer& initializer() const {
    return *_init;
  }

  const HTShape& shape() const {
    return _shape;
  }

  DataType dtype() const {
    return _dtype;
  }

  Device device() const {
    return _device;
  }

  bool requires_grad() const {
    return _requires_grad;
  }

  void set_requires_grad(bool requires_grad) {
    _requires_grad = requires_grad;
  }

 protected:
  std::shared_ptr<Initializer> _init;
  NDArray _provided_data;
  bool _copy_provided_data;
  HTShape _shape;
  DataType _dtype;
  Device _device;
  bool _requires_grad;
};

class ParallelVariableOpImpl : public OpInterface {
 public:
  ParallelVariableOpImpl(const Initializer& init, HTShape global_shape, 
                         const DistributedStatesHierarchy& ds_hierarchy, std::vector<int64_t> local_idx,
                         DataType dtype = kFloat32, bool requires_grad = false,
                         ParameterDict parameter_dict = {})
  : OpInterface(quote(ParallelVariableOp)), _init(init.copy()), 
    _global_shape(std::move(global_shape)), _local_idx(std::move(local_idx)), 
    _dtype(dtype), _ds_hierarchy(ds_hierarchy), _requires_grad(requires_grad),
    _parameter_dict(parameter_dict) {
      _local_shape = get_local_shape(_global_shape, _ds_hierarchy.get_default_ds()); // deduce local shape default by ds[0]
    }

  ParallelVariableOpImpl(NDArray provided_data, bool copy_provided_data, 
                         const DistributedStatesHierarchy& ds_hierarchy, DataType dtype, bool requires_grad,
                         ParameterDict parameter_dict = {}) 
  : OpInterface(quote(ParallelVariableOp)), _provided_data(provided_data),
    _copy_provided_data(copy_provided_data), _local_shape(provided_data->shape()),
    _dtype(_InferDataType(provided_data, dtype)), _ds_hierarchy(ds_hierarchy), _requires_grad(requires_grad),
    _parameter_dict(parameter_dict) {
      _global_shape = get_global_shape(_local_shape, _ds_hierarchy.get_default_ds());
    }

  // todo: if need provide multi shape for multi ds?
  ParallelVariableOpImpl(NDArrayList multi_provided_data, bool copy_provided_data, 
                         const DistributedStatesHierarchy& ds_hierarchy, DataType dtype, bool requires_grad,
                         ParameterDict parameter_dict = {}) 
  : OpInterface(quote(ParallelVariableOp)), _multi_provided_data(std::move(multi_provided_data)),
    _copy_provided_data(copy_provided_data), _local_shape(_multi_provided_data[0]->shape()), // use the first strategy shape
    _dtype(_InferDataType(_multi_provided_data[0], dtype)), _ds_hierarchy(ds_hierarchy), _requires_grad(requires_grad),
    _parameter_dict(parameter_dict) {
      _global_shape = get_global_shape(_local_shape, _ds_hierarchy.get_default_ds());
    }    

  uint64_t op_indicator() const noexcept override {
    return VARIABLE_OP;
  }  

 protected:
  HTShape get_global_shape(HTShape& local_shape, const DistributedStates& ds) {
    if (!_global_shape.empty())
      return _global_shape;
    HTShape shape(local_shape.size());
    for (size_t d = 0; d < local_shape.size(); d++) {
      shape[d] = local_shape[d] * ds.get_dim(d);
    }
    return shape;
  }

  HTShape get_local_shape(HTShape& global_shape, const DistributedStates& ds) {
    if (!_local_shape.empty())
      return _local_shape;
    HTShape shape(global_shape.size());
    for (size_t d = 0; d < global_shape.size(); d++) {
      shape[d] = global_shape[d] / ds.get_dim(d);
    }
    return shape;    
  }

  std::vector<NDArrayMeta>
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override {
    const auto& cur_ds_union = _ds_hierarchy.get(Graph::GetGraph(Graph::cur_graph_ctx()).CUR_STRATEGY_ID);
    // workaround 
    // 这里不得不使用CUR_HETERO_ID（就算外部没有进行USE_HETERO_ID）
    // 在define graph中，该值一定是0
    // 在exec graph中，该值会在MakeOpInner时被合理地设置
    const auto& cur_ds = cur_ds_union.is_hetero() ? cur_ds_union.get(Graph::GetGraph(Graph::cur_graph_ctx()).CUR_HETERO_ID) : cur_ds_union.get(0);
    HT_ASSERT(!_global_shape.empty())
      << "global shape should be initialized";
    HTShape cur_local_shape(_global_shape.size());
    for (size_t d = 0; d < _global_shape.size(); d++) {
      cur_local_shape[d] = _global_shape[d] / cur_ds.get_dim(d);
    } 
    return {NDArrayMeta().set_shape(cur_local_shape).set_dtype(dtype())};
  }                     

  TensorList DoGradient(Operator& op,
                        const TensorList& grad_outputs) const override {
    return {Tensor()};
  }

  void DoSpecialMergeStrategy(Operator& op, Operator& another_op) {
    HT_ASSERT(is_variable_op(op) && is_variable_op(another_op))
      << "two ops should both be variables";
    auto& another_op_impl = dynamic_cast<ParallelVariableOpImpl&>(another_op->body());
    _multi_provided_data.insert(_multi_provided_data.end(), another_op_impl._multi_provided_data.begin(), another_op_impl._multi_provided_data.end());
    _local_idx.insert(_local_idx.end(), another_op_impl._local_idx.begin(), another_op_impl._local_idx.end());
    for (const auto& ds_union : another_op_impl._ds_hierarchy.raw_data()) {
      _ds_hierarchy.add(ds_union);
    }
    HT_ASSERT((_multi_provided_data.size() == 0 || _multi_provided_data.size() == op->graph().NUM_STRATEGY)
              && _local_idx.size() == op->graph().NUM_STRATEGY
              && _ds_hierarchy.size() == op->graph().NUM_STRATEGY)
      << "size mismatch";
  }

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override {
    auto cur_ds = op->output(0)->inferred_cur_ds();
    HT_ASSERT(!_global_shape.empty())
      << "global shape should be initialized";
    HTShape cur_local_shape(_global_shape.size());
    for (size_t d = 0; d < _global_shape.size(); d++) {
      cur_local_shape[d] = _global_shape[d] / cur_ds.get_dim(d);
    } 
    return {cur_local_shape};
  }

  NDArrayList DoAllocOutputs(Operator& op, const NDArrayList& inputs,
                             RuntimeContext& runtime_ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override {}  

 public:
  bool operator==(const OpInterface& rhs) const override {
    return false;
  }

  const Initializer& initializer() const {
    return *_init;
  }

  const HTShape& global_shape() const {
    return _global_shape;
  }

  const HTShape& local_shape() const {
    return _local_shape;
  }

  // Used for parallel plan changing test case
  void set_ds(const DistributedStates& ds) {
    HT_ASSERT(_ds_hierarchy.size() == 1)
      << "ParallelVariableOp set ds can only used in exec graph";
    _ds_hierarchy.get(0).get(0)= ds;
    HT_ASSERT(_local_shape.size() == _global_shape.size())
      << "something wrong, the local shape and global shape dims are mismatched";
    for (size_t d = 0; d < _global_shape.size(); d++) {
      _local_shape[d] = _global_shape[d] / ds.get_dim(d);
    }
  }

  // Used for applying zero
  void set_ds_hierarchy(const DistributedStatesHierarchy& ds_hierarchy) {
    _ds_hierarchy = ds_hierarchy;
  }

  DataType dtype() const {
    return _dtype;
  }

  bool requires_grad() const {
    return _requires_grad;
  }

  void set_requires_grad(bool requires_grad) {
    _requires_grad = requires_grad;
  }

  ParameterDict get_parameter_dict() {
    return _parameter_dict;    
  }


  std::shared_ptr<Initializer> _init;
  mutable NDArray _provided_data; // local_data
  mutable NDArrayList _multi_provided_data; // local_data
  bool _copy_provided_data;  
  HTShape _global_shape;
  HTShape _local_shape;
  DistributedStatesHierarchy _ds_hierarchy;
  // _local_idx seems useless
  std::vector<int64_t> _local_idx; // _local_idx only be assigned when op is in pipeline device_group, and return local_device index in the device_group
  DataType _dtype;
  bool _requires_grad;
  ParameterDict _parameter_dict;
};

Tensor MakeVariableOp(const Initializer& init, HTShape shape, 
                      DataType dtype = kFloat32, bool requires_grad = false, 
                      const DistributedStatesHierarchy& ds_hierarchy = DistributedStatesHierarchy(), 
                      OpMeta op_meta = OpMeta());

Tensor MakeVariableOp(NDArray provided_data, bool copy_provided_data = false,
                      DataType dtype = kUndeterminedDataType, bool requires_grad = false, 
                      const DistributedStatesHierarchy& ds_hierarchy = DistributedStatesHierarchy(), 
                      OpMeta op_meta = OpMeta());

Tensor MakeParameterOp(const Initializer& init, HTShape shape,
                       DataType dtype = kFloat32, bool requires_grad = false, 
                       const DistributedStatesHierarchy& ds_hierarchy = DistributedStatesHierarchy(), 
                       OpMeta op_meta = OpMeta());

Tensor MakeParameterOp(NDArray provided_data, bool copy_provided_data = false, 
                       DataType dtype = kUndeterminedDataType, bool requires_grad = false, 
                       const DistributedStatesHierarchy& ds_hierarchy = DistributedStatesHierarchy(),
                       OpMeta op_meta = OpMeta());

Tensor MakeParallelVariableOp(const Initializer& init, HTShape global_shape, 
                              const DistributedStatesHierarchy& ds_hierarchy, std::vector<int64_t> local_idx={-1},
                              DataType dtype = kFloat32, bool requires_grad = false,
                              ParameterDict parameter_dict = {}, 
                              OpMeta op_meta = OpMeta());

Tensor MakeParallelVariableOp(NDArray provided_data, const DistributedStatesHierarchy& ds_hierarchy, 
                              bool copy_provided_data = false, DataType dtype = kUndeterminedDataType, 
                              bool requires_grad = false, ParameterDict parameter_dict = {}, 
                              OpMeta op_meta = OpMeta());

Tensor MakeParallelVariableOp(NDArrayList multi_provided_data, DistributedStatesHierarchy ds_hierarchy, 
                              bool copy_provided_data = false, DataType dtype = kUndeterminedDataType, 
                              bool requires_grad = false,
                              ParameterDict parameter_dict = {}, 
                              OpMeta op_meta = OpMeta());

Tensor MakeParallelParameterOp(const Initializer& init, HTShape global_shape, 
                               const DistributedStatesHierarchy& ds_hierarchy, std::vector<int64_t> local_idx={-1},
                               DataType dtype = kFloat32, bool requires_grad = false,
                               ParameterDict parameter_dict = {}, 
                               OpMeta op_meta = OpMeta());
// provided_data is local_data!
Tensor MakeParallelParameterOp(NDArray provided_data, const DistributedStatesHierarchy& ds_hierarchy, 
                               bool copy_provided_data = false, DataType dtype = kUndeterminedDataType, 
                               bool requires_grad = false, ParameterDict parameter_dict = {},  
                               OpMeta op_meta = OpMeta());

// provided_data is local_data!
Tensor MakeParallelParameterOp(NDArrayList multi_provided_data, DistributedStatesHierarchy ds_hierarchy, 
                               bool copy_provided_data = false, DataType dtype = kUndeterminedDataType, 
                               bool requires_grad = false, ParameterDict parameter_dict = {}, 
                               OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu
