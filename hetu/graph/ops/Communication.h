#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/graph.h"
#include "hetu/graph/utils/tensor_utils.h"
#include "hetu/graph/distributed_states.h"
#include "hetu/impl/communication/comm_group.h"

namespace hetu {
namespace graph {

class CommOpImpl;
class AllReduceOpImpl;
class P2PSendOpImpl;
class P2PRecvOpImpl;
class BatchedISendIRecvOpImpl;
class AllGatherOpImpl;
class ReduceScatterOpImpl;
class ScatterOpImpl;

class CommOpInfo {
 public:
  bool is_empty;
  DeviceGroupUnion src_group_union;
  DeviceGroupUnion dst_group_union;
  DistributedStatesUnion src_ds_union;
  DistributedStatesUnion dst_ds_union;
  size_t src_union_idx;
  size_t dst_union_idx;
  int32_t placement_pos;
  DeviceGroup src_group;
  DeviceGroup dst_group;
  DistributedStates src_ds;
  DistributedStates dst_ds;
  DistributedStates local_src_ds;
  DistributedStates local_dst_ds;
  
  CommOpInfo()
    : is_empty(true) {}

  CommOpInfo(const DeviceGroupUnion& src_group_union_, const DeviceGroupUnion& dst_group_union_,
             const DistributedStatesUnion& src_ds_union_, const DistributedStatesUnion& dst_ds_union_,
             size_t src_union_idx_, size_t dst_union_idx_, int32_t placement_pos_)
    : is_empty(false),
      src_group_union(src_group_union_),
      dst_group_union(dst_group_union_),
      src_ds_union(src_ds_union_),
      dst_ds_union(dst_ds_union_),
      src_union_idx(src_union_idx_),
      dst_union_idx(dst_union_idx_),
      placement_pos(placement_pos_) {
    src_group = src_group_union.get(src_union_idx);
    dst_group = dst_group_union.get(dst_union_idx);
    src_ds = src_ds_union.get(src_union_idx);
    dst_ds = dst_ds_union.get(dst_union_idx);
    local_src_ds = src_ds_union.get_local(src_union_idx);
    local_dst_ds = dst_ds_union.get_local(dst_union_idx);
  }
};

std::ostream& operator<<(std::ostream& os, const CommOpInfo& info);

class CommOpImpl final: public OpInterface {
 public:
  // dst_group is only for exec graph instantiate, at this time there will only exists one ds strategy
  // also, dst_ds_hierarchy should contains only one dst_ds_union for comm op which created in exec graph instantiate
  CommOpImpl(DistributedStatesHierarchy dst_ds_hierarchy, DeviceGroupHierarchy dst_group_hierarchy = DeviceGroupHierarchy(), 
             ReductionType red_type = kSUM) : OpInterface(quote(CommOp)), 
             _dst_ds_hierarchy(std::move(dst_ds_hierarchy)), _dst_group_hierarchy(std::move(dst_group_hierarchy)), _red_type(red_type) {
    auto& graph = Graph::GetGraph(Graph::cur_graph_ctx());
    if (graph.type() == GraphType::DEFINE_AND_RUN) {
      HT_ASSERT(_dst_ds_hierarchy.size() == graph.NUM_STRATEGY)
        << "Size of dst ds hierarchy should be equal to the strategy num in define graph";
    } else if (graph.type() == GraphType::EXECUTABLE) {
      HT_ASSERT(_dst_ds_hierarchy.size() == 1)
        << "Size of dst ds hierarchy should be equal to 1 in exec graph"; 
    } else {
      HT_NOT_IMPLEMENTED << "Currently not support";
    }
    if (_dst_group_hierarchy.size() != 0) {
      if (graph.type() == GraphType::DEFINE_AND_RUN) {
        HT_ASSERT(_dst_group_hierarchy.size() == graph.NUM_STRATEGY)
          << "Size of dst group hierarchy should be equal to the strategy num in define graph";
      } else if (graph.type() == GraphType::EXECUTABLE) {
        HT_ASSERT(_dst_group_hierarchy.size() == 1)
          << "Size of dst group hierarchy should be equal to 1 in exec graph"; 
      } else {
        HT_NOT_IMPLEMENTED << "Currently not support";
      }
    }
  }      

  uint64_t op_indicator() const noexcept override {
    return COMM_OP;
  }  

 protected:
  bool DoMapToParallelDevices(Operator& op,
                              const DeviceGroupUnion& pg_union) const override;

  void DoSpecialMergeStrategy(Operator& op, Operator& another_op) override;

  bool DoInstantiate(Operator& op, const Device& placement,
                     StreamIndex stream_index) const override;                              

  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override;

  void DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                      const OpMeta& op_meta,
                      const InstantiationContext& inst_ctx) const override;

  void DoDeduceHeterProp(const std::vector<int32_t>& inputs_hetero_dim,
                         TensorList& outputs, const OpMeta& op_meta,
                         const InstantiationContext& inst_ctx) const override;  

  TensorList DoGradient(Operator& op,
                        const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const {}

 public: 
  const DistributedStatesHierarchy& dst_ds_hierarchy() {
    return _dst_ds_hierarchy;
  }

  const DistributedStatesUnion& get_dst_ds_union(Operator& op) const {
    auto& graph = op->graph();
    HT_ASSERT(_dst_ds_hierarchy.size() == 1 || _dst_ds_hierarchy.size() == graph.NUM_STRATEGY)
      << "CommOp get dst ds error!";
    if (_dst_ds_hierarchy.size() == 1) { // for comm op created in exec_graph, without multi ds
      return _dst_ds_hierarchy.get(0);
    } else { // for comm op created in define_and_run_graph, with multi ds
      return _dst_ds_hierarchy.get(graph.CUR_STRATEGY_ID);
    }
  }

  const DistributedStates& get_dst_distributed_states(Operator& op) const {
    auto& graph = op->graph();
    HT_ASSERT(_dst_ds_hierarchy.size() == 1 || _dst_ds_hierarchy.size() == graph.NUM_STRATEGY)
      << "CommOp get dst ds error!";
    if (graph.USE_HETERO_ID) {
      if (_dst_ds_hierarchy.size() == 1) { // for comm op created in exec_graph, without multi ds
        return _dst_ds_hierarchy.get(0).is_hetero() ? _dst_ds_hierarchy.get(0).get(graph.CUR_HETERO_ID) : _dst_ds_hierarchy.get(0).get(0);
      } else { // for comm op created in define_and_run_graph, with multi ds
        // HT_LOG_DEBUG << "get " << graph.CUR_HETERO_ID << " from " << _dst_ds_hierarchy.get(graph.CUR_STRATEGY_ID).ds_union_info();
        const auto& ds_union = _dst_ds_hierarchy.get(graph.CUR_STRATEGY_ID);
        return ds_union.is_hetero() ? ds_union.get(graph.CUR_HETERO_ID) : ds_union.get(0);
      }
    } else {
      auto idx = op->output(0)->inferred_local_placement_group_idx();
      if (_dst_ds_hierarchy.size() == 1) { // for comm op created in exec_graph, without multi ds
        return _dst_ds_hierarchy.get(0).is_hetero() ? _dst_ds_hierarchy.get(0).get(idx) : _dst_ds_hierarchy.get(0).get(0);
      } else { // for comm op created in define_and_run_graph, with multi ds
        return _dst_ds_hierarchy.get(graph.CUR_STRATEGY_ID).get(idx);
      }
    }
  }

  // deprecated: used for parallel plan changing test case
  DistributedStates& set_dst_distributed_states(const DistributedStates& ds) {
    HT_ASSERT(_dst_ds_hierarchy.size() == 1)
      << "CommOp set dst ds can only used in exec graph";
    _dst_ds_hierarchy.get(0).get(Graph::GetGraph(Graph::cur_graph_ctx()).CUR_HETERO_ID) = ds;
  }

  ReductionType reduction_type() const {
    return _red_type;
  }
  
  const DeviceGroupHierarchy& dst_group_hierarchy() {
    return _dst_group_hierarchy;
  }
  
  // only used in exec graph
  const DeviceGroupUnion& get_src_group_union(Operator& op) const {
    HT_ASSERT(op->graph().type() == GraphType::EXECUTABLE && op->input(0)->has_placement_group())
      << "get_src_group_union can only used in exec graph and op should have placement group";
    return op->input(0)->placement_group_union();
  }

  // only used in exec graph
  const DeviceGroupUnion& get_dst_group_union(Operator& op) const {
    HT_ASSERT(op->graph().type() == GraphType::EXECUTABLE && op->output(0)->has_placement_group())
      << "get_dst_group_union can only used in exec graph and op should have placement group";
    return op->output(0)->placement_group_union();
  }

  CommOpInfo get_comm_info(Operator& op, const Device& inferred) const;

  uint64_t get_comm_type(Operator& op, const Device& inferred, const CommOpInfo& comm_info = {});

  DeviceGroup get_devices_by_dim(Operator& op, int32_t dim) const; 

  std::tuple<size_t, std::vector<DeviceGroupList>> get_split_comm_groups_list(Operator& op, const DeviceGroupUnion& dg_union,
                                                                              const DistributedStatesUnion& ds_union) const;

 protected:
  uint64_t _comm_type{UNKNOWN_OP};
  // DistributedStates _dst_ds;
  DistributedStatesHierarchy _dst_ds_hierarchy;
  DeviceGroupHierarchy _dst_group_hierarchy;
  ReductionType _red_type{kNONE}; // only used for AllReduce, ReduceScatter
};

Tensor MakeCommOp(Tensor input, DistributedStatesHierarchy dst_ds_hierarchy, 
                  ReductionType red_type, OpMeta op_meta = OpMeta());

Tensor MakeCommOp(Tensor input, DistributedStatesHierarchy dst_ds_hierarchy,
                  const std::string& mode, OpMeta op_meta = OpMeta());

Tensor MakeCommOp(Tensor input, DistributedStatesHierarchy dst_ds_hierarchy, 
                  DeviceGroupHierarchy dst_group_hierarchy, bool is_pipeline_op=true, OpMeta op_meta = OpMeta());

Tensor MakeCommOp(Tensor input, DistributedStatesHierarchy dst_ds_hierarchy, 
                  OpMeta op_meta = OpMeta());

// single strategy
Tensor MakeCommOp(Tensor input, DistributedStates dst_ds, 
                  ReductionType red_type, OpMeta op_meta = OpMeta());

Tensor MakeCommOp(Tensor input, DistributedStates dst_ds,
                  const std::string& mode, OpMeta op_meta = OpMeta());

Tensor MakeCommOp(Tensor input, DistributedStates dst_ds, DeviceGroup dst_group,
                  OpMeta op_meta = OpMeta());

Tensor MakeCommOp(Tensor input, DistributedStates dst_ds, OpMeta op_meta = OpMeta());

class AllReduceOpImpl final : public OpInterface {
 public:
  AllReduceOpImpl(DeviceGroup comm_group, ReductionType red_type = kSUM, bool inplace = false)
  : OpInterface(quote(AllReduceOp)), _comm_group(comm_group), _red_type(red_type), _inplace(inplace) {
    HT_ASSERT(_comm_group.num_devices() >= 2)
             << "AllReduce requires two or more comm devices. Got " << _comm_group;
  }

  inline uint64_t op_indicator() const noexcept override {
    return _inplace ? ALL_REDUCE_OP | INPLACE_OP : ALL_REDUCE_OP;
  }

  inline bool inplace() const {
    return _inplace;
  }

  ReductionType reduction_type() const {
    return _red_type;
  }

 protected:
  bool DoMapToParallelDevices(Operator& op,
                              const DeviceGroupUnion& pg_union) const override;

  bool DoInstantiate(Operator& op, const Device& placement,
                     StreamIndex stream_index) const override;

  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;

  NDArrayList DoCompute(Operator& op,
                        const NDArrayList& inputs,
                        RuntimeContext& ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;
  
  bool _inplace{false};

 public:
  const DeviceGroup& comm_group() const {
    return _comm_group;
  }

 protected:
  DeviceGroup _comm_group;
  ReductionType _red_type{kNONE};
};

Tensor MakeAllReduceOp(Tensor input, DeviceGroup comm_group, 
                       bool inplace = false, OpMeta op_meta = OpMeta());

Tensor MakeAllReduceOp(Tensor input, DeviceGroup comm_group, ReductionType red_type, 
                       bool inplace = false, OpMeta op_meta = OpMeta());

class P2PSendOpImpl final : public OpInterface {
 public:
  P2PSendOpImpl(DeviceGroup dst_group, int dst_device_index = -1, std::vector<int> all_ranks = {})
  : OpInterface(quote(P2PSendOp)), _dst_group(std::move(dst_group)), 
    _dst_device_index(dst_device_index), _all_ranks(std::move(all_ranks)) {
    HT_ASSERT(!_dst_group.empty())
      << "Please provide the \"dst_group\" argument to indicate "
      << "the destination devices for P2PSend";
  }

  uint64_t op_indicator() const noexcept override {
    return PEER_TO_PEER_SEND_OP;
  }

 protected:
  bool DoMapToParallelDevices(Operator& op,
                              const DeviceGroupUnion& pg_union) const override;

  bool DoInstantiate(Operator& op, const Device& placement,
                     StreamIndex stream_index) const override;

  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;  

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;
                        
 public:
  const DeviceGroup& dst_group() const {
    return _dst_group;
  }

  int dst_device_index() const {
    return _dst_device_index;
  }  

 protected:
  DeviceGroup _dst_group;
  int _dst_device_index{-1};
  std::vector<int> _all_ranks;
};

Tensor MakeP2PSendOp(Tensor input, DeviceGroup dst_group, 
                     int dst_device_index = -1, std::vector<int> all_ranks = {}, OpMeta op_meta = OpMeta());

class P2PRecvOpImpl final : public OpInterface {
 public:
  // symbolic shape constructor
  P2PRecvOpImpl(DeviceGroup src_group, DataType dtype,
                SyShape shape, int src_device_index = -1, std::vector<int> all_ranks = {})
  : OpInterface(quote(P2PRecvOp)), _src_group(std::move(src_group)), _dtype(dtype),
                _shape(std::move(shape)), _src_device_index(src_device_index), _all_ranks(std::move(all_ranks)) {
    HT_ASSERT(!_src_group.empty())
      << "Please provide the \"src_group\" argument to indicate "
      << "the source devices for P2PRecv";
    HT_ASSERT(!_shape.empty())
      << "P2P RecvOp require determined tensor shape to recv. Got empty shape param!";
    HT_ASSERT(_dtype != kUndeterminedDataType)
      << "Please specify data type for P2P communication";
  }
  // fixed shape constructor
  P2PRecvOpImpl(DeviceGroup src_group, DataType dtype,
                HTShape shape, int src_device_index = -1, std::vector<int> all_ranks = {})
  : OpInterface(quote(P2PRecvOp)), _src_group(std::move(src_group)), _dtype(dtype),
                _shape(shape.begin(), shape.end()), _src_device_index(src_device_index), _all_ranks(std::move(all_ranks)) {
    HT_ASSERT(!_src_group.empty())
      << "Please provide the \"src_group\" argument to indicate "
      << "the source devices for P2PRecv";
    HT_ASSERT(!_shape.empty())
      << "P2P RecvOp require determined tensor shape to recv. Got empty shape param!";
    HT_ASSERT(_dtype != kUndeterminedDataType)
      << "Please specify data type for P2P communication";
  }

  uint64_t op_indicator() const noexcept override {
    return PEER_TO_PEER_RECV_OP;
  }

 protected:
  bool DoMapToParallelDevices(Operator& op,
                              const DeviceGroupUnion& pg_union) const override;

  bool DoInstantiate(Operator& op, const Device& placement,
                     StreamIndex stream_index) const override;

  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;

 public:
  const DeviceGroup& src_group() const {
    return _src_group;
  }

  int src_device_index() {
    return _src_device_index;
  } 

  HTShape get_shape() const {
    return get_HTShape_from_SyShape(_shape);
  }

  const SyShape& get_symbolic_shape() const {
    return _shape;
  }

 protected:
  DeviceGroup _src_group;
  int _src_device_index{-1};
  DataType _dtype;
  SyShape _shape;  
  std::vector<int> _all_ranks;         
};

Tensor MakeP2PRecvOp(DeviceGroup src_group, DataType dtype,
                     HTShape shape, int src_device_index = -1, 
                     std::vector<int> all_ranks = {}, OpMeta op_meta = OpMeta());

// symbolic shape
Tensor MakeP2PRecvOp(DeviceGroup src_group, DataType dtype,
                     SyShape shape, int src_device_index = -1, 
                     std::vector<int> all_ranks = {}, OpMeta op_meta = OpMeta());     

class BatchedISendIRecvOpImpl final : public OpInterface {
 public:
  // symbolic shape constructor
  BatchedISendIRecvOpImpl(std::vector<Device> dst_devices, 
                          SyShapeList outputs_shape,
                          std::vector<Device> src_devices, 
                          std::vector<Device> comm_devices,
                          DataType dtype)
  : OpInterface(quote(BatchedISendIRecvOp)), _dst_devices(std::move(dst_devices)), 
  _outputs_shape(std::move(outputs_shape)), _src_devices(std::move(src_devices)), 
  _comm_devices(std::move(comm_devices)), _dtype(dtype) {}
  // fixed shape constructor
  BatchedISendIRecvOpImpl(std::vector<Device> dst_devices, 
                          HTShapeList outputs_shape,
                          std::vector<Device> src_devices, 
                          std::vector<Device> comm_devices,
                          DataType dtype)
  : OpInterface(quote(BatchedISendIRecvOp)), _dst_devices(std::move(dst_devices)), 
  _src_devices(std::move(src_devices)), 
  _comm_devices(std::move(comm_devices)), _dtype(dtype) {
    _outputs_shape.reserve(outputs_shape.size());
    for (auto& output_shape : outputs_shape) {
      _outputs_shape.emplace_back(SyShape(output_shape.begin(), output_shape.end()));
    }
  }

  uint64_t op_indicator() const noexcept override {
    return BATCHED_ISEND_IRECV_OP;
  }

 public:
  void print_mesg(Operator& op) {
    std::ostringstream os;
    os << "dst devices =";
    for (auto& d : _dst_devices) {
      os << " device_" << hetu::impl::comm::DeviceToWorldRank(d);
    }
    os << "src devices =";
    for (auto& s : _src_devices) {
      os << " device_" << hetu::impl::comm::DeviceToWorldRank(s);
    }
    HT_LOG_DEBUG << hetu::impl::comm::GetLocalDevice() 
                 << ": BatchedISendIRecvOp definition: " << op->name() << ": " << os.str();    
  }

  const std::vector<Device>& src_devices() const {
    return _src_devices;
  }

  std::vector<Device>& src_devices() {
    return _src_devices;
  }  

  const std::vector<Device>& dst_devices() const {
    return _dst_devices;
  }

  std::vector<Device>& dst_devices() {
    return _dst_devices;
  }

  const std::vector<Device>& comm_devices() const {
    return _comm_devices;
  }

  std::vector<Device>& comm_devices() {
    return _comm_devices;
  }

  HTShapeList get_outputs_shape() const {
    HTShapeList outputs_shape;
    outputs_shape.reserve(outputs_shape.size());
    for (auto& output_shape : _outputs_shape) {
      outputs_shape.emplace_back(get_HTShape_from_SyShape(output_shape));
    }
    return outputs_shape;
  }

  const SyShapeList& get_symbolic_outputs_shape() const {
    return _outputs_shape;
  }

 protected:
  bool DoInstantiate(Operator& op, const Device& placement,
                     StreamIndex stream_index) const override;
                    
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override; 

  HTShapeList DoInferDynamicShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;   

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;

 protected:
  std::vector<Device> _dst_devices; 
  std::vector<Device> _src_devices;
  std::vector<Device> _comm_devices;
  SyShapeList _outputs_shape;
  DataType _dtype;
};

Tensor MakeBatchedISendIRecvOp(TensorList inputs, 
                               std::vector<Device> dst_devices, 
                               HTShapeList outputs_shape, 
                               std::vector<Device> src_devices, 
                               std::vector<Device> comm_devices, 
                               DataType dtype, OpMeta op_meta = OpMeta());

// symbolic shape
Tensor MakeBatchedISendIRecvOp(TensorList inputs, 
                               std::vector<Device> dst_devices, 
                               SyShapeList outputs_shape, 
                               std::vector<Device> src_devices, 
                               std::vector<Device> comm_devices, 
                               DataType dtype, OpMeta op_meta = OpMeta());

class AllGatherOpImpl final : public OpInterface {
 public:
  AllGatherOpImpl(const DeviceGroup& comm_group, int32_t gather_dim = 0)
  : OpInterface(quote(AllGatherOp)), _gather_dim(gather_dim), _comm_group(std::move(comm_group)) {
    HT_ASSERT(_comm_group.num_devices() >= 2)
      << "AllGather requires two or more devices. Got " << _comm_group;
  }

  uint64_t op_indicator() const noexcept override {
    return ALL_GATHER_OP;
  }

  int32_t get_gather_dim() const {
    return _gather_dim;
  }

 protected:
  bool DoMapToParallelDevices(Operator& op,
                              const DeviceGroupUnion& pg_union) const override;

  bool DoInstantiate(Operator& op, const Device& placement,
                     StreamIndex stream_index) const override;

  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;  

  NDArrayList DoCompute(Operator& op, const NDArrayList& inputs,
                        RuntimeContext& runtime_ctx) const override;
  
  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;

 protected:
  DeviceGroup _comm_group;
  static NDArray _buffer_for_allgather; // workaround for allgather activation when use sp
  int32_t _gather_dim;
};

Tensor MakeAllGatherOp(Tensor input, const DeviceGroup& comm_group, int32_t gather_dim = 0,
                       OpMeta op_meta = OpMeta());

class ReduceScatterOpImpl final : public OpInterface {
 public:
  ReduceScatterOpImpl(const DeviceGroup& comm_group, int32_t scatter_dim = 0, ReductionType red_type = kSUM,
                      bool inplace = false)
  : OpInterface(quote(ReduceScatterOp)), _scatter_dim(scatter_dim),
    _comm_group(std::move(comm_group)), _red_type(red_type), _inplace(inplace) {
    HT_ASSERT(_comm_group.num_devices() >= 2)
      << "ReduceScatter requires two or more devices. Got " << _comm_group;            
  }

  inline bool inplace() const {
    return _inplace;
  }

  inline uint64_t inplace_pos() const {
    return 0;
  }

  inline bool inplace_at(size_t input_position) const override {
    return inplace() && input_position == inplace_pos();
  }

  inline uint64_t op_indicator() const noexcept override {
    return _inplace ? REDUCE_SCATTER_OP | INPLACE_OP : REDUCE_SCATTER_OP;
  }

  int32_t get_scatter_dim() const {
    return _scatter_dim;
  }

  ReductionType reduction_type() const {
    return _red_type;
  }

  const DeviceGroup& comm_group() const {
    return _comm_group;
  } 

 protected:
  bool DoMapToParallelDevices(Operator& op,
                              const DeviceGroupUnion& pg_union) const override;

  bool DoInstantiate(Operator& op, const Device& placement,
                     StreamIndex stream_index) const override;
                                                    
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;  

  NDArrayList DoCompute(Operator& op,
                        const NDArrayList& inputs,
                        RuntimeContext& ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const override;

  bool _inplace;

 protected:
  DeviceGroup _comm_group;
  int32_t _scatter_dim;
  ReductionType _red_type{kNONE};
};

Tensor MakeReduceScatterOp(Tensor input, const DeviceGroup& comm_group, int32_t scatter_dim = 0,
                           bool inplace = false, OpMeta op_meta = OpMeta());

Tensor MakeReduceScatterOp(Tensor input, DeviceGroup comm_group, ReductionType red_type, int32_t scatter_dim = 0, 
                           bool inplace = false, OpMeta op_meta = OpMeta());

class SplitAllGatherOpImpl final : public OpInterface {
 public:
  SplitAllGatherOpImpl(std::vector<DeviceGroupList> comm_groups_list, size_t split_num, bool inplace = false)
  : OpInterface(quote(SplitAllGatherOp)), _comm_groups_list(comm_groups_list), _split_num(split_num), _inplace(inplace) {
    for (auto& comm_groups : _comm_groups_list) {
      for (auto& comm_group : comm_groups) {
        HT_ASSERT(comm_group.num_devices() >= 2)
          << "SplitAllGather requires two or more comm devices in each comm group. Got " << _comm_groups_list;
      }
    }
  }

  inline uint64_t op_indicator() const noexcept override {
    return _inplace ? SPLIT_ALL_GATHER_OP | INPLACE_OP : SPLIT_ALL_GATHER_OP;
  }

  inline size_t split_num() const {
    return _split_num;
  }

  inline bool inplace() const {
    return _inplace;
  }

 protected:
  bool DoMapToParallelDevices(Operator& op,
                              const DeviceGroupUnion& pg_union) const override;

  bool DoInstantiate(Operator& op, const Device& placement,
                     StreamIndex stream_index) const override;

  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;

  NDArrayList DoCompute(Operator& op,
                        const NDArrayList& inputs,
                        RuntimeContext& ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const {};
  
  bool _inplace{false};

 public:
  const std::vector<DeviceGroupList>& comm_groups_list() const {
    return _comm_groups_list;
  }

 protected:
  std::vector<DeviceGroupList> _comm_groups_list;
  size_t _split_num;
};

Tensor MakeSplitAllGatherOp(Tensor input, std::vector<DeviceGroupList> comm_groups_list, size_t split_num,
                            bool inplace = false, OpMeta op_meta = OpMeta());

class SplitAllReduceOpImpl final : public OpInterface {
 public:
  SplitAllReduceOpImpl(std::vector<DeviceGroupList> comm_groups_list, size_t split_num, ReductionType red_type = kSUM, bool inplace = false)
  : OpInterface(quote(SplitAllReduceOp)), _comm_groups_list(comm_groups_list), _split_num(split_num), _red_type(red_type), _inplace(inplace) {
    for (auto& comm_groups : _comm_groups_list) {
      for (auto& comm_group : comm_groups) {
        HT_ASSERT(comm_group.num_devices() >= 2)
          << "SplitAllReduce requires two or more comm devices in each comm group. Got " << _comm_groups_list;
      }
    }
  }

  inline uint64_t op_indicator() const noexcept override {
    return _inplace ? SPLIT_ALL_REDUCE_OP | INPLACE_OP : SPLIT_ALL_REDUCE_OP;
  }

  inline size_t split_num() const {
    return _split_num;
  }

  inline bool inplace() const {
    return _inplace;
  }

  ReductionType reduction_type() const {
    return _red_type;
  }

 protected:
  bool DoMapToParallelDevices(Operator& op,
                              const DeviceGroupUnion& pg_union) const override;

  bool DoInstantiate(Operator& op, const Device& placement,
                     StreamIndex stream_index) const override;

  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;

  NDArrayList DoCompute(Operator& op,
                        const NDArrayList& inputs,
                        RuntimeContext& ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const {};
  
  bool _inplace{false};

 public:
  const std::vector<DeviceGroupList>& comm_groups_list() const {
    return _comm_groups_list;
  }

 protected:
  std::vector<DeviceGroupList> _comm_groups_list;
  size_t _split_num;
  ReductionType _red_type{kNONE};
};

Tensor MakeSplitAllReduceOp(Tensor input, std::vector<DeviceGroupList> comm_groups_list, size_t split_num,
                            bool inplace = false, OpMeta op_meta = OpMeta());

Tensor MakeSplitAllReduceOp(Tensor input, std::vector<DeviceGroupList> comm_groups_list, size_t split_num, ReductionType red_type, 
                            bool inplace = false, OpMeta op_meta = OpMeta());

class SplitReduceScatterOpImpl final : public OpInterface {
 public:
  SplitReduceScatterOpImpl(std::vector<DeviceGroupList> comm_groups_list, size_t split_num, ReductionType red_type = kSUM, bool inplace = false)
  : OpInterface(quote(SplitReduceScatterOp)), _comm_groups_list(comm_groups_list), _split_num(split_num), _red_type(red_type), _inplace(inplace) {
    for (auto& comm_groups : _comm_groups_list) {
      for (auto& comm_group : comm_groups) {
        HT_ASSERT(comm_group.num_devices() >= 2)
          << "SplitReduceScatter requires two or more comm devices in each comm group. Got " << _comm_groups_list;
      }
    }
  }

  inline uint64_t op_indicator() const noexcept override {
    return _inplace ? SPLIT_REDUCE_SCATTER_OP | INPLACE_OP : SPLIT_REDUCE_SCATTER_OP;
  }

  inline size_t split_num() const {
    return _split_num;
  }

  inline bool inplace() const {
    return _inplace;
  }

  ReductionType reduction_type() const {
    return _red_type;
  }

 protected:
  bool DoMapToParallelDevices(Operator& op,
                              const DeviceGroupUnion& pg_union) const override;

  bool DoInstantiate(Operator& op, const Device& placement,
                     StreamIndex stream_index) const override;

  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs, const InstantiationContext& inst_ctx) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;

  NDArrayList DoCompute(Operator& op,
                        const NDArrayList& inputs,
                        RuntimeContext& ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& ctx) const {};
  
  bool _inplace{false};

 public:
  const std::vector<DeviceGroupList>& comm_groups_list() const {
    return _comm_groups_list;
  }

 protected:
  std::vector<DeviceGroupList> _comm_groups_list;
  size_t _split_num;
  ReductionType _red_type{kNONE};
};

Tensor MakeSplitReduceScatterOp(Tensor input, std::vector<DeviceGroupList> comm_groups_list, size_t split_num,
                                bool inplace = false, OpMeta op_meta = OpMeta());

Tensor MakeSplitReduceScatterOp(Tensor input, std::vector<DeviceGroupList> comm_groups_list, size_t split_num, ReductionType red_type, 
                                bool inplace = false, OpMeta op_meta = OpMeta());



}
}